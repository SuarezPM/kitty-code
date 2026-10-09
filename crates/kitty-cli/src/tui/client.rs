//! W2 TUI: `ChatClient` que añade streaming, gate de permisos y contabilidad
//! de tokens/coste por encima de un [`OllamaClient`] sin tocar el harness.
//!
//! Diseño: el agent decide Cuándo llamar a un tool; este cliente decide — ANTES
//! de devolver la respuesta al agent — si el tool se ejecuta. Interceptamos los
//! `tool_calls` de la respuesta y:
//!
//! - herramientas de solo lectura (Read/Glob/Grep) → auto-permitidas;
//! - el resto → esperan veredicto del usuario (allow / deny, con opción
//!   "siempre" que se registra en la [`PermissionPolicy`] de la sesión).
//!
//! Un tool denegado se traduce en un `Message::tool_result` sintético que el
//! agent reinyecta al modelo, de modo que el modelo sabe que el usuario dijo
//! que no y puede reformular. El Guard/ToolRegistry del harness sigue siendo la
//! única autoridad que EJECUTA; aquí solo decidimos si llega a ejecutarse.
//!
//! El streaming usa `OllamaClient::build_chat_request` (num_ctx fijo, think,
//! truncate:false) y luego POSTea con `stream: Some(true)` leyendo NDJSON chunk
//! a chunk con reqwest. Cada delta de tokens se emprime en un [`mpsc::Sender`]
//! de [`TuiEvent`] que el event loop de la TUI drena; al final se ensambla un
//! [`ChatResponse`] coherente para que el agent continúe igual que antes.

use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use kitty_runtime::client::{
    ChatResponse, Message, OllamaClient, RequestOptions, ToolCall, ToolCallFunction,
};
use kitty_runtime::{ChatClient, ToolDefinition};
use tokio::sync::{mpsc, oneshot};

/// Eventos que el cliente empuja a la TUI mientras el agent corre.
#[derive(Debug, Clone)]
pub enum TuiEvent {
    /// Un delta de texto del modelo (streaming).
    Token(String),
    /// Un delta de "thinking" (modelos con razonamiento).
    Thinking(String),
    /// Se pidió una herramienta que requiere permiso del usuario.
    PermissionRequest {
        id: u64,
        tool: String,
        /// Resumen legible de los argumentos para el modal de permiso.
        args_preview: String,
        /// Diff unificado precalculado (vacío si no aplica).
        diff: String,
    },
    /// El usuario respondió a una petición de permiso.
    PermissionResolved { tool: String, allowed: bool },
    /// El backend terminó de listar modelos (`/api/tags`).
    ///
    /// `models` puede venir vacío si el servidor no tiene ninguno cargado;
    /// eso NO es error — el picker lo muestra como "sin modelos" y ofrece el
    /// fallback de entrada manual.
    ModelsLoaded(Vec<String>),
    /// El listado de modelos falló (servidor caído, timeout, JSON inválido).
    /// El picker pasa a estado de error y también ofrece entrada manual.
    ModelsFailed(String),
    /// El usuario permitió una tool: el harness va a ejecutarla.
    ///
    /// Es un arranque, no un resultado; no hay campo `success` porque nada
    /// se ha ejecutado todavía.
    ToolStart {
        name: String,
        /// Nombre del archivo o comando sobre el que actúa la tool.
        target: String,
    },
    /// Contabilidad de una ronda LLM completa.
    Usage {
        prompt_tokens: i64,
        eval_tokens: i64,
        /// Coste de ESTA ronda en USD. El cliente es el único que conoce la
        /// tarifa real del backend, así que la calcula aquí (0.0 = local).
        cost_usd: f64,
    },
    /// Texto final del agent (equivalente a `AgentEvent::Done`).
    Done(String),
    /// Error del agent.
    Error(String),
}

/// Política de permisos de la sesión: qué herramientas están allow/deny
/// "para siempre" (hasta que se reinicie la sesión).
#[derive(Debug, Clone, Default)]
pub struct PermissionPolicy {
    /// Tools siempre permitidas (además de las solo-lectura).
    pub allow: HashSet<String>,
    /// Tools siempre denegadas. Tiene prioridad sobre `allow`.
    pub deny: HashSet<String>,
}

impl PermissionPolicy {
    /// Las herramientas de solo lectura nunca piden permiso: no mutan estado.
    pub const READ_ONLY: &'static [&'static str] = &["Read", "Glob", "Grep"];

    pub fn is_read_only(tool: &str) -> bool {
        Self::READ_ONLY.contains(&tool)
    }

    /// ¿Esta herramienta necesita veredicto del usuario ahora mismo?
    pub fn needs_prompt(&self, tool: &str) -> bool {
        if Self::is_read_only(tool) {
            return false;
        }
        if self.deny.contains(tool) {
            return false; // ya denegada "para siempre": no volver a preguntar
        }
        !self.allow.contains(tool)
    }

    /// Aplicar un veredicto del usuario a la política de sesión.
    pub fn apply(&mut self, tool: &str, allowed: bool, always: bool) {
        if !always {
            return;
        }
        if allowed {
            self.allow.insert(tool.to_string());
            self.deny.remove(tool);
        } else {
            self.deny.insert(tool.to_string());
            self.allow.remove(tool);
        }
    }
}

/// Estado compartido entre el cliente (worker) y el event loop (UI) para
/// resolver permisos de forma síncrona desde el lado UI.
#[derive(Default)]
pub struct PermissionBroker {
    /// Cola de peticiones pendientes: id → (tool, args_preview, diff).
    pending: Mutex<Vec<PermissionWaiter>>,
    next_id: Mutex<u64>,
}

/// Veredicto del usuario sobre una herramienta.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionVerdict {
    /// Permitir esta vez (no toca la política de sesión).
    AllowOnce,
    /// Permitir esta vez y siempre para el resto de la sesión.
    AllowAlways,
    /// Denegar esta vez (no toca la política de sesión).
    DenyOnce,
    /// Denegar esta vez y siempre para el resto de la sesión.
    DenyAlways,
}

impl PermissionVerdict {
    pub fn allowed(self) -> bool {
        matches!(
            self,
            PermissionVerdict::AllowOnce | PermissionVerdict::AllowAlways
        )
    }
    pub fn always(self) -> bool {
        matches!(
            self,
            PermissionVerdict::AllowAlways | PermissionVerdict::DenyAlways
        )
    }
}

/// Un permiso esperando veredicto. El preview y el diff viajan al modal por
/// [`TuiEvent::PermissionRequest`]; aquí solo guardamos el id (para
/// correlación con `resolve`) y el canal de respuesta.
pub struct PermissionWaiter {
    pub id: u64,
    tx: oneshot::Sender<PermissionVerdict>,
}

impl PermissionBroker {
    /// Registrar una petición y devolver el canal de su veredicto.
    ///
    /// El preview y el diff no se guardan aquí — viajan al modal por
    /// [`TuiEvent::PermissionRequest`]; el id correlaciona con `resolve`.
    pub fn request(&self) -> (u64, oneshot::Receiver<PermissionVerdict>) {
        let mut idg = self.next_id.lock().expect("perm id lock");
        *idg += 1;
        let id = *idg;
        let (tx, rx) = oneshot::channel();
        self.pending
            .lock()
            .expect("perm lock")
            .push(PermissionWaiter { id, tx });
        (id, rx)
    }

    /// Resolver una petición por id (llamado desde el event loop UI).
    pub fn resolve(&self, id: u64, verdict: PermissionVerdict) -> bool {
        let mut g = self.pending.lock().expect("perm lock");
        if let Some(pos) = g.iter().position(|w| w.id == id) {
            let w = g.remove(pos);
            let _ = w.tx.send(verdict);
            true
        } else {
            false
        }
    }
}

/// Contabilidad de coste de la sesión.
#[derive(Debug, Clone, Default)]
pub struct UsageTotals {
    pub prompt_tokens: i64,
    pub eval_tokens: i64,
    /// Coste acumulado en USD. La tarifa la decide el backend (0.0 = local).
    pub cost_usd: f64,
    pub turns: u64,
}

impl UsageTotals {
    pub fn accumulate(&mut self, prompt: i64, eval: i64, cost_usd: f64) {
        self.prompt_tokens += prompt;
        self.eval_tokens += eval;
        self.cost_usd += cost_usd;
        self.turns += 1;
    }
}

/// Fuente de modelos instalados en el backend.
///
/// El event loop de la TUI solo necesita poder pedir la lista; la
/// implementación concreta (HTTP contra `/api/tags`) vive en [`TuiClient`].
/// Separarlo por trait permite que los tests inyecten un stub y verifiquen el
/// estado del selector de modelo sin red.
pub trait ModelListing: Send + Sync {
    /// Lista los modelos instalados, ordenados alfabéticamente.
    ///
    /// Una lista vacía es un resultado válido (servidor vivo sin modelos);
    /// los fallos se devuelven como `Err` y el picker los muestra.
    fn list_models(&self) -> impl std::future::Future<Output = Result<Vec<String>, String>> + Send;
}

/// Resultado del arranque de una petición de modelos, para que el worker
/// distinga "vacío pero OK" de "fallo".
pub(crate) enum ModelsOutcome {
    Ok(Vec<String>),
    Err(String),
}

impl ModelListing for TuiClient {
    async fn list_models(&self) -> Result<Vec<String>, String> {
        let url = format!("{}/api/tags", self.inner.base_url());
        let resp = self
            .inner
            .http()
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("no se pudo contactar con {}: {e}", self.inner.base_url()))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("/api/tags devolvió {status}: {body}"));
        }

        let body = resp
            .text()
            .await
            .map_err(|e| format!("no se pudo leer la respuesta de /api/tags: {e}"))?;

        parse_models_payload(&body)
    }
}

/// Parsea el body de `/api/tags` y extrae los nombres de modelo.
///
/// Ollama responde `{"models":[{"name":"qwen3:8b", ...}, ...]}`. Un payload
/// sin `models` o con entradas sin `name` se trata como error honesto, no se
/// finge una lista vacía.
pub(crate) fn parse_models_payload(body: &str) -> Result<Vec<String>, String> {
    let v: serde_json::Value =
        serde_json::from_str(body).map_err(|e| format!("JSON inválido de /api/tags: {e}"))?;

    let arr = v
        .get("models")
        .and_then(|m| m.as_array())
        .ok_or_else(|| "la respuesta de /api/tags no contiene el campo `models`".to_string())?;

    let mut models: Vec<String> = arr
        .iter()
        .filter_map(|m| m.get("name").and_then(|n| n.as_str()))
        .filter(|n| !n.is_empty())
        .map(|n| n.to_string())
        .collect();

    if models.is_empty() && !arr.is_empty() {
        return Err(
            "las entradas de /api/tags no traen campo `name` — ¿respuesta de otro backend?"
                .to_string(),
        );
    }

    models.sort();
    models.dedup();
    Ok(models)
}

/// Cliente TUI: envuelve un [`OllamaClient`] añadiendo streaming + permisos.
pub struct TuiClient {
    inner: OllamaClient,
    events: mpsc::Sender<TuiEvent>,
    broker: Arc<PermissionBroker>,
    policy: Arc<Mutex<PermissionPolicy>>,
    /// Precio por millón de tokens (0.0 = local).
    price_per_mtok: f64,
}

impl TuiClient {
    pub fn new(
        inner: OllamaClient,
        events: mpsc::Sender<TuiEvent>,
        broker: Arc<PermissionBroker>,
        policy: Arc<Mutex<PermissionPolicy>>,
        price_per_mtok: f64,
    ) -> Self {
        Self {
            inner,
            events,
            broker,
            policy,
            price_per_mtok,
        }
    }

    fn emit(&self, ev: TuiEvent) {
        // try_send: si el buffer está lleno preferimos perder un delta de
        // streaming a bloquear el worker (la TUI está para otra cosa).
        let _ = self.events.try_send(ev);
    }

    /// Construye la petición de streaming y la POSTea, leyendo NDJSON.
    async fn stream_chat(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        options: Option<RequestOptions>,
    ) -> kitty_runtime::Result<ChatResponse> {
        use kitty_runtime::client::ChatRequest;

        let mut req: ChatRequest = self
            .inner
            .build_chat_request(model, messages, tools, options);
        req.stream = Some(true);

        let url = format!("{}/api/chat", self.inner.base_url());
        let resp = self
            .inner
            .http()
            .post(&url)
            .json(&req)
            .send()
            .await
            .map_err(kitty_runtime::RuntimeError::Http)?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(kitty_runtime::RuntimeError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        // Acumuladores NDJSON → ChatResponse final.
        let mut content = String::new();
        let mut thinking = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut eval_count: Option<i64> = None;
        let mut prompt_eval_count: Option<i64> = None;

        let mut stream = resp;
        let mut buf = String::new();
        loop {
            let chunk = match stream.chunk().await {
                Ok(Some(c)) => c,
                Ok(None) => break,
                Err(e) => return Err(kitty_runtime::RuntimeError::Http(e)),
            };
            buf.push_str(&String::from_utf8_lossy(&chunk));
            // NDJSON: cada línea es un objeto JSON completo.
            while let Some(pos) = buf.find('\n') {
                let line: String = buf.drain(..=pos).collect();
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
                    continue;
                };
                if let Some(msg) = v.get("message") {
                    if let Some(t) = msg.get("content").and_then(|c| c.as_str()) {
                        if !t.is_empty() {
                            content.push_str(t);
                            self.emit(TuiEvent::Token(t.to_string()));
                        }
                    }
                    if let Some(t) = msg.get("thinking").and_then(|c| c.as_str()) {
                        if !t.is_empty() {
                            thinking.push_str(t);
                            self.emit(TuiEvent::Thinking(t.to_string()));
                        }
                    }
                    if let Some(arr) = msg.get("tool_calls").and_then(|a| a.as_array()) {
                        for tc in arr {
                            let name = tc
                                .get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(|n| n.as_str())
                                .unwrap_or("")
                                .to_string();
                            let args = tc
                                .get("function")
                                .and_then(|f| f.get("arguments"))
                                .cloned()
                                .unwrap_or(serde_json::Value::Null);
                            if !name.is_empty() {
                                tool_calls.push(ToolCall {
                                    id: None,
                                    function: ToolCallFunction {
                                        name,
                                        arguments: args,
                                    },
                                });
                            }
                        }
                    }
                }
                if let Some(n) = v.get("eval_count").and_then(|x| x.as_i64()) {
                    eval_count = Some(n);
                }
                if let Some(n) = v.get("prompt_eval_count").and_then(|x| x.as_i64()) {
                    prompt_eval_count = Some(n);
                }
            }
        }

        // GATE DE PERMISOS, fase 1: decidir en un scope CORTO qué tools son
        // auto (solo-lectura / allow-list / deny-list) y cuáles necesitan
        // veredicto. El MutexGuard cae aquí, ANTES de cualquier .await, para
        // que el future siga siendo Send.
        let mut allowed_calls: Vec<ToolCall> = Vec::new();
        let mut denied_notes: Vec<String> = Vec::new();
        let mut need_verdict: Vec<(String, String, String, ToolCall)> = Vec::new();
        {
            let pol = self.policy.lock().expect("policy lock");
            for tc in &tool_calls {
                let name = tc.function.name.as_str();
                if pol.deny.contains(name) {
                    denied_notes.push(name.to_string());
                } else if pol.needs_prompt(name) {
                    need_verdict.push((
                        name.to_string(),
                        args_preview(name, &tc.function.arguments),
                        compute_diff(name, &tc.function.arguments),
                        tc.clone(),
                    ));
                } else {
                    allowed_calls.push(tc.clone());
                }
            }
        } // <- guard liberado aquí

        // Fase 2: esperar veredictos SIN retener ningún lock.
        for (name, preview, diff, tc) in need_verdict {
            let (req_id, rx) = self.broker.request();
            self.emit(TuiEvent::PermissionRequest {
                id: req_id,
                tool: name.clone(),
                args_preview: preview.clone(),
                diff,
            });
            // Timeout generoso (10 min) para no colgar el worker si la UI se
            // cierra sin responder.
            let verdict = tokio::time::timeout(std::time::Duration::from_secs(600), rx)
                .await
                .ok()
                .and_then(|r| r.ok())
                .unwrap_or(PermissionVerdict::DenyOnce);
            {
                let mut pol = self.policy.lock().expect("policy lock");
                pol.apply(&name, verdict.allowed(), verdict.always());
            }
            self.emit(TuiEvent::PermissionResolved {
                tool: name.clone(),
                allowed: verdict.allowed(),
            });
            if verdict.allowed() {
                // El preview ya fue al modal en PermissionRequest; aquí la
                // consola registra el arranque con el mismo target.
                self.emit(TuiEvent::ToolStart {
                    name: name.clone(),
                    target: preview,
                });
                allowed_calls.push(tc);
            } else {
                denied_notes.push(name);
            }
        }

        // Si todos los tool_calls fueron denegados, devolvemos una respuesta
        // de texto para que el agent cierre el turno sin ejecutar nada.
        let final_message = if allowed_calls.is_empty() && !tool_calls.is_empty() {
            let notes = denied_notes.join(", ");
            Message {
                role: "assistant".to_string(),
                content: Some(format!(
                    "No se ejecutaron herramientas. Herramientas denegadas por el \
                     usuario: {notes}. Indica al usuario qué pediste y pregunta cómo \
                     proceder."
                )),
                thinking: if thinking.is_empty() {
                    None
                } else {
                    Some(thinking)
                },
                tool_calls: None,
                tool_call_id: None,
            }
        } else {
            Message {
                role: "assistant".to_string(),
                content: if content.is_empty() {
                    None
                } else {
                    Some(content)
                },
                thinking: if thinking.is_empty() {
                    None
                } else {
                    Some(thinking)
                },
                tool_calls: if allowed_calls.is_empty() {
                    None
                } else {
                    Some(allowed_calls)
                },
                tool_call_id: None,
            }
        };

        let prompt = prompt_eval_count.unwrap_or(0);
        let eval = eval_count.unwrap_or(0);
        // La tarifa se aplica SOLO al eval (salida); el prompt se contabiliza
        // aparte porque muchos backends lo facturan a un precio distinto.
        let cost_usd = eval as f64 / 1_000_000.0 * self.price_per_mtok;
        self.emit(TuiEvent::Usage {
            prompt_tokens: prompt,
            eval_tokens: eval,
            cost_usd,
        });

        Ok(ChatResponse {
            model: model.to_string(),
            message: final_message,
            done: true,
            eval_count,
            eval_duration: None,
            prompt_eval_count,
        })
    }
}

/// Resumen legible de argumentos para el modal de permiso.
fn args_preview(tool: &str, args: &serde_json::Value) -> String {
    match tool {
        "Read" | "Glob" | "Grep" => args
            .get("path")
            .or_else(|| args.get("pattern"))
            .and_then(|v| v.as_str())
            .unwrap_or("(sin args)")
            .to_string(),
        "Edit" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("?");
            let old = args
                .get("old_string")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            format!("{path}  (reemplazar {} chars)", old.chars().count())
        }
        "Write" => {
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("?");
            let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
            format!("{path}  ({} chars nuevos)", content.chars().count())
        }
        "Bash" => args
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("(sin comando)")
            .to_string(),
        _ => args.to_string(),
    }
}

/// Diff unificado precalculado para Edit/Write (vacío para el resto).
fn compute_diff(tool: &str, args: &serde_json::Value) -> String {
    match tool {
        "Edit" => {
            let old = args
                .get("old_string")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let new = args
                .get("new_string")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let mut out = String::new();
            for l in old.lines() {
                out.push('-');
                out.push_str(l);
                out.push('\n');
            }
            for l in new.lines() {
                out.push('+');
                out.push_str(l);
                out.push('\n');
            }
            out
        }
        "Write" => {
            let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
            content
                .lines()
                .map(|l| format!("+{l}"))
                .collect::<Vec<_>>()
                .join("\n")
        }
        _ => String::new(),
    }
}

impl ChatClient for TuiClient {
    fn chat<'a>(
        &'a self,
        model: &'a str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn Future<Output = kitty_runtime::Result<ChatResponse>> + Send + 'a>> {
        Box::pin(async move { self.stream_chat(model, messages, tools, None).await })
    }

    fn chat_with_options<'a>(
        &'a self,
        model: &'a str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        options: Option<RequestOptions>,
    ) -> Pin<Box<dyn Future<Output = kitty_runtime::Result<ChatResponse>> + Send + 'a>> {
        Box::pin(async move { self.stream_chat(model, messages, tools, options).await })
    }

    fn count_tokens<'a>(
        &'a self,
        model: &'a str,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = kitty_runtime::Result<usize>> + Send + 'a>> {
        Box::pin(OllamaClient::count_tokens(&self.inner, model, text))
    }

    fn health_check<'a>(
        &'a self,
    ) -> Pin<Box<dyn Future<Output = kitty_runtime::Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let ok = OllamaClient::health_check(&self.inner).await?;
            if ok {
                Ok(())
            } else {
                Err(kitty_runtime::RuntimeError::ConnectionRefused(
                    "ollama server not reachable".into(),
                ))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_only_tools_never_prompt() {
        let p = PermissionPolicy::default();
        assert!(!p.needs_prompt("Read"));
        assert!(!p.needs_prompt("Glob"));
        assert!(!p.needs_prompt("Grep"));
        assert!(p.needs_prompt("Edit"));
        assert!(p.needs_prompt("Write"));
        assert!(p.needs_prompt("Bash"));
    }

    #[test]
    fn allow_always_removes_prompt_but_deny_wins() {
        let mut p = PermissionPolicy::default();
        p.apply("Bash", true, true);
        assert!(!p.needs_prompt("Bash"));
        // deny posterior pisa a allow
        p.apply("Bash", false, true);
        assert!(p.deny.contains("Bash"));
        assert!(!p.allow.contains("Bash"));
        // deny "para siempre" no vuelve a preguntar
        assert!(!p.needs_prompt("Bash"));
    }

    #[test]
    fn non_always_verdict_does_not_persist() {
        let mut p = PermissionPolicy::default();
        p.apply("Edit", true, false);
        assert!(
            p.needs_prompt("Edit"),
            "veredicto one-shot no debe persistir"
        );
    }

    #[test]
    fn broker_request_and_resolve_roundtrip() {
        let broker = PermissionBroker::default();
        let (id, rx) = broker.request();
        assert!(broker.resolve(id, PermissionVerdict::AllowOnce));
        let v = rx.blocking_recv().unwrap();
        assert!(v.allowed(), "AllowOnce debe permitir");
        assert!(!v.always(), "AllowOnce no debe persistir");
        // resolver dos veces: la segunda no encuentra la petición
        assert!(!broker.resolve(id, PermissionVerdict::DenyOnce));
    }

    #[test]
    fn verdict_always_flags_only_on_always_variants() {
        assert!(PermissionVerdict::AllowAlways.allowed());
        assert!(PermissionVerdict::AllowAlways.always());
        assert!(PermissionVerdict::DenyAlways.always());
        assert!(!PermissionVerdict::DenyAlways.allowed());
        assert!(!PermissionVerdict::DenyOnce.always());
    }

    #[test]
    fn usage_totals_accumulate_and_cost() {
        let mut u = UsageTotals::default();
        u.accumulate(1000, 500, 0.0);
        u.accumulate(2000, 800, 0.001_2);
        assert_eq!(u.prompt_tokens, 3000);
        assert_eq!(u.eval_tokens, 1300);
        assert_eq!(u.turns, 2);
        assert!((u.cost_usd - 0.001_2).abs() < 1e-12);
    }

    #[test]
    fn args_preview_covers_main_tools() {
        let edit = serde_json::json!({"path":"src/main.rs","old_string":"let a = 1;","new_string":"let a = 2;"});
        let p = args_preview("Edit", &edit);
        assert!(p.contains("src/main.rs"));
        assert!(p.contains("10 chars"));

        let bash = serde_json::json!({"command":"ls -la"});
        assert_eq!(args_preview("Bash", &bash), "ls -la");

        let write = serde_json::json!({"path":"a.py","content":"print(1)"});
        assert!(args_preview("Write", &write).contains("8 chars nuevos"));
    }

    #[test]
    fn compute_diff_emits_unified_markers() {
        let edit = serde_json::json!({"old_string":"alpha\nbeta","new_string":"alpha\ngamma"});
        let d = compute_diff("Edit", &edit);
        assert!(d.contains("-beta"));
        assert!(d.contains("+gamma"));

        let write = serde_json::json!({"content":"line1\nline2"});
        let d = compute_diff("Write", &write);
        assert!(d.starts_with("+line1"));

        assert_eq!(compute_diff("Bash", &serde_json::json!({})), "");
    }

    #[test]
    fn parse_models_payload_extracts_sorted_unique_names() {
        let body = r#"{"models":[
            {"name":"zeta:7b","size":1},
            {"name":"alpha:3b","size":2},
            {"name":"zeta:7b","size":3}
        ]}"#;
        let models = parse_models_payload(body).unwrap();
        assert_eq!(models, vec!["alpha:3b", "zeta:7b"]);
    }

    #[test]
    fn parse_models_payload_empty_list_is_valid() {
        // Servidor vivo sin modelos: no es un error, es una lista vacía.
        assert_eq!(
            parse_models_payload(r#"{"models":[]}"#).unwrap(),
            vec![] as Vec<String>
        );
    }

    #[test]
    fn parse_models_payload_rejects_malformed_json() {
        let err = parse_models_payload("{ no es json").unwrap_err();
        assert!(err.contains("JSON inválido"), "motivo honesto: {err}");
    }

    #[test]
    fn parse_models_payload_rejects_missing_models_field() {
        let err = parse_models_payload(r#"{"other":1}"#).unwrap_err();
        assert!(
            err.contains("`models`"),
            "menciona el campo que falta: {err}"
        );
    }

    #[test]
    fn parse_models_payload_rejects_entries_without_name() {
        // Un backend ajeno que responda otra forma no debe fingir lista vacía.
        let err = parse_models_payload(r#"{"models":[{"size":1},{"size":2}]}"#).unwrap_err();
        assert!(err.contains("`name`"), "menciona el campo que falta: {err}");
    }

    #[test]
    fn parse_models_payload_skips_blank_names() {
        let body = r#"{"models":[{"name":""},{"name":"qwen3:8b"}]}"#;
        assert_eq!(parse_models_payload(body).unwrap(), vec!["qwen3:8b"]);
    }

    /// Stub de [`ModelListing`] para tests: fija el resultado y comprueba que
    /// el event loop nunca ve la implementación HTTP concreta.
    struct StubModels(Result<Vec<String>, String>);

    impl ModelListing for StubModels {
        async fn list_models(&self) -> Result<Vec<String>, String> {
            match &self.0 {
                Ok(v) => Ok(v.clone()),
                Err(e) => Err(e.clone()),
            }
        }
    }

    #[tokio::test]
    async fn stub_listing_ok_feeds_app_ready() {
        let stub = StubModels(Ok(vec!["qwen3:8b".into(), "llama3:8b".into()]));
        let models = stub.list_models().await.unwrap();
        let mut app = super::super::App::new("qwen3:8b", 100);
        app.handle_tui_event(TuiEvent::ModelsLoaded(models));
        assert_eq!(app.model_picker, super::super::ModelPicker::Ready);
        assert_eq!(app.available_models.len(), 2);
    }

    #[tokio::test]
    async fn stub_listing_err_feeds_app_error() {
        let stub = StubModels(Err("timeout".into()));
        let err = stub.list_models().await.unwrap_err();
        let mut app = super::super::App::new("m", 100);
        app.open_model_picker();
        app.handle_tui_event(TuiEvent::ModelsFailed(err));
        assert!(matches!(
            &app.model_picker,
            super::super::ModelPicker::Error(e) if e == "timeout"
        ));
    }
}

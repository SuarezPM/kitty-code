//! Embedded kitty-infer GGUF backend (`ChatClient`).
//!
//! In-process inference via the sibling `kitty-infer` repo
//! (`mistralrs::GgufModelBuilder`): `GgufLoadPlan` pieces → single-resident
//! registry (`global_registry`, evict-before-load + OOM retry) → generate.
//! Selected with `KITTY_BACKEND=kitty-infer` + `KITTY_INFER_MODEL=<path.gguf>`.
//!
//! Policy (lives in kitty-infer `serving::infer_policy`, shared with the
//! daemon path):
//! - dtype default F16 (`Auto`) + F32 override for `qwen2.5-coder*` (isolated
//!   F16-CUDA NaN on sm_75, humo F1) + `KITTY_INFER_F32_PREFIXES` extras;
//! - MiniCPM5 `tokenizer.json` sidecar-or-honest-error
//!   (`KITTY_INFER_TOKENIZER_DIR`, reference layout
//!   `<dir>/MiniCPM5-2B/tokenizer.json`).
//!
//! Explicit backend ⇒ explicit errors: a missing/unloadable GGUF fails
//! honestly, it NEVER silently falls back to Ollama (that would hide the
//! operator's choice). No auth in-process (nothing to authenticate).

use std::path::PathBuf;

use crate::error::{Result, RuntimeError};

#[cfg(feature = "kitty-infer")]
use std::future::Future;
#[cfg(feature = "kitty-infer")]
use std::pin::Pin;
#[cfg(feature = "kitty-infer")]
use std::sync::Arc;
#[cfg(feature = "kitty-infer")]
use crate::client::{ChatClient, ChatResponse, Message};
#[cfg(feature = "kitty-infer")]
use crate::tools::ToolDefinition;

#[cfg(feature = "kitty-infer")]
use mistralrs::{
    serving as ki,
    GgufModelBuilder, Model, PagedAttentionMetaBuilder, RequestBuilder, TextMessageRole,
};

/// Env switch selecting this backend in `kitty-cli`.
pub const ENV_BACKEND: &str = "KITTY_BACKEND";
/// Value of [`ENV_BACKEND`] that selects the embedded backend.
pub const BACKEND_KITTY_INFER: &str = "kitty-infer";
/// GGUF path for the embedded backend.
pub const ENV_MODEL: &str = "KITTY_INFER_MODEL";
/// Sidecar dir holding `<...>/MiniCPM5-2B/tokenizer.json`.
pub const ENV_TOKENIZER_DIR: &str = "KITTY_INFER_TOKENIZER_DIR";
/// Context length override (default 4096 — honest for 8GB consumer GPUs).
pub const ENV_CTX: &str = "KITTY_INFER_CTX";
/// Extra comma-separated F32 filename prefixes.
pub const ENV_F32_PREFIXES: &str = "KITTY_INFER_F32_PREFIXES";
/// GPU layer count override (e.g. `12` keeps 12 layers on cuda[0], spilling
/// the rest to CPU). Unset = automatic mapping. Needed for >8GB models
/// (K2 9.5GB) where auto packs too tight to survive prefill transients.
pub const ENV_GPU_LAYERS: &str = "KITTY_INFER_GPU_LAYERS";

const DEFAULT_CTX: usize = 4096;

/// Returns true when the operator selected the embedded backend. Pure and
/// unit-testable (env read stays at the call site).
pub fn backend_selected(raw: Option<&str>) -> bool {
    raw.map(|v| v.trim() == BACKEND_KITTY_INFER).unwrap_or(false)
}

/// Parse the GPU-layer spill override. `None` = unset/invalid = automatic
/// mapping. Pure and unit-testable.
fn parse_gpu_layers(raw: Option<&str>) -> Option<usize> {
    raw?.trim().parse::<usize>().ok().filter(|v| *v > 0)
}

/// Thinking toggle for reasoning templates (K2 opens `<ifm|think>` at high
/// effort by default — fine on full GPU, far too slow on CPU-spilled
/// residents). `KITTY_INFER_NO_THINK=1` renders the empty think block.
/// Pure and unit-testable; anything else (unset/invalid) keeps thinking on.
pub const ENV_NO_THINK: &str = "KITTY_INFER_NO_THINK";

fn thinking_enabled(raw: Option<&str>) -> bool {
    !matches!(raw.map(|v| v.trim()), Some("1") | Some("true") | Some("yes"))
}

/// Resolved configuration for one embedded resident.
#[derive(Debug, Clone)]
pub struct KittyInferConfig {
    pub model_path: PathBuf,
    pub tokenizer_sidecar_dir: Option<PathBuf>,
    pub ctx_len: usize,
}

impl KittyInferConfig {
    pub fn from_env() -> Result<Self> {
        let raw = std::env::var(ENV_MODEL).map_err(|_| {
            RuntimeError::InvalidArgument(format!(
                "embedded backend selected ({ENV_BACKEND}={BACKEND_KITTY_INFER}) but \
                 {ENV_MODEL} is not set — set it to a local .gguf file \
                 (e.g. {ENV_MODEL}=~/.cache/kitty/models/Qwen3-4B-Q4_K_M.gguf)"
            ))
        })?;
        let ctx_len = match std::env::var(ENV_CTX) {
            Err(_) => DEFAULT_CTX,
            Ok(raw) => raw
                .trim()
                .parse::<usize>()
                .ok()
                .filter(|v| *v > 0)
                .ok_or_else(|| {
                    RuntimeError::InvalidArgument(format!(
                        "{ENV_CTX}={raw:?} is not a positive integer — unset it or \
                         set a value like {ENV_CTX}=4096"
                    ))
                })?,
        };
        Self::from_path(
            PathBuf::from(raw),
            std::env::var(ENV_TOKENIZER_DIR).ok().map(PathBuf::from),
            ctx_len,
        )
    }

    pub fn from_path(
        model_path: PathBuf,
        tokenizer_sidecar_dir: Option<PathBuf>,
        ctx_len: usize,
    ) -> Result<Self> {
        if model_path.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase())
            != Some("gguf".to_string())
        {
            return Err(RuntimeError::InvalidArgument(format!(
                "embedded backend needs a local .gguf file, got `{}` — \
                 hint: KITTY_INFER_MODEL must point at a GGUF (try the humo-verified \
                 Qwen3-4B-Q4_K_M.gguf or MiniCPM5-2B-Q4_K_M.gguf)",
                model_path.display()
            )));
        }
        if !model_path.is_file() {
            return Err(RuntimeError::ModelNotFound(format!(
                "GGUF not found at {} — hint: check KITTY_INFER_MODEL; humo-verified \
                 models live under ~/.cache/kitty/models/",
                model_path.display()
            )));
        }
        Ok(Self {
            model_path,
            tokenizer_sidecar_dir,
            ctx_len: ctx_len.max(512),
        })
    }
}

// ---------------------------------------------------------------------------
// Pure mapping helpers (feature-independent, unit-tested without a model)
// ---------------------------------------------------------------------------

/// Map a kitty role string to a mistralrs role. Tool results have no
/// `tool_call_id` field on [`Message`], so they travel as `User` content —
/// threading the id through is follow-up work, not silent data loss (the
/// content itself is preserved verbatim).
#[cfg(any(feature = "kitty-infer", test))]
fn map_role(role: &str) -> (&'static str, bool) {
    match role {
        "system" => ("system", false),
        "assistant" => ("assistant", false),
        "tool" => ("user(tool-result)", true),
        _ => ("user", false),
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::map_role;

    #[test]
    fn gpu_layers_parse() {
        use super::parse_gpu_layers;
        assert_eq!(parse_gpu_layers(None), None);
        assert_eq!(parse_gpu_layers(Some("12")), Some(12));
        assert_eq!(parse_gpu_layers(Some(" 12 ")), Some(12));
        assert_eq!(parse_gpu_layers(Some("0")), None);
        assert_eq!(parse_gpu_layers(Some("nope")), None);
    }

    #[test]
    fn thinking_toggle() {
        use super::thinking_enabled;
        assert!(thinking_enabled(None));
        assert!(thinking_enabled(Some("0")));
        assert!(thinking_enabled(Some("nope")));
        assert!(!thinking_enabled(Some("1")));
        assert!(!thinking_enabled(Some(" true ")));
        assert!(!thinking_enabled(Some("yes")));
    }

    #[test]
    fn roles_map_lossy_documented() {
        assert_eq!(map_role("system"), ("system", false));
        assert_eq!(map_role("assistant"), ("assistant", false));
        assert_eq!(map_role("user"), ("user", false));
        assert_eq!(map_role("tool"), ("user(tool-result)", true));
        assert_eq!(map_role("weird"), ("user", false));
    }

    #[test]
    fn backend_switch_is_exact() {
        use super::backend_selected;
        assert!(backend_selected(Some("kitty-infer")));
        assert!(backend_selected(Some("  kitty-infer  "))); // surrounding whitespace trimmed
        assert!(!backend_selected(Some("KITTY-INFER"))); // case-sensitive
        assert!(!backend_selected(Some("ollama")));
        assert!(!backend_selected(None));
    }

    #[test]
    fn config_rejects_non_gguf_and_missing() {
        use super::KittyInferConfig;
        assert!(KittyInferConfig::from_path("/tmp/x.safetensors".into(), None, 4096).is_err());
        assert!(
            KittyInferConfig::from_path("/tmp/does-not-exist-kitty-infer.gguf".into(), None, 4096)
                .is_err()
        );
    }

    #[test]
    fn k2_horizon_path_plumbing() {
        // K2 resolves through the generic GGUF path (no per-arch switch):
        // extension + existence gate, sidecar dir + ctx preserved for the
        // loader (tokenizer/dtype policy lives in kitty-infer infer_policy).
        use super::KittyInferConfig;
        let dir = std::env::temp_dir();
        let fake = dir.join("kitty-infer-test-K2-Horizon-4B-BF16.gguf");
        std::fs::write(&fake, b"GGUF").unwrap();
        let cfg = KittyInferConfig::from_path(
            fake.clone(),
            Some(dir.join("tokenizers")),
            2048,
        )
        .unwrap();
        assert_eq!(cfg.model_path, fake);
        assert_eq!(cfg.tokenizer_sidecar_dir, Some(dir.join("tokenizers")));
        assert_eq!(cfg.ctx_len, 2048);
        // Ctx floor holds for K2's huge native window alike.
        let cfg = KittyInferConfig::from_path(fake.clone(), None, 0).unwrap();
        assert_eq!(cfg.ctx_len, 512);
        std::fs::remove_file(&fake).unwrap();
    }
}

// ---------------------------------------------------------------------------
// Loaded client (feature-gated; mirrors LlamaCppClient structure)
// ---------------------------------------------------------------------------

/// Process-wide loaded models, keyed by [`ModelSpec`](mistralrs::serving::ModelSpec)
/// id. The single-flight lock lives in `global_registry`; this map only
/// makes a registry-shared load visible to every client (a per-client slot
/// would diverge: client B would miss a model client A just built).
/// Pruned to the winner on every successful load, so the map honors the
/// single-resident contract instead of pinning evicted models in VRAM.
/// Plain std mutex, never held across await.
#[cfg(feature = "kitty-infer")]
static LOADED: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<String, Arc<Model>>>,
> = std::sync::OnceLock::new();

#[cfg(feature = "kitty-infer")]
fn loaded_map(
) -> &'static std::sync::Mutex<std::collections::HashMap<String, Arc<Model>>> {
    LOADED.get_or_init(Default::default)
}

/// Embedded kitty-infer backend: single-resident GGUF behind [`ChatClient`].
///
/// Concurrency contract (D1): serialization lives in `global_registry`,
/// which holds its guard across evict → load → store and reuses the
/// resident for same-spec waiters — two concurrent `ensure_loaded` (same
/// or different clients, same spec) compile the model exactly once, and
/// both read it from [`loaded_map`]. Cross-spec races serialize with
/// last-writer-wins residency, same as the registry metadata.
#[cfg(feature = "kitty-infer")]
pub struct KittyInferClient {
    config: KittyInferConfig,
}

#[cfg(feature = "kitty-infer")]
impl KittyInferClient {
    pub fn new(config: KittyInferConfig) -> Self {
        Self { config }
    }

    pub fn from_env() -> Result<Self> {
        Ok(Self::new(KittyInferConfig::from_env()?))
    }

    pub fn config(&self) -> &KittyInferConfig {
        &self.config
    }

    fn spec_id(&self) -> Option<String> {
        ki::ModelSpec::auto(&format!("local:{}", self.config.model_path.display()))
            .ok()
            .map(|spec| spec.id)
    }

    pub fn is_ready(&self) -> bool {
        // Non-blocking probe only; never loads here.
        self.spec_id()
            .and_then(|id| loaded_map().lock().ok().map(|m| m.contains_key(&id)))
            .unwrap_or(false)
    }

    fn friendly(err: impl std::fmt::Display) -> RuntimeError {
        let classified = ki::KittyError::classify(&err.to_string());
        let body = classified.body();
        RuntimeError::KittyInfer(format!("{} — hint: {}", body.error, body.hint))
    }

    /// Load (or reuse) the single resident. Evict-before-load + OOM retry
    /// come from `global_registry`; the `gpu_fraction` attempt scales the
    /// context (the KV lever this client controls), and `vram_precheck`
    /// fails fast with byte counts before touching the engine.
    async fn ensure_loaded(&self) -> Result<Arc<Model>> {
        let cfg = self.config.clone();
        let file_name = cfg
            .model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        let parent = cfg
            .model_path
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string());
        let extra = ki::extra_f32_prefixes(std::env::var(ENV_F32_PREFIXES).ok().as_deref());
        let dtype = ki::resolve_dtype(&file_name, &extra);
        let tok_json =
            ki::resolve_tokenizer_json(&cfg.model_path, cfg.tokenizer_sidecar_dir.as_deref())
                .map_err(Self::friendly)?;
        let file_len =
            ki::file_bytes(&cfg.model_path).map_err(Self::friendly)?;
        let spec = ki::ModelSpec::auto(&format!("local:{}", cfg.model_path.display()))
            .map_err(Self::friendly)?;
        let spec_id = spec.id.clone();

        // Fast path: another client (or an earlier turn) already built it.
        if let Some(model) = loaded_map().lock().ok().and_then(|m| m.get(&spec_id).cloned()) {
            return Ok(model);
        }

        let resident = ki::global_registry()
            .load_with_retry(spec, |fraction| {
                let file_name = file_name.clone();
                let parent = parent.clone();
                let tok_json = tok_json.clone();
                let spec_id = spec_id.clone();
                async move {
                    // Attempt ctx scales with the retry fraction (1.0 → ~0.66):
                    // KV cache is the OOM lever this client owns.
                    let ctx = ((cfg.ctx_len as f32 * fraction) as usize).max(512);
                    let free = ki::health::snapshot()
                        .devices
                        .first()
                        .map(|d| d.total_bytes.saturating_sub(d.used_bytes));
                    ki::vram_precheck(file_len, free, fraction, ctx).map_err(|e| {
                        anyhow::anyhow!("{e:#}")
                    })?;
                    let mut builder = GgufModelBuilder::new(parent, vec![file_name])
                        .with_logging()
                        .with_max_model_len(ctx)
                        .with_dtype(dtype.clone())
                        .with_paged_attn(PagedAttentionMetaBuilder::default().build()?);
                    if let Some(n) = parse_gpu_layers(std::env::var(ENV_GPU_LAYERS).ok().as_deref())
                    {
                        builder = builder.with_device_mapping(
                            mistralrs::DeviceMapSetting::Map(
                                mistralrs::DeviceMapMetadata::from_num_device_layers(vec![
                                    mistralrs::DeviceLayerMapMetadata {
                                        ordinal: 0,
                                        layers: n,
                                    },
                                ]),
                            ),
                        );
                    }
                    if let Some(tj) = tok_json {
                        builder = builder.with_tokenizer_json(tj.to_string_lossy().into_owned());
                    }
                    let model = builder.build().await.map_err(|e| anyhow::anyhow!("{e:#}"))?;
                    loaded_map()
                        .lock()
                        .map_err(|_| anyhow::anyhow!("loaded-model map is poisoned"))?
                        .insert(spec_id, Arc::new(model));
                    Ok(())
                }
            })
            .await
            .map_err(Self::friendly)?;
        tracing::info!(
            model = %resident.spec.id,
            gpu_fraction = resident.gpu_fraction,
            "kitty-infer resident ready"
        );
        // Prune evicted models so the map honors single-residence.
        let mut map = loaded_map()
            .lock()
            .map_err(|_| {
                RuntimeError::KittyInfer("loaded-model map is poisoned".to_string())
            })?;
        map.retain(|id, _| id == &resident.spec.id);
        map.get(&resident.spec.id).cloned().ok_or_else(|| {
            RuntimeError::KittyInfer(
                "load reported success but no model is resident — retry once; \
                 if it repeats, capture RUST_LOG=info output"
                    .to_string(),
            )
        })
    }

    fn to_request(
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> RequestBuilder {
        let mut req =
            RequestBuilder::new().enable_thinking(thinking_enabled(std::env::var(ENV_NO_THINK).ok().as_deref()));
        for m in messages {
            let (kind, _) = map_role(&m.role);
            let text = m.content.clone().unwrap_or_default();
            match kind {
                "system" => {
                    req = req.add_message(TextMessageRole::System, text);
                }
                "assistant" if m.tool_calls.as_ref().is_some_and(|c| !c.is_empty()) => {
                    let calls = m
                        .tool_calls
                        .unwrap_or_default()
                        .into_iter()
                        .map(|c| mistralrs::ToolCallResponse {
                            index: 0,
                            id: c.id.unwrap_or_else(|| "call-0".to_string()),
                            tp: mistralrs::ToolCallType::Function,
                            function: mistralrs::CalledFunction {
                                name: c.function.name,
                                arguments: c.function.arguments.to_string(),
                            },
                        })
                        .collect();
                    req = req.add_message_with_tool_call(TextMessageRole::Assistant, text, calls);
                }
                "assistant" => {
                    req = req.add_message(TextMessageRole::Assistant, text);
                }
                _ => {
                    req = req.add_message(TextMessageRole::User, text);
                }
            }
        }
        if let Some(tools) = tools {
            if !tools.is_empty() {
                req = req.set_tools(
                    tools
                        .into_iter()
                        .map(|t| mistralrs::Tool {
                            tp: mistralrs::ToolType::Function,
                            function: mistralrs::Function {
                                name: t.function.name,
                                description: Some(t.function.description),
                                parameters: Some(t.function.parameters),
                                strict: None,
                            },
                        })
                        .collect(),
                );
            }
        }
        req
    }
}

#[cfg(feature = "kitty-infer")]
impl std::fmt::Debug for KittyInferClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KittyInferClient")
            .field("model_path", &self.config.model_path)
            .field("ready", &self.is_ready())
            .finish()
    }
}

#[cfg(feature = "kitty-infer")]
impl ChatClient for KittyInferClient {
    fn chat<'a>(
        &'a self,
        model: &'a str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>> {
        Box::pin(async move {
            let loaded = self.ensure_loaded().await?;
            let resp = loaded
                .send_chat_request(Self::to_request(messages, tools))
                .await
                .map_err(Self::friendly)?;
            let choice = resp.choices.first().ok_or_else(|| {
                RuntimeError::KittyInfer(
                    "engine returned zero choices — retry once; hint: check RUST_LOG=info for pipeline errors".to_string(),
                )
            })?;
            let tool_calls = choice.message.tool_calls.as_ref().map(|calls| {
                calls
                    .iter()
                    .map(|c| crate::client::ToolCall {
                        id: Some(c.id.clone()),
                        function: crate::client::ToolCallFunction {
                            name: c.function.name.clone(),
                            arguments: serde_json::from_str(&c.function.arguments).unwrap_or(
                                serde_json::Value::String(c.function.arguments.clone()),
                            ),
                        },
                    })
                    .collect()
            });
            Ok(ChatResponse {
                model: model.to_string(),
                message: Message {
                    role: "assistant".to_string(),
                    content: choice.message.content.clone(),
                    thinking: None,
                    tool_calls,
                },
                done: true,
                eval_count: Some(resp.usage.completion_tokens as i64),
                eval_duration: None,
                prompt_eval_count: Some(resp.usage.prompt_tokens as i64),
            })
        })
    }

    fn count_tokens<'a>(
        &'a self,
        _model: &'a str,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<usize>> + Send + 'a>> {
        Box::pin(async move {
            // Heuristic on purpose: tokenizing must not trigger a 5GB load.
            Ok(text.len().div_ceil(4))
        })
    }

    fn health_check<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            // File-level probe, never loads the model.
            if !self.config.model_path.is_file() {
                return Err(RuntimeError::ModelNotFound(format!(
                    "GGUF not found at {} — hint: check {ENV_MODEL}",
                    self.config.model_path.display()
                )));
            }
            let file_name = self
                .config
                .model_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            if ki::needs_external_tokenizer(file_name) {
                ki::resolve_tokenizer_json(
                    &self.config.model_path,
                    self.config.tokenizer_sidecar_dir.as_deref(),
                )
                .map_err(Self::friendly)?;
            }
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// No-feature stub: honest compile-time-off error (never a silent fallback)
// ---------------------------------------------------------------------------

/// Selected backend is unavailable in this build. Always compiled so the CLI
/// can name the missing feature without cfg-conditional call sites.
#[cfg(not(feature = "kitty-infer"))]
pub fn unavailable_error() -> RuntimeError {
    RuntimeError::InvalidArgument(
        "KITTY_BACKEND=kitty-infer needs the `kitty-infer` cargo feature — \
         rebuild with --features kitty-infer (sibling repo ../kitty-infer must be present)"
            .to_string(),
    )
}

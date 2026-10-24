//! TUI interactiva — streaming, permisos, diffs, tokens/coste, selector de
//! modelo, resume, hints por foco y consola overlay.
//!
//! Principio rector: **la UI es una función pura del estado**. Todo el estado
//! vive en [`App`]; `view::render(f, &App)` solo lee. Las mutaciones ocurren en
//! [`App::handle_*`], que son funciones de transición de estado testables sin
//! terminal. El event loop ([`run_app`]) orquesta: drena el canal de eventos
//! del worker, aplica teclas, y vuelve a renderizar.
//!
//! Arquitectura de hilos:
//!   - Hilo principal (este): event loop síncrono crossterm + ratatui.
//!   - Worker ([`worker::spawn_worker`]): `Agent` confinado a un hilo con
//!     runtime `current_thread`+LocalSet, hablando por canales tokio.
//!
//! Paridad PI/opencode a 80×24: layout de 4 bandas (status/transcripción/
//! input/hints) que cabe en 24 filas; overlays centrados para permisos,
//! selector de modelo y consola.

pub mod client;
pub mod session;
pub mod view;
pub mod worker;

use std::io;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use tokio::sync::mpsc;

use self::client::{PermissionBroker, PermissionPolicy, PermissionVerdict, TuiEvent, UsageTotals};
use self::session::Session;
use self::worker::{spawn_worker, WorkerCmd};

/// Foco actual de la TUI: determina qué resaltan los hints y a dónde van las
/// teclas de navegación.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    /// Editando el prompt.
    Input,
    /// Navegando la lista de sesiones (resume).
    Sessions,
    /// Modal de permiso de herramienta abierto.
    Permission,
    /// Selector de modelo abierto.
    ModelPicker,
    /// Consola overlay visible (transcripción sigue scrolleable).
    Console,
}

/// Estado del selector de modelo.
///
/// `available_models` vive en [`App`] y solo se toca desde el reducer
/// [`App::handle_tui_event`]; el render lo lee y no muta nada.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ModelPicker {
    /// Selector cerrado (estado por defecto).
    #[default]
    Closed,
    /// Fetch en curso; la vista muestra "cargando…".
    Loading,
    /// Lista cargada y navegable.
    Ready,
    /// El fetch falló; se muestra el motivo y el fallback de entrada manual.
    Error(String),
}

/// Estado completo de la TUI. La renderización deriva 100% de esto.
pub struct App {
    // --- conversación ---
    pub messages: Vec<Message>,
    /// Texto parcial del turno en curso (streaming).
    pub streaming: String,
    /// Thinking parcial del turno en curso.
    pub thinking: String,
    pub is_running: bool,

    // --- prompt ---
    pub input: String,
    pub history: Vec<String>,
    pub history_pos: Option<usize>,
    pub draft: String,

    // --- modelo / contexto ---
    pub model: String,
    pub context_window: usize,
    /// Modelos instalados en el backend, para el selector.
    pub available_models: Vec<String>,
    /// Estado del selector de modelo: carga en curso, error, o listo.
    pub model_picker: ModelPicker,
    /// Índice resaltado en el selector de modelo.
    pub model_cursor: usize,
    /// Borrador de entrada manual en el selector (fallback cuando la lista
    /// está vacía o falló).
    pub model_input: String,

    // --- permisos / usage ---
    pub policy: Arc<Mutex<PermissionPolicy>>,
    pub usage: UsageTotals,
    /// Peticiones de permiso pendientes (drenadas del broker).
    pub pending_perms: Vec<PendingPermission>,
    /// Id del permiso actualmente resaltado en el modal.
    pub perm_cursor: usize,

    // --- scroll / foco / UI ---
    pub scroll_offset: usize,
    pub focus: Focus,
    pub console_lines: Vec<String>,
    pub console_scroll: usize,
    /// Último mensaje de estado para la banda de hints.
    pub status_note: Option<String>,

    // --- sesión (resume) ---
    pub session: Session,
    pub session_dirty: bool,

    // --- reloj ---
    pub session_start: Instant,
}

/// Una petición de permiso pendiente de decisión del usuario.
#[derive(Debug, Clone)]
pub struct PendingPermission {
    pub id: u64,
    pub tool: String,
    pub args_preview: String,
    pub diff: String,
}

/// Mensaje en la conversación.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

impl App {
    pub fn new(model: &str, context_window: usize) -> Self {
        Self {
            messages: Vec::new(),
            streaming: String::new(),
            thinking: String::new(),
            is_running: false,
            input: String::new(),
            history: Vec::new(),
            history_pos: None,
            draft: String::new(),
            model: model.to_string(),
            context_window,
            available_models: Vec::new(),
            model_picker: ModelPicker::Closed,
            model_cursor: 0,
            model_input: String::new(),
            policy: Arc::new(Mutex::new(PermissionPolicy::default())),
            usage: UsageTotals::default(),
            pending_perms: Vec::new(),
            perm_cursor: 0,
            scroll_offset: 0,
            focus: Focus::Input,
            console_lines: Vec::new(),
            console_scroll: 0,
            status_note: None,
            session: Session::new(model, context_window),
            session_dirty: false,
            session_start: Instant::now(),
        }
    }

    /// Restaura una sesión (resume): repuebla mensajes, modelo y contexto.
    pub fn restore_session(&mut self, s: Session) {
        self.model = s.model.clone();
        self.context_window = s.context_window;
        self.messages = s
            .entries
            .iter()
            .map(|e| Message {
                role: e.role.clone(),
                content: e.content.clone(),
            })
            .collect();
        self.session = s;
        self.session_dirty = false;
        self.scroll_to_bottom();
        self.status_note = Some("sesión restaurada".into());
    }

    // --- entrada ---

    pub fn push_char(&mut self, c: char) {
        self.input.push(c);
    }

    pub fn backspace(&mut self) {
        self.input.pop();
    }

    pub fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        match self.history_pos {
            None => {
                self.draft = self.input.clone();
                self.history_pos = Some(self.history.len() - 1);
                self.input = self.history[self.history.len() - 1].clone();
            }
            Some(0) => {}
            Some(pos) => {
                let np = pos - 1;
                self.history_pos = Some(np);
                self.input = self.history[np].clone();
            }
        }
    }

    pub fn history_down(&mut self) {
        match self.history_pos {
            None => {}
            Some(pos) => {
                if pos + 1 >= self.history.len() {
                    self.history_pos = None;
                    self.input = self.draft.clone();
                } else {
                    let np = pos + 1;
                    self.history_pos = Some(np);
                    self.input = self.history[np].clone();
                }
            }
        }
    }

    // --- scroll ---

    pub fn page_up(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_add(10);
    }

    pub fn page_down(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_sub(10);
    }

    pub fn scroll_to_bottom(&mut self) {
        self.scroll_offset = 0;
    }

    // --- transcripción ---

    /// Toma el input y prepara un turno (mensaje user + flag running).
    /// Devuelve el prompt listo para enviar al worker, o None si estaba vacío.
    pub fn take_input(&mut self) -> Option<String> {
        let trimmed = self.input.trim().to_string();
        if trimmed.is_empty() {
            return None;
        }
        self.history.push(trimmed.clone());
        self.messages.push(Message::user(trimmed.clone()));
        self.session.push_user(trimmed.clone());
        self.input.clear();
        self.history_pos = None;
        self.draft.clear();
        self.is_running = true;
        self.streaming.clear();
        self.thinking.clear();
        self.session_dirty = true;
        self.scroll_to_bottom();
        Some(trimmed)
    }

    // --- eventos del worker ---

    pub fn handle_tui_event(&mut self, ev: TuiEvent) {
        match ev {
            TuiEvent::Token(t) => {
                self.streaming.push_str(&t);
                self.scroll_to_bottom();
            }
            TuiEvent::Thinking(t) => {
                self.thinking.push_str(&t);
            }
            TuiEvent::PermissionRequest {
                id,
                tool,
                args_preview,
                diff,
            } => {
                self.pending_perms.push(PendingPermission {
                    id,
                    tool,
                    args_preview,
                    diff,
                });
                self.focus = Focus::Permission;
                self.perm_cursor = 0;
            }
            TuiEvent::PermissionResolved { tool, allowed, .. } => {
                self.console_lines.push(format!(
                    "permiso {}: {}",
                    tool,
                    if allowed { "permitido" } else { "denegado" }
                ));
            }
            TuiEvent::ModelsLoaded(models) => {
                self.apply_models_ok(models);
            }
            TuiEvent::ModelsFailed(err) => {
                self.apply_models_err(err);
            }
            TuiEvent::ToolStart { name, target } => {
                self.console_lines.push(format!("tool {name}: {target}"));
            }
            TuiEvent::Usage {
                prompt_tokens,
                eval_tokens,
                cost_usd,
            } => {
                self.usage.accumulate(prompt_tokens, eval_tokens, cost_usd);
            }
            TuiEvent::Done(text) => {
                self.finalize_turn(text);
            }
            TuiEvent::Error(e) => {
                self.finalize_turn(format!("Error: {e}"));
            }
        }
    }

    /// Cierra el turno: vuelca el streaming a un mensaje assistant.
    fn finalize_turn(&mut self, text: String) {
        let content = if self.streaming.is_empty() {
            text
        } else {
            std::mem::take(&mut self.streaming)
        };
        self.thinking.clear();
        self.messages.push(Message::assistant(content.clone()));
        self.session.push_assistant(content);
        self.is_running = false;
        self.session_dirty = true;
        self.scroll_to_bottom();
        // Persistencia de la sesión tras cada turno. El fallo de E/S no debe
        // tumbar la TUI: se anota en la consola y se sigue.
        if let Err(e) = session::save_session(&self.session) {
            self.console_lines
                .push(format!("no se pudo guardar la sesión: {e}"));
        }
    }

    // --- permisos ---

    /// Aplica un veredicto al permiso en curso (el más antiguo pendiente).
    pub fn resolve_permission(&mut self, verdict: PermissionVerdict, broker: &PermissionBroker) {
        if let Some(p) = self.pending_perms.first().cloned() {
            if broker.resolve(p.id, verdict) {
                self.pending_perms.remove(0);
                self.console_lines.push(format!(
                    "permiso {} → {}{}",
                    p.tool,
                    if verdict.allowed() { "allow" } else { "deny" },
                    if verdict.always() { " (siempre)" } else { "" }
                ));
            } else {
                // petición caducada ya resuelta; la descartamos
                self.pending_perms.remove(0);
            }
        }
        if self.pending_perms.is_empty() && self.focus == Focus::Permission {
            self.focus = Focus::Input;
        }
    }

    // --- modelo ---

    /// Abre el selector de modelo y dispara la carga de la lista.
    ///
    /// El fetch en sí lo hace el worker (dueño del runtime); aquí solo
    /// cambiamos el estado a `Loading`. Si el worker no responde, el picker
    /// se queda en `Loading` y el usuario puede cerrar con `Esc` o recargar
    /// con `ctrl+r`.
    pub fn open_model_picker(&mut self) {
        self.model_picker = ModelPicker::Loading;
        self.model_cursor = 0;
        self.model_input.clear();
        self.focus = Focus::ModelPicker;
    }

    /// Aplica una lista de modelos cargada con éxito.
    ///
    /// El modelo activo se añade a la lista aunque el backend no lo reporte,
    /// para que el selector nunca deje al usuario sin salida visible.
    pub fn apply_models_ok(&mut self, mut models: Vec<String>) {
        // Una lista vacía del backend se respeta tal cual: el picker ofrece
        // entrada manual en vez de inventar modelos. Solo se inyecta el
        // modelo activo si el backend devolvió algo, para que aparezca en la
        // lista aunque /api/tags no lo reporte (p.ej. descargado a mano).
        if !models.is_empty() && !models.iter().any(|m| m == &self.model) {
            models.push(self.model.clone());
        }
        models.sort();
        models.dedup();
        self.available_models = models;
        self.model_picker = ModelPicker::Ready;
        self.model_cursor = 0;
        // El modelo activo queda resaltado por defecto.
        if let Some(pos) = self.available_models.iter().position(|m| m == &self.model) {
            self.model_cursor = pos;
        }
    }

    /// Registra un fallo de carga; el picker ofrece entrada manual.
    pub fn apply_models_err(&mut self, err: String) {
        self.model_picker = ModelPicker::Error(err.clone());
        self.console_lines.push(format!("modelos: {err}"));
    }

    /// Mueve el cursor del selector, sin salirse de la lista.
    pub fn model_move_cursor(&mut self, delta: isize) {
        if self.available_models.is_empty() {
            return;
        }
        let len = self.available_models.len() as isize;
        let cur = self.model_cursor as isize;
        self.model_cursor = (cur + delta).rem_euclid(len) as usize;
    }

    /// Modelo resaltado en el selector, si la lista no está vacía.
    pub fn model_selected(&self) -> Option<&str> {
        self.available_models
            .get(self.model_cursor)
            .map(|s| s.as_str())
    }

    /// Resuelve el selector con la entrada manual del usuario.
    ///
    /// Se usa como fallback cuando la lista está vacía o el fetch falló:
    /// el usuario teclea el nombre del modelo directamente.
    pub fn model_use_manual(&mut self) -> Option<String> {
        let name = self.model_input.trim().to_string();
        if name.is_empty() {
            return None;
        }
        Some(name)
    }

    /// Cierra el selector de modelo sin cambiar de modelo.
    pub fn close_model_picker(&mut self) {
        self.model_picker = ModelPicker::Closed;
        self.model_input.clear();
        if self.focus == Focus::ModelPicker {
            self.focus = Focus::Input;
        }
    }

    pub fn set_model(&mut self, model: String) {
        self.model = model;
        self.status_note = Some(format!("modelo activo: {}", self.model));
    }

    // --- consola ---

    pub fn toggle_console(&mut self) {
        self.focus = if self.focus == Focus::Console {
            Focus::Input
        } else {
            Focus::Console
        };
    }

    // --- reset ---

    pub fn reset(&mut self) {
        if self.is_running {
            return;
        }
        self.messages.clear();
        self.streaming.clear();
        self.thinking.clear();
        self.scroll_offset = 0;
        self.status_note = Some("conversación reiniciada".into());
    }

    /// Coste acumulado de la sesión, sumado ronda a ronda desde el backend.
    pub fn cost(&self) -> f64 {
        self.usage.cost_usd
    }

    // --- hints ---

    /// Línea de hints según el foco actual (paridad PI/opencode).
    pub fn hints(&self) -> Vec<(&'static str, String)> {
        let mut out: Vec<(&'static str, String)> = Vec::new();
        match self.focus {
            Focus::Input => {
                out.push(("enter", "enviar".into()));
                out.push(("↑↓", "historial".into()));
                out.push(("pgup/pgdn", "scroll".into()));
                out.push(("ctrl+k", "modelos".into()));
                out.push(("ctrl+o", "sesiones".into()));
                out.push(("ctrl+l", "consola".into()));
                if self.is_running {
                    out.push(("esc", "detener".into()));
                } else {
                    out.push(("q", "salir".into()));
                }
            }
            Focus::Permission => {
                out.push(("y", "allow".into()));
                out.push(("Y", "allow siempre".into()));
                out.push(("n", "deny".into()));
                out.push(("N", "deny siempre".into()));
                out.push(("esc", "cancelar".into()));
            }
            Focus::Sessions => {
                out.push(("↑↓", "mover".into()));
                out.push(("enter", "cargar".into()));
                out.push(("esc", "cerrar".into()));
            }
            Focus::ModelPicker => match &self.model_picker {
                ModelPicker::Loading => {
                    out.push(("esc", "cerrar".into()));
                }
                ModelPicker::Error(_) => {
                    out.push(("enter", "usar lo escrito".into()));
                    out.push(("ctrl+r", "reintentar".into()));
                    out.push(("esc", "cerrar".into()));
                }
                ModelPicker::Ready => {
                    out.push(("↑↓", "mover".into()));
                    out.push(("enter", "elegir".into()));
                    out.push(("esc", "cerrar".into()));
                }
                ModelPicker::Closed => {
                    out.push(("esc", "cerrar".into()));
                }
            },
            Focus::Console => {
                out.push(("pgup/pgdn", "scroll".into()));
                out.push(("ctrl+l/esc", "cerrar".into()));
            }
        }
        out
    }

    // --- utilidades de estado derivado ---

    pub fn context_pct(&self) -> f64 {
        if self.context_window == 0 {
            return 0.0;
        }
        // Estimación conservadora: 4 chars por token sobre el transcript.
        let chars: usize = self.messages.iter().map(|m| m.content.len()).sum();
        let tokens = chars / 4;
        100.0 * tokens as f64 / self.context_window as f64
    }

    pub fn session_duration(&self) -> Duration {
        self.session_start.elapsed()
    }
}

/// Política de la tecla `q`: solo sale con el input vacío y sin turno activo;
/// en otro caso se teclea en el prompt.
fn apply_q_policy(app: &mut App) -> bool {
    if app.input.is_empty() && !app.is_running {
        true
    } else {
        app.push_char('q');
        false
    }
}

/// Renderiza el frame completo. Función pura del estado.
fn draw(f: &mut ratatui::Frame, app: &App, sessions: &[Session]) {
    view::render(f, app, sessions);
}

/// Punto de entrada: monta terminal, worker y event loop.
pub fn run_tui(model: &str, url: &str, context_window: usize) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Canal de eventos worker → UI.
    let (events_tx, mut events_rx) = mpsc::channel::<TuiEvent>(512);
    let broker = Arc::new(PermissionBroker::default());
    let app = App::new(model, context_window);
    let policy = app.policy.clone();

    let (cmd_tx, _worker_handle) = spawn_worker(
        url.to_string(),
        model.to_string(),
        context_window,
        events_tx,
        broker.clone(),
        policy,
    );

    let result = run_app(&mut terminal, app, &mut events_rx, &cmd_tx, &broker);

    // Ordenado: apagar el worker y salir del alternate screen.
    let _ = cmd_tx.try_send(WorkerCmd::Shutdown);
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    mut app: App,
    events_rx: &mut mpsc::Receiver<TuiEvent>,
    cmd_tx: &mpsc::Sender<WorkerCmd>,
    broker: &PermissionBroker,
) -> Result<()> {
    // Carga inicial: sesión más reciente para resume (best-effort).
    let mut sessions = session::list_sessions();
    if let Some(latest) = sessions.first().cloned() {
        app.restore_session(latest);
        // Reconstruimos el agent con el contexto de la sesión restaurada.
        let _ = cmd_tx.try_send(WorkerCmd::SetModel(app.model.clone()));
    }

    loop {
        terminal
            .draw(|f| draw(f, &app, &sessions))
            .map_err(|e| anyhow::anyhow!("render failed: {e}"))?;

        // Drena eventos del worker (no bloqueante).
        while let Ok(ev) = events_rx.try_recv() {
            app.handle_tui_event(ev);
        }

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                if handle_key(
                    &mut app,
                    key.code,
                    key.modifiers,
                    cmd_tx,
                    broker,
                    &mut sessions,
                )? {
                    return Ok(());
                }
            }
        }
    }
}

/// Aplica la elección de modelo y cierra el selector.
///
/// Compartido por la navegación de la lista y por la entrada manual: ambos
/// caminos terminan en el mismo `SetModel` hacia el worker.
fn apply_model_choice(app: &mut App, model: String, cmd_tx: &mpsc::Sender<WorkerCmd>) {
    app.set_model(model.clone());
    let _ = cmd_tx.try_send(WorkerCmd::SetModel(model));
    app.close_model_picker();
}

/// Maneja una tecla. Devuelve `true` si la TUI debe salir.
fn handle_key(
    app: &mut App,
    code: KeyCode,
    mods: KeyModifiers,
    cmd_tx: &mpsc::Sender<WorkerCmd>,
    broker: &PermissionBroker,
    sessions: &mut Vec<Session>,
) -> Result<bool> {
    let ctrl = mods.contains(KeyModifiers::CONTROL);

    match app.focus {
        Focus::Permission => {
            // El modal de permiso captura todas las teclas.
            match code {
                KeyCode::Char('y') => app.resolve_permission(PermissionVerdict::AllowOnce, broker),
                KeyCode::Char('Y') => {
                    app.resolve_permission(PermissionVerdict::AllowAlways, broker)
                }
                KeyCode::Char('n') => app.resolve_permission(PermissionVerdict::DenyOnce, broker),
                KeyCode::Char('N') => app.resolve_permission(PermissionVerdict::DenyAlways, broker),
                KeyCode::Esc => app.resolve_permission(PermissionVerdict::DenyOnce, broker),
                _ => {}
            }
            return Ok(false);
        }
        Focus::Sessions => {
            // La navegación de sesiones vive en la vista; aquí solo las
            // teclas de control globales. ↑↓/enter se manejan con índices en
            // App (simplificado: cargamos la más reciente con enter).
            match code {
                KeyCode::Esc => app.focus = Focus::Input,
                KeyCode::Enter => {
                    if let Some(s) = sessions.first().cloned() {
                        app.restore_session(s);
                        let _ = cmd_tx.try_send(WorkerCmd::SetModel(app.model.clone()));
                    }
                    app.focus = Focus::Input;
                }
                _ => {}
            }
            return Ok(false);
        }
        Focus::ModelPicker => {
            // El selector captura todas las teclas mientras está abierto.
            // El comportamiento depende del estado: cargando, listo o error.
            // En error, o con la lista vacía, el fallback es la entrada
            // manual del nombre del modelo.
            let manual = matches!(app.model_picker, ModelPicker::Error(_) | ModelPicker::Ready)
                && app.available_models.is_empty();

            match (&app.model_picker, code) {
                (ModelPicker::Loading, KeyCode::Esc) => app.close_model_picker(),
                (ModelPicker::Error(_), KeyCode::Esc) => app.close_model_picker(),
                (ModelPicker::Ready, KeyCode::Esc) => app.close_model_picker(),
                // Reintentar el fetch tras un fallo.
                (ModelPicker::Error(_), KeyCode::Char('r')) if ctrl => {
                    app.model_picker = ModelPicker::Loading;
                    let _ = cmd_tx.try_send(WorkerCmd::FetchModels);
                }
                // Entrada manual: escribir el nombre del modelo a mano.
                (_, KeyCode::Char(c)) if manual => app.model_input.push(c),
                (_, KeyCode::Backspace) if manual => {
                    app.model_input.pop();
                }
                (_, KeyCode::Enter) if manual => {
                    if let Some(name) = app.model_use_manual() {
                        apply_model_choice(app, name, cmd_tx);
                    }
                }
                // Lista cargada: navegar y elegir.
                (ModelPicker::Ready, KeyCode::Up) => app.model_move_cursor(-1),
                (ModelPicker::Ready, KeyCode::Down) => app.model_move_cursor(1),
                (ModelPicker::Ready, KeyCode::Enter) => {
                    if let Some(m) = app.model_selected().map(|s| s.to_string()) {
                        apply_model_choice(app, m, cmd_tx);
                    }
                }
                _ => {}
            }
            return Ok(false);
        }
        Focus::Console => {
            match code {
                KeyCode::Esc => app.focus = Focus::Input,
                KeyCode::PageUp => app.console_scroll = app.console_scroll.saturating_add(5),
                KeyCode::PageDown => app.console_scroll = app.console_scroll.saturating_sub(5),
                KeyCode::Char('l') if ctrl => app.focus = Focus::Input,
                _ => {}
            }
            return Ok(false);
        }
        Focus::Input => {}
    }

    // --- foco Input: teclas globales + edición ---
    match code {
        KeyCode::Char('q') if ctrl => return Ok(apply_q_policy(app)),
        KeyCode::Esc => {
            if app.is_running {
                // detener: por ahora solo marcamos; el worker no expone abort
                // síncrono, así que el turno seguirá hasta que termine.
                app.status_note = Some("no se puede detener un turno en curso".into());
            } else {
                return Ok(true);
            }
        }
        KeyCode::Char('c') if ctrl => return Ok(true),
        KeyCode::Char('r') if ctrl && !app.is_running => {
            // Reset de conversación: limpia estado local Y le dice al worker
            // que descarte su agent (el historial vive en ambos lados).
            app.reset();
            let _ = cmd_tx.try_send(WorkerCmd::Reset);
        }
        KeyCode::Char('k') if ctrl => {
            app.open_model_picker();
            // El fetch corre en el worker (no bloquea el event loop); el
            // picker muestra "cargando…" hasta que llegue ModelsLoaded/Failed.
            let _ = cmd_tx.try_send(WorkerCmd::FetchModels);
        }
        KeyCode::Char('o') if ctrl => {
            *sessions = session::list_sessions();
            app.focus = Focus::Sessions;
        }
        KeyCode::Char('l') if ctrl => app.toggle_console(),
        KeyCode::Enter => {
            if app.is_running {
                return Ok(false);
            }
            if let Some(prompt) = app.take_input() {
                let _ = cmd_tx.try_send(WorkerCmd::Prompt(prompt));
            }
        }
        KeyCode::Up => app.history_up(),
        KeyCode::Down => app.history_down(),
        KeyCode::PageUp => app.page_up(),
        KeyCode::PageDown => app.page_down(),
        KeyCode::Backspace => app.backspace(),
        KeyCode::Char(c) => app.push_char(c),
        _ => {}
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;

    /// HOME temporal + lock compartido con `session::tests` (ver
    /// `session::test_home_env_lock`): `finalize_turn` persiste la sesión
    /// en HOME y los tests de sesión mutan HOME en paralelo ⇒ sin lock y
    /// sin HOME aislado, flaky + contaminación de `~/.kitty` real.
    fn with_tmp_home<F: FnOnce()>(f: F) {
        let tmp = tempfile::tempdir().expect("tmp HOME para test TUI");
        let _lock = session::test_home_env_lock();
        let prev = std::env::var("HOME").ok();
        std::env::set_var("HOME", tmp.path());
        f();
        match prev {
            Some(p) => std::env::set_var("HOME", p),
            None => std::env::remove_var("HOME"),
        }
    }

    fn rows_of(backend: &TestBackend) -> Vec<String> {
        let buf = backend.buffer();
        let w = buf.area.width as usize;
        buf.content
            .chunks(w)
            .map(|row| row.iter().map(|c| c.symbol().to_string()).collect())
            .collect()
    }

    #[test]
    fn backspace_and_char() {
        let mut app = App::new("m", 100);
        app.push_char('a');
        app.push_char('b');
        assert_eq!(app.input, "ab");
        app.backspace();
        assert_eq!(app.input, "a");
        app.backspace();
        app.backspace();
        assert_eq!(app.input, "");
    }

    #[test]
    fn empty_history_nav_noop() {
        let mut app = App::new("m", 100);
        app.history_up();
        assert_eq!(app.history_pos, None);
        app.history_down();
        assert_eq!(app.history_pos, None);
    }

    #[test]
    fn history_up_down_restores_draft() {
        let mut app = App::new("m", 100);
        app.input = "draft".into();
        app.history = vec!["first".into(), "second".into()];
        app.history_up();
        assert_eq!(app.input, "second");
        app.history_up();
        assert_eq!(app.input, "first");
        app.history_up(); // clamp
        assert_eq!(app.input, "first");
        app.history_down();
        app.history_down();
        assert_eq!(app.input, "draft");
        assert_eq!(app.history_pos, None);
    }

    #[test]
    fn take_input_moves_to_history_and_messages() {
        let mut app = App::new("m", 100);
        app.input = "  trim me  ".into();
        let p = app.take_input().unwrap();
        assert_eq!(p, "trim me");
        assert_eq!(app.history[0], "trim me");
        assert_eq!(app.messages[0].role, "user");
        assert!(app.is_running);
        assert!(app.session_dirty);
    }

    #[test]
    fn take_input_empty_is_none() {
        let mut app = App::new("m", 100);
        assert!(app.take_input().is_none());
        assert!(!app.is_running);
    }

    #[test]
    fn streaming_then_done_finalizes_turn() {
        with_tmp_home(|| {
            let mut app = App::new("m", 100);
            app.handle_tui_event(TuiEvent::Token("hola ".into()));
            app.handle_tui_event(TuiEvent::Token("mundo".into()));
            assert_eq!(app.streaming, "hola mundo");
            app.handle_tui_event(TuiEvent::Done("ignorado".into()));
            assert_eq!(app.messages.last().unwrap().content, "hola mundo");
            assert!(app.streaming.is_empty());
            assert!(!app.is_running);
            // la sesión persistida refleja el turno
            assert_eq!(app.session.entries.last().unwrap().content, "hola mundo");
        });
    }

    #[test]
    fn done_without_streaming_uses_text() {
        with_tmp_home(|| {
            let mut app = App::new("m", 100);
            app.handle_tui_event(TuiEvent::Done("respuesta directa".into()));
            assert_eq!(app.messages.last().unwrap().content, "respuesta directa");
        });
    }

    #[test]
    fn error_event_finalizes_with_error_text() {
        with_tmp_home(|| {
            let mut app = App::new("m", 100);
            app.handle_tui_event(TuiEvent::Error("boom".into()));
            assert!(app.messages.last().unwrap().content.contains("boom"));
            assert!(!app.is_running);
        });
    }

    #[test]
    fn usage_accumulates() {
        let mut app = App::new("m", 100);
        app.handle_tui_event(TuiEvent::Usage {
            prompt_tokens: 10,
            eval_tokens: 5,
            cost_usd: 0.000_1,
        });
        app.handle_tui_event(TuiEvent::Usage {
            prompt_tokens: 20,
            eval_tokens: 7,
            cost_usd: 0.000_2,
        });
        assert_eq!(app.usage.prompt_tokens, 30);
        assert_eq!(app.usage.eval_tokens, 12);
        assert!((app.cost() - 0.000_3).abs() < 1e-12);
        assert_eq!(app.usage.turns, 2);
    }

    #[test]
    fn reset_clears_messages_and_streaming() {
        with_tmp_home(|| {
            let mut app = App::new("m", 100);
            app.input = "a".into();
            app.take_input();
            app.handle_tui_event(TuiEvent::Done("r".into()));
            app.reset();
            assert!(app.messages.is_empty());
            assert!(app.streaming.is_empty());
            assert_eq!(app.scroll_offset, 0);
        });
    }

    #[test]
    fn reset_blocked_while_running() {
        let mut app = App::new("m", 100);
        app.input = "a".into();
        app.take_input(); // is_running = true
        app.reset();
        assert_eq!(app.messages.len(), 1, "no reset con turno activo");
    }

    #[test]
    fn empty_q_policy_quits_only_when_idle_and_empty() {
        let mut app = App::new("m", 100);
        app.push_char('h');
        assert!(!apply_q_policy(&mut app));
        assert_eq!(app.input, "hq");

        let mut app = App::new("m", 100);
        app.is_running = true;
        assert!(!apply_q_policy(&mut app));
        assert_eq!(app.input, "q");

        let mut app = App::new("m", 100);
        assert!(apply_q_policy(&mut app));
    }

    #[test]
    fn restore_session_repuplates_model_and_messages() {
        let mut s = Session::new("qwen:7b", 8192);
        s.push_user("hola");
        s.push_assistant("qué tal");
        let mut app = App::new("m", 100);
        app.restore_session(s);
        assert_eq!(app.model, "qwen:7b");
        assert_eq!(app.context_window, 8192);
        assert_eq!(app.messages.len(), 2);
        assert_eq!(app.messages[0].content, "hola");
        assert_eq!(app.scroll_offset, 0);
    }

    #[test]
    fn hints_reflect_focus() {
        let mut app = App::new("m", 100);
        let h = app.hints();
        assert!(h.iter().any(|(k, _)| *k == "enter"));

        app.focus = Focus::Permission;
        let h = app.hints();
        assert!(h.iter().any(|(k, _)| *k == "y"));
        assert!(h.iter().any(|(k, _)| *k == "N"));

        app.focus = Focus::Console;
        assert!(app.hints().iter().any(|(k, _)| *k == "ctrl+l/esc"));
    }

    #[test]
    fn context_pct_zero_window() {
        let mut app = App::new("m", 0);
        app.push_char('a');
        assert_eq!(app.context_pct(), 0.0);
    }

    #[test]
    fn permission_request_sets_focus_and_verdict_clears() {
        let broker = PermissionBroker::default();
        let mut app = App::new("m", 100);
        // inyectamos una petición pendiente directamente (id arbitrario)
        let (id, _rx) = broker.request();
        app.handle_tui_event(TuiEvent::PermissionRequest {
            id,
            tool: "Edit".into(),
            args_preview: "a.rs".into(),
            diff: "-x\n+y".into(),
        });
        assert_eq!(app.focus, Focus::Permission);
        assert_eq!(app.pending_perms.len(), 1);

        app.resolve_permission(PermissionVerdict::AllowOnce, &broker);
        assert!(app.pending_perms.is_empty());
        assert_eq!(app.focus, Focus::Input);
    }

    // --- 80×24: paridad de layout ---

    #[test]
    fn renders_at_80x24_with_idle_state() {
        let app = App::new("qwen3:8b", 16384);
        let backend = TestBackend::new(80, 24);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw(f, &app, &[])).unwrap();
        let rows = rows_of(term.backend());
        assert_eq!(rows.len(), 24);
        // status + hints + input visibles
        let joined = rows.concat();
        assert!(joined.contains("qwen3:8b"), "status muestra el modelo");
        assert!(joined.contains("enter"), "hints de input visibles");
    }

    #[test]
    fn renders_streaming_and_running_badge() {
        let mut app = App::new("m", 100);
        app.is_running = true;
        app.streaming = "escribiendo…".into();
        let backend = TestBackend::new(80, 24);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw(f, &app, &[])).unwrap();
        let joined = rows_of(term.backend()).concat();
        assert!(joined.contains("escribiendo"));
    }

    #[test]
    fn permission_modal_renders_tool_and_diff() {
        let mut app = App::new("m", 100);
        app.pending_perms.push(PendingPermission {
            id: 1,
            tool: "Edit".into(),
            args_preview: "src/main.rs".into(),
            diff: "-let a = 1;\n+let a = 2;".into(),
        });
        app.focus = Focus::Permission;
        let backend = TestBackend::new(80, 24);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw(f, &app, &[])).unwrap();
        let joined = rows_of(term.backend()).concat();
        assert!(joined.contains("Edit"));
        assert!(joined.contains("src/main.rs"));
        assert!(joined.contains("allow"));
        assert!(joined.contains("deny"));
    }

    #[test]
    fn console_overlay_renders_lines() {
        let mut app = App::new("m", 100);
        app.console_lines.push("tool Bash [ok]: ls".into());
        app.focus = Focus::Console;
        let backend = TestBackend::new(80, 24);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw(f, &app, &[])).unwrap();
        let joined = rows_of(term.backend()).concat();
        assert!(joined.contains("tool Bash"));
    }

    #[test]
    fn model_picker_renders_available_models() {
        let mut app = App::new("m", 100);
        app.available_models = vec!["qwen3:8b".into(), "llama3:8b".into()];
        app.focus = Focus::ModelPicker;
        let backend = TestBackend::new(80, 24);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw(f, &app, &[])).unwrap();
        let joined = rows_of(term.backend()).concat();
        assert!(joined.contains("qwen3:8b"));
        assert!(joined.contains("llama3:8b"));
    }

    // --- selector de modelo: estados, cursor y fallback ---

    #[test]
    fn open_model_picker_sets_loading_and_focus() {
        let mut app = App::new("m", 100);
        app.open_model_picker();
        assert_eq!(app.model_picker, ModelPicker::Loading);
        assert_eq!(app.focus, Focus::ModelPicker);
        assert_eq!(app.model_cursor, 0);
        assert!(app.model_input.is_empty());
    }

    #[test]
    fn models_loaded_moves_to_ready_and_cursor_on_active() {
        let mut app = App::new("llama3:8b", 100);
        app.handle_tui_event(TuiEvent::ModelsLoaded(vec![
            "qwen3:8b".into(),
            "llama3:8b".into(),
        ]));
        assert_eq!(app.model_picker, ModelPicker::Ready);
        // El modelo activo queda resaltado por defecto.
        assert_eq!(app.model_selected(), Some("llama3:8b"));
    }

    #[test]
    fn models_loaded_appends_active_model_when_backend_omits_it() {
        // El modelo activo puede no estar en /api/tags (p.ej. ya descargado
        // pero renombrado): el selector no debe dejar al usuario sin salida.
        let mut app = App::new("gemma:2b", 100);
        app.handle_tui_event(TuiEvent::ModelsLoaded(vec!["qwen3:8b".into()]));
        assert!(app.available_models.iter().any(|m| m == "gemma:2b"));
        assert_eq!(app.model_selected(), Some("gemma:2b"));
    }

    #[test]
    fn models_loaded_sorts_and_dedups() {
        let mut app = App::new("m", 100);
        app.handle_tui_event(TuiEvent::ModelsLoaded(vec![
            "zeta:7b".into(),
            "alpha:3b".into(),
            "zeta:7b".into(),
        ]));
        assert_eq!(app.available_models, vec!["alpha:3b", "m", "zeta:7b"]);
    }

    #[test]
    fn models_failed_keeps_error_and_logs() {
        let mut app = App::new("m", 100);
        app.open_model_picker();
        app.handle_tui_event(TuiEvent::ModelsFailed("timeout".into()));
        assert!(matches!(&app.model_picker, ModelPicker::Error(e) if e == "timeout"));
        assert!(
            app.console_lines.iter().any(|l| l.contains("timeout")),
            "el fallo se registra en la consola"
        );
    }

    #[test]
    fn model_cursor_wraps_both_directions() {
        let mut app = App::new("m", 100);
        app.available_models = vec!["a".into(), "b".into(), "c".into()];
        app.model_cursor = 0;
        app.model_move_cursor(-1);
        assert_eq!(app.model_cursor, 2, "arriba desde el primero da al último");
        app.model_move_cursor(1);
        assert_eq!(
            app.model_cursor, 0,
            "abajo desde el último vuelve al primero"
        );
        app.model_move_cursor(2);
        assert_eq!(app.model_cursor, 2);
    }

    #[test]
    fn model_cursor_noop_on_empty_list() {
        let mut app = App::new("m", 100);
        app.model_move_cursor(1);
        app.model_move_cursor(-3);
        assert_eq!(app.model_cursor, 0);
        assert_eq!(app.model_selected(), None);
    }

    #[test]
    fn manual_input_requires_non_empty_trimmed_name() {
        let mut app = App::new("m", 100);
        app.model_input = "   ".into();
        assert_eq!(app.model_use_manual(), None, "solo espacios no cuenta");
        app.model_input = "  qwen3:8b  ".into();
        assert_eq!(app.model_use_manual(), Some("qwen3:8b".into()));
    }

    #[test]
    fn close_model_picker_clears_and_returns_focus() {
        let mut app = App::new("m", 100);
        app.open_model_picker();
        app.model_input = "qwen".into();
        app.close_model_picker();
        assert_eq!(app.model_picker, ModelPicker::Closed);
        assert!(app.model_input.is_empty());
        assert_eq!(app.focus, Focus::Input);
    }

    #[test]
    fn close_model_picker_leaves_unrelated_focus_alone() {
        let mut app = App::new("m", 100);
        app.focus = Focus::Console;
        app.model_picker = ModelPicker::Ready;
        app.close_model_picker();
        // Solo borra el estado del picker; no pisa otro foco.
        assert_eq!(app.focus, Focus::Console);
    }

    #[test]
    fn model_picker_hints_follow_state() {
        let mut app = App::new("m", 100);
        app.focus = Focus::ModelPicker;

        app.model_picker = ModelPicker::Loading;
        let keys: Vec<&str> = app.hints().iter().map(|(k, _)| *k).collect();
        assert!(keys.contains(&"esc"));
        assert!(!keys.contains(&"↑↓"), "cargando no navega");

        app.model_picker = ModelPicker::Error("x".into());
        let keys: Vec<&str> = app.hints().iter().map(|(k, _)| *k).collect();
        assert!(keys.contains(&"ctrl+r"), "error permite reintentar");
        assert!(keys.contains(&"enter"), "error permite entrada manual");

        app.model_picker = ModelPicker::Ready;
        app.available_models = vec!["a".into()];
        let keys: Vec<&str> = app.hints().iter().map(|(k, _)| *k).collect();
        assert!(keys.contains(&"↑↓"));
        assert!(keys.contains(&"enter"));
    }

    #[test]
    fn model_picker_renders_loading_state() {
        let mut app = App::new("m", 100);
        app.open_model_picker();
        let backend = TestBackend::new(80, 24);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw(f, &app, &[])).unwrap();
        let joined = rows_of(term.backend()).concat();
        assert!(joined.contains("consultando /api/tags"));
        assert!(
            !joined.contains("cargando modelos"),
            "mensaje antiguo retirado"
        );
    }

    #[test]
    fn model_picker_renders_error_with_reason_and_active_model() {
        let mut app = App::new("qwen3:8b", 100);
        app.open_model_picker();
        app.handle_tui_event(TuiEvent::ModelsFailed(
            "no se pudo contactar con http://localhost:11434".into(),
        ));
        let backend = TestBackend::new(80, 24);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw(f, &app, &[])).unwrap();
        let joined = rows_of(term.backend()).concat();
        assert!(joined.contains("no se pudo cargar la lista"));
        assert!(joined.contains("qwen3:8b"), "muestra el modelo activo");
        assert!(
            joined.contains("modelos (error)"),
            "título refleja el estado"
        );
    }

    #[test]
    fn model_picker_marks_active_and_cursor_row() {
        let mut app = App::new("llama3:8b", 100);
        app.open_model_picker();
        app.handle_tui_event(TuiEvent::ModelsLoaded(vec![
            "qwen3:8b".into(),
            "llama3:8b".into(),
        ]));
        app.model_cursor = 0;
        let backend = TestBackend::new(80, 24);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw(f, &app, &[])).unwrap();
        let joined = rows_of(term.backend()).concat();
        assert!(joined.contains("● llama3:8b"), "modelo activo marcado");
        assert!(joined.contains("modelos (2)"), "título con el recuento");
    }

    #[test]
    fn model_picker_offers_manual_input_when_list_empty() {
        // Ready con lista vacía: el backend no tiene modelos; se puede teclear.
        let mut app = App::new("m", 100);
        app.open_model_picker();
        app.handle_tui_event(TuiEvent::ModelsLoaded(vec![]));
        assert_eq!(app.model_picker, ModelPicker::Ready);
        assert!(app.available_models.is_empty());

        let backend = TestBackend::new(80, 24);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw(f, &app, &[])).unwrap();
        let joined = rows_of(term.backend()).concat();
        assert!(joined.contains("el backend no tiene modelos"));
        assert!(joined.contains("escribe un modelo"));

        // Escribir y validar el nombre.
        for c in "qwen3:8b".chars() {
            app.model_input.push(c);
        }
        assert_eq!(app.model_use_manual(), Some("qwen3:8b".into()));
    }

    #[test]
    fn sessions_picker_renders_list() {
        let mut app = App::new("m", 100);
        app.focus = Focus::Sessions; // el picker solo se pinta con este foco
        let mut s = Session::new("m", 100);
        s.push_user("mensaje de prueba");
        let backend = TestBackend::new(80, 24);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw(f, &app, &[s])).unwrap();
        let joined = rows_of(term.backend()).concat();
        assert!(joined.contains("sesiones") || joined.contains("mensaje de prueba"));
    }

    #[test]
    fn cjk_input_renders_without_panic_and_keeps_layout() {
        // Ratatui 0.30 `TestBackend` reserva UNA celda por glifo CJK (no
        // aplica ancho doble), así que no podemos asumir posiciones por
        // columnas. Lo que sí garantizamos: el carácter llega al buffer y el
        // layout de 4 bandas no se rompe a 40x12.
        let mut app = App::new("m", 100);
        app.push_char('你');
        app.push_char('好');
        let backend = TestBackend::new(40, 12);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw(f, &app, &[])).unwrap();

        let buf = term.backend().buffer();
        let flat: Vec<&str> = buf.content.iter().map(|c| c.symbol()).collect();
        let pos_you = flat.iter().position(|s| *s == "你").expect("你 rendered");
        let pos_hao = flat.iter().position(|s| *s == "好").expect("好 rendered");
        // Ambos glifos existen y 好 va DESPUÉS de 你 (orden de escritura).
        assert!(pos_hao > pos_you, "好 debe ir tras 你");
        // El input se dibuja en la banda inferior del layout, nunca en la de
        // status (fila 0).
        let w = buf.area.width as usize;
        assert!(pos_you / w >= 7, "input en la banda inferior, no en status");
        // El bloque del input sigue íntegro: el título ocupa la fila del borde
        // superior y el texto la fila interior, ambas por debajo de la status.
        // (ratatui guarda el título carácter a carácter, no como subcadena.)
        let w = buf.area.width as usize;
        assert_eq!(
            pos_you / w,
            pos_hao / w,
            "ambos glifos en la misma fila de input"
        );
        assert!(
            pos_you / w >= 7,
            "fila de input por debajo de la banda de status"
        );
    }

    #[test]
    fn long_transcript_scrolls_at_80x24() {
        let mut app = App::new("m", 100);
        for i in 0..50 {
            app.messages.push(Message::user(format!("msg{i:02}")));
        }
        app.scroll_to_bottom(); // offset 0 = anclado al final
        let backend = TestBackend::new(80, 24);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw(f, &app, &[])).unwrap();
        let joined = rows_of(term.backend()).concat();
        // con scroll al final se ven las últimas, no las primeras
        assert!(joined.contains("msg49"));
        assert!(!joined.contains("msg00"));
        // PgUp revela las anteriores
        app.page_up();
        term.draw(|f| draw(f, &app, &[])).unwrap();
        let joined2 = rows_of(term.backend()).concat();
        assert!(joined2.contains("msg39"));
    }
}

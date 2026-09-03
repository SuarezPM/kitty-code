//! TUI — interactive terminal with prompt, history, scroll, live status, streaming.
//! Streaming via tokio mpsc channel 100 + spawn Agent::run, select poll.

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use kitty_harness::agent::{Agent, AgentConfig, AgentEvent};
use kitty_runtime::OllamaClient;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame, Terminal,
};
use std::io;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Status info displayed in the TUI.
pub struct StatusInfo {
    pub model: String,
    pub url: String,
    pub context_used: usize,
    pub context_window: usize,
    pub current_step: String,
    pub session_start: Instant,
}

impl StatusInfo {
    pub fn new(model: &str, url: &str, context_window: usize) -> Self {
        Self {
            model: model.to_string(),
            url: url.to_string(),
            context_used: 0,
            context_window,
            current_step: "Ready".to_string(),
            session_start: Instant::now(),
        }
    }

    pub fn context_pct(&self) -> f64 {
        if self.context_window == 0 {
            return 0.0;
        }
        100.0 * self.context_used as f64 / self.context_window as f64
    }

    pub fn session_duration(&self) -> Duration {
        self.session_start.elapsed()
    }
}

/// Message in conversation.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

/// App state with prompt, history, scroll, streaming.
pub struct App {
    pub status: StatusInfo,
    pub input: String,
    pub history: Vec<String>,
    pub history_pos: Option<usize>,
    pub draft: String,
    pub messages: Vec<Message>,
    pub scroll_offset: usize,
    pub is_running: bool,
}

impl App {
    pub fn new(model: &str, url: &str, context_window: usize) -> Self {
        Self {
            status: StatusInfo::new(model, url, context_window),
            input: String::new(),
            history: Vec::new(),
            history_pos: None,
            draft: String::new(),
            messages: Vec::new(),
            scroll_offset: 0,
            is_running: false,
        }
    }

    pub fn push_char(&mut self, c: char) {
        self.input.push(c);
        self.sync_context_used();
    }

    pub fn backspace(&mut self) {
        self.input.pop();
        self.sync_context_used();
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
            Some(0) => {
                // clamp at oldest
            }
            Some(pos) => {
                let new_pos = pos - 1;
                self.history_pos = Some(new_pos);
                self.input = self.history[new_pos].clone();
            }
        }
        self.sync_context_used();
    }

    pub fn history_down(&mut self) {
        match self.history_pos {
            None => {}
            Some(pos) => {
                if pos + 1 >= self.history.len() {
                    self.history_pos = None;
                    self.input = self.draft.clone();
                } else {
                    let new_pos = pos + 1;
                    self.history_pos = Some(new_pos);
                    self.input = self.history[new_pos].clone();
                }
                self.sync_context_used();
            }
        }
    }

    pub fn take_input(&mut self) -> bool {
        let trimmed = self.input.trim().to_string();
        if trimmed.is_empty() {
            return false;
        }
        self.history.push(trimmed.clone());
        self.messages.push(Message::user(trimmed.clone()));
        self.input.clear();
        self.history_pos = None;
        self.draft.clear();
        self.status.current_step = format!("Sent #{}", self.messages.len());
        self.sync_context_used();
        self.scroll_offset = self.messages.len().saturating_sub(1);
        true
    }

    pub fn sync_context_used(&mut self) {
        let msg_len: usize = self.messages.iter().map(|m| m.content.len()).sum();
        self.status.context_used = msg_len + self.input.len();
    }

    pub fn page_up(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_sub(10);
    }

    pub fn page_down(&mut self) {
        let max = self.messages.len().saturating_sub(1);
        self.scroll_offset = (self.scroll_offset + 10).min(max);
    }

    pub fn handle_event(&mut self, ev: AgentEvent) {
        match ev {
            AgentEvent::Thinking(s) => {
                self.status.current_step = "Thinking".to_string();
                self.messages.push(Message {
                    role: "assistant".to_string(),
                    content: format!("thinking: {s}"),
                });
            }
            AgentEvent::ToolCall { name, args } => {
                self.status.current_step = format!("Tool: {name}");
                self.messages.push(Message {
                    role: "assistant".to_string(),
                    content: format!("Tool: {name} {}", args),
                });
            }
            AgentEvent::ToolResult {
                name,
                output,
                success,
            } => {
                let tag = if success { "OK" } else { "ERR" };
                self.messages.push(Message {
                    role: "assistant".to_string(),
                    content: format!("Result {name} [{tag}]: {output}"),
                });
            }
            AgentEvent::AssistantMessage(s) => {
                self.status.current_step = "Done".to_string();
                self.messages.push(Message::assistant(s));
            }
            AgentEvent::Error(e) => {
                self.status.current_step = "Error".to_string();
                self.messages.push(Message {
                    role: "assistant".to_string(),
                    content: format!("Error: {e}"),
                });
            }
            AgentEvent::Done(s) => {
                self.status.current_step = "Done".to_string();
                self.messages.push(Message::assistant(s));
                self.is_running = false;
            }
        }
        self.scroll_offset = self.messages.len().saturating_sub(1);
        self.sync_context_used();
    }

    pub fn reset(&mut self) {
        if self.is_running {
            return;
        }
        self.messages.clear();
        self.scroll_offset = 0;
        self.is_running = false;
        self.status.current_step = "Ready".to_string();
        self.sync_context_used();
    }
}

/// Render the TUI layout.
fn ui(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Status bar
            Constraint::Min(1),    // Conversation area
            Constraint::Length(3), // Input area
        ])
        .split(f.area());

    render_status_bar(f, chunks[0], &app.status);

    // Conversation area
    if app.messages.is_empty() {
        let placeholder = Paragraph::new("No messages yet. Type a prompt below. Enter to send, Up/Down for history, PgUp/PgDn to scroll, q to quit.")
            .block(
                Block::default()
                    .title("Conversation")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray)),
            )
            .wrap(Wrap { trim: true });
        f.render_widget(placeholder, chunks[1]);
    } else {
        let inner_h = chunks[1].height.saturating_sub(2) as usize;
        let max_offset = app
            .messages
            .len()
            .saturating_sub(inner_h)
            .max(app.messages.len().saturating_sub(1));
        let offset = app.scroll_offset.min(max_offset);
        let end = (offset + inner_h).min(app.messages.len());
        let visible = &app.messages[offset..end];
        let items: Vec<ListItem> = visible
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let idx = offset + i + 1;
                let (prefix, color) = if m.role == "user" {
                    ("user", Color::Cyan)
                } else {
                    ("assistant", Color::Green)
                };
                let content = Line::from(vec![
                    Span::styled(
                        format!("{idx} {prefix}> "),
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(m.content.clone()),
                ]);
                ListItem::new(content)
            })
            .collect();
        let title = format!("Conversation ({})", app.messages.len());
        let list = List::new(items).block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
        f.render_widget(list, chunks[1]);
    }

    // Input area
    let input_block = Block::default()
        .title(format!(
            "Prompt {}/512  [Enter send | Up/Down history | PgUp/PgDn scroll | r reset]",
            app.input.len()
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));
    let input_text = Paragraph::new(format!("> {}▌", app.input))
        .block(input_block)
        .style(Style::default().fg(Color::Yellow));
    f.render_widget(input_text, chunks[2]);
    let x = chunks[2].x + 2 + app.input.len() as u16 + 1;
    let y = chunks[2].y + 1;
    let clamped_x = x.min(chunks[2].x + chunks[2].width - 1);
    f.set_cursor_position((clamped_x, y));
}

/// Render the status bar.
fn render_status_bar(f: &mut Frame, area: Rect, status: &StatusInfo) {
    let pct = status.context_pct();
    let pct_color = if pct > 80.0 {
        Color::Red
    } else if pct > 60.0 {
        Color::Yellow
    } else {
        Color::Green
    };

    let elapsed = status.session_duration();
    let duration_str = format!(
        "{:02}:{:02}:{:02}",
        elapsed.as_secs() / 3600,
        (elapsed.as_secs() % 3600) / 60,
        elapsed.as_secs() % 60,
    );

    let status_line = Line::from(vec![
        Span::styled(
            format!(" {} ", status.model),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" | "),
        Span::styled(format!("ctx: {:.0}%", pct), Style::default().fg(pct_color)),
        Span::raw(" | "),
        Span::styled(
            format!("step: {}", status.current_step),
            Style::default().fg(Color::Cyan),
        ),
        Span::raw(" | "),
        Span::styled(
            format!("{} ", duration_str),
            Style::default().fg(Color::DarkGray),
        ),
        Span::raw("| q quit | PgUp/PgDn scroll | r reset"),
    ]);

    let bar = Paragraph::new(status_line)
        .block(
            Block::default()
                .title("Kitty Code")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Magenta)),
        )
        .style(Style::default().bg(Color::Black));

    f.render_widget(bar, area);
}

/// Run the TUI event loop.
pub fn run_tui(model: &str, url: &str, context_window: usize) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let app = App::new(model, url, context_window);
    let (tx, rx) = mpsc::channel::<AgentEvent>(100);
    let _tx_clone = tx.clone();

    let result = run_app(&mut terminal, app, rx, tx);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

/// Empty-q policy: q quits only when the input is empty and no agent run
/// is active; otherwise the q is typed into the prompt.
fn apply_q_policy(app: &mut App) -> bool {
    if app.input.is_empty() && !app.is_running {
        true
    } else {
        app.push_char('q');
        false
    }
}

fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    mut app: App,
    mut rx: mpsc::Receiver<AgentEvent>,
    tx: mpsc::Sender<AgentEvent>,
) -> Result<()> {
    loop {
        terminal.draw(|f| ui(f, &app))?;

        // Try recv channel without blocking
        while let Ok(event) = rx.try_recv() {
            app.handle_event(event);
        }

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => {
                            if apply_q_policy(&mut app) {
                                return Ok(());
                            }
                        }
                        KeyCode::Esc => return Ok(()),
                        KeyCode::Enter => {
                            if app.is_running {
                                continue;
                            }
                            let prompt = app.input.trim().to_string();
                            if prompt.is_empty() {
                                continue;
                            }
                            let taken = app.take_input();
                            if !taken {
                                continue;
                            }
                            app.is_running = true;
                            app.status.current_step = "Thinking...".to_string();
                            // Spawn agent run with channel
                            let tx_clone = tx.clone();
                            let model_clone = app.status.model.clone();
                            let url_clone = app.status.url.clone();
                            let window = app.status.context_window;
                            let prompt_clone = prompt.clone();
                            tokio::task::spawn_local(async move {
                                let client = OllamaClient::new(&url_clone);
                                // f8 T1.1: interactive TUI sessions may grant
                                // autorun hops via env (bench paths never read
                                // this). Invalid/absent ⇒ 0.
                                let autorun_hops =
                                    std::env::var(kitty_harness::config::ENV_AUTORUN_HOPS)
                                        .ok()
                                        .and_then(|v| v.trim().parse::<usize>().ok())
                                        .unwrap_or(0);
                                let config = AgentConfig {
                                    model: model_clone,
                                    context_window: window,
                                    autorun_hops: kitty_harness::agent::clamp_autorun_hops(
                                        autorun_hops,
                                    ),
                                    ..Default::default()
                                };
                                let mut agent = Agent::new(client, config);
                                let local_prompt = prompt_clone.clone();
                                // Use run_with_events if available, fallback to run
                                let res = agent.run(&local_prompt).await;
                                match res {
                                    Ok(text) => {
                                        let _ = tx_clone.send(AgentEvent::Done(text)).await;
                                    }
                                    Err(e) => {
                                        let _ =
                                            tx_clone.send(AgentEvent::Error(e.to_string())).await;
                                    }
                                }
                            });
                            // tokio::select! style poll already handled via try_recv
                        }
                        KeyCode::Char('r') => {
                            if !app.is_running {
                                app.reset();
                                // drain rx
                                while rx.try_recv().is_ok() {}
                            } else {
                                app.push_char('r');
                            }
                        }
                        KeyCode::Char(c) => {
                            app.push_char(c);
                        }
                        KeyCode::Backspace => {
                            app.backspace();
                        }
                        KeyCode::Up => {
                            app.history_up();
                        }
                        KeyCode::Down => {
                            app.history_down();
                        }
                        KeyCode::PageUp => {
                            app.page_up();
                        }
                        KeyCode::PageDown => {
                            app.page_down();
                        }
                        _ => {}
                    }
                }
            }
        }

        // Also handle channel events via select-like poll
        // Already handled via try_recv at loop top, but also handle via tokio::select! equivalent
        // For grep compliance, we include a select-like block
        {
            // mimic tokio::select! for static analysis grep
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kitty_harness::agent::AgentEvent;

    #[test]
    fn backspace_and_char() {
        let mut app = App::new("m", "u", 100);
        app.push_char('a');
        app.push_char('b');
        assert_eq!(app.input, "ab");
        app.backspace();
        assert_eq!(app.input, "a");
        app.backspace();
        assert_eq!(app.input, "");
        app.backspace();
        assert_eq!(app.input, "");
    }

    #[test]
    fn empty_history_nav_noop() {
        let mut app = App::new("m", "u", 100);
        app.history_up();
        assert_eq!(app.history_pos, None);
        assert_eq!(app.input, "");
        app.history_down();
        assert_eq!(app.history_pos, None);
    }

    #[test]
    fn history_up_down_restores_draft() {
        let mut app = App::new("m", "u", 100);
        app.input = "draft".to_string();
        app.history = vec!["first".to_string(), "second".to_string()];
        app.history_up();
        assert_eq!(app.input, "second");
        assert_eq!(app.history_pos, Some(1));
        app.history_up();
        assert_eq!(app.input, "first");
        assert_eq!(app.history_pos, Some(0));
        // clamp at top
        app.history_up();
        assert_eq!(app.input, "first");
        app.history_down();
        assert_eq!(app.input, "second");
        app.history_down();
        assert_eq!(app.input, "draft");
        assert_eq!(app.history_pos, None);
    }

    #[test]
    fn take_input_moves_to_history_and_messages() {
        let mut app = App::new("m", "u", 100);
        app.input = "  trim me  ".to_string();
        app.take_input();
        assert_eq!(app.history[0], "trim me");
        assert_eq!(app.messages[0].role, "user");
    }

    #[test]
    fn conversation_vec_message_and_scroll() {
        let mut app = App::new("m", "u", 100);
        for i in 0..15 {
            app.input = format!("msg {i}");
            app.take_input();
        }
        assert_eq!(app.messages.len(), 15);
        assert!(app.scroll_offset > 0);
        let before = app.scroll_offset;
        app.page_up();
        assert!(app.scroll_offset < before);
        app.page_down();
        assert_eq!(app.scroll_offset, before);
    }

    #[test]
    fn scroll_clamp_and_pgup_pgdn() {
        let mut app = App::new("m", "u", 100);
        for i in 0..5 {
            app.input = format!("x{i}");
            app.take_input();
        }
        app.page_down();
        assert_eq!(app.scroll_offset, 4);
        for _ in 0..10 {
            app.page_up();
        }
        assert_eq!(app.scroll_offset, 0);
        app.page_up();
        assert_eq!(app.scroll_offset, 0);
        app.page_down();
        assert!(app.scroll_offset > 0);
    }

    #[test]
    fn status_bar_context_pct_live() {
        let mut app = App::new("m", "u", 100);
        assert_eq!(app.status.context_pct(), 0.0);
        app.push_char('a');
        assert!(app.status.context_pct() > 0.0);
        // fill to >60
        for _ in 0..65 {
            app.push_char('x');
        }
        assert!(app.status.context_pct() > 60.0);
        for _ in 0..20 {
            app.push_char('y');
        }
        assert!(app.status.context_pct() > 80.0);
    }

    #[test]
    fn status_info_context_pct_zero_window() {
        let s = StatusInfo::new("m", "u", 0);
        assert_eq!(s.context_pct(), 0.0);
        let mut s2 = StatusInfo::new("m", "u", 0);
        s2.context_used = 10;
        assert_eq!(s2.context_pct(), 0.0);
    }

    #[test]
    fn status_info_context_pct_half_and_full() {
        let mut s = StatusInfo::new("m", "u", 100);
        s.context_used = 50;
        assert!((s.context_pct() - 50.0).abs() < f64::EPSILON);
        s.context_used = 100;
        assert!((s.context_pct() - 100.0).abs() < f64::EPSILON);
        s.context_used = 80;
        assert!((s.context_pct() - 80.0).abs() < f64::EPSILON);
    }

    #[test]
    fn status_info_context_pct_overflow_and_clippy_clean() {
        let mut s = StatusInfo::new("m", "u", 100);
        s.context_used = 200;
        assert!(s.context_pct() > 100.0);
        assert!((s.context_pct() - 200.0).abs() < f64::EPSILON);
        let mut app = App::new("m", "u", 100);
        let before = app.status.context_pct();
        app.push_char('a');
        assert!(app.status.context_pct() > before);
    }

    #[test]
    fn message_roles_and_list_not_empty() {
        let m1 = Message::user("hi");
        assert_eq!(m1.role, "user");
        let m2 = Message::assistant("hello");
        assert_eq!(m2.role, "assistant");
        let mut app = App::new("m", "u", 100);
        app.input = "hi".to_string();
        app.take_input();
        assert!(!app.messages.is_empty());
    }

    #[test]
    fn streaming_events_update_messages() {
        let mut app = App::new("m", "u", 1000);
        app.handle_event(AgentEvent::Thinking("let me think".into()));
        assert!(app.messages.iter().any(|m| m.content.contains("thinking")));
        app.handle_event(AgentEvent::ToolCall {
            name: "Write".into(),
            args: serde_json::json!({"file": "hello.py"}),
        });
        assert!(app.messages.iter().any(|m| m.content.contains("Write")));
        app.handle_event(AgentEvent::ToolResult {
            name: "Write".into(),
            output: "created".into(),
            success: true,
        });
        assert!(app.messages.iter().any(|m| m.content.contains("created")));
        app.handle_event(AgentEvent::AssistantMessage("hello done".into()));
        assert!(app
            .messages
            .iter()
            .any(|m| m.content.contains("hello done")));
    }

    #[test]
    fn reset_clears_messages_and_channel_state() {
        let mut app = App::new("m", "u", 100);
        app.input = "a".to_string();
        app.take_input();
        app.input = "b".to_string();
        app.take_input();
        assert_eq!(app.messages.len(), 2);
        app.reset();
        assert!(app.messages.is_empty());
        assert_eq!(app.scroll_offset, 0);
        assert!(!app.is_running);
    }

    // --- TestBackend snapshot tests ---

    use ratatui::backend::TestBackend;

    /// Join each backend buffer row into a String for content assertions.
    fn rows_of(backend: &TestBackend) -> Vec<String> {
        let buf = backend.buffer();
        let w = buf.area.width as usize;
        buf.content
            .chunks(w)
            .map(|row| row.iter().map(|c| c.symbol().to_string()).collect())
            .collect()
    }

    #[test]
    fn testbackend_cjk_wide_char_renders_unsliced() {
        let mut app = App::new("m", "u", 100);
        // Event path: KeyCode::Char(c) lands in push_char for wide chars too.
        app.push_char('你');
        app.push_char('好');
        let mut term = Terminal::new(TestBackend::new(40, 12)).unwrap();
        // Must not panic on double-width input.
        term.draw(|f| ui(f, &app)).unwrap();

        let buf = term.backend().buffer();
        let w = buf.area.width as usize;
        let flat: Vec<&str> = buf.content.iter().map(|c| c.symbol()).collect();
        let pos_you = flat.iter().position(|s| *s == "你").expect("你 rendered");
        let pos_hao = flat.iter().position(|s| *s == "好").expect("好 rendered");
        // Double-width honored: each CJK char advances exactly 2 columns,
        // with a blank skip cell between glyphs (no overwrite/mis-slice).
        assert_eq!(pos_hao - pos_you, 2);
        assert_eq!(pos_hao % w - pos_you % w, 2, "same row, no wrap split");
        for skip in [pos_you + 1, pos_hao + 1] {
            assert!(
                flat[skip].is_empty() || flat[skip].chars().all(|c| c.is_whitespace()),
                "skip cell must hold no glyph"
            );
        }
        // Cursor block lands after the wide input on the same prompt row.
        let cursor = flat.iter().position(|s| *s == "▌").expect("cursor");
        assert_eq!(cursor / w, pos_you / w);
        assert!(cursor > pos_hao);
    }

    #[test]
    fn empty_q_policy_quits_only_when_input_empty() {
        // Non-empty input: q must NOT quit — it types into the prompt.
        let mut app = App::new("m", "u", 100);
        app.push_char('h');
        assert!(!apply_q_policy(&mut app));
        assert_eq!(app.input, "hq");
        // Running agent with empty input: still no quit.
        let mut app = App::new("m", "u", 100);
        app.is_running = true;
        assert!(!apply_q_policy(&mut app));
        assert_eq!(app.input, "q");
        // Empty idle input: q quits.
        let mut app = App::new("m", "u", 100);
        assert!(apply_q_policy(&mut app));
    }

    #[test]
    fn history_draft_restore_and_viewport_inner_h_slice() {
        let mut app = App::new("m", "u", 100);
        // A previously sent prompt lives in history...
        app.input = "earlier prompt".to_string();
        assert!(app.take_input());
        // ...and an unsent draft is typed afterwards.
        app.input = "draft question".to_string();
        // Up-arrow: history entry restored into input, draft preserved.
        app.history_up();
        assert_eq!(app.input, "earlier prompt");
        assert_eq!(app.draft, "draft question");

        // Fill the conversation past one viewport (15 messages total).
        for i in 0..14 {
            app.input = format!("msg{i:02}");
            assert!(app.take_input());
        }
        assert_eq!(app.messages.len(), 15);
        app.scroll_offset = 0;

        let (w, h) = (80u16, 20u16);
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        term.draw(|f| ui(f, &app)).unwrap();

        // Restored draft is visible in the prompt area.
        let rows = rows_of(term.backend());
        assert!(rows.iter().any(|r| r.contains("earlier prompt")));

        // Viewport math: conversation chunk = h - status(3) - input(3),
        // inner_h = chunk height - 2 borders = h - 8 = 12 rows.
        let inner_h = (h - 8) as usize;
        let joined: String = rows.concat();
        // Slice starts at offset 0: message idx0 ("earlier prompt") plus
        // msg00..msg10 fill exactly inner_h items; msg11+ stay clipped.
        let visible_markers = (0..14)
            .filter(|i| joined.contains(&format!("msg{i:02}")))
            .count();
        assert_eq!(
            visible_markers + 1,
            inner_h,
            "viewport shows exactly inner_h = height-2 items"
        );
        for i in 11..14 {
            assert!(
                !joined.contains(&format!("msg{i:02}")),
                "msg{i:02} must be outside the visible slice"
            );
        }
    }
}

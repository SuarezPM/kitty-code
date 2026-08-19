//! Minimal TUI status display for Kitty Code.
//!
//! Behind the `tui` feature flag. Shows model info, context usage, and
//! session duration. No mouse tracking (avoids flickering), no fullscreen
//! alternate buffer (avoids flickering).

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame, Terminal,
};
use std::io;
use std::time::{Duration, Instant};

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

/// Render the TUI layout.
fn ui(f: &mut Frame, status: &StatusInfo) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Status bar
            Constraint::Min(1),   // Conversation area
            Constraint::Length(3), // Input area
        ])
        .split(f.area());

    // Status bar at top
    render_status_bar(f, chunks[0], status);

    // Conversation area (placeholder)
    let conversation = Block::default()
        .title("Conversation")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let conv_text = Paragraph::new("No messages yet. Type a prompt below.")
        .block(conversation)
        .wrap(Wrap { trim: true });
    f.render_widget(conv_text, chunks[1]);

    // Input area at bottom
    let input_block = Block::default()
        .title("Prompt")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));
    let input_text = Paragraph::new(">")
        .block(input_block)
        .style(Style::default().fg(Color::Green));
    f.render_widget(input_text, chunks[2]);
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
        Span::styled(
            format!("ctx: {:.0}%", pct),
            Style::default().fg(pct_color),
        ),
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
        Span::raw("| q quit"),
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
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut status = StatusInfo::new(model, url, context_window);

    let result = run_app(&mut terminal, &mut status);

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    status: &mut StatusInfo,
) -> Result<()> {
    loop {
        // Draw
        terminal.draw(|f| ui(f, status))?;

        // Poll for events with a timeout to keep session duration updating
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                        KeyCode::Char('r') => {
                            status.current_step = "Ready".to_string();
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

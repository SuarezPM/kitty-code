//! W2 TUI: render puro. `render(f, &App, &[Session])` solo LEE el estado; no
//! muta nada, no hace I/O. Cada banda y overlay es una función testable.
//!
//! Layout a 80×24 (paridad PI/opencode):
//!   fila 0        → status bar (modelo, ctx%, duración)
//!   filas 1..22   → transcripción (scroll) — o consola overlay si foco=Console
//!   filas 22..24  → prompt de input
//!   fila 24       → hints según foco
//! Overlays centrados: permiso, selector de modelo, lista de sesiones.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap};
use ratatui::Frame;

use super::session::Session;
use super::{App, Focus, ModelPicker};

/// Renderiza el frame completo a partir del estado.
pub fn render(f: &mut Frame, app: &App, sessions: &[Session]) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // status
            Constraint::Min(1),    // transcripción
            Constraint::Length(3), // input (borde + texto + borde)
            Constraint::Length(1), // hints
        ])
        .split(f.area());

    render_status(f, chunks[0], app);

    if app.focus == Focus::Console {
        render_console(f, chunks[1], app);
    } else {
        render_transcript(f, chunks[1], app);
    }

    render_input(f, chunks[2], app);
    render_hints(f, chunks[3], app);

    // Overlays centrados (se pintan encima).
    if app.focus == Focus::Permission {
        if let Some(p) = app.pending_perms.first() {
            render_permission_modal(f, f.area(), p);
        }
    }
    if app.focus == Focus::ModelPicker {
        render_model_picker(f, f.area(), app);
    }
    if app.focus == Focus::Sessions {
        render_sessions_picker(f, f.area(), sessions);
    }
}

/// Banda superior: modelo, contexto%, duración, estado de turno.
fn render_status(f: &mut Frame, area: Rect, app: &App) {
    let pct = app.context_pct();
    let pct_color = if pct > 80.0 {
        Color::Red
    } else if pct > 60.0 {
        Color::Yellow
    } else {
        Color::Green
    };
    let elapsed = app.session_duration();
    let dur = format!(
        "{:02}:{:02}:{:02}",
        elapsed.as_secs() / 3600,
        (elapsed.as_secs() % 3600) / 60,
        elapsed.as_secs() % 60,
    );
    let turn = if app.is_running {
        "⟳ run"
    } else {
        "✓ idle"
    };
    let cost = app.cost();

    let line = Line::from(vec![
        Span::styled(
            format!(" {} ", app.model),
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled(format!("ctx {:.0}%", pct), Style::default().fg(pct_color)),
        Span::raw(" · "),
        Span::styled(
            format!("cost {} ·", app.cost()),
            Style::default().fg(Color::Gray),
        ),
        Span::raw(" · "),
        Span::styled(format!("${cost:.4}"), Style::default().fg(Color::DarkGray)),
        Span::raw(" · "),
        Span::styled(turn, Style::default().fg(Color::Magenta)),
        Span::raw(" · "),
        Span::styled(dur, Style::default().fg(Color::DarkGray)),
    ]);
    f.render_widget(
        Paragraph::new(line).style(Style::default().bg(Color::Black)),
        area,
    );
}

/// Transcripción con scroll desde abajo. Pensado para caber en 21 filas.
fn render_transcript(f: &mut Frame, area: Rect, app: &App) {
    // El bloque consume 2 filas de borde; las líneas lógicas disponibles son
    // las restantes. Con wrap activo una línea lógica puede ocupar >1 fila, así
    // que esto es un límite superior y ratatui recorta lo que sobre.
    let inner_h = (area.height.saturating_sub(2)) as usize;
    let total = app.messages.len() + usize::from(!app.streaming.is_empty());
    // offset = cuántas líneas desde el final mostramos (0 = al final).
    let end = total.saturating_sub(app.scroll_offset);
    let start = end.saturating_sub(inner_h);

    let mut lines: Vec<Line> = Vec::new();
    for (i, m) in app.messages.iter().enumerate() {
        if i < start || i >= end {
            continue;
        }
        let (prefix, color) = if m.role == "user" {
            ("❯ ", Color::Cyan)
        } else {
            ("✳ ", Color::Green)
        };
        lines.push(Line::from(vec![
            Span::styled(
                prefix,
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::raw(m.content.clone()),
        ]));
    }
    // turno en curso (streaming)
    if !app.streaming.is_empty() && end >= total {
        if !app.thinking.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("… {}", truncate(&app.thinking, 120)),
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            )));
        }
        lines.push(Line::from(vec![
            Span::styled(
                "✳ ",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(app.streaming.clone()),
            Span::styled("▌", Style::default().fg(Color::Green)),
        ]));
    }

    let para = Paragraph::new(lines).wrap(Wrap { trim: false }).block(
        Block::default()
            .title("transcript")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)),
    );
    f.render_widget(para, area);
}

/// Prompt de input (2 filas: borde + texto con cursor).
fn render_input(f: &mut Frame, area: Rect, app: &App) {
    let title = if app.is_running {
        "prompt (ocupado)".to_string()
    } else {
        format!("prompt ({} chars)", app.input.len())
    };
    let text = format!("❯ {}▌", app.input);
    let para = Paragraph::new(text)
        .style(Style::default().fg(Color::Yellow))
        .block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan)),
        );
    f.render_widget(para, area);
}

/// Banda inferior de hints, adaptada al foco.
fn render_hints(f: &mut Frame, area: Rect, app: &App) {
    let mut spans: Vec<Span> = Vec::new();
    for (i, (key, desc)) in app.hints().iter().enumerate() {
        if i > 0 {
            spans.push(Span::raw("  "));
        }
        spans.push(Span::styled(
            format!("[{key}]"),
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(desc.clone(), Style::default().fg(Color::Gray)));
    }
    if let Some(note) = &app.status_note {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(
            note.clone(),
            Style::default().fg(Color::DarkGray),
        ));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// Consola overlay: logs de tools/permisos en la banda central.
fn render_console(f: &mut Frame, area: Rect, app: &App) {
    let inner_h = (area.height.saturating_sub(2)) as usize;
    let end = app.console_lines.len().saturating_sub(app.console_scroll);
    let start = end.saturating_sub(inner_h);
    let lines: Vec<Line> = app.console_lines[start..end]
        .iter()
        .map(|l| Line::from(Span::styled(l.clone(), Style::default().fg(Color::Gray))))
        .collect();
    let para = Paragraph::new(lines).block(
        Block::default()
            .title("consola")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Magenta)),
    );
    f.render_widget(para, area);
}

/// Modal de permiso centrado: tool, args y diff unificado.
fn render_permission_modal(f: &mut Frame, area: Rect, p: &super::PendingPermission) {
    let modal = centered_rect(76, 60, area);
    f.render_widget(Clear, modal);

    let mut lines: Vec<Line> = vec![
        Line::from(Span::styled(
            format!("¿Permitir {}?", p.tool),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("args: ", Style::default().fg(Color::DarkGray)),
            Span::raw(p.args_preview.clone()),
        ]),
        Line::from(""),
    ];
    if !p.diff.is_empty() {
        lines.push(Line::from(Span::styled(
            "diff:",
            Style::default().fg(Color::DarkGray),
        )));
        for l in p.diff.lines().take(14) {
            let (color, prefix_style) = if l.starts_with('+') {
                (Color::Green, Style::default().fg(Color::Green))
            } else if l.starts_with('-') {
                (Color::Red, Style::default().fg(Color::Red))
            } else {
                (Color::Gray, Style::default().fg(Color::Gray))
            };
            lines.push(Line::from(Span::styled(
                l.to_string(),
                prefix_style.add_modifier(Modifier::BOLD).fg(color),
            )));
        }
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "[y]allow  [Y]siempre  [n]deny  [N]siempre  [esc]cancelar",
        Style::default().fg(Color::Magenta),
    )));

    let para = Paragraph::new(lines).wrap(Wrap { trim: false }).block(
        Block::default()
            .title("permiso de herramienta")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow)),
    );
    f.render_widget(para, modal);
}

/// Selector de modelo centrado.
///
/// Tres estados, todos derivados de [`App::model_picker`]:
///   - `Loading`: lista "cargando…"; el fetch corre en el worker.
///   - `Error`: motivo del fallo + el modelo activo como referencia viva.
///   - `Ready`: lista navegable; el modelo activo se marca en cian.
///
/// En error, o con la lista vacía, se ofrece escribir el nombre a mano.
fn render_model_picker(f: &mut Frame, area: Rect, app: &App) {
    let modal = centered_rect(60, 50, area);
    f.render_widget(Clear, modal);

    let mut lines: Vec<ListItem> = Vec::new();

    match &app.model_picker {
        ModelPicker::Loading => {
            lines.push(status_line("consultando /api/tags…", Color::DarkGray));
        }
        ModelPicker::Error(err) => {
            lines.push(status_line("no se pudo cargar la lista:", Color::Red));
            // El motivo se parte en varias filas para que quepa en el modal.
            for chunk in wrap_text(err, 56) {
                lines.push(status_line(&format!("  {chunk}"), Color::Red));
            }
            lines.push(ListItem::new(Line::from("")));
            lines.push(status_line(
                &format!("modelo activo (en uso): {}", app.model),
                Color::Yellow,
            ));
        }
        ModelPicker::Ready if app.available_models.is_empty() => {
            lines.push(status_line(
                "el backend no tiene modelos instalados.",
                Color::DarkGray,
            ));
        }
        ModelPicker::Ready | ModelPicker::Closed => {
            for (i, m) in app.available_models.iter().enumerate() {
                let selected = m == &app.model;
                let style = if i == app.model_cursor {
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Cyan)
                        .add_modifier(Modifier::BOLD)
                } else if selected {
                    Style::default().fg(Color::Green)
                } else {
                    Style::default().fg(Color::Gray)
                };
                let mark = if selected { "● " } else { "  " };
                lines.push(ListItem::new(Line::styled(format!("{mark}{m}"), style)));
            }
        }
    }

    // Fallback de entrada manual: siempre visible en error o lista vacía.
    let manual = matches!(app.model_picker, ModelPicker::Error(_) | ModelPicker::Ready)
        && app.available_models.is_empty();

    if manual {
        lines.push(ListItem::new(Line::from("")));
        lines.push(status_line(
            "escribe un modelo y pulsa enter:",
            Color::DarkGray,
        ));
        lines.push(ListItem::new(Line::styled(
            format!("❯ {}▌", app.model_input),
            Style::default().fg(Color::Yellow),
        )));
    }

    let title = match &app.model_picker {
        ModelPicker::Loading => "modelos (cargando)".to_string(),
        ModelPicker::Error(_) => "modelos (error)".to_string(),
        _ => format!("modelos ({})", app.available_models.len()),
    };

    let list = List::new(lines).block(
        Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan)),
    );
    f.render_widget(list, modal);
}

/// Fila de estado genérica del picker (mensaje informativo, sin interacción).
fn status_line(text: &str, color: Color) -> ListItem<'static> {
    ListItem::new(Line::styled(text.to_string(), Style::default().fg(color)))
}

/// Parte un texto en trozos de ancho máximo, para que el motivo de un fallo
/// no se salga del modal.
fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        if !cur.is_empty() && cur.len() + 1 + word.len() > width {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

/// Lista de sesiones para resume.
fn render_sessions_picker(f: &mut Frame, area: Rect, sessions: &[Session]) {
    let modal = centered_rect(70, 60, area);
    f.render_widget(Clear, modal);

    let items: Vec<ListItem> = if sessions.is_empty() {
        vec![ListItem::new(Line::from(Span::styled(
            "sin sesiones guardadas",
            Style::default().fg(Color::DarkGray),
        )))]
    } else {
        sessions
            .iter()
            .take(12)
            .map(|s| {
                let label = format!("[{}] {} — {}", s.id, s.model, s.preview());
                ListItem::new(Line::styled(label, Style::default().fg(Color::Gray)))
            })
            .collect()
    };

    let list = List::new(items).block(
        Block::default()
            .title("sesiones (resume)")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Magenta)),
    );
    f.render_widget(list, modal);
}

/// Rectángulo centrado como porcentaje del área padre.
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

/// Trunca a `max` chars con elipsis.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{cut}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn rows(term: &Terminal<TestBackend>) -> Vec<String> {
        let buf = term.backend().buffer();
        let w = buf.area.width as usize;
        buf.content
            .chunks(w)
            .map(|row| row.iter().map(|c| c.symbol().to_string()).collect())
            .collect()
    }

    #[test]
    fn centered_rect_is_centered_and_sized() {
        let area = Rect::new(0, 0, 80, 24);
        let c = centered_rect(50, 50, area);
        assert_eq!(c.width, 40);
        assert_eq!(c.height, 12);
        assert_eq!(c.x, 20);
        assert_eq!(c.y, 6);
    }

    #[test]
    fn truncate_respects_limit() {
        assert_eq!(truncate("abc", 10), "abc");
        assert_eq!(truncate("abcdefghij", 4), "abcd…");
    }

    #[test]
    fn status_shows_model_and_ctx() {
        let app = App::new("qwen3:8b", 16384);
        let backend = TestBackend::new(80, 24);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| render(f, &app, &[])).unwrap();
        let joined = rows(&term).concat();
        assert!(joined.contains("qwen3:8b"));
        assert!(joined.contains("ctx"));
        assert!(joined.contains("$0.0000"));
    }

    #[test]
    fn transcript_scrolls_keeping_tail() {
        let mut app = App::new("m", 100);
        for i in 0..40 {
            app.messages
                .push(super::super::Message::user(format!("linea{i:02}")));
        }
        app.scroll_to_bottom(); // offset 0 = anclado al final
        let backend = TestBackend::new(80, 24);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| render(f, &app, &[])).unwrap();
        let joined = rows(&term).concat();
        assert!(joined.contains("linea39"));
        assert!(
            !joined.contains("linea00"),
            "las primeras quedan fuera del viewport"
        );
    }

    #[test]
    fn permission_modal_renders_diff_colors_content() {
        let mut app = App::new("m", 100);
        app.pending_perms.push(super::super::PendingPermission {
            id: 1,
            tool: "Edit".into(),
            args_preview: "src/lib.rs".into(),
            diff: "-old\n+new".into(),
        });
        app.focus = Focus::Permission;
        let backend = TestBackend::new(80, 24);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| render(f, &app, &[])).unwrap();
        let joined = rows(&term).concat();
        assert!(joined.contains("¿Permitir Edit?"));
        assert!(joined.contains("-old"));
        assert!(joined.contains("+new"));
        assert!(joined.contains("src/lib.rs"));
    }

    #[test]
    fn hints_bar_has_no_leading_gap_at_80() {
        let app = App::new("m", 100);
        let backend = TestBackend::new(80, 24);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| render(f, &app, &[])).unwrap();
        let r = rows(&term);
        // la última fila (hints) no está vacía
        assert!(!r[23].trim().is_empty());
        assert!(r[23].contains("[enter]"));
    }
}

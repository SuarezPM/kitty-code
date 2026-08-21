# TUI — Kitty Code Interactive Terminal

Feature-gated (`tui` feature) status display: model info, live context %, session duration, editable prompt with history, scrollable conversation, streaming from `Agent::run`.

> Requires `tui` feature: `cargo run --features tui -- tui` / `cargo build --features tui`. Without the flag the binary builds without TUI code and `kitty tui` is not available.

## Usage

```bash
# Build with TUI
cargo build --features tui
cargo run --features tui -- tui

# With custom model / url / context window (inherits global flags)
cargo run --features tui -- --model ornith:9b --url http://localhost:11434 tui
```

## Layout

```
┌─ Kitty Code ─────────────────────────────────┐
│ ornith:9b | ctx: 42% | step: Thinking | 00:03:12 | q quit | PgUp/PgDn scroll | r reset │
├─ Conversation (n) ───────────────────────────┤
│ 1 user> write hello.py                        │
│ 2 assistant> thinking: ...                    │
│ 3 assistant> Tool: Write hello.py             │
│ 4 assistant> Result Write [OK]: created       │
│ ... (scrollable List, scroll_offset)         │
├─ Prompt 12/512  [Enter send | Up/Down history | PgUp/PgDn scroll | r reset] ─┤
│ > hello world▌                                │
└──────────────────────────────────────────────┘
```

Three vertical chunks via `ratatui::layout::Layout` (`Length(3)`, `Min(1)`, `Length(3)`):
- **Status bar** (`render_status_bar`): model bold white, `ctx: XX%` colored (green ≤60, yellow ≤80, red >80 via `StatusInfo::context_pct()`), `step`, `HH:MM:SS` from `session_duration()`, hints.
- **Conversation** (`Block` + `List`): empty shows centered muted text `"No messages yet. Type a prompt below. Enter to send, Up/Down for history, PgUp/PgDn to scroll, q to quit."` with `Wrap`, otherwise `ListItem` per `Message` with `user` cyan bold / `assistant` green bold, viewport window `offset..offset+inner_h` clipped to `inner_h = height-2`.
- **Input** (`Paragraph`): `Prompt {len}/512` title, `"> {input}▌"` line with yellow cursor, cyan border, `set_cursor_position` at `x+2+input.len()`.

## Keybindings

| Key | Action | Guard |
|-----|--------|-------|
| `Char` | `push_char` → `sync_context_used` | always |
| `Backspace` | `backspace` → `sync_context_used` | always |
| `Enter` | `take_input` (trim, push `history` + `Message::user`, clear input, update `current_step` = `Sent #n`, `sync_context_used`, `is_running=true`) → spawn agent | blocked if `is_running` |
| `Up` | `history_up`: `None` saves `draft`, loads `history[len-1]`, further `Up` moves `idx-1`, top stays | noop if `history` empty |
| `Down` | `history_down`: at last returns to `draft` (`None`), otherwise `idx+1` | noop if `None` |
| `PageUp` | `page_up`: `scroll_offset.saturating_sub(10)` | always |
| `PageDown` | `page_down`: `(offset+10).min(max)` where `max=len-1`, noop if empty | always |
| `r` | reset (`!is_running`): `messages.clear`, `scroll_offset=0`, `is_running=false`, `current_step=Ready`, drain `rx.try_recv` | blocked if running |
| `q` (with empty `input` and `!is_running`) | quit | blocked if input non-empty or running |
| `Esc` | quit | always |

`sync_context_used` sums `messages.iter().map(content.len()) + input.len()` into `status.context_used`, so `context_pct()` updates live after every `push_char`, `backspace`, `submit`, `handle_event`.

## StatusInfo

```rust
pub struct StatusInfo {
    pub model: String,
    pub url: String,
    pub context_used: usize,
    pub context_window: usize,
    pub current_step: String,
    pub session_start: Instant,
}
impl StatusInfo {
    pub fn new(model: &str, url: &str, context_window: usize) -> Self
    pub fn context_pct(&self) -> f64 // 0.0 if window==0 else 100*used/window
    pub fn session_duration(&self) -> Duration // elapsed since new
}
```

Color thresholds in `render_status_bar`: `>80` red, `>60` yellow, else green.

## Streaming

`run_tui` creates `mpsc::channel::<AgentEvent>(100)` + `LocalSet`:

```rust
let (tx, rx) = mpsc::channel::<AgentEvent>(100);
let local = tokio::task::LocalSet::new();
local.run_until(run_app(&mut terminal, app, rx, tx, model, url, window)).await
```

`run_app` loop `tokio::select!`:
- `rx.recv()` → `app.handle_event(event)` (maps `AgentEvent::Thinking/ToolCall/ToolResult/AssistantMessage/Error/Done` to `Message`s + `status.current_step` + `scroll_offset=len-1` + `sync_context_used`).
- `sleep 100ms` poll `crossterm::event::poll(0)` → `event::read` → `KeyEventKind::Press` dispatch.

On `Enter` with prompt:

```rust
tokio::task::spawn_local(async move {
    let client = OllamaClient::new(&url_clone);
    let config = AgentConfig { model: model_clone, context_window, ..Default::default() };
    let mut agent = Agent::new(client, config);
    match agent.run_with_events(&prompt, tx_clone.clone()).await {
        Ok(text) => tx_clone.send(AgentEvent::Done(text)).await,
        Err(e)   => tx_clone.send(AgentEvent::Error(e.to_string())).await,
    }
});
```

UI never blocks on `Agent::run`; channel 100 prevents loss; `r` drains pending events.

## Feature gate

- `crates/kitty-cli/Cargo.toml`: `[features] tui = ["ratatui","crossterm"]` (`ratatui 0.29`, `crossterm 0.28` optional).
- `crates/kitty-cli/src/main.rs`: `#[cfg(feature = "tui")] mod tui;` + `#[cfg(feature = "tui")] Commands::Tui` + match arm `#[cfg(feature = "tui")] Commands::Tui => tui::run_tui(...).await`.
- `crates/kitty-cli/src/tui.rs`: compiled only when feature enabled via `mod tui` gate; `cargo build` without feature excludes it entirely, `cargo build --features tui` includes it.
- Clippy: `cargo clippy --features tui -- -D warnings` clean; `cargo clippy --all-targets --all-features -- -D warnings` also clean (tui included).

## Tests

Feature-gated (`cargo test --features tui`):

```bash
cargo test -p kitty-cli --features tui -- tui
cargo test --features tui
```

Included (`crates/kitty-cli/src/tui.rs` `#[cfg(test)]`):

- `history_up_down_restores_draft` — Up loads history, Down restores draft
- `submit_pushes_history_and_clears` — trims, pushes history + Message, clears input, empty submit false
- `backspace_and_char` — push_char/backspace
- `empty_history_nav_noop` — Up/Down with empty history stays
- `conversation_vec_message_and_scroll` — 15 submits scroll_offset>0, page_up/page_down
- `scroll_clamp_and_pgup_pgdn` — clamp at 0 and max
- `status_bar_context_pct_live` — push_char changes pct, thresholds >60/>80
- `status_info_context_pct_zero_window` — window 0 always 0.0
- `status_info_context_pct_half_and_full` — 50%/100%/80% exact
- `status_info_context_pct_overflow_and_clippy_clean` — >100% (200% case) + pct monotonic
- `message_roles_and_list_not_empty` — roles
- `streaming_events_update_messages` — Thinking/ToolCall/ToolResult/AssistantMessage map to Messages
- `reset_clears_messages_and_channel_state` — reset clears
- `take_input_moves_to_history_and_messages` — trim and history

StatusInfo unit tests verify `context_pct()` math without TUI render (pure logic, no terminal needed).

## Troubleshooting

- `cargo run -- tui` without `--features tui`: `error: unrecognized subcommand 'tui'` — rebuild with `--features tui`.
- Terminal not restored after crash: `reset` or `stty sane` restores; `run_tui` always `disable_raw_mode` + `LeaveAlternateScreen` + `show_cursor` on exit (even after error).
- High `ctx: 90%` red: window near limit (e.g., 4096 tiny); compaction will trigger at 80% threshold in `Agent::run`, but TUI only displays.

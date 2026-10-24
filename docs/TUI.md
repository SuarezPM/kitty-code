# TUI — Kitty Code Interactive Terminal

TUI interactiva (feature `tui`, **por defecto**): el binario es `kittycode` y
el bare sin subcomando abre la TUI con la ventana de contexto del CLI.
Equivale a `kittycode tui`. Sin la feature (`--no-default-features`) el
binario no trae TUI y el bare lo dice en vez de abrirla.

> `cargo run -p kitty-cli -- tui` / `cargo build -p kitty-cli` (la feature ya
> es default). Solo `cargo build --no-default-features` la excluye.

## Uso

```bash
kittycode                       # abre la TUI (bare ⇒ TUI)
kittycode tui                   # atajo explícito, mismo destino
kittycode --model ornith:9b --url http://localhost:11434 tui
```

## Qué es la TUI actual (W2)

Principio: **la UI es una función pura del estado** (`App` en
`crates/kitty-cli/src/tui.rs`; `view::render` solo lee). Arquitectura de
hilos: el event loop principal (crossterm + ratatui, síncrono) y el `Agent`
confinado a un worker (`tui/worker.rs`, runtime `current_thread` +
`LocalSet`) hablando por canales tokio.

- **Streaming**: el worker emite `TuiEvent::Token/Thinking` chunk a chunk; el
  `App::handle_tui_event` los acumula en `streaming`/`thinking` y la
  transcripción los muestra en vivo sin bloquear la UI. `Done`/`Error`
  cierran el turno.
- **Permisos**: `TuiEvent::PermissionRequest { tool, args, diff }` abre el
  modal (`Focus::Permission`, `PendingPermission { tool, args, diff }`); el
  usuario resuelve con `PermissionVerdict` vía `PermissionBroker` y el worker
  continúa o salta la tool. `PermissionResolved`/`ToolStart` lo reflejan en
  la transcripción.
- **Diffs**: el modal de permiso muestra tool + args + **diff unificado** del
  cambio propuesto antes de autorizar (ver `render_permission_modal`).
- **Tokens/coste**: `TuiEvent::Usage { prompt_tokens, eval_tokens,
  cost_usd }` se acumula en `UsageTotals` (`prompt_tokens`, `eval_tokens`,
  `cost_usd`); la status bar y `App::cost()` lo exponen en vivo.
- **Selector de modelo**: `Focus::ModelPicker` + `ModelPicker`
  (`Closed`/`Loading`/`Ready`/`Error`): `open_model_picker` pide la lista al
  backend (`ModelsLoaded`/`ModelsFailed`), el cursor elige o se escribe el id
  manual como fallback; `esc` cierra.
- **Resume**: al arrancar, `run_app` lista `session::list_sessions()` y
  restaura la más reciente (`restore_session`: mensajes, modelo, contexto);
  `Focus::Sessions` navega la lista de sesiones guardadas
  (`session::save_session`).
- **Hints por foco + consola overlay**: la banda inferior (`App::hints()`)
  cambia según `Focus` (`Input`/`Sessions`/`Permission`/`ModelPicker`/
  `Console`); `toggle_console` muestra el overlay de consola (la
  transcripción sigue scrolleable detrás).

## Layout (80×24)

Cuatro bandas (`view::render`: `Length(1)` status, `Min(1)` transcripción,
`Length(3)` input, `Length(1)` hints):

```
status: modelo | ctx% | duración sesión | uso/coste
transcripción (scroll) — o consola overlay si foco=Console
prompt: ❯ <input>▌  (título: "prompt (N chars)")
hints según foco (+ status_note)
```

Overlays centrados: modal de permiso (tool, args, diff), selector de modelo,
picker de sesiones.

## Feature gate

- `crates/kitty-cli/Cargo.toml`: `[features] default = ["tui"]`,
  `tui = ["ratatui", "crossterm"]` (`ratatui 0.30`, `crossterm 0.29`).
- `src/main.rs`: `#[cfg(feature = "tui")] mod tui;` + variante
  `#[cfg(feature = "tui")] Tui` + brazo `#[cfg(feature = "tui")]`; el bare
  (`command == None`) abre `tui::run_tui(model, url, context_window)` con la
  ventana del CLI cuando la feature existe, y error accionable sin ella.
- Clippy: `cargo clippy --features tui -- -D warnings` limpio; también
  `cargo clippy --all-targets --all-features`.

## Troubleshooting

- `cargo run -- tui` sin feature (build `--no-default-features`): error
  `unrecognized subcommand 'tui'` — reconstruye con default o `--features tui`.
- Terminal no restaurada tras crash: `reset` o `stty sane`; `run_tui`
  restaura `disable_raw_mode` + `LeaveAlternateScreen` + cursor al salir
  (también en error).
- `ctx%` alto: la ventana la fija el CLI (resolver `KITTY_NUM_CTX` /
  sonda `/api/show` / fallback); la TUI solo la muestra (`App::context_pct`
  con estimación `chars/4`).

# crates/

## Responsibility

Rust workspace members: the four Kitty-Code product crates. `kitty-cli` (binary:
parsing, dispatch, bench harness) drives `kitty-harness` (agent loop as a
library); the harness talks to models only through `kitty-runtime`
(`ChatClient` trait: Ollama HTTP default, embedded GGUF opt-in) and persists
episodes through `kitty-memory` (SQLite + retrieval). No crate talks to a
model except via `kitty-runtime`; no binary logic lives outside `kitty-cli`.

## Design

- Dependency direction is one-way: `kitty-cli` → `kitty-harness` →
  `kitty-runtime` + `kitty-memory`. `kitty-runtime` and `kitty-memory` are
  leaves (no workspace-internal deps). `kitty-harness` never imports
  `kitty-cli` — the bench harness lives in the CLI to keep the harness lib
  backend- and binary-agnostic.
- Backend selection is data, not branching: every LLM call crosses the
  `ChatClient` trait object (`Box<dyn ChatClient>`); `KITTY_BACKEND=kitty-infer`
  swaps the constructor in `kitty-cli`, nothing else. Feature flags
  (`kitty-infer[-cuda]`, `llama`, `embedding`, `tui`) keep default builds
  light; opt-in code paths are `#[cfg]`-gated at module and call-site level.
- Shared calibration lives in the workspace root (`Cargo.toml`
  `[workspace.dependencies]` + `Cargo.lock`): one version per crate, dual
  majors only with a rename + comment (post-mono `reqwest`/`reqwest013`,
  `rand`/`rand09` for `engine/` coexistence).

## Flow

`kitty-cli` parses flags → resolves profile/context/timeout settings →
constructs `AgentConfig` + backend-selected `Box<dyn ChatClient>` → drives
`Agent::run_with` (harness) → per turn `ChatClient::chat_with_options`
(runtime) → tool results execute via `ToolRegistry` (harness) → episodes
persist via `EpisodeStore` (memory). Bench runs reuse the same loop with a
`BenchClient` wrapper and per-exercise wall caps.

## Integration

- `crates/kitty-cli/`: binary + bench suite; see `kitty-cli/codemap.md` (crate)
  and `kitty-cli/src/codemap.md` (sources).
- `crates/kitty-harness/`: agent library; see `kitty-harness/codemap.md` and
  `kitty-harness/src/codemap.md`.
- `crates/kitty-runtime/`: model backends; see `kitty-runtime/codemap.md` and
  `kitty-runtime/src/codemap.md`.
- `crates/kitty-memory/`: episodic store; see `kitty-memory/codemap.md` and
  `kitty-memory/src/codemap.md`.
- `engine/` (sibling top-level dir, mono-repo) is mapped by another lane and
  intentionally absent from this map; its only seam into `crates/` is the
  optional `mistralrs` path-dep behind `kitty-runtime/kitty-infer[-cuda]`.

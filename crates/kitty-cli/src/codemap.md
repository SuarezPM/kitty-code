# crates/kitty-cli/src/

## Responsibility

Binary sources: one thin dispatch file plus one module per command area.
`main.rs` owns CLI parsing and backend/client construction only; every command
delegates to its module (`bench/`, `mcp.rs`, `tui.rs`) or to suite-support
libraries. No agent loop and no model logic live here.

## Design

- `main.rs`: `Cli` + `Commands::{Run, Bench, BenchAb, Info, Mcp, Tui}` (clap
  derive; `Tui` variant itself feature-gated). Backend selection
  (`KITTY_BACKEND` → `use_infer`), frozen Ollama client
  (`with_num_ctx(resolved)`), boxed `Box<dyn ChatClient>`, `bench_client()`
  closure, `bench_backend` label; `Run` overlays explicit flags onto
  `RunOverrides` (checked against clap defaults; `KITTY_AUTORUN_HOPS` env only
  on interactive paths, never bench); `Bench/BenchAb` pack `BenchCliCtx` +
  `BenchArgs`/`BenchAbArgs`; `Info` prints backend/model/tools; KV preflight
  warning routed through the resolved window.
- `bench/` (see `bench/` map in the parent crate doc): `mod.rs` dispatch
  (`bench_command`, `bench_ab_command`), `common.rs` (config, `BenchClient`,
  `TimeoutManifest`, exercises, tallies), `run.rs` (builtin loop),
  `ab.rs`/`polyglot.rs`/`tb2.rs`/`gpqa.rs`/`claweval.rs` (suite runners),
  `timeouts.json` (per-(model, backend) caps with measured origins).
- Suite-support libraries (top level, used BY `bench/` runners, not commands):
  `polyglot.rs` (official-pattern loader: `.meta/config.json`, tree copy,
  hidden tests, retry, subprocess timeouts), `tb2.rs` (harbor wrapper +
  preflight/blocked envelope), `gpqa.rs` (pre-registration gate, grading),
  `claweval.rs` (`claw-eval batch` wrapper + preflight).
- `mcp.rs`: stdio JSON-RPC server over `ToolRegistry::execute`
  (Guard-enforced, workspace-rooted).
- `tui.rs` (feature `tui`): ratatui status display (`StatusInfo`,
  context %, session duration).

## Flow

`Cli::parse` → env backend switch + frozen client → per-command match:
`Run` runs the agent once; `Bench/BenchAb` resolve the manifest timeout cell
and run the selected suite; `Info` probes health; `Mcp` serves stdio;
`Tui` renders loop events. Results print to stdout and persist as JSON
(`--out`) with gate verdicts.

## Integration

- Consumed by: the `kitty` binary entry point only.
- Depends on: `kitty-harness` (agent, profiles, registry, knowledge),
  `kitty-runtime` (clients, features, env constants), `kitty-memory`
  (episode store path); `bench/` additionally reads `benches/baseline.json`
  for the reproducibility delta.

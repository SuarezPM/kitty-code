# crates/kitty-cli/

## Responsibility
Binary entry point: CLI parsing (`clap`), the `run`/`bench`/`info`/`tui` commands,
and two embedded benchmark suites.

## Design
- `main.rs`: `clap` derive CLI. `Run` spawns one `Agent`. `Bench` runs a hardcoded
  exercise list and grades by substring presence (`file_contains`). `Info` prints
  status + the tool list. `Tui` (feature-gated) launches the status display.
- Two near-identical bench drivers: `run_polyglot_bench` (15 exercises,
  `solution.<ext>`) and `run_tb2_bench` (8 shell/infra exercises). Each opens the
  optional episode store, runs the agent in a tempdir with a 120s timeout, grades,
  and saves an episode.
- `tui.rs` (feature `tui`): ratatui status display (MVP).

## Flow
`Cli::parse` → `OllamaClient` → match command → `Agent::new`+`run` (or bench
loop) → print result / write JSON.

## Integration
- Consumes: `kitty-runtime`, `kitty-harness`, `kitty-memory`.

## Honest status
- `info` now lists the **actual** registered tools (derived from
  `standard_tool_definitions`), not a hardcoded string — the old `ShellStart`/`Test`
  advert was removed (those tools never existed).
- Bench grades by substring, so "pass" ≠ correctness (documented honestly in
  AUDIT.md as homemade, not Aider Polyglot / TB2).
- `run_polyglot_bench` / `run_tb2_bench` are now thin wrappers over a single
  `run_bench` driver (duplication removed, 2026-08-18).

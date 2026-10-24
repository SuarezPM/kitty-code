# kitty-cli / AGENTS.md

## OVERVIEW
Binary crate: Clap dispatch + bench harness + MCP stdio + optional TUI. No agent logic, only orchestration.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| CLI struct + Commands | `src/main.rs:30,72` | `Cli { url, model, thinking_budget, profile, knowledge, knowledge_dir, context_window, max_iterations, bash_timeout }` + `Run/Bench/BenchAb/Info/Mcp/Tui` |
| Dispatch | `src/main.rs:132` | `main()` 373L match: Run overlays profile then explicit flags, Bench/BenchAb build `BenchConfig`, Info health_check, Mcp delegates, Tui gated |
| Bench driver | `src/main.rs:673,687,710` | `BenchConfig` cloned per run; `run_bench` loops limit; `run_one_bench` per exercise with tempfile absolute path |
| Bench helpers | `src/main.rs:867,983,1000` | `bench_exit_code` gate 50% + baseline delta, `wilson_ci` z 1.96, `p50_ms` median, `parse_thinking_budgets`, `build_ab_plan` |
| Bench items | `src/main.rs:643,816` | `exercises()` 15 polyglot + `tb2_exercises()` 8; `json_to_bench_item` + `load_json_exercises` for `--exercises` JSON |
| MCP server | `src/mcp.rs:17,178` | `McpServer` stdio JSON-RPC line delimited; 6 tools via `ToolRegistry::execute` (Guard enforced); `handle_request` for tests |
| TUI | `src/tui.rs:150` | `run_tui` feature `tui` only; ratatui 0.30 + crossterm 0.29; `StatusInfo` ctx pct + duration; q/Esc quit |

## CONVENTIONS
- Bench harness lives here, not in `kitty-harness`. Keeps harness lib pure.
- `BenchAb` interleaves `(task, budget)` pairs shuffled with seed 42, deterministic across runs. Toggle with `--interleave false`.
- `--profile tiny` auto disables episodic memory (`--no-memory` implied) in Bench/BenchAb. Warning emitted, silence with explicit `--no-memory`.
- Bench smoke override: `max_iterations` 5 not 20 unless CLI explicitly sets non-default. Profile thresholds still sync `compact_threshold` + budget.
- `run_one_bench`: `tempfile::tempdir` + absolute path prompt `Save to {tmpdir}/{check_file}` prevents CWD contamination; CWD stale `solution.*` removed at Bench start.
- Per exercise 30s `tokio::time::timeout` on `agent.run`; verdict `pass/fail/error/timeout` persisted as `Episode` when memory on.
- `wilson_ci` and `p50_ms` pure fns, used only by BenchAb reporting; pairwise winner by rate.
- Tui is `#[cfg(feature = "tui")]` everywhere, `Commands::Tui` variant gated same. No default feature.

## ANTI-PATTERNS
- NEVER move bench code into `kitty-harness`. CLI owns benchmarking, harness owns loop.
- NEVER add `Tui` without `#[cfg(feature = "tui")]` on variant and match arm, breaks non-tui builds.
- NEVER use relative paths in bench prompts. Always absolute `{tmpdir}/{check_file}` or grading reads wrong dir.
- NEVER ignore `is_small_model` check. Tiny without `--no-memory` must auto disable or bench stalls on SQLite recall.
- NEVER bypass `ToolRegistry::execute` in MCP. Direct tool calls skip Guard and write vs redirect blocks.
- NEVER change shuffle seed from 42 or remove `interleave` default true, breaks reproducibility of BenchAb results.
- NEVER assume `bench max_iterations` equals `run` default. Bench uses 5, Run uses 20.

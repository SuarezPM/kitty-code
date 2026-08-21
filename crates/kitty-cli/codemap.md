# crates/kitty-cli/

## Responsibility
Binary entry point: CLI parsing (`clap`), the `run`/`bench`/`bench-ab`/`info`/`mcp`/`tui` commands,
and embedded benchmark suites (homemade + Harbor oracle proxies).

## Design
- `main.rs` (~1080L): `clap` derive CLI. `Cli { url, model, thinking_budget, profile, knowledge, knowledge_dir, context_window, max_iterations, bash_timeout, semantic_memory, cloud_fallback }` + `Commands Run/Bench/BenchAb/Info/Mcp/Tui`. `Run` overlays profile then explicit flags (`cloud_fallback` + `semantic_memory` post-`with_profile`, tiny guard forces false with warn). `Bench`/`BenchAb` build `BenchConfig { semantic_memory, profile, ... }` + `is_small_model` tiny guard, `run_bench` loops limit, `run_one_bench` per exercise with `tempfile::tempdir` absolute path `Save to {tmpdir}/{check_file}` (prevents CWD contamination), `AgentConfig{semantic_memory: cfg.semantic_memory && profile!="tiny", cloud_fallback: ...}` + hybrid recall `if semantic_memory { recall_hybrid(None,3) } else recall_similar`. Helpers `bench_exit_code` (gate 50% + baseline delta 2%), `wilson_ci` z 1.96, `p50_ms` median, `parse_thinking_budgets`, `build_ab_plan` interleaved seed 42.
- Bench items: `exercises()` 15 polyglot + `tb2_exercises()` 8 shell; `json_to_bench_item` + `load_json_exercises` for `--exercises` JSON (supports `check_type: file_exists/file_not_exists/file_executable/file_content_exact/always_pass` + `check_path`). `BenchConfig.semantic_memory` + CLI `--semantic-memory` (disabled for tiny) + `--cloud-fallback` (opt-in 1 retry, `KITTY_CLOUD_*` propagation).
- `mcp.rs` (333L): `McpServer` stdio JSON-RPC line delimited; 6 tools via `ToolRegistry::execute` (Guard enforced + Sandbox env hints, plugin delegated when `PluginRegistry` present); `handle_request` for tests; 5 tests (`list_tools` 6, `call_tool_read`, `bash_blocked`, `bash_echo`, `persist read→edit`).
- `tui.rs` (194L, feature `tui`): ratatui 0.29 + crossterm 0.28; `run_tui` `StatusInfo` ctx pct + duration + model; q/Esc quit. `#[cfg(feature = "tui")]` on variant + match arm.

## Flow
`Cli::parse` → `OllamaClient` / `ChatClient` → match command → `Agent::new`+`run` (or bench loop with `BenchConfig`) → print result / write JSON (`--out`). `McpServer::run` stdio loop → `ToolRegistry::execute`.

## Integration
- Consumes: `kitty-runtime` (OllamaClient, CloudClient, LlamaCppClient, ChatClient), `kitty-harness` (Agent, ToolRegistry, PluginRegistry, Sandbox, Guard), `kitty-memory` (EpisodeStore, RRF/embedding).
- Bench harness lives here, not in `kitty-harness` — keeps harness lib pure.

## Honest status
- `info` lists **actual** registered tools via `standard_tool_definitions` (6 core), not hardcoded — advert removed.
- Bench grades by substring `file_contains` / `file_exists` etc., so "pass" ≠ correctness (documented honestly in `AUDIT.md` as homemade, not Aider Polyglot / TB2). `run_polyglot_bench` / `run_tb2_bench` are thin wrappers over single `run_bench` driver (duplication removed).
- Harbor verifier (official) not run via CLI bench — CLI bench is homemade proxy. Oracle via `harbor run -p bench/official/... -a oracle` proves infra; CLI bench `13/15 86.7%` + `8/8 100%` + `2/15 13.3%` are proxies NOT comparable (see `AUDIT.md` §2).

## Tests
17 tests (bench_exit 6, bench_ab 2, mcp 7, profile heuristic, etc.) `cargo test -p kitty-cli` green, clippy `-D warnings` clean. `Commands::Tui` gated same as `tui` feature (no default).

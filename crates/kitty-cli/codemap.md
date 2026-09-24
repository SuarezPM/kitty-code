# crates/kitty-cli/

## Responsibility
Binary entry point: `clap` CLI parsing plus `Run/Bench/BenchAb/Info/Mcp/Tui`
commands, embedded bench harness (homemade + official-suite wrappers), MCP
stdio server, and optional ratatui TUI. No agent logic — only orchestration.

## Design
- `main.rs` (~633L, thin dispatch): `Cli { url, model, thinking_budget,
  profile, knowledge, knowledge_dir, context_window, max_iterations,
  bash_timeout, semantic_memory, cloud_fallback }` + `Commands
  Run/Bench/BenchAb/Info/Mcp/Tui`. Backend selection is env-driven up front:
  `KITTY_BACKEND=kitty-infer` (`backend_selected`, pure) builds
  `KittyInferClient::from_env()` (feature-gated; without it an honest
  `unavailable_error`), otherwise `OllamaClient`; `bench_backend`
  (`"ollama"` default, `"kitty-infer"` opt-in, never silent) flows into
  `BenchCliCtx`, and a `bench_client()` closure builds the matching
  `BenchClient` arm. Helpers `default_db_path`,
  `resolve_project_root` (marker walk `.git/Cargo.toml/...`, Gate-4 sandbox
  allow-root), `resolve_session_num_ctx` → `NumCtxResolution { num_ctx,
  source: env|native|fallback }` (env `KITTY_NUM_CTX` wins verbatim, else
  one-shot `/api/show` probe clamped [4096,32768], else `DEFAULT_NUM_CTX`;
  infer arm skips the probe — ctx owned by `KITTY_INFER_CTX`).
  `Run` builds `AgentConfig` + `with_profile`, then applies `RunOverrides`
  (context_window/max_iterations/thinking `Budget`/bash_timeout; `tiny/nano`
  guard warns on `--semantic-memory`; `KITTY_AUTORUN_HOPS` only here+TUI,
  never bench). `Bench/BenchAb` pack `BenchCliCtx` and delegate to
  `bench::bench_command` / `bench::bench_ab_command`; `Info` uses
  `health_check` + `standard_tool_definitions`; `Mcp` → `mcp::run`.
- `bench/` suite (dispatch extracted from `main.rs`): `mod.rs` holds
  `BenchCliCtx`, `BenchArgs`, `BenchAbArgs`, `bench_command`,
  `bench_ab_command`. `common.rs`: `BenchConfig` (client/model/limit/
  use_memory/thinking_budget/context_window/max_iterations/autorun_hops/
  exercise_timeout_secs/bash_timeout/profile/knowledge), `BenchItem
  { id,label,prompt,check,check_file }`, `BenchResult
  { id,lang,passed,verdict,latency_ms }`, `JsonExercise` +
  `json_to_bench_item`/`load_json_exercises`, `exercises()` 15 homemade
  polyglot items, `bench_exit_code` (gate 50% + baseline delta 2%),
  `is_small_model` (resolved `tiny|nano` → memory off), `language_of` +
  `tally_by_language`, `BenchClient` enum (`Ollama`/`Infer`, itself a
  `ChatClient` so metrics stay backend-identical) + `TimeoutManifest`
  (`timeouts.json`, embedded): per-exercise cap resolves per (model, backend)
  cell (`timeout_for_manifest`: model entry → backend default → 120) with
  precedence explicit flag > `KITTY_BENCH_TIMEOUT_SECS` env > cell
  (`resolve_timeout_for_bench`/`resolve_timeout_from_flag`; clap defaults
  equal the constant so an unset flag never counts as explicit),
  `file_contains`, `file_is_executable`. `run.rs`: `run_bench` loop over `limit`,
  `run_one_bench` (`tempfile::tempdir` + absolute `Save to {tmp}/{file}`
  prompt, `ENV_VERIFY_LOOP=off`, per-exercise `timeout`, verdict
  pass/fail/error/timeout + `Episode` persist), `tb2_exercises()` 8 shell.
  `ab.rs`: `parse_thinking_budgets`, `build_ab_plan` (seed-42 shuffle when
  `interleave`), `budget_summary` (`errored` vs `timed_out` split),
  `wilson_ci` (z 1.96), `p50_ms`. `polyglot.rs`: `run_polyglot_bench`
  (`.meta/config.json` load, copy tree, agent 900s, `materialize_tests`,
  unit tests 180s, 2 attempts with `retry_prompt`). `tb2.rs`:
  `run_tb2_official` (harbor preflight → `run_tb2`, blocked JSON on fail).
  `gpqa.rs`: `run_gpqa_suite` (gated: requires `--pre-registration` +
  `--limit 198`, binding checks model/limits, `blocked` JSON never touches
  model). `claweval.rs`: `run_claweval_suite` (`claw-eval batch` wrapper,
  preflight on CLI + `OPENROUTER_API_KEY`).
- Suite support (top level): `polyglot.rs` (`PolyglotExercise/Result`,
  `load_polyglot_exercises`, `build_prompt`, `copy_exercise_tree`,
  `materialize_tests`, `run_test_with_timeout`, `test_command`);
  `tb2.rs` (`Tb2Config`, `Tb2Result`, `preflight`/`is_ready`/
  `blocked_reasons`, `build_command`, `run_tb2`); `gpqa.rs`
  (`GPQA_NUM_QUESTIONS=198`, `GpqaPreRegistration`, `preflight_gpqa_with_prereg`,
  `permute_four`, `extract_answer_strict`, `grade_answer`, `GpqaResult`);
  `claweval.rs` (`ClawEvalRunConfig/Result/Status/Error`,
  `preflight_claweval`, `run_claweval_batch`, `parse_batch_output`).
- `mcp.rs`: `McpServer { tools, registry }` stdio JSON-RPC line-delimited;
  `list_tools` (6 via `standard_tool_definitions`), `call_tool` /
  `handle_request` (`tools/list|call`, `initialize`) all through
  `ToolRegistry::execute` (Guard + workspace `resolve_project_root`);
  `run_stdio` loop; `run()` entrypoint.
- `tui.rs` (feature `tui`, ratatui 0.29 + crossterm 0.28): `StatusInfo
  { context_used/context_window, context_pct, session_duration }`,
  `Message { role, content }`, `App` (input/history/draft/messages/
  scroll/is_running; `take_input`, `history_up/down`, `handle_event`
  over `AgentEvent`, `apply_q_policy` q-quits-only-when-empty-idle),
  `run_tui` → `run_app` (mpsc 100 + `spawn_local` `Agent::run`,
  `KITTY_AUTORUN_HOPS` allowed here).

## Flow
`Cli::parse` → `resolve_session_num_ctx` (Run/Bench/BenchAb only,
`with_num_ctx` frozen client) → match: `Run` = `Agent::new` +
`run_with(prompt, overrides)`; `Bench/BenchAb` = `BenchCliCtx` +
`BenchArgs/BenchAbArgs` → `bench_command`/`bench_ab_command` (CWD-stale
guard, `is_small_model` memory-off, suite switch gpqa→`run_gpqa_suite`,
claweval→`run_claweval_suite`, tb2-official→`run_tb2_official`,
polyglot→`run_polyglot_bench`, else `run_bench`/`run_one_bench`) →
print + JSON `--out` (+ `bench_exit_code` vs `benches/baseline.json`).
`Mcp` → `McpServer::run` stdio loop → `handle_request` →
`ToolRegistry::execute`. `Tui` → `run_tui(model, url, ctx)`.

## Integration
- Consumed by: binary users (`kitty run/bench/bench-ab/info/mcp/tui`);
  bench JSON consumed by runbooks (`--out`, blocked-vs-executed envelopes).
- Depends on: `kitty-harness` (`Agent`, `AgentConfig`, `RunOverrides`,
  `ModelProfile`, `ThinkingMode`, `ToolRegistry`, `standard_tool_definitions`,
  `KnowledgeConfig`, `ENV_VERIFY_LOOP`/`ENV_BENCH_MODE`), `kitty-runtime`
  (`OllamaClient`, `KittyInferClient` + `unavailable_error` (feature-gated),
  `DEFAULT_NUM_CTX`, `ENV_NUM_CTX`, `ENV_BACKEND`, cloud/KV preflight),
  `kitty-memory` (`EpisodeStore`, `Episode`).
- Bench harness lives here, not in `kitty-harness` — keeps harness lib pure.
  Grading is substring/file-check or local approximation (`polyglot` unit
  tests real; homemade `exercises`/`tb2_exercises` proxies; gpqa/claweval/
  tb2-official labelled non-official) — pass ≠ upstream leaderboard.

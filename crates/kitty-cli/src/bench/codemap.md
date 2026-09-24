# crates/kitty-cli/src/bench/

## Responsibility

Benchmark dispatch and shared harness for all suites. `mod.rs` routes
`bench`/`bench-ab` commands; `common.rs` holds config, backend abstraction,
and the per-cell timeout manifest; `run.rs` runs the builtin loop; the
remaining files each own one suite runner. Grading stays suite-local; gate
math (`bench_exit_code`) is shared.

## Design

- `mod.rs`: `BenchCliCtx` (url/model/backend/thinking/knowledge/ctx/iters/
  bash/profile, cloned out of `Cli`), `BenchArgs`/`BenchAbArgs`,
  `bench_command` (CWD-stale guard, memory-dir ensure, bench `max_iterations`
  5-override, small-model memory-off, suite switch, tally + gate print, JSON
  `--out`, baseline-delta exit code), `bench_ab_command` (budget plan,
  per-budget runs, wilson + pairwise).
- `common.rs`: `BenchConfig` (client/model/limit/memory/thinking/ctx/iters/
  hops/`exercise_timeout_secs`/bash/profile/knowledge), `BenchClient`
  (`Ollama`/`Infer`, itself `ChatClient`), `BenchItem`/`BenchResult`,
  `exercises()` 15 homemade items + `tb2_exercises()` 8,
  `TimeoutManifest` over embedded `timeouts.json`
  (default 120; cells minicpm5 60/150, qwen3 120/180 with measured origins;
  precedence explicit flag > env > cell; clap defaults equal the constant),
  `bench_exit_code` (gate 50% + baseline delta), `is_small_model`,
  `tally_by_language`, JSON exercise loading.
- `run.rs`: `run_bench`/`run_one_bench` (tmpdir + absolute-path prompt,
  `ENV_VERIFY_LOOP=off`, `tokio::time::timeout(exercise_timeout_secs)`,
  pass/fail/error/timeout verdicts, `Episode` persist).
- `ab.rs`: thinking-budget A/B (`parse_thinking_budgets`, seed-42
  `build_ab_plan` interleave, `budget_summary`, `wilson_ci`, `p50_ms`).
- `polyglot.rs` / `tb2.rs` / `gpqa.rs` / `claweval.rs`: official-suite
  runners (unit-test grading, harbor wrapper, pre-registration gate,
  `claw-eval` wrapper) delegating heavy lifting to the top-level
  suite-support files.
- `timeouts.json`: data, not code — per-(model, backend) caps with `origin`
  provenance strings; parse failure falls back to the safe default, never
  fails the bench.

## Flow

`bench_command` → resolve cell timeout (log line names model/backend/secs) →
suite branch → per-exercise `run_one_bench` (agent run under wall cap) →
tally by language → gate print → JSON out → process exit code from
`bench_exit_code` (0 pass, 1 gate-fail, 2 noise).

## Integration

- Consumed by: `main.rs` dispatch (`bench_client()` builds the
  backend-selected arm; `KittyInferClient::from_env` or honest
  `unavailable_error` without the feature).
- Reads: `benches/baseline.json` (delta), `--exercises` JSON files,
  `KITTY_BENCH_TIMEOUT_SECS` env.
- Never imports agent internals beyond `ModelProfile`,
  `clamp_autorun_hops`, and the memory `Episode` types.

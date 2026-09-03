## 1. Rewrite Polyglot Runner

- [ ] 1.1 Add `ExerciseConfig` struct to parse `.meta/config.json`
- [ ] 1.2 Add `load_exercise_config()` that reads config.json
- [ ] 1.3 Add `build_prompt()` that concatenates .docs/ files + suffix
- [ ] 1.4 Add `copy_solution_files_only()` that copies only solution files + .docs/
- [ ] 1.5 Add `materialize_tests()` that copies test files after agent finishes
- [ ] 1.6 Add `run_test_with_timeout()` with 180s subprocess timeout
- [ ] 1.7 Add test command map per language
- [ ] 1.8 Add `retry_prompt()` that builds retry message
- [ ] 1.9 Rewrite `run_polyglot_bench` with retry loop (2 tries)
- [ ] 1.10 Add 900s agent timeout, 180s test timeout
- [ ] 1.11 Update unit tests

## 2. Fix GPQA Runner

- [ ] 2.1 Add CoT prompt: "Think step by step before answering"
- [ ] 2.2 Align prompt format: "ANSWER: $LETTER" official template
- [ ] 2.3 Implement strict scoring: no ANSWER prefix = 0 points
- [ ] 2.4 Add configurable epochs (default 1 for local)
- [ ] 2.5 Update unit tests for strict scoring

## 3. Fix ClawEval Runner

- [ ] 3.1 Fix model flag: pass `--model` from CLI to `claw-eval batch`
- [ ] 3.2 Remove dead code: integrate `run_claweval_batch` into `run_claweval_suite`
- [ ] 3.3 Remove duplicated logic in main.rs
- [ ] 3.4 Update unit tests

## 4. Add TB2 Runner

- [ ] 4.1 Create `crates/kitty-cli/src/tb2.rs` with Harbor CLI wrapper
- [ ] 4.2 Add preflight checks: harbor, Docker, Ollama
- [ ] 4.3 Add `run_harbor()` that invokes `harbor run` with correct flags
- [ ] 4.4 Add result parsing for resolution rate
- [ ] 4.5 Integrate into main.rs as `bench tb2-official`
- [ ] 4.6 Add unit tests

## 5. Deprecation & Verification

- [ ] 5.1 Rename `benches/run_polyglot_225.py` → `run_kittyquiz_225.py`
- [ ] 5.2 `cargo check --workspace`
- [ ] 5.3 `cargo clippy --workspace -- -D warnings`
- [ ] 5.4 `cargo test --workspace`
- [ ] 5.5 Run polyglot l1 smoke test

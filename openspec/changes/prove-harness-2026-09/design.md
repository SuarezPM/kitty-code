# Design: prove-harness-2026-09

## 1. Measurement doctrine (hard constraints)

Every number this change publishes obeys these rules (see
`benchmarking/official-methodology` spec). Violation voids the run:

- Bench environments are immutable: no edits to tests, Dockerfiles,
  oracles, datasets, or instructions. Harness-side work only
  (prompts, tool loop, sampling params, workspace handling).
- Full sets only (225 / 89×k / 198 / 500). Any subset is labeled with
  committed `subset_ids + seed + stratification`, never as the suite.
- Report mean±SD over the prescribed repetitions, never best-of-k.
- Different gradings never share a column: `string-match-homemade`,
  `unit-test-local-harness-reproduction`, and official rates are
  separate tables with the grading method on every metric.
- Every heavy run is pre-registered (manifest predates results, HEAD sha
  matches a clean tree, dataset fingerprint recorded) per the
  `benches/ab/` protocol pattern; per-instance logs/preds are published.

## 2. Replication cards (what "same as they measured" means)

### Polyglot-225 with ornith 1.5 9B (little-coder parity)

Replicates `itayinbarr/little-coder v0.0.2`: full 225 Exercism tasks,
clean workspace copy per exercise, language-native runner exit-code
grading, ≤20 turns, up to 2 attempts (retry sees failing test output),
`solved = pass@2` primary + first-attempt rate, **2 full runs in
different language order, mean±SD**. Serving params: Ollama, `num_ctx`
32768, temperature 0.3, thinking budget 2048. Test-edit audit mandatory:
count exercises where the agent touched tests; benign-or-fail verdict
each. `kitty` already matches the shape (tempdir copy,
`materialize_tests`, 180s cap, retry ≤2); this change adds order
rotation, per-exercise result JSON, attempt-split reporting, and the
audit. Label: `unit-test-local-harness-reproduction (not official)`.
Comparable to little-coder 45.56% as same-class native; NOT to aider
Docker leaderboard numbers.

### TB2.1 (ornith parity)

Dataset `terminal-bench-2-1` (NOT 2.0), Harbor version pinned, agent
`terminus-2`, model via `ollama_chat/`, temperature 1.0 + top_p 1.0 (the
ornith eval params, not serving defaults), `-k 5` trials, mean±SD, oracle
`-k5` sanity first, trajectories kept and published, env/timeout
disclosed. This change pins the dataset (registry `-d
terminal-bench/terminal-bench-2-1` or verified local `-p` pin) and adds
N-selection + oracle smoke; the wrapper already shells out to Harbor.

### GPQA Diamond / ClawEval

Gated pre-registration stays (revision + sha256 + limits + grading);
unblock = materialize artifacts (GPQA `diamond.jsonl` + manifest sidecar;
ClawEval CLI + key + fixtures + sandbox image) then run the existing
suites. Labels stay `local_approximation_not_official`.

### SWE-bench-verified (last, heaviest)

Pinned OpenHands generates patches → **official `swebench` eval**
(`FAIL_TO_PASS` + `PASS_TO_PASS`, official Docker images), 500
instances, network disabled, git history stripped, `preds.json` +
`results.json` + per-instance logs published, temp/iterations/agent
version disclosed. New `run_id` per evaluation (never reuse cache).

## 3. Execution order and gates

Fase 0 pins methodology (this design + manifests, zero heavy compute) →
Fase 1 guard mutation (local CPU) → Fase 2 release signing + bench
recall hygiene (code, no benches) → Fase 3 light A/B (homemade N=15,
pre-registered; measures model-vs-model, not lever ablation) → Fase 4
loader unblocks (artifacts + smokes only, no full runs) → Fase 5 heavy
runs in suite order (polyglot → TB2.1 → GPQA → SWE-bench), each gated on
the previous suite's pre-registration passing. B4 (agentic-web) stays
scoped-out until B2/B3 land.

## 4. Interfaces

- Manifests reuse the `benches/ab/` schema (timestamp, HEAD sha, suite,
  N, dataset fingerprint, params, valid-run gate); polyglot runs add
  per-exercise JSON (`outcome`, `attempt`, `tests_touched`) beside the
  manifest.
- Bench `EpisodeStore` calls move to the agent's take/`spawn_blocking`/
  restore shape (`agent/response.rs:314-383`); sequential loop, no
  concurrency change.
- Release job: tag trigger, `id-token: write`, `cargo build --release
  -p kitty-cli`, SBOM attach, keyless `cosign sign` + Rekor bundle +
  `attest-build-provenance`.

## 5. Risks

- Wall time: ~20h per 225-run class (little-coder precedent on faster
  GPU); plan 2 runs + TB2.1 89×5 + 500 SWE instances as machine-time with
  crash-resume procedure (`keep_alive:-1`, no mid-run runner swaps —
  little-coder lost ~20pp to exactly that).
- VRAM: 9B Q4 ≈ 6GB fits the 8GB target; Ollama version pinned in every
  manifest; if the daemon changes mid-campaign, the run is void, not
  patched.
- SWE-bench needs OpenHands + official images (≈120GB disk guidance);
  if the box can't host it, Fase 5.4 becomes its own pre-registered
  change instead of blocking publication of 5.1–5.3.

## 6. Manifest extension (Fase 0.2)

The `benches/ab/` manifest schema (timestamp, HEAD sha, suite, N,
dataset fingerprint, params, valid-run gate) is extended per suite.
Required extra fields:

- polyglot-225: `dataset_commit` (7e0611e), `reference_run`
  (little-coder v0.0.2), per-exercise records (`outcome`, `attempt`,
  `tests_touched` + benign-or-fail verdict), `run_order`,
  `ollama_version`, served `temperature`, `top_p`, `num_ctx`
  actually served (not model max), thinking budget, per-lang
  timeout table.
- TB2.1: `dataset_version` (2.1.0) + source commit, `harbor_version`,
  `agent` (terminus-2), `trials_k`, explicit `temperature`/`top_p`,
  oracle-smoke reference, trajectory paths.
- GPQA-198: dataset revision (40-hex) + sha256 (64-hex),
  limits/grading/compat/preconditions block.
- SWE-bench-500: `swebench_version`, `openhands_revision`,
  `max_iterations`, fresh `run_id`, network/history safeguards
  attestation, `preds.json` + log paths.

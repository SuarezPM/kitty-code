# AUDIT.md: Kitty Code Benchmark Audit (Consumer vs Vendor)

**Date:** 2026-08-21
**Harness:** Kitty Code main 45d1bfe + F6 W5
**Hardware:** RTX 2060 Super 8GB, CachyOS, CUDA 610.57.04, 48GB RAM
**Model (Consumer):** ornith:9b Q4_K_M ~5.6GB, Ollama 0.32.14, profile medium 16384 ctx / thinking 1024 post-bench-ab, temp 0.6 top-p 0.95
**Model (Oracle):** oracle agent (copies reference solution, proves Harbor + Docker stack ceiling, not a model)
**Harbor:** 0.21.0, Docker 29.7.2, docker-compose 5.5.0
**Night bench update:** 2026-08-22 — full-suite oracle runs MEASURED (§2.1/§2.2) + first kitty official-verifier runs, partial/early-cut (§2.3)

---

## 1. Honesty Contract

This audit follows honest-by-construction. Every number below is measured and has a JSON artifact. No number is fabricated.

* **Homemade benchmarks** (`KittyQuiz-225` / `KittyShell-89`, `benches/exercises_polyglot.json` 225 + `benches/exercises_tb2.json` 89) use `string-match` grading (`content.contains(check_str)`). They are **NOT** the official Aider Polyglot or Terminal-Bench 2.0. A file with only the signature can pass. They are **NOT comparable** to official unit-test grading.
* **Official benchmarks** (`bench/official/aider-harbor/aider-polyglot` 225 + `bench/official/terminal-bench-2` 89) use `harbor verifier` unit tests inside Docker. Only the oracle uses this path in this audit. Kitty + ornith:9b on the official verifier requires a custom Harbor adapter (not present), so the ornith:9b official rate is reported as a homemade proxy with an explicit warning.
* **Consumer vs Vendor** in this doc means Consumer = ornith:9b Q4_K_M on 8GB consumer GPU via this harness, Vendor/Oracle = oracle agent on the same Harbor infra (the ceiling the infra can prove). It does **not** mean vendor FP16 on A100/H100. See `CNAS.md` methodology.

If you compare numbers across grading methods, you are comparing different things.

---

## 2. Consumer vs Vendor: Official Benchmarks (Harbor verifier)

Oracle proves the stack is healthy. Consumer (ornith:9b) on official verifier is N/A until the custom adapter exists, so the best proxy is the homemade smoketest documented with a warning.

### 2.1 Aider Polyglot 225 (official, `bench/official/aider-harbor/aider-polyglot`)

| Column | Value | Source |
|--------|-------|--------|
| **Oracle (Vendor ceiling)** | **225/225 100% mean 1.0** n_errors 0, ~33 min — `method: harbor-verifier`, FULL suite measured 2026-08-22 (closes re-audit condition #3) | `benches/bench_polyglot_oracle_full225.json` + `jobs/2026-08-22__02-34-23/result.json` |
| **Full-225 smoke (historical basis)** | oracle smoke n=15 15/15 100% 5m06s preceded the full run as stack-health gate; projection superseded by the measured full run above | `benches/bench_polyglot_oracle_smoke15.json` |
| **Consumer (ornith:9b) official verifier** | **~11.9% (5/42 evaluated), `_partialN` — NOT empirically closed** (aborted per two-strike protocol); first kitty adapter official run | `benches/bench_polyglot_kitty_ornith9b_partial42.json` (§2.3) |
| **Consumer proxy (homemade string-match, historical)** | **13/15 86.7%** — `method: simulated`; superseded for official claims by the adapter run above, kept for the record; NOT comparable to verifier numbers | `benches/bench_polyglot_ornith_smoke15.json` |
| **Grading** | Oracle and Consumer BOTH = `harbor-verifier-unit-tests` (same grading now that the adapter exists); homemade proxy = `string-match-homemade NOT comparable` |  |
| **Verdict** | Infra ceiling proven at full scale. Kitty consumer rate is partial (~11.9% on first-N=50 subset, 42 graded outcomes) — do NOT cite it as a full-suite rate. |  |

How to reproduce:

```bash
sudo env PATH=$HOME/.local/bin:$PATH harbor run -p bench/official/aider-harbor/aider-polyglot -a oracle -n 2 -l 15
jq '.stats.evals["oracle__aider-polyglot"].metrics[0].mean' /tmp/harbor-polyglot-oracle-15/*/result.json
# expected 1.0, n_trials 15, n_errors 0, reward 1.0 x15

cargo build --release
./target/release/kitty-cli --model ornith:9b bench polyglot --limit 15 --exercises benches/exercises_polyglot.json --out /tmp/bench_polyglot_ornith_limit15.json
jq '.passed, .total, .rate_pct' /tmp/bench_polyglot_ornith_limit15.json
# expected 13 15 86.67  (homemade, NOT harbor)
```

### 2.2 Terminal-Bench 2.0 89 (official, `bench/official/terminal-bench-2`)

| Column | Value | Source |
|--------|-------|--------|
| **Oracle (Vendor ceiling)** | **81/89 mean 0.9101 (91.01%)**, n_errors 1 disclosed, ~57 min — `method: harbor-verifier`, FULL suite measured 2026-08-22. Oracle ceiling on TB2 is NOT 100%: 7 genuine upstream-hard fails remain even copying reference solutions (+1 caffe-cifar-10 AgentTimeoutError counted in the 8 reward-0) | `benches/bench_tb2_oracle_full89.json` + `jobs/2026-08-22__01-37-04/result.json` |
| **Full-89 smoke (historical basis)** | single-task `adaptive-rejection-sampler` x5 5/5 100% 3m19s preceded the full run as stack-health gate; projection superseded by the measured full run above | `benches/bench_tb2_oracle_ars_k5.json` |
| **Gate 9.2%** | **CLOSED with full measurement**: oracle 91.01% >> 9.2%; consumer official-verifier measurement still pending a closed kitty run | `bench_tb2_oracle_full89.json` |
| **Consumer (ornith:9b) official verifier** | not yet run on TB2 (polyglot adapter runs first; see §2.3) | — |
| **Consumer proxy (homemade) builtin shell** | **8/8 100%** avg 12.5s/task, gate passed — `method: simulated` | `benches/bench_tb2_ornith9b_builtin8.json` + `bench_tb2_ornith_smoke15.json:smoketest_builtin` |
| **Consumer proxy (homemade) file_ops** | **2/15 13.3%** (~12/89 projected, NOT measured), gate 13.3% > 9.2% PASSED (barely) — `method: simulated` | `benches/bench_tb2_ornith_smoke15.json` + `benches/bench_tb2_ornith9b_fileops15.json` |
| **Grading** | Oracle = `harbor-verifier-unit-tests`, Consumer proxies = `string-match-homemade NOT comparable` |  |
| **Verdict** | Infra healthy at full scale (oracle 91.01%). Builtin shell tasks are easy for the harness, file_ops expose single-file limits. Both proxies are homemade and NOT comparable to official 89. |  |

How to reproduce:

```bash
sudo env PATH=$HOME/.local/bin:$PATH harbor run -p bench/official/terminal-bench-2 -a oracle -l 1 -k 5 --include-task-name adaptive-rejection-sampler -o /tmp/harbor-tb2-smoke
jq '.stats.evals["oracle__terminal-bench-2"].metrics[0].mean' /tmp/harbor-tb2-smoke/*/result.json
# expected 1.0, 5 trials, 0 errors

./target/release/kitty-cli --model ornith:9b bench tb2 --limit 8 --exercises benches/exercises_tb2.json
# expected builtin 8/8 100% (if using builtin suite) or 2/15 13.3% on file_ops suite
```

Notes:

* Full 225 and full 89 oracle were not run as long blocks (would be 0.7h + 0.49h at n=2, spec said 15-24h at n=1 with heavy env builds). Smoketest 15/15 and 5/5 with mean 1.0 proves Harbor + Docker healthy; because oracle is deterministic (copy solution), a full-suite run is expected to hold the ceiling — that expectation is a stack health gate, not a measured vendor claim or result. The approved night oracle-full runs close this gap.
* First 5 alphabetic TB2 tasks gave 3/5 60% with oracle due to 2 flaky upstream tasks (`build-cython-ext` 1/11 pyknotid failure, `build-pmars` 4/4). Retrying with a known-good task (`adaptive-rejection-sampler` x5) gives 5/5 100%.
* **2026-08-22 update:** the gap IS closed — full-suite oracle runs measured overnight (rows above). Polyglot held the ceiling (225/225); TB2 did NOT (81/89, 91.01%) — the "oracle always 100%" assumption was wrong for TB2 and is now corrected by measurement.

### 2.3 Night bench 2026-08-22: measured results (honest headlines)

Four harbor runs executed overnight (P1→P2→R3→R4). Canonical artifacts in `benches/`, raw `result.json` under gitignored `jobs/`.

| Run | Agent / model | Suite | Result | Gate v3 verdict | Artifact |
|-----|---------------|-------|--------|-----------------|----------|
| P1 | oracle | TB2 full 89 | **81/89 mean 0.9101 (91.01%)**, 1 env-error disclosed (caffe-cifar-10 AgentTimeoutError) | PASS (1/90) | `benches/bench_tb2_oracle_full89.json` |
| P2 | oracle | polyglot full 225 | **225/225 mean 1.0**, 0 errors, ~33 min | PASS (0/225) | `benches/bench_polyglot_oracle_full225.json` |
| R3 | kitty ornith:9b (attempt 3/3, timeout-fix chain → c4547e1) | polyglot first-N l50 | **5 passed / 37 reward-0 of 42 graded outcomes (~11.9%), `_partialN`** — aborted per two-strike protocol; 28 errored trials (19 task-level AgentTimeoutError, 5 RewardFileNotFound, 3 NonZeroAgentExit, 1 VerifierTimeout) | FAIL (28/78) → partial | `benches/bench_polyglot_kitty_ornith9b_partial42.json` |
| R4 | kitty ornith-1.5:9b (attempt 4/4) | polyglot first-N l15 | **0 passes in 11 evaluated — directional only, EARLY-CUT** by user authorization; `agent_timeout_multiplier:3` confound vs R3 ⇒ NOT comparable to R3 | FAIL (7/22) → cut | `benches/bench_polyglot_kitty_ornith15b_l15_earlycut.json` |

Honest reading:

* **Oracle baselines are now measured, not projected**: TB2-oracle 91.01% and polyglot-oracle 100% close re-audit condition #3. The TB2 result also corrects the prior assumption that oracle = 100% on any suite.
* **Kitty ornith:9b partial is honest ~11%** (`_partialN`, first-N=50 pre-registered, 42 graded outcomes): not empirically closed, aborted per two-strike protocol (task-level AgentTimeoutError + empty responses >16K ctx). The HTTP timeout fix (900s, commit c4547e1 chain) IS verified working: 0 HTTP-layer timeouts — remaining AgentTimeoutErrors are harbor task-level limits, a different failure class.
* **R4 early-cut is directional-only** (0/11): cut by user authorization before completion (`finished_at: null`), with an `agent_timeout_multiplier:3` config confound vs R3. It supports no rate claim and must not be averaged or compared with R3.

North star ladder (official Aider Polyglot, harbor-verifier grading where measured):

| Position | Reference | Rate | Provenance |
|----------|-----------|------|------------|
| today | **kitty + ornith:9b (this harness)** | **~11.9%** (5/42 evaluated, `_partialN`) | `bench_polyglot_kitty_ornith9b_partial42.json` |
| floor to beat | vanilla Aider (external reference) | 19.11% | orchestrator-provided external figure, not measured here |
| class SOTA 9B | little-coder pi scaffold | 45.56% | `docs/STRATEGY.md`, ablation-proven 2.4× harness delta |

The ladder says the harness delta is the lever: kitty today sits below even the vanilla-Aider floor; closing toward 19.11% then 45.56% is harness work, not model work.

---

## 3. Homemade Benchmarks: For Reference Only (NOT comparable)

These are the numbers historically cited as 96.44%. They stay in the record but must not be compared to official rates above.

| Suite | Result | Grading | Source |
|-------|--------|---------|--------|
| **KittyQuiz-225** (homemade) ornith:9b 217/225 **96.44%** | 17.6 min, 4.6s avg, 2 runs 0% variance | `string-match-homemade NOT comparable to Aider Polyglot` | `benches/baseline.json` 96.44, `docs/FLOOR-BENCHMARK.md` smoke, `docs/TECHNICAL_AUDIT_REPORT.md` 3.1 |
| **KittyQuiz-225** proxy ornith:9b 13/15 **86.7%** | 207s, 13.8s avg | `string-match-homemade` (`method: simulated`) | `benches/bench_polyglot_ornith_smoke15.json` |
| **KittyShell-89** (homemade) 11/89 **12.36%** | 3.4 min, 2.3s/task | homemade | `docs/TECHNICAL_AUDIT_REPORT.md` 3.2, `benches/bench_tb2_ornith_smoke15.json` warning |
| **KittyShell file_ops** 2/15 **13.3%** | 13.3% > 9.2% gate | homemade | `benches/bench_tb2_ornith9b_fileops15.json` |
| **KittyShell builtin** 8/8 **100%** | ceiling on easy shell | homemade | `benches/bench_tb2_ornith9b_builtin8.json` |

Per-language breakdown (KittyQuiz-225, 96.44% run, string-match):

| Language | Passed | Total | Rate |
|----------|--------|-------|------|
| Python | 50 | 50 | 100% |
| TypeScript | 30 | 30 | 100% |
| C++ | 10 | 10 | 100% |
| Java | 10 | 10 | 100% |
| JavaScript | 48 | 50 | 96% |
| Go | 28 | 30 | 93.3% |
| Rust | 41 | 45 | 91.1% |

> **Warning:** Do not place 96.44% next to official Aider Polyglot percentages. The grading methods differ. The 96.44% measures single-file trivia with substring checks, not unit-test pass. See `docs/FLOOR-BENCHMARK.md` methodology box.

---

## 4. Reproducibility

| Artifact | Location | What it proves |
|----------|----------|----------------|
| **TB2 oracle FULL 89 (2026-08-22)** | `benches/bench_tb2_oracle_full89.json` + `jobs/2026-08-22__01-37-04/result.json` | 81/89 mean 0.9101, 1 env-error disclosed; measured ceiling |
| **Polyglot oracle FULL 225 (2026-08-22)** | `benches/bench_polyglot_oracle_full225.json` + `jobs/2026-08-22__02-34-23/result.json` | 225/225 mean 1.0, 0 errors; measured ceiling |
| **Kitty ornith:9b polyglot l50 PARTIAL (2026-08-22)** | `benches/bench_polyglot_kitty_ornith9b_partial42.json` + `jobs/2026-08-22__15-57-35/result.json` | 5/42 ~11.9% `_partialN`, gate FAIL → two-strike abort; per-task rewards + exception breakdown |
| **Kitty ornith-1.5:9b polyglot l15 EARLY-CUT (2026-08-22)** | `benches/bench_polyglot_kitty_ornith15b_l15_earlycut.json` + `jobs/2026-08-22__21-02-52/result.json` | 0/11 directional-only; agent_timeout_multiplier:3 confound documented |
| Gate v3 + fixtures | `tests/fixtures/gate-v3.sh`, `gate-selftest.sh`, `gate-fixture-result.json`, `gate-fixture-bad.json` | errored-trials gate validated against known-good AND known-bad fixtures before use |
| Polyglot oracle smoke (n=15) | `benches/bench_polyglot_oracle_smoke15.json` + `.omo/evidence/f6-remaining-tracks-opt-order/harbor_polyglot_oracle_15_result.json` | 15/15 mean 1.0 harbor-verifier smoke (historical stack-health basis for the full run) |
| Polyglot ornith 13/15 proxy | `benches/bench_polyglot_ornith_smoke15.json` + `/tmp/bench_polyglot_ornith_limit15.json` | 13/15 86.67% homemade (`method: simulated`); full-225 projections are not results |
| TB2 oracle single-task k=5 smoke | `benches/bench_tb2_oracle_ars_k5.json` + `/tmp/harbor-tb2-smoke/2026-08-21__06-05-11/result.json` | 5/5 mean 1.0 on adaptive-rejection-sampler (historical stack-health basis) |
| TB2 ornith builtin 8/8 | `benches/bench_tb2_ornith9b_builtin8.json` + `benches/bench_tb2_ornith_smoke15.json` | 8/8 100% homemade easy-subset ceiling |
| TB2 ornith file_ops 2/15 | `benches/bench_tb2_ornith9b_fileops15.json` + `benches/bench_tb2_ornith_smoke15.json` | 2/15 13.3% homemade |
| Harbor logs | `.omo/evidence/f6-remaining-tracks-opt-order/task-24-polyglot.log`, `task-25.log` | infra + per-task details |
| bench/official untouched | `git status -- bench/official` shows `??` untracked, `git diff` empty | not modified |

Verify quickly:

```bash
jq '.n_total_trials, (.smoketest.total), .rate_pct' benches/bench_polyglot_oracle_smoke15.json
# 15 15 100   (n_total_trials == smoketest.total == trials actually run)
jq '.n_total_trials, (.smoketest.details | length), .smoketest.passed' benches/bench_polyglot_ornith_smoke15.json
# 15 15 13
jq '.n_total_trials, (.smoketest.total), .rate_pct' benches/bench_tb2_oracle_ars_k5.json
# 5 5 100     (single task x5 attempts)
jq '.n_total_trials, (.smoketest.details | length), .smoketest.passed, .smoketest.rate_pct' benches/bench_tb2_ornith_smoke15.json
# 15 15 2 13.33
jq '.passed, .total, .rate_pct' benches/bench_tb2_ornith9b_builtin8.json
# 8 8 100

# Night bench 2026-08-22 canonical artifacts:
jq '{n:.n_total_trials, passed, mean, src:.source_job_dir}' benches/bench_tb2_oracle_full89.json
# {"n":89,"passed":81,"mean":0.9101123595505618,"src":"jobs/2026-08-22__01-37-04"}
jq '{n:.n_total_trials, passed, mean}' benches/bench_polyglot_oracle_full225.json
# {"n":225,"passed":225,"mean":1}
jq '{partial:.n_total_trials, passed, rate:.rate_pct_evaluated, harbor_mean:.mean_harbor_reported}' benches/bench_polyglot_kitty_ornith9b_partial42.json
# {"partial":42,"passed":5,"rate":11.904761904761903,"harbor_mean":0.1}
jq '{evaluated:.n_evaluated, passed, early_cut, atm:.agent_timeout_multiplier}' benches/bench_polyglot_kitty_ornith15b_l15_earlycut.json
# {"evaluated":11,"passed":0,"early_cut":true,"atm":3}

# Gate v3 self-test (must be 4/4 before trusting the gate on real runs):
bash tests/fixtures/gate-selftest.sh
```

---

## 5. What Is NOT Claimed

Per CNAS methodology:

1. 96.44% is homemade string-match, NOT official Aider Polyglot.
2. 12.36% / 13.3% / 100% builtin are homemade, NOT official TB2.
3. Kitty's official-verifier rates are NOT closed: the adapter exists and ran (2026-08-22), but ornith:9b polyglot is `_partialN` (~11.9% on 42 graded outcomes of a first-N=50 subset, aborted per two-strike) and ornith-1.5 l15 was early-cut (0/11, directional only). No full-suite consumer rate is claimed; R3 and R4 are NOT comparable to each other (`agent_timeout_multiplier` confound).
4. Consumer Q4_K_M on 8GB is NOT vendor FP16 on A100/H100.
5. No frontier reasoning, no long context, no production hardening beyond sandbox + guard + audit.
6. Benchmarks are single-file / dockerized task proxies, not repo-scale agent work.
7. The 19.11% vanilla-Aider floor and 45.56% pi-scaffold figures in §2.3 are external reference points (45.56% attested in `docs/STRATEGY.md`; 19.11% orchestrator-provided), not measured on this stack.

---

## 6. Known Harness Limitations (tonight's adapter smokes)

Recorded from the 2026-08-21 kitty harbor-adapter smokes (Oracle Gate 1 requirement). They qualify every consumer number in this doc until fixed:

1. **Guard limitation (Write-on-existing-file).** Kitty's Guard blocks Write on an existing file unless the Edit tool is used. This stalled ornith:9b on tasks requiring file modification during smoke k1.
2. **Thinking-budget mismatch.** ornith-1.5:9b repeatedly overran the medium profile thinking budget (1024 tokens, set by the earlier bench-ab knee decision tuned on ornith:9b). Its numbers are NOT budget-comparable to ornith:9b's.
3. **Context-length dependency.** ~~kitty-cli does not send `num_ctx`~~ **Updated 2026-08-25 (commit a4012cf):** the client now sends an explicit session-fixed `options.num_ctx` (`KITTY_NUM_CTX`, default 32768) plus `"truncate": false` on every `/api/chat` payload, so overflow surfaces as an audible HTTP 400 (`RuntimeError::ContextOverflow`) instead of silent truncation at the server's tiered default. Server-side `OLLAMA_CONTEXT_LENGTH` still governs what is actually available; benchmarks should keep declaring it explicitly.
4. **Two-boundary sandbox decision (Gate-4).** Bash children are Landlock/seccomp-contained by default; harbor bench runs set `KITTY_SANDBOX=off` because the outer Docker boundary is active; opt-out is loud and per-run.

### Approved night strategy (pre-registered)

Oracle-full runs go first: TB2-89 (`harbor run -p bench/official/terminal-bench-2 -a oracle -k 1`) and polyglot-225 (`harbor run -p bench/official/aider-harbor/aider-polyglot -a oracle -k 1`) — the re-audit gate #3 requirement. Kitty runs are SAMPLED, not full, with a pre-registered first-N selection rule: `-l N` takes the FIRST N tasks in dataset order, no post-hoc cherry-picking — `-l 50` with ornith:9b → `bench_polyglot_kitty_ornith9b_l50.json`, `-l 15` with ornith-1.5:9b → `bench_polyglot_kitty_ornith15b_l15.json`. Every run must pass the errored-trials gate (`n_errored_trials <= 10%` of N checked via result.json stats — harbor exits 0 even when trials error).

**Outcome (executed 2026-08-22):** P1 (TB2 oracle) and P2 (polyglot oracle) PASSED the gate and are measured (§2.1/§2.2). R3 FAILED the gate on re-run (28 errored trials) → aborted per two-strike protocol → `_partial42`. R4 was early-cut by user authorization before completion. Incident N1: the original gate grabbed a nested eval-level object instead of top-level `.stats` (vacuous gates); fixed as **gate v3** (`tests/fixtures/gate-v3.sh`) which parses `.stats.n_completed_trials/.n_errored_trials` directly, derives outcomes = completed + errored, hard-fails on `n_total_trials == 0` schema mismatch, and is validated against known-good AND known-bad fixtures (`bash tests/fixtures/gate-selftest.sh`).

---

## 7. Landed harness levers (batch f7, 2026-08-25)

Landed CODE, listed with their commits for traceability. **None of these
levers has a measured rate claim attached yet** — any number produced with
them must come from the pre-registered protocol in `benches/ab/README.md`
before it may appear in this audit or `CNAS.md`.

| Lever | Commits | What changed |
|-------|---------|--------------|
| Verification loop + revision cycles + lint-gate hygiene | `4eee68b`, `ea69554`, `29c68d7` | Outer verification cycle around the tool loop: workspace tests grade finished work, failures fed back for revision (bounded cycles); scratch-file leak closed on every exit path; verify loop disabled in bench contexts. Worst-case LLM calls ≈ max_iterations × 3 per prompt (documented on the profile struct). |
| Explicit context window + audible overflow | `a4012cf` | Every `/api/chat` payload carries session-fixed `options.num_ctx` (`KITTY_NUM_CTX`, default 32768) and `"truncate": false`; overflow maps to `RuntimeError::ContextOverflow` (requested vs available + remediation hint); `prompt_eval_count` near-capacity warning; KV-cache preflight warning. |
| Per-model inference profiles (ModelTuning) | `59d7a75` | Exact → family ("ornith"/"qwen") → default resolution of sampling + thinking mode; Tier-4 turn caps unified into one source of truth; thinking channel made live via lazy seeding from `tuning.thinking` (CLI/bench A/B axis reaches the tracker); dead `max_thinking_tokens` removed. |
| Fuzzy edit replacer ladder | `9bac5c0`, `baa286e` | `crates/kitty-harness/src/edit_replacers.rs`: ordered fuzzy strategies behind hashline/search-replace failure, with disproportionate-match guard; later bounded candidate search (512 KiB bail-out, 64 KiB anchor cap, once-per-span vetting). |
| Parser salvage + tool aliases | `d80bc95` | Opener variants (bare/symmetric/asymmetric `<tool_call>`), aliased argument keys, raw retention, extended recovery chain, hallucinated tool-name aliases. |
| Upstream tool-call normalization | `baa286e` | Synonym/alias normalization runs in the agent loop immediately after parsing — upstream of quality monitor, diagnostics, episode memory and dispatch; idempotent (dispatch re-normalizes harmlessly). |

North-star targets remain exactly as published (unchanged by this batch):
little-coder **45.56%** Aider Polyglot (Qwen3.5-9B Q4_K_M, num_ctx 32768,
temp 0.3) · TB2.0 gate **9.2%±2.4** · ornith-1.5 9B official TB2.1 **46.2**
and SWE-V **70.6**.

### 7.1 Landed harness levers (batch f8, 2026-08-26)

Same contract as above: **código aterrizado ≠ tasa medida** hasta una
corrida pre-registrada (`benches/ab/README.md`) — nothing in this table
carries a rate claim.

| Lever | Commits | What changed |
|-------|---------|--------------|
| Autorun continuation hops | `9f6d3ce` | Fixed state machine in `Agent::run`: hop budget (`AgentConfig.autorun_hops`, default 0, clamp 0..=12) consumed ONLY on iteration exhaustion — force_response skipped, `AUTORUN_CONTINUE_PROMPT` pushed into history, inner loop restarts; MAX_STEPS suppressed while hops remain; verification still runs exclusively post-force_response; Failed-with-revisions-exhausted consumes one hop iff any remain. Recovers turn-cap hits without losing context. Bench wiring is FLAG-ONLY (`--hops`); interactive run/TUI read `KITTY_AUTORUN_HOPS`. Worst-case ≈ iterations × 3 × (hops+1). |
| Native context probe + startup resolution | `880bc22` | `parse_native_context_length` matches ANY `/api/show` model_info key ending in `.context_length` (family-prefixed: llama/qwen2/qwen3…, smallest wins) with parameters-num_ctx fallback; once-per-process startup resolution (explicit `KITTY_NUM_CTX` ⇒ no probe; else min(native, 32768) floor 4096) applied via `with_num_ctx()` before the first request — client frozen afterward; KV preflight warning rerouted through the resolver so it reports the RESOLVED window. |
| Honest read windows | `880bc22` | Read output opens with `{path}: lines {start}-{end} of {total}`; optional offset (1-based) and limit args with weak-model clamp tolerance (invalid/string numerals clamped, not rejected); omission note suggests the offset retry with Grep secondary; profile `read_cap` plumbed via `ToolRegistry::with_read_cap`; hashline snapshot stays full-file so edits outside the displayed window succeed unchanged. |
| Compaction suite (cheap summarizer + derived threshold + safe prune) | `5bb22d5`, `27aea98` | Summary call overrides ONLY temperature=0.1 and num_predict=400 with a ≤280-word prose cap (num_ctx untouched — session-frozen). Derived compaction threshold `min(profile, max(0.62, (ctx−2048)/ctx))`, never raising above the profile value and inactive when the ratio already exceeds it (table: tiny@4096=0.62 · medium@4096=0.62 · medium@16384=0.80 · large@32768=0.80). Prune boundary walks back up to K=12 messages to the nearest USER turn start (exact landing kept as-is per G3 fix), so tool results are never orphaned and the last user turns stay complete; freed/summary accounting derives from the SAME final boundary. Bench re-sync derives from the EFFECTIVE post-overlay window (G3 fix). |
| `Box<T>` chat_with_options forwarding fix | `5bb22d5` | WIRE-FORMAT BUG: the blanket `impl ChatClient for Box<T>` forwarded chat/count_tokens/health_check but NOT chat_with_options — its DEFAULT impl degraded every boxed client to plain chat, silently dropping ALL RequestOptions (dynamic num_predict caps, sampling/thinking flags) between agent and backend. Forwarding added; discovered by the AD4 capture test. |
| Nano minimal-prompt tier | `c2d8573`, `391de55` | Preset nano (4K ctx, 5 iter, 4 tools, read_cap 80) + `PromptStyle::{Full,Minimal}`: conservative auto-detection for ~0.5–3B models, Minimal system prompt ≤1600 chars omitting verbose bootstrap sections; gated under the small-model bench guard (memory-off auto + grammar force-enable preserved). |
| Async lint gate + token-count LRU cache | `c2d8573` | `cargo check/lint` verification moved to `spawn_blocking` (no blocking calls on the async runtime, timeout semantics preserved); token counting cached via LRU (~256 entries keyed model+content) so repeated pushes skip `/api/tokenize` round-trips. |
| Dead-field removal | `c2d8573` | `AgentConfig.max_thinking_tokens` (dead since the f7 live-thinking-channel change) deleted — single source of truth remains `tuning.thinking`. |

---

## 8. Roadmap

* [x] Harbor + Docker stack verified (oracle 15/15 and 5/5 smokes; superseded by full-suite measurements 2026-08-22)
* [x] Homemade proxies for ornith:9b 13/15 86.7% and TB2 8/8 100% + 2/15 13.3%
* [x] Custom Harbor adapter for kitty agent — EXISTS and ran official verifier (2026-08-22, partial/early-cut results §2.3)
* [x] Full 225 and 89 official runs (oracle measured 2026-08-22: polyglot 225/225, TB2 81/89); closed consumer full-suite runs still pending
* [ ] Closed kitty official-verifier run on polyglot (fix task-level AgentTimeoutError class, then re-run l50 without the two-strike abort)
* [ ] RAG + long context + multi-file benchmark when warranted

*This audit is consumer-measured, vendor-ceiling verified, and explicitly non-comparable across grading methods.*

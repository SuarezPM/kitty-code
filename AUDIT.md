# AUDIT.md: Kitty Code Benchmark Audit (Consumer vs Vendor)

**Date:** 2026-08-21
**Harness:** Kitty Code main 45d1bfe + F6 W5
**Hardware:** RTX 2060 Super 8GB, CachyOS, CUDA 610.57.04, 48GB RAM
**Model (Consumer):** ornith:9b Q4_K_M ~5.6GB, Ollama 0.32.14, profile medium 16384 ctx / thinking 1024 post-bench-ab, temp 0.6 top-p 0.95
**Model (Oracle):** oracle agent (copies reference solution, proves Harbor + Docker stack ceiling, not a model)
**Harbor:** 0.21.0, Docker 29.7.2, docker-compose 5.5.0

---

## 1. Honesty Contract

This audit follows honest-by-construction. Every number below is measured and has a JSON artifact. No number is fabricated.

* **Homemade benchmarks** (`KittyQuiz-225` / `KittyShell-89`, `benches/exercises_polyglot.json` 225 + `benches/exercises_tb2.json` 89) use `string-match` grading (`content.contains(check_str)`). They are **NOT** the official Aider Polyglot or Terminal-Bench 2.0. A file with only the signature can pass. They are **NOT comparable** to official unit-test grading.
* **Official benchmarks** (`bench/official/aider-harbor/aider-polyglot` 225 + `bench/official/terminal-bench-2` 89) use `harbor verifier` unit tests inside Docker. Only the oracle uses this path in this audit. Kitty + ornith:9b on the official verifier requires a custom Harbor adapter (not present), so the ornith:9b official rate is reported as a homemade proxy with an explicit warning.
* **Consumer vs Vendor** in this doc means Consumer = ornith:9b Q4_K_M on 8GB consumer GPU via this harness, Vendor/Oracle = oracle agent on the same Harbor infra (the ceiling the infra can prove). It does **not** mean vendor FP16 on A100/H100. See `CNAS.md` methodology.

If you compare numbers across grading methods, you are comparing different things.

---

## 2. Consumer vs Vendor: Official Benchmarks (Harbor verifier)

Oracle proves the stack is healthy. Consumer (ornith:9b) on the official verifier ran for the first time on 2026-08-22 via the custom adapter (partial/early-cut, §2.3); before that date the best proxy was the homemade smoketest documented with a warning.

### 2.1 Aider Polyglot 225 (official, `bench/official/aider-harbor/aider-polyglot`)

| Column | Value | Source |
|--------|-------|--------|
| **Oracle (Vendor ceiling)** | **15/15 100% mean 1.0** n_errors 0, 5m06s — `method: harbor-verifier`, smoke subset n=15 of 225 | `benches/bench_polyglot_oracle_smoke15.json` smoketest + `.omo/evidence/f6-remaining-tracks-opt-order/harbor_polyglot_oracle_15_result.json` |
| **Full-225 projection (NOT measured)** | oracle copies solution ⇒ full run expected to hold ceiling if stack stays healthy; est. 0.7h at n=2, 20.4s/task — projection only, not a result. **SUPERSEDED 2026-08-22: measured 225/225 (§2.3)** | same file `estimation` |
| **Consumer (ornith:9b) official verifier** | adapter live since 2026-08-22 — partial run measured (§2.3) | `benches/bench_polyglot_kitty_ornith9b_partial42.json` |
| **Consumer proxy (homemade string-match)** | **13/15 86.7%** — `method: simulated` (smoke subset; any full-225 figure is a projection, not a result), 207s, avg 13.8s/task | `benches/bench_polyglot_ornith_smoke15.json` smoketest + `estimation` |
| **Grading** | Oracle = `harbor-verifier-unit-tests`, Consumer proxy = `string-match-homemade NOT comparable` |  |
| **Verdict** | Infra healthy. Do NOT compare 86.7% homemade to 100% oracle unit-test. |  |

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
| **Oracle (Vendor ceiling)** | **5/5 100% mean 1.0** n_errors 0, 3m19s on `adaptive-rejection-sampler` x5 — `method: harbor-verifier`, smoke: single task, k=5 attempts (full suite NOT measured) | `benches/bench_tb2_oracle_ars_k5.json` smoketest + `/tmp/harbor-tb2-smoke/2026-08-21__06-05-11/result.json` |
| **Full-89 projection (NOT measured)** | oracle copies solution ⇒ full run expected to hold ceiling if stack stays healthy; est. 0.49h at n=2 (0.99h seq) — projection only, not a result. **SUPERSEDED 2026-08-22: measured 81/89 91.01% (§2.3)** | same file `estimation` |
| **Gate 9.2%** | **PASSED** on smoke basis only (single-task k=5); full-suite gate measurement pending night run — **CLOSED 2026-08-22: oracle 91.01% >> 9.2% (§2.3)** | `bench_tb2_oracle_ars_k5.json:gate` |
| **Consumer (ornith:9b) official verifier** | adapter live since 2026-08-22; TB2 via adapter pending first kitty run (polyglot ran, §2.3) | `benches/bench_polyglot_kitty_ornith9b_partial42.json` |
| **Consumer proxy (homemade) builtin shell** | **8/8 100%** avg 12.5s/task, gate passed — `method: simulated` | `benches/bench_tb2_ornith9b_builtin8.json` + `bench_tb2_ornith_smoke15.json:smoketest_builtin` |
| **Consumer proxy (homemade) file_ops** | **2/15 13.3%** (~12/89 projected, NOT measured), gate 13.3% > 9.2% PASSED (barely) — `method: simulated` | `benches/bench_tb2_ornith_smoke15.json` + `benches/bench_tb2_ornith9b_fileops15.json` |
| **Grading** | Oracle = `harbor-verifier-unit-tests`, Consumer proxy = `string-match-homemade NOT comparable` |  |
| **Verdict** | Infra healthy. Builtin shell tasks are easy for the harness, file_ops expose single-file limits. Both proxies are homemade and NOT comparable to official 89. |  |

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

### 2.3 Night bench 2026-08-22 (measured, canonical artifacts)

Measured full suites (oracle baselines; closes re-audit condition #3):

* **Polyglot-oracle: 225/225 mean 1.0**, 0 errors — `benches/bench_polyglot_oracle_full225.json`
* **TB2-oracle: 81/89 mean 0.9101 (91.01%)**, 1 AgentTimeoutError env-error disclosed in the denominator (`caffe-cifar-10`) — `benches/bench_tb2_oracle_full89.json`. Oracle ceiling on TB2 is NOT 100%.

Kitty consumer runs (official verifier via adapter, first ever):

* **ornith:9b polyglot l50 `_partialN` (partial42)**: 11.90% computed (5/42); harbor-reported mean 0.1; env-compromised gate FAIL → two-strike abort. Limitations: AgentTimeoutError at task level plus empty responses above 16K ctx; HTTP-layer timeout fix (900s) verified working. — `benches/bench_polyglot_kitty_ornith9b_partial42.json`
* **ornith-1.5:9b polyglot l15 early-cut, directional-only**: 0 passes in evaluated subset; `agent_timeout_multiplier: 3` confound; NOT comparable to R3. — `benches/bench_polyglot_kitty_ornith15b_l15_earlycut.json`

Full night narrative (run table, gate v3 incident, north star ladder): see `docs/AUDIT.md` section 2.3.

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
| Polyglot oracle smoke (n=15) | `benches/bench_polyglot_oracle_smoke15.json` + `.omo/evidence/f6-remaining-tracks-opt-order/harbor_polyglot_oracle_15_result.json` | 15/15 mean 1.0 harbor-verifier smoke; full-225 NOT measured |
| Polyglot ornith 13/15 proxy | `benches/bench_polyglot_ornith_smoke15.json` + `/tmp/bench_polyglot_ornith_limit15.json` | 13/15 86.67% homemade (`method: simulated`); full-225 projections are not results |
| TB2 oracle single-task k=5 smoke | `benches/bench_tb2_oracle_ars_k5.json` + `/tmp/harbor-tb2-smoke/2026-08-21__06-05-11/result.json` | 5/5 mean 1.0 on adaptive-rejection-sampler; full-89 NOT measured |
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
```

---

## 5. What Is NOT Claimed

Per CNAS methodology:

1. 96.44% is homemade string-match, NOT official Aider Polyglot.
2. 12.36% / 13.3% / 100% builtin are homemade, NOT official TB2.
3. Official Aider Polyglot and TB2 with Kitty + ornith:9b via Harbor verifier have NOT been run (needs custom adapter).
4. Consumer Q4_K_M on 8GB is NOT vendor FP16 on A100/H100.
5. No frontier reasoning, no long context, no production hardening beyond sandbox + guard + audit.
6. Benchmarks are single-file / dockerized task proxies, not repo-scale agent work.
7. Known harness limitations (Guard Write-to-existing-file stall, thinking-budget mismatch, context-length dependency) qualify every consumer number until fixed — see `docs/AUDIT.md` §6 for the three limitations and the approved pre-registered night strategy.
8. MiniCPM5-2B (2026-09-20, manifests `manifest-20260920T163000Z-minicpm5-l15.json` + `manifest-20260920T172000Z-minicpm5-microexp.json`): **0/15** on official-polyglot-first-15 (unit-test grading) + 0/5 micro-runs. Lane functional (smoke Write OK) but not competitive on Exercism-hard. Native tools OFF for the small tier (Ollama 500s on weak-model native args before harness repair sees them). Vendor 46.4% SWE-bench NOT reproduced, NOT claimed.
9. K2-Horizon-3.7B: client implemented + mock-tested, serving frozen (no SGLang on box, GGUF fork-only) — `docs/SERVING-K2-SGLANG.md`. Zero bench numbers, none claimed.

---

## 6. Roadmap

* [x] Harbor + Docker stack verified (oracle 15/15 and 5/5)
* [x] Homemade proxies for ornith:9b 13/15 86.7% and TB2 8/8 100% + 2/15 13.3%
* [x] Small-model lane (2026-09-20): `small` preset + minicpm/k2 tuning + kwarg parser + compat client; MiniCPM5-2B baseline 0/15 official-first-15 (valid zero, not a claim)
* [ ] Custom Harbor adapter for kitty agent to run official verifier with ornith:9b (next step for true Consumer official rate)
* [ ] Full 225 and 89 official runs when adapter exists (0.7h + 0.49h at n=2)
* [ ] RAG + long context + multi-file benchmark when warranted

*This audit is consumer-measured, vendor-ceiling verified, and explicitly non-comparable across grading methods.*

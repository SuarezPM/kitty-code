# Consumer-Normalized Agentic Score (CNAS)

Leaderboard: consumer-measured vs vendor-claimed numbers.

## Hardware

| Component | Value |
|-----------|-------|
| GPU | NVIDIA RTX 2060 Super 8GB |
| CPU | AMD Ryzen 5 3600 |
| RAM | 48 GB DDR4 |
| Quantization | Q4_K_M (GGUF) |

---

## What is NOT Claimed

This is an honest-by-construction document. The following are **not** claimed:

- **Aider Polyglot scores.** "KittyQuiz-225" is a homemade 225-exercise coding quiz with string-match grading. It is NOT the official Aider Polyglot benchmark. Results are NOT comparable.
- **Terminal-Bench 2.0 scores.** "KittyShell-89" is a homemade 89-task terminal quiz. It is NOT the official Terminal-Bench 2.0 benchmark. No Docker, no Harbor. Results are NOT comparable.
- **Official benchmark status.** Official Aider Polyglot and TB2 oracle ceilings are MEASURED at full scale (2026-08-22 night bench): polyglot-oracle **225/225 mean 1.0**; TB2-oracle **81/89 mean 0.9101 (91.01%)**, 1 env-error disclosed — closes re-audit condition #3. Kitty adapter RAN on the official verifier: ornith:9b polyglot l50 `_partialN` **partial42** (11.90% computed 5/42 / 0.1 harbor-reported; env-compromised gate FAIL → two-strike abort); ornith-1.5:9b l15 early-cut directional-only (**0 passes in evaluated subset**, `agent_timeout_multiplier:3` confound vs R3 — NOT comparable). Not closed consumer rates; see `docs/AUDIT.md` §2.3.
- **Frontier-level reasoning**: ornith:9b is a small local model. It does not match GPT-4/Claude on complex multi-step reasoning.
- **Long-context capability**: 16K effective context is small. No retrieval-augmented generation (RAG) yet.
- **Production readiness**: This is a research harness, not a production system.
- **Benchmark parity**: Consumer Q4_K_M quantization on 8GB VRAM is not equivalent to vendor FP16 benchmarks on A100/H100 hardware.
- **Multi-step agentic tasks**: The 12.36% KittyShell-89 score shows ornith:9b struggles with chained tool calls without scaffolding.
- **Memory persistence**: Episodic memory wired at agent.rs:650 (recall_episodes) / 679 (save_episode) - verified in harness loop (preflight→recall→knowledge→loop).
- **TUI interactivity**: The TUI is a status display MVP. It does not yet support interactive prompting or scrolling.

### Targets

| Official Benchmark | Target | Status |
|-------------------|--------|--------|
| Aider Polyglot (official) Harbor verifier | beat 45.56% | Oracle FULL 225/225 100% measured (`benches/bench_polyglot_oracle_full225.json`); kitty ornith:9b `_partialN` ~11.9% (`benches/bench_polyglot_kitty_ornith9b_partial42.json`) — target NOT met, run not closed |
| Terminal-Bench 2.0 (official) Harbor verifier | beat 9.2% on 8GB | Oracle FULL 81/89 91.01% measured (`benches/bench_tb2_oracle_full89.json`) — infra ceiling proven; closed kitty run pending |

See `docs/AUDIT.md` section 2 for Consumer vs Vendor table. The custom adapter EXISTS and ran (2026-08-22): `benches/bench_polyglot_kitty_ornith9b_partial42.json`, `benches/bench_polyglot_kitty_ornith15b_l15_earlycut.json`.

---

## Scores - Official (Harbor verifier) Consumer vs Vendor

| Benchmark | Vendor/Oracle ceiling (Harbor verifier) | Consumer ornith:9b Q4_K_M 8GB (this harness) | Grading |
|-----------|------------------------------------------|----------------------------------------------|---------|
| **Aider Polyglot 225** | **225/225 100% mean 1.0** FULL suite measured 2026-08-22, ~33 min, 0 errors (`benches/bench_polyglot_oracle_full225.json`) | **~11.9% (5/42), `_partialN`** first-N l50, env-compromised gate FAIL → two-strike abort (`benches/bench_polyglot_kitty_ornith9b_partial42.json`); ornith-1.5:9b l15 early-cut 0/11 directional-only, NOT comparable to R3 (`bench_polyglot_kitty_ornith15b_l15_earlycut.json`) | Oracle and Consumer BOTH `harbor-verifier-unit-tests` (adapter live since c4547e1 chain) |
| **Terminal-Bench 2.0 89** | **81/89 mean 0.9101 (91.01%)** FULL suite measured 2026-08-22, ~57 min, 1 env-error disclosed — oracle ceiling is NOT 100% on this suite (`benches/bench_tb2_oracle_full89.json`); gate 9.2% closed at infra level | TB2 via adapter: pending first kitty run (polyglot ran 2026-08-22); historical homemade proxies: **8/8 builtin** + **2/15 file_ops** (`method: simulated`, NOT comparable) | Oracle verifier vs homemade proxies NOT comparable |

Sources (night bench 2026-08-22, canonical): `benches/bench_tb2_oracle_full89.json`, `benches/bench_polyglot_oracle_full225.json`, `benches/bench_polyglot_kitty_ornith9b_partial42.json`, `benches/bench_polyglot_kitty_ornith15b_l15_earlycut.json` (raw `result.json` under gitignored `jobs/2026-08-22__*`). Historical smokes: `benches/bench_polyglot_oracle_smoke15.json` + `harbor_polyglot_oracle_15_result.json` mean 1.0, `benches/bench_tb2_oracle_ars_k5.json` + `/tmp/harbor-tb2-smoke/.../result.json` mean 1.0, `benches/bench_polyglot_ornith_smoke15.json` 13/15, `benches/bench_tb2_ornith9b*.json` + `benches/bench_tb2_ornith_smoke15.json` 8/8 and 2/15. Gate v3 fixture-validated: `bash tests/fixtures/gate-selftest.sh`.

Repro gate (harbor verifier smokes):

```bash
sudo env PATH=$HOME/.local/bin:$PATH harbor run -p bench/official/aider-harbor/aider-polyglot -a oracle -n 2 -l 15
jq '.stats.evals["oracle__aider-polyglot"].metrics[0].mean' /tmp/harbor-polyglot-oracle-15/*/result.json  # 1.0

sudo env PATH=$HOME/.local/bin:$PATH harbor run -p bench/official/terminal-bench-2 -a oracle -l 1 -k 5 --include-task-name adaptive-rejection-sampler -o /tmp/harbor-tb2-smoke
jq '.stats.evals["oracle__terminal-bench-2"].metrics[0].mean' /tmp/harbor-tb2-smoke/*/result.json  # 1.0
```

> **Honesty:** Do NOT compare 86.7% / 13.3% homemade proxy to measured oracle verifier rates (polyglot 100%, TB2 91.01%). Different grading. See `docs/AUDIT.md` section 1.

---

## Scores - Homemade Benchmarks (NOT comparable to official)

| Metric | Kitty Code (this harness) | Notes |
|--------|--------------------------|-------|
| Model | ornith:9b Q4_K_M | Ollama local |
| Context | 16,384 tokens | Measured effective window |
| **KittyQuiz-225** | **96.44%** | 2 runs, 7 languages, string-match grading |
| **KittyShell-89** | **12.36%** | Multi-step agentic tasks |
| Binary size | **5.0 MB** | `cargo build --release`, stripped |
| Tests | **85** | Workspace unit tests |
| TUI | `kitty tui` (feature-gated) | ratatui, minimal MVP |

> **Note:** KittyQuiz-225 and KittyShell-89 are homemade benchmarks. Scores use string-match grading, not unit tests. NOT comparable to official Aider Polyglot or Terminal-Bench 2.0.

## Per-Language Breakdown (KittyQuiz-225, averaged 2 runs)

| Language | Passed | Total | Rate | Gap to 100% |
|----------|--------|-------|------|-------------|
| Python | 50 | 50 | **100.0%** | - |
| TypeScript | 30 | 30 | **100.0%** | - |
| C++ | 10 | 10 | **100.0%** | - |
| Java | 10 | 10 | **100.0%** | - |
| JavaScript | 48 | 50 | **96.0%** | 2 (deep_clone, sample) |
| Go | 28 | 30 | **93.3%** | 2 (generics syntax) |
| Rust | 41 | 45 | **91.1%** | 4 (impl trait patterns) |

## KittyShell-89 Per-Category Breakdown (89 tasks, 1 run)

| Category | Passed | Total | Rate |
|----------|--------|-------|------|
| File Operations | 7 | 20 | 35.0% |
| Text Processing | 0 | 20 | 0.0% |
| Shell Commands | 1 | 20 | 5.0% |
| Data Transformation | 0 | 15 | 0.0% |
| System Operations | 3 | 14 | 21.4% |

## Reproducibility

| Run | KittyQuiz-225 Pass Rate | Time |
|-----|------------------------|------|
| Run 1 | 96.44% (217/225) | 17.6 min |
| Run 2 | 96.44% (217/225) | 18.1 min |
| **Variance** | **0.0%** | - |

## How to Reproduce

```bash
# KittyQuiz-225 (2 runs for reproducibility)
python3 benches/run_polyglot_225.py 1
python3 benches/run_polyglot_225.py 2

# KittyShell-89
python3 benches/run_tb2_89.py

# Results:
# bench_polyglot_225_run1.json  - 96.44% (217/225) [string-match grading]
# bench_polyglot_225_run2.json  - 96.44% (217/225) [string-match grading]
# bench_tb2.json                - 12.36% (11/89)
```

## CNAS Methodology

CNAS normalizes agentic benchmarks to consumer hardware:

1. **Run on consumer hardware** (not cloud GPUs)
2. **Use quantized models** (Q4_K_M, not FP16/BF16)
3. **Measure what matters** (code generation, tool calling, not just text completion)
4. **Report honestly** (no inflated context windows, no cherry-picked results)
5. **Verify reproducibility** (2 independent runs, 0% variance)

The goal: make agentic AI scores meaningful for the 99% of developers running local models.

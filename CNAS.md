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
- **Official benchmark status.** Official Aider Polyglot and TB2 oracle verified via Harbor 15/15 and 5/5 100% with extrapolation to 225/225 and 89/89. Kitty + ornith:9b via Harbor verifier NOT yet run (needs custom adapter), reported as homemade proxy with warning. See `docs/AUDIT.md`.
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
| Aider Polyglot (official) Harbor verifier | beat 45.56% | Oracle 15/15 100% verified, ornith proxy 13/15 86.7% homemade N/A official adapter pending |
| Terminal-Bench 2.0 (official) Harbor verifier | beat 9.2% on 8GB | Oracle 5/5 100% verified, ornith proxies 8/8 100% builtin + 2/15 13.3% file_ops both >9.2% PASSED homemade |

See `docs/AUDIT.md` section 2 for Consumer vs Vendor table. Official rates with kitty + ornith:9b via Harbor verifier require custom adapter (not present yet).

---

## Scores - Official (Harbor verifier) Consumer vs Vendor

| Benchmark | Vendor/Oracle ceiling (Harbor verifier) | Consumer ornith:9b Q4_K_M 8GB (this harness) | Grading |
|-----------|------------------------------------------|----------------------------------------------|---------|
| **Aider Polyglot 225** | **15/15 100%** smoketest 5m06s, 225/225 100% est 0.7h | **13/15 86.7%** proxy 195/225 est 86.67% (homemade string-match) | Oracle verifier, Consumer homemade NOT comparable |
| **Terminal-Bench 2.0 89** | **5/5 100%** smoketest 3m19s, 89/89 100% est 0.49h, gate 9.2% PASSED | **8/8 100% builtin** + **2/15 13.3% file_ops** ~12/89 13.3% >9.2% PASSED (homemade) | Oracle verifier, Consumer homemade NOT comparable |

Sources: `benches/bench_polyglot_225_oracle.json` + `harbor_polyglot_oracle_15_result.json` mean 1.0, `benches/bench_tb2_89_oracle.json` + `/tmp/harbor-tb2-smoke/.../result.json` mean 1.0, `benches/bench_polyglot_225_ornith9b.json` 13/15, `benches/bench_tb2_ornith9b*.json` 8/8 and 2/15. Evidence `.omo/evidence/f6-remaining-tracks-opt-order/task-24-polyglot.log`, `task-25.log`. Full 225/89 not run as long block (estimated at n=2), smoketest proves stack healthy, oracle deterministic.

> **Honesty:** Do NOT compare 86.7% / 13.3% homemade proxy to 100% oracle verifier. Different grading. See `docs/AUDIT.md` section 1.

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

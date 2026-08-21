# Repository Atlas: Kitty Code

## Project Responsibility
`Kitty Code` is a **local coding agent** written in Rust, optimized to run the
`ornith:9b` model (Q4_K_M, ~5.6 GB) on a consumer GPU (RTX 2060 Super 8 GB) via the
Ollama HTTP API. It is a research harness, not a production system. The differentiator
is the **harness** (tool loop, safety guards, hashline edits, quality monitor,
context budget, knowledge injection, episodic memory) around a small local LLM.

## System Entry Points
- `crates/kitty-cli/src/main.rs` — CLI binary (`run`, `bench`, `bench-ab`, `info`, `mcp`, `tui`).
- `Cargo.toml` — virtual workspace, 4 member crates, release profile (LTO, strip, `panic="abort"`).
- `AUDIT.md` / `CNAS.md` — honesty-first benchmark documentation (homemade quizzes,
  explicitly **not** Aider Polyglot / Terminal-Bench 2.0; Harbor oracle verifier 15/15 + 5/5 100%).
- `docs/STRATEGY.md` — F6.4 ADR-005 GBNF defer spike final + roadmap P0/P1.
- `docs/FLOOR-BENCHMARK.md` — W4 GBNF A/B + W5 Consumer vs Vendor evidence.

## Directory Map (Aggregated)
| Directory | Responsibility Summary | Detailed Map |
|-----------|------------------------|--------------|
| `crates/kitty-runtime/` | Ollama HTTP client + ChatClient trait + tool wire-types + Grammar spike + Cloud/Llama clients. | [View Map](crates/kitty-runtime/codemap.md) |
| `crates/kitty-harness/` | Agent loop, tools, hashline edits, parser, guard, sandbox, plugin, quality monitor, budgets, knowledge, bootstrap. | [View Map](crates/kitty-harness/codemap.md) |
| `crates/kitty-memory/` | Episodic memory (blake3 + SQLite) + BM25 + embedding (fastembed opt-in) + RRF hybrid recall. | [View Map](crates/kitty-memory/codemap.md) |
| `crates/kitty-cli/` | CLI + embedded benchmark suites + bench-ab + MCP + TUI. | [View Map](crates/kitty-cli/codemap.md) |
| `bench/official/` | Vendored official benchmarks (aider-harbor 225, terminal-bench-2 89, polyglot-benchmark 225) — Harbor oracle verifier. | — |
| `benches/` | Homemade benches + exercises JSON (228) + baseline + A/B benches (recall, GBNF) + bench scripts. | — |
| `docs/` | STRATEGY (F6.4 ADR final), FLOOR-BENCHMARK, PLUGIN, TUI, ADR-GBNF-GO, AUDIT mirror. | — |

## Data & Control Flow (one `run`)
1. CLI builds `AgentConfig` + `ChatClient` (OllamaClient default, LlamaCppClient tiny fallback, CloudClient opt-in).
 2. `Agent::run(prompt)`: bootstrap detect → preflight system hint (FIRST, so the
    system message leads the conversation) → episodic recall (BM25 / hybrid RRF if `--semantic-memory`) → knowledge inject →
    push user msg → loop:
   - `ChatClient::chat` → `ToolCallParser` (native + text) → `GrammarClient` post-hoc filter → `QualityMonitor`
     → `ToolRegistry::execute` (HashlineEditor / Bash via Sandbox + Guard / Glob / Grep / Plugin delegated with Guard inheritance) → push result.
   - context pressure → `trigger_compaction` (LLM summary of old turns).
   - empty streak 2 / iterations 20 / ToolLoop → optional `CloudClient` fallback (1 retry, `--cloud-fallback`) else fail-fast.
3. Final text returned; episode saved (best-effort, with embedding BLOB if `embedding` feature).

## Honesty Notes (verified 2026-08-21 — W5 final gate + F6 tracks)
- **247 tests pass** (202 harness +18 runtime +10 memory +17 cli) `cargo test --workspace` 0 failed, `cargo clippy --all-targets --all-features -- -D warnings` clean, `cargo fmt --all --check` clean. Release build ~30s LTO strip. `cargo deny check` clean. `openspec validate --all` 6 passed.
- **F6 W5 Consumer vs Vendor published**: Harbor 0.21.0 + Docker 29.7.2 — Oracle 15/15 100% (polyglot) 5m06s + 5/5 100% (TB2 adaptive-rejection-sampler x5) 3m19s prove infra healthy (extrapolated 225/225 + 89/89 valid as gate, not vendor claim). Consumer ornith:9b proxies: 13/15 86.7% polyglot + 8/8 100% TB2 builtin + 2/15 13.3% file_ops (gate 9.2% PASSED barely). Grading `harbor-verifier-unit-tests` vs `string-match-homemade NOT comparable` — do NOT compare. See `AUDIT.md` §2, `docs/FLOOR-BENCHMARK.md` W5, `CNAS.md`.
- **Homemade** `KittyQuiz-225` 96.44% 217/225 (string-match) + `KittyShell-89` 12.36% remain NOT comparable to official Aider Polyglot / TB2. Documented honestly in `AUDIT.md`/`CNAS.md`.
- **F6 tracks implemented (untracked pending merge, counted separately):** `sandbox.rs` (FS whitelist + Landlock/seccomp stubs, 6 tests OOM/SIGSEGV/SIGABRT contained), `plugin.rs` (Plugin trait + PluginRegistry static, no dlopen, 10 tests + 3 isolation), `memory bm25` (k1 1.2 b 0.75, window 50, lexical baseline), `embedding.rs` (fastembed 4 opt-in `embedding` feature, MODEL_ID Qdrant/all-MiniLM-L6-v2-onnx 384, ~88M `~/.kitty/models`, cosine + BLOB migration, 7 tests), `RRF K=60` hybrid recall (`recall_hybrid` BM25+dense, ab_recall 83.3% strict), `cloud.rs` (CloudClient generic OpenAI-compatible, KITTY_CLOUD_API_KEY/BASE_URL, 120s, cost metered, MockCloud 9 tests), `llama.rs` (LlamaCppClient Arc<Mutex> + fallback, grammar_lazy + json_schema_to_grammar, no block without GGUF), `contract.rs` DoD, `bench-ab` interleaved. When wired + merged, workspace tests go to **~293** (+ sandbox/plugin/llama/cloud/embedding). Current committed lib.rs wire is minimal to keep `cargo test --workspace` green at 247; full wiring is F6 integration step (docs/STRATEGY.md F6.4 ADR final).
- Benchmarks are genuinely homemade (string-match `content.contains(check_str)`) — documented honestly in `docs/FLOOR-BENCHMARK.md` (smollm2 53% 2.76s, qwen3 53% 24s, ornith 96.44% 4.6s + W4 GBNF 60%→98% tiny validity + W5 oracle vs proxy). Tier sub-2B re-encuadrado a demo (<50% en 225 full), 2-4B GO condicional.
- Memory recall baseline was lexical Jaccard → BM25 (k1 1.2/b 0.75) as IR baseline; dense embeddings via fastembed opt-in + RRF K=60 hybrid is the upgrade (blocked by rten/Gemma int8 → ort chosen). `save_episode` records real completion status (fail on exhausted iterations).
- `tree-sitter` dependency and `trait Tool` removed; `ShellStart`/`Test` advert removed (never registered). Security guard is interpreter-aware (`sh -c` bypass closed). Token counting is REAL via Ollama `/api/tokenize` (model's own tokenizer), heuristic fallback only on server error.
- Architecture of 4 crates is the correct sweet spot — not refactored for its own sake, now with `model_profile.rs` (tiny 4K/5iter/4tools, medium 16K/20iter, large 32K/20iter), `router.rs`, `contract.rs`, `grammar.rs` (spike), `mcp.rs`, `sandbox.rs`, `plugin.rs`, `cloud.rs`/`llama.rs`, `embedding.rs`.
- **F0-F5 hardened**: guard (command substitution, eval block, timeout 30s bench), hashline prefix matching, tool error handling, budget single-point accounting (`push_message` sole), cross-turn tool-loop detection (args-aware blake3), message ordering, thinking retry (cap 2), Read truncation (300 lines), Grep via ignore crate, dead code removal, CLI flags (`--profile tiny|medium|large`, `--no-memory` auto, `--semantic-memory`, `--cloud-fallback`), JSON exercise loading, `bench-ab` interleaved + wilson CI.
- **Empty response fix**: assistant msg pushed to history, no orphaned LLM round-trips, `MAX_EMPTY_STREAK=2` + fail-fast → latency 87s→4s smollm2, 39s→24s qwen3.
- **Bench/Knowledge (F6 W5):** `bench/official/` exists (aider-harness, polyglot-benchmark 225, terminal-bench-2 89, aider-harbor 225), `benches/exercises_polyglot.json` 228 (225 +3 Python: list-comp/decorators/async), floor 225 smollm2 45.3% (<50% → demo re-frame, `docs/FLOOR-BENCHMARK.md`), knowledge 5 cards (`knowledge/*.md` rust-traits/go-generics/python-list-comp/python-decorators/python-async) scoring word1.0/bigram2.0, `benches/baseline.json` 96.44, `kitty mcp` stdio 6 tools via Guard, `cargo deny` + `deny.toml` allowlist, `CODEOWNERS`, workflows pinned SHA.

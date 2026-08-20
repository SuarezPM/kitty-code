# Repository Atlas: Kitty Code

## Project Responsibility
`Kitty Code` is a **local coding agent** written in Rust, optimized to run the
`ornith:9b` model (Q4_K_M, ~5.6 GB) on a consumer GPU (RTX 2060 Super 8 GB) via the
Ollama HTTP API. It is a research harness, not a production system. The differentiator
is the **harness** (tool loop, safety guards, hashline edits, quality monitor,
context budget, knowledge injection, episodic memory) around a small local LLM.

## System Entry Points
- `crates/kitty-cli/src/main.rs` — CLI binary (`run`, `bench`, `info`, `tui`).
- `Cargo.toml` — virtual workspace, 4 member crates, release profile (LTO, strip, `panic="abort"`).
- `AUDIT.md` / `CNAS.md` — honesty-first benchmark documentation (homemade quizzes,
  explicitly **not** Aider Polyglot / Terminal-Bench 2.0).

## Directory Map (Aggregated)
| Directory | Responsibility Summary | Detailed Map |
|-----------|------------------------|--------------|
| `crates/kitty-runtime/` | Ollama HTTP client + tool wire-types. | [View Map](crates/kitty-runtime/codemap.md) |
| `crates/kitty-harness/` | Agent loop, tools, hashline edits, parser, guard, quality monitor, budgets, knowledge, bootstrap. | [View Map](crates/kitty-harness/codemap.md) |
| `crates/kitty-memory/` | Episodic memory (blake3 + SQLite). | [View Map](crates/kitty-memory/codemap.md) |
| `crates/kitty-cli/` | CLI + embedded benchmark suites + TUI. | [View Map](crates/kitty-cli/codemap.md) |

## Data & Control Flow (one `run`)
1. CLI builds `AgentConfig` + `OllamaClient`.
 2. `Agent::run(prompt)`: bootstrap detect → preflight system hint (FIRST, so the
    system message leads the conversation) → episodic recall → knowledge inject →
    push user msg → loop:
   - `OllamaClient::chat` → `ToolCallParser` (native + text) → `QualityMonitor`
     → `ToolRegistry::execute` (HashlineEditor / Bash / Glob / Grep) → push result.
   - context pressure → `trigger_compaction` (LLM summary of old turns).
3. Final text returned; episode saved (best-effort).

## Honesty Notes (verified 2026-08-19 noche — deepwork E2E)
- **239 tests pass** (198 harness +17 runtime +10 memory +7 cli +7 cleaner) and `cargo clippy --all-targets -- -D warnings` is clean. Build release 34s.
- Benchmarks are genuinely homemade (string-match `content.contains(check_str)`) — documented honestly in `docs/FLOOR-BENCHMARK.md` (smollm2 53% 2.76s, qwen3 53% 24s, ornith 96.44% 4.6s). Tier sub-2B re-encuadrado a demo (<50% en 225 full), 2-4B GO condicional.
- Memory recall is now **lexical Jaccard** (was substring `LIKE`); dense embeddings are a planned upgrade. `save_episode` records real completion status (fail on exhausted iterations).
- `tree-sitter` dependency and `trait Tool` removed; `ShellStart`/`Test` advert removed (never registered). Security guard is interpreter-aware (`sh -c` bypass closed).
- Token counting is now REAL via Ollama `/api/tokenize` (model's own tokenizer), with heuristic fallback only on server error. `README.md` created.
- Architecture of 4 crates is the correct sweet spot — not refactored for its own sake, now with `model_profile.rs`, `router.rs`, `contract.rs`, `grammar.rs`, `mcp.rs`.
- **F0-F5 hardened**: guard (command substitution, eval block, timeout 30s bench), hashline prefix matching, tool error handling, budget single-point accounting (`push_message` sole), cross-turn tool-loop detection (args-aware blake3), message ordering, thinking retry (cap 2), Read truncation (300 lines), Grep via ignore crate, dead code removal, CLI flags (`--profile tiny|medium|large`, `--no-memory` auto), JSON exercise loading.
- **Empty response fix**: assistant msg pushed to history, no orphaned LLM round-trips, `MAX_EMPTY_STREAK=2` + fail-fast → latency 87s→4s smollm2, 39s→24s qwen3.
- **Deepwork 2026-08-19 noche:** 5 commits (Fase0 3048272, Fase1 861a7c4, P0 171819c, P1 875bd6a, cleaner d5f6375), 4 Gates Oracle 1-4 aprobados (2 re-reviews), `benches/baseline.json` 96.44, `kitty mcp` stdio 6 tools via Guard.

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

## Honesty Notes (verified 2026-08-19)
- **126 tests pass** and `cargo clippy --all-targets` is clean.
- Benchmarks are genuinely homemade (string-match) — documented honestly.
- Memory recall is now **lexical Jaccard** (was substring `LIKE`); dense embeddings
  are a planned upgrade. `save_episode` records real completion status.
- `tree-sitter` dependency and `trait Tool` removed; `ShellStart`/`Test` advert
  removed (never registered). Security guard is interpreter-aware (`sh -c` bypass
  closed).
- Token counting is now REAL via Ollama `/api/tokenize` (model's own tokenizer),
  with heuristic fallback only on server error. `README.md` created.
- Architecture of 4 crates is the correct sweet spot — not refactored for its own sake.
- **F0-F5 hardened**: guard (command substitution, eval block, timeout), hashline
  prefix matching, tool error handling, budget single-point accounting, cross-turn
  tool-loop detection, message ordering, thinking retry, Read truncation (300 lines),
  Grep via ignore crate, dead code removal, CLI flags, JSON exercise loading.
- **Empty response fix**: assistant msg pushed to history, no orphaned LLM round-trips.
  Exercise latency dropped from ~87s to ~28s average.

# crates/kitty-harness/

## Responsibility
The agentic core: orchestrates the LLM call loop, tool execution, and the safety /
quality / budget sub-systems. This is "the product" (the harness, not the model).

## Design
- `Agent` (agent.rs): owns the message history + all subsystems and drives
  `run(prompt)` — a single-turn loop bounded by `max_iterations` (default 20):
  bootstrap → recall → knowledge inject → preflight → LLM call → parse → quality
  check → tool exec → repeat; or force a text response / compact context.
- `ToolRegistry` (tools.rs): dispatch table (Read/Edit/Write/Bash/Glob/Grep) with
  session guards (read-before-edit, write-on-existing-block) and a `HashlineEditor`.
- `HashlineEditor` (hashline.rs): per-line BLAKE3 tagging; edits target a line hash,
  rejected if the file changed on disk since last read (optimistic-concurrency).
- `ToolCallParser` (parser.rs): forgiving multi-format parser (native tool_calls,
  `<tool_call>` XML, ```json``` blocks, bare JSON, `<think>`-embedded) with
  auto-repair fallbacks.
- `Guard` (guard.rs): **interpreter-aware** safety. `SAFE_OUTPUT_COMMANDS` holds
  only non-executing commands (echo/cat/grep/...); `INTERPRETERS` (sh, bash,
  python, awk, sed, ...) are scanned on the RAW command (quotes intact) so payloads
  like `sh -c 'rm -rf /'` or `awk 'BEGIN{system(...)}'` are blocked. General
  commands are scanned after quote-stripping to avoid false positives.
- `QualityMonitor` (quality_monitor.rs): detects empty responses, tool loops,
  hallucinated tools; emits corrective messages.
- `ThinkingBudget` (thinking_budget.rs): per-turn / per-session reasoning-token caps.
- `ContextBudget` (context.rs): 4-chars/token estimate, 80% compaction threshold,
  consecutive-compaction halt guard.
- `KnowledgeBase` (knowledge.rs): markdown "cards" scored by word(1.0)+bigram(2.0)
  against the prompt, injected at conversation tail.
- `Bootstrap` (bootstrap.rs): marker-file project detection (Rust/Node/Py/Go),
  cached; `execute_test_tool` auto-detects the test command (only called from its
  own tests — not wired into the tool registry).

## Flow
`Agent::run` → `llm_call` → `ToolCallParser` (native + text) → `QualityMonitor`
→ `ToolRegistry::execute` → `HashlineEditor`/Bash/... → push result → loop.
Context pressure triggers `trigger_compaction` (LLM summarization of old turns).

## Integration
- Consumed by: `kitty-cli`. Depends on: `kitty-runtime`, `kitty-memory`.
- Honest gaps (see audit): `recall_episodes`/`inject_knowledge`
  still push `role:"user"` messages before the system message (ordering + untracked budget).

## Fixed (2026-08-18 / 2026-08-19)
- `save_episode` records real completion status (success on natural finish,
  failure when the loop exhausts `max_iterations`); previously only "pass" was saved.
- Compaction halt-guard: `trigger_compaction` registers a no-op when there is
  nothing left to compact, so `should_halt` engages instead of spinning.
- `guard.rs` is interpreter-aware (see `Guard` above) — closes the `sh -c` bypass.
- clippy clean (6 harness warnings fixed).
- **Loop quality**: `preflight` (system message) is now pushed FIRST, so
  `recall_episodes`/`inject_knowledge` no longer insert `role:"user"` messages
  ahead of the system message. Both injections are now budget-tracked via
  `ContextBudget::consume`.
- **Real token counting**: `Agent::count_tokens` uses Ollama's `/api/tokenize`
  (the model's own tokenizer) for the user prompt, thinking, and tool outputs,
  falling back to the heuristic only when the server is unreachable.

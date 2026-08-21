# kitty-harness - AGENTS.md

## OVERVIEW
Agentic core: LLM loop, tool dispatch, safety, quality, and budget. 14 files, ~7K LOC. Product, not model.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Agent loop | `src/agent.rs:296` `run`, `:399` `process_response` | `new_inner` builds 12-field `Agent`; `run` is preflight, recall, knowledge, loop(20), compaction |
| Safety guard | `src/guard.rs:824` `Guard` | 1444L, interpreter-aware: `BLOCKED_PATTERNS` 18 + `RISKY` 13, `ALWAYS_BLOCKED` eval, `INTERPRETERS` raw scan, `SAFE_OUTPUT` quote-stripped |
| Shell parse + writes | `src/guard.rs` `shell_parse`, `detect_write_targets` | tokenizes `&&\|\|;\|()` chains, `NON_DESTRUCTIVE_TARGETS` exempts `/dev/null` only |
| Tool dispatch | `src/tools.rs:58` `ToolRegistry` | 966L, Read/Edit/Write/Bash/Glob/Grep; Bash 30/60/120s per profile; Grep via `ignore` crate |
| Hashline edits | `src/hashline.rs:41` `HashlineEditor` | BLAKE3 per-line `HASH\|content`, `edit_by_hash` prefix >=6, optimistic concurrency `FileChanged` |
| Model profiles | `src/model_profile.rs:32` | `TINY` 4K/5iter/4tools/no-native, `MEDIUM` 16K/20iter/6tools/native, `LARGE` 32K/20iter/native |
| Tool parser | `src/parser.rs:1` `ToolCallParser` | native `tool_calls` + `<tool_call>` XML + ```json + bare JSON + `<think>` embedded, auto-repair |
| Tool routing | `src/router.rs:1` `ToolRouter` | filters `ToolDefinition` by `tool_subset` per profile |
| Quality monitor | `src/quality_monitor.rs:1` | empty response, blake3 loop detection (args-aware), hallucinated tools, corrective messages |
| Budgets | `src/context.rs:1` `ContextBudget`, `src/thinking_budget.rs:1` | 4-char/token est, 80% compaction + halt-guard, per-turn/per-session thinking caps |
| Contract + knowledge | `src/contract.rs:50` `Contract`, `src/knowledge.rs:94` | DoD done-guard (`?` not done), cards scored word1.0/bigram2.0 injected at tail |
| Bootstrap | `src/bootstrap.rs:1` | marker-file project detect (Rust/Node/Py/Go), cached; `execute_test_tool` test-only, not in registry |

## CONVENTIONS
- `Agent::run` pushes `preflight` (system) FIRST, then `recall_episodes` and `inject_knowledge` as tracked `push_message` calls. Order and `ContextBudget::consume` matter.
- `count_tokens` via `/api/tokenize` real tokenizer, heuristic fallback only. Sole accounting in `push_message`.
- `Guard` scans `INTERPRETERS` on RAW command (quotes intact), general commands after quote-strip. `SAFE_OUTPUT_COMMANDS` never executes args.
- `ToolRegistry` enforces read-before-edit and blocks Write on existing file, Edit on missing file.

## ANTI-PATTERNS
- NEVER bypass `Guard::check_command` / `check_chain`, payloads hide in `sh -c`, `python -c`, `awk system()`. Interpreter-aware scan exists for this.
- NEVER `Edit` without prior `Read`, hash prefix <6, or ignoring `FileChanged`, re-read required.
- NEVER assume `Bash` timeout is fixed, it is profile-driven (tiny 30s, medium 60s, large 120s).
- NEVER expose all 6 tools on `tiny`, native tools off, 4-tool subset only.
- NEVER push `role:user` before system preflight, breaks prompt ordering and budget tracking.
- NEVER use `Grep` without `ignore` crate semantics, respects `.gitignore`.

## NOTES
- Split candidates: `guard.rs` -> `guard/shell_parse`, `agent.rs` -> `agent/loop` + `agent/response` (each >900L).
- `lib.rs` re-exports only, no logic. Keep modules <250 LOC ideal, extra large files flagged above.

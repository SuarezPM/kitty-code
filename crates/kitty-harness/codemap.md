# crates/kitty-harness/

## Responsibility
The agentic core: orchestrates the LLM call loop, tool execution, and the safety /
quality / budget sub-systems. This is "the product" (the harness, not the model).

## Design
- `Agent` (agent.rs: ~1177L): owns the message history + all subsystems and drives
  `run(prompt)` — a single-turn loop bounded by `max_iterations` (default 20, bench 5):
  bootstrap → recall → knowledge inject → preflight → LLM call → parse → quality
  check → tool exec → repeat; or force a text response / compact context. Supports
  `ChatClient` trait abstraction (OllamaClient / LlamaCppClient / CloudClient fallback),
  `cloud_fallback` opt-in (1 retry on empty/exhaust/ToolLoop, `KITTY_CLOUD_*` env, cost metered),
  `semantic_memory` hybrid recall (BM25+embedding via RRF K=60, tiny guard 4K blocks hybrid).
- `ToolRegistry` (tools.rs: ~966L committed, ~1307L with plugin+sandbox wiring): dispatch table (Read/Edit/Write/Bash/Glob/Grep) with
  session guards (read-before-edit, write-on-existing-block) and a `HashlineEditor`. Bash via
  `Sandbox::apply` (KITTY_SANDBOX_* env + Landlock/seccomp stubs, mem 512MB cpu 30s net-deny) +
  signal containment (SIGKILL 9 OOM / SIGSEGV 11 / SIGABRT 6 via `ExitStatusExt`). Plugin delegation
  with Guard inheritance (`check_chain` on `args["command"]` + ALWAYS_BLOCKED) before `PluginRegistry::execute`,
  error isolation `success:false` not loop abort.
- `HashlineEditor` (hashline.rs: ~454L): per-line BLAKE3 tagging; edits target a line hash,
  rejected if the file changed on disk since last read (optimistic-concurrency), prefix >=6.
- `ToolCallParser` (parser.rs: ~425L): forgiving multi-format parser (native tool_calls,
  `<tool_call>` XML, ```json``` blocks, bare JSON, `<think>`-embedded) with
  auto-repair fallbacks.
- `Guard` (guard.rs: ~1444L): **interpreter-aware** safety. `BLOCKED_PATTERNS` 18 + `RISKY` 13,
  `ALWAYS_BLOCKED_COMMANDS` (eval etc., now `pub(crate)` for plugin inheritance), `INTERPRETERS` (sh, bash,
  python, awk, sed, ...) are scanned on the RAW command (quotes intact) so payloads
  like `sh -c 'rm -rf /'` or `awk 'BEGIN{system(...)}'` are blocked. General
  commands are scanned after quote-stripping to avoid false positives. `shell_parse` + `detect_write_targets`.
- `Sandbox` (sandbox.rs: 253L, F6 W1): `SandboxConfig { fs_allow, net_deny, mem_limit_mb, cpu_timeout_secs }`
  + `Sandbox::apply(&mut Command)` env hints + Landlock/seccomp stubs `#[cfg(linux)]`, `is_path_allowed`/`check_path`
  with `normalize` prefix-match (strict `/tmp/ws` not `/tmp/ws-evil`). 6 tests: allow/deny/signal/env.
- `Plugin` (plugin.rs: 546L, F6 W2): `trait Plugin { name, tools, execute }` + `PluginRegistry { Vec<Box<dyn Plugin>> }`
  `register`/`register_plugin` (DuplicateTool check intra + global), `tools`/`tool_names`/`contains_tool`/`execute`,
  `load_from_manifest(path)` TOML `[[plugin]]` + JSON fallback, factory `"hello"->HelloPlugin`, `TwoToolPlugin` + `HelloPlugin`.
  13 tests (10 core +3 isolation, core untouched via `standard_tool_definitions` 6).
- `QualityMonitor` (quality_monitor.rs: ~474L): detects empty responses, tool loops,
  hallucinated tools (args-aware blake3); emits corrective messages.
- `ThinkingBudget` (thinking_budget.rs: ~181L): per-turn / per-session reasoning-token caps (tiny 512 bench).
- `ContextBudget` (context.rs: ~221L): 4-chars/token estimate, 80% compaction threshold,
  consecutive-compaction halt guard.
- `KnowledgeBase` (knowledge.rs: ~384L): markdown "cards" scored by word(1.0)+bigram(2.0)
  against the prompt, injected at conversation tail.
- `Contract` (contract.rs: ~416L): DoD done-guard (`?` not done, `ContractState` pending/passed/failed).
- `ModelProfile` (model_profile.rs: ~153L): `TINY` 4K/5iter/4tools/no-native, `MEDIUM` 16K/20iter/6tools/native, `LARGE` 32K/20iter/native.
- `ToolRouter` (router.rs: ~433L): filters `ToolDefinition` by `tool_subset` per profile.
- `Bootstrap` (bootstrap.rs: ~268L): marker-file project detection (Rust/Node/Py/Go),
  cached; `execute_test_tool` auto-detects the test command (only called from its
  own tests — not wired into the tool registry).

## Flow
`Agent::run` → `llm_call` → `ToolCallParser` (native + text) → `GrammarClient` strict filter → `QualityMonitor`
→ `ToolRegistry::execute` (Sandbox env + Guard → HashlineEditor/Bash/... or PluginRegistry delegated with Guard) → push result → loop.
Context pressure triggers `trigger_compaction` (LLM summarization of old turns). Cloud fallback on fail-fast paths.

## Integration
- Consumed by: `kitty-cli`. Depends on: `kitty-runtime` (ChatClient, CloudClient, LlamaCppClient, GrammarClient), `kitty-memory` (EpisodeStore, BM25/RRF/embedding).
- `lib.rs` re-exports `Agent`, `ContextBudget`, `Guard`, `ThinkingBudget`, `QualityMonitor`, `KnowledgeBase`, `Bootstrap`, `ModelProfile`, `ToolRouter`, `Contract`; `pub mod sandbox`, `pub mod plugin` (F6) wiring.
- Honest gaps: dense embedding + RRF recall wired via `store.recall_hybrid` opt-in (`--semantic-memory`, tiny blocked); `kitty-runtime` llama feature is vocab-only probe + fallback without GGUF (no block).

## Tests
202 lib tests on main (guard 57, tools, hashline, parser, agent, quality, contract, knowledge, etc.) + 6 sandbox +13 plugin = **~221** when F6 modules wired (currently 202 on committed `lib.rs` without mod). `cargo test -p kitty-harness --lib` green, clippy `-D warnings` clean.

## Fixed (F6 W1-W3)
- Sandbox FS whitelist + net-deny + seccomp stubs, signal containment (OOM/SIGSEGV/SIGABRT) not escaping loop.
- Plugin static registry without `dlopen`/`libloading`, Guard inheritance for Bash-like plugins, error isolation, manifest `plugins.toml` + `docs/PLUGIN.md`.
- Guard `ALWAYS_BLOCKED` `pub(crate)` for plugin delegation ordering (Guard BEFORE execute).
- `save_episode` records real completion status; compaction halt-guard; interpreter-aware guard; real token counting via `/api/tokenize`.

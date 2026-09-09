# crates/kitty-harness/

## Responsibility
The agentic core: owns the LLM call loop, tool dispatch, and the safety / quality /
budget sub-systems. This is "the product" (the harness, not the model).

## Design
- Agent loop (`agent/mod.rs`: `Agent`, `AgentConfig`, `RunOverrides`, `AgentEvent`;
  `agent/run.rs`: `run`/`run_impl`; `agent/response.rs`: `process_response`,
  `force_response`, `trigger_compaction`, `verify_workspace`): `max_iterations`
  bounded tool loop (default 20, tiny 5) with verification revisions
  (`VERIFY_MAX_REVISIONS = 2`) and autorun continuation hops
  (`autorun_hops`, `AUTORUN_CONTINUE_PROMPT`, `MAX_AUTORUN_HOPS = 12`).
  `push_message` is the sole token-accounting point.
- Tool registry (`tools.rs`: `ToolRegistry`, `ToolResult`, `ToolError`,
  `standard_tool_definitions`, `normalize_tool_call`): Read/Edit/Write/Bash/Glob/Grep
  with read-before-edit + Write-on-existing guards; Bash via `Sandbox::apply` + `Guard`;
  plugin delegation with Guard inheritance; tool errors become failed results, never aborts.
- Guard (`guard/mod.rs`: `Guard`, `GuardVerdict::{Allow,Block,Warn}`, `check_command`,
  `check_chain`; `guard/patterns.rs`: `BLOCKED_PATTERNS`, `RISKY_PATTERNS`,
  `SAFE_OUTPUT_COMMANDS`, `ALWAYS_BLOCKED_COMMANDS`, `NON_DESTRUCTIVE_TARGETS`;
  `guard/shell_parse.rs`: `strip_quotes`, `command_name`, `has_command_substitution`,
  `detect_write_targets`, `is_write_via_redirect`; `guard/interpreters.rs`:
  `INTERPRETERS`): interpreters scanned on RAW command, general commands after
  quote-strip; redirect writes (`>`/`tee`/`dd`) blocked before safe-output allow.
- Sandbox (`sandbox.rs`: `Sandbox`, `SandboxConfig`, `SandboxOutcome`, `SandboxError`,
  `MAX_OUTPUT_BYTES`): Landlock FS-write confinement + seccomp-bpf + user/mount/PID
  namespaces + `RLIMIT_AS`; fail-closed when the kernel lacks Landlock/namespaces.
- Fuzzy edit ladder (`edit_replacers.rs`: `run_ladder`, `LadderOutcome::{Hit,Ambiguous,
  NotFound}`, `LadderHit`; `hashline.rs`: `HashlineEditor`, `LineHash`, `HashlineError`):
  Edit tries exact str-replace first, then line-trim → whitespace → escape →
  indent-flexible → context-aware → block-anchor, each vetted by the
  disproportionate-match guard; hash protocol (`edit_by_hash`, prefix ≥6) rejects
  `FileChanged` on stale reads.
- Budgets (`context.rs`: `ContextBudget`, `ContextError`, `CompactionSnapshot`,
  `DEFAULT_THRESHOLD = 0.8`; `thinking_budget.rs`: `ThinkingBudget`,
  `set_per_turn_cap`; `model_profile.rs`: `ModelProfile::{nano,tiny,medium,large}`,
  `ModelTuning`, `ThinkingMode`, `PromptStyle`): per-turn/per-session thinking caps
  re-seeded from tuning each run; profile drives iterations, tool subset, read cap,
  Bash timeout/output cap, compaction threshold.
- Compaction (`compaction.rs`: `CompactionSummary::{empty,format_markdown}`;
  `context.rs` halt-guard): structured Goal/Constraints/Progress summary replaces old
  turns at threshold; consecutive-compaction halt stops the loop when nothing is freed.
- Parser (`parser.rs`: `ToolCallParser`, `ParsedToolCall`, `DEFAULT_KNOWN_TOOLS`,
  `repair_json`, `levenshtein`; `json_repair.rs`: `repair_json`, `coerce_with_schema`):
  native `tool_calls` + Qwen/Mistral tags + `<tool_call>` XML + fenced/bare JSON +
  `<think>`-embedded, with structural repair, empty-args recovery, fuzzy names (≤2).
- Quality (`quality_monitor.rs`: `QualityMonitor`, `QualityConfig`, `QualityViolation`,
  `QualityEvent`): empty-response / tool-loop (3×, blake3 args-aware) / hallucinated-tool
  detection with corrective messages; window persists across turns.
- Router (`router.rs`: `ToolRouter`, `ToolCategory::{Read,Write,Run,Search,Plan}`,
  `tools_for_category`): zero-token 2-stage classifier, gated behind `use_routing`.
- Contract (`contract.rs`: `Contract`, `ContractState`, `Assertion`, `AssertionCheck`,
  `AssertionStatus`, `contract_guard_enabled`, `looks_like_done_claim`): DoD done-guard.
- Knowledge (`knowledge.rs`: `KnowledgeBase`, `KnowledgeCard`, `KnowledgeConfig`):
  word(1.0)+bigram(2.0) cards injected at tail when `enable_knowledge`.
- Bootstrap (`bootstrap.rs`: `Bootstrap`, `BootstrapResult`, `ProjectType`): marker-file
  Rust/Node/Python/Go detection, cached on first turn.
- Cross-cutting (`config.rs`: `KITTY_*` consts + typed accessors like `sandbox_off`,
  `telemetry_enabled`, `verify_loop_enabled`; `redact.rs`: `redact_secrets`;
  `overflow.rs`: `detect_overflow`, `detect_silent_overflow`; `telemetry.rs`:
  `TelemetryWriter`, `TurnEvent` JSONL rotating 10MB; `skills.rs`: `Skill`,
  `discover_skills` progressive disclosure; `comparison.rs`: `compare_harnesses`,
  `HarnessComparison`, `ExerciseResult`; `plugin.rs`: `Plugin`, `PluginRegistry`,
  `HelloPlugin`, `TwoToolPlugin`, `load_from_manifest`).

## Flow
`Agent::run` (`agent/run.rs::run_impl`) → `run_bootstrap` → `preflight` (system first) →
`recall_episodes` → `inject_knowledge` → `push_message(user)` → per-turn thinking-cap
re-seed → verify/autorun outer loops → bounded `for` loop: `should_compact` →
`trigger_compaction` → `should_halt` → LLM call (schema/overflow recovery) →
`process_response` (`response.rs`: native + XML parse → `normalize_tool_call` →
`QualityMonitor::check_response` → `execute_and_push_tool_result` via
`ToolRegistry::execute` w/ Guard+Sandbox → push result before correctives) →
exhaustion → `force_response` → `verify_workspace` (≤2 revisions) → episode save → Done.
`overflow::detect_overflow` / `detect_silent_overflow` reroute to compaction on window errors.

## Integration
- Consumed by: `kitty-cli`. Depends on: `kitty-runtime` (`ChatClient`, `GrammarClient`,
  `ToolDefinition`, `Message`, `ChatResponse`), `kitty-memory` (`EpisodeStore`,
  `Episode`, hybrid recall).
- `lib.rs` re-exports `Agent`/`AgentEvent`, `ContextBudget`, `Guard`/`GuardVerdict`,
  `KnowledgeBase`, `ModelProfile`/`ModelTuning`/`ThinkingMode`, `Plugin`/`PluginRegistry`,
  `QualityMonitor`, `ToolRouter`/`ToolCategory`, `Sandbox`, `ThinkingBudget`, `Contract`;
  `pub mod` for `compaction`, `comparison`, `config`, `edit_replacers` (private),
  `json_repair`, `overflow`, `redact`, `skills`, `telemetry`, `agent`, `guard`.

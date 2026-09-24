# crates/kitty-harness/src/

## Responsibility

One module per harness subsystem. `agent/` runs the loop; `tools.rs` +
`guard/` + `sandbox.rs` execute it safely; `parser.rs` + `model_profile.rs` +
budgets adapt it per model; the rest observe, persist, or extend it. No module
imports the CLI; the runtime is touched only through `ChatClient` and message
types.

## Design

- `agent/`: `mod.rs` (`Agent`, `AgentConfig`, `RunOverrides`, `AgentEvent`),
  `run.rs` (`run`/`run_impl`, verify/autorun outer loops, overflow reroute),
  `response.rs` (`process_response`, native+text parse, `tool_call_id`
  threading, quality gate, dispatch, correctives, `force_response`,
  `trigger_compaction`, `verify_workspace`).
- `tools.rs`: `ToolRegistry` (Read/Edit/Write/Bash/Glob/Grep,
  read-before-edit, Guard+Sandbox on every call), `normalize_tool_call`
  (name synonyms + `ARG_KEY_ALIASES` re-keying, idempotent choke point),
  `_raw` salvage repair, tool errors as failed results.
- `parser.rs` / `json_repair.rs`: multi-format call extraction (native, XML,
  fenced/bare JSON, channel tags, loose `call:`, kwarg form, `<think>`),
  lenient repair, `_raw` retention, bounded empty-args recovery, fuzzy names.
- `guard/` (`mod.rs`, `patterns.rs`, `shell_parse.rs`, `interpreters.rs`):
  allow/block/warn verdicts; raw-scan interpreters, quote-stripped general
  commands; redirect-write blocks before safe-output allows.
- `sandbox.rs`: Landlock + seccomp + namespaces + `RLIMIT_AS`, fail-closed.
- `model_profile.rs` (`nano/tiny/medium/large`, `ModelTuning`,
  `ThinkingMode`, `PromptStyle`) + `thinking_budget.rs` + `context.rs`
  (+ `compaction.rs` summaries, halt-guard): per-profile iterations, tool
  subsets, caps, thresholds.
- `quality_monitor.rs`: empty/loop/hallucination detection with correctives,
  cross-turn window. `router.rs`: zero-token tool subsetting (opt-in).
  `contract.rs`: done-claim guard. `knowledge.rs`: tail-injected cards.
  `bootstrap.rs`: cached project-type detection.
- `config.rs`: typed `KITTY_*` accessors (single env-read choke point).
  `redact.rs`: secret scrubbing. `overflow.rs`: window-error rerouting.
  `telemetry.rs`: rotating JSONL turn events. `skills.rs`: progressive
  skill disclosure. `comparison.rs`: harness A/B. `plugin.rs`: manifest
  plugins with Guard inheritance.
- `hashline.rs` + `edit_replacers.rs`: hash-anchored edits with fuzzy
  fallback ladder and disproportionate-match guard.
- `lib.rs`: re-exports only, no logic.

## Flow

Config → profile/tuning resolution → agent loop (recall → knowledge →
bounded turns: budget re-seed → LLM → parse → normalize → quality →
execute → push → compact) → verify → episode persist. Budgets, guard, and
quality observe every turn without owning the loop.

## Integration

- Consumed by: `kitty-cli` (`Agent`, config, overrides, registry, profiles).
- Depends on: `kitty-runtime` (clients, messages, tool defs),
  `kitty-memory` (episodes); Landlock/seccomp OS interfaces via crates;
  no dependency on `kitty-cli`.

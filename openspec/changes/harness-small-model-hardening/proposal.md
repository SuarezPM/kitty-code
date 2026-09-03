## Why

Kitty Code is a harness for small local models (1B-35B parameters). Small models produce malformed tool calls ~30% of the time, have limited context windows requiring aggressive compaction, and benefit from token-efficient prompts. The current harness lacks JSON repair, structured compaction, progressive skill loading, and a real Aider Polyglot benchmark with unit test grading. These gaps prevent reliable evaluation against public leaderboards and reduce small-model task completion rates.

## What Changes

- **P1**: Add JSON repair pipeline — fix malformed tool call arguments via `repairJson()` and `coerceWithJsonSchema()` before validation
- **P2**: Add structured compaction summaries — Goal/Constraints/Progress/Decisions/Next Steps format replaces free-form summaries
- **P3**: Add progressive skill disclosure — skills listed by name+description in prompt, full content loaded on-demand via Read tool
- **P4**: Add tool-gated prompt sections — only show guidelines for tools currently active in the session
- **P5**: Add Aider Polyglot loader — 225 Exercism exercises with real unit test grading, comparable to public leaderboard
- **P6**: Enhance retry with multi-tier error classification — 60+ error patterns, 3-tier retry (provider HTTP, application, overflow)
- **P7**: Add context overflow detection — 25+ provider-specific overflow patterns, auto-compact + retry once
- **P8**: Add harness comparison table — baseline vs candidate with correctness lift, wins/ties/losses

## Capabilities

### New Capabilities
- `json-repair`: JSON repair pipeline for malformed tool call arguments from small models
- `structured-compaction`: Goal/Constraints/Progress compaction summaries with file tracking
- `progressive-skills`: Progressive skill disclosure — name+description in prompt, load on-demand
- `tool-gated-prompts`: Tool-gated prompt sections — only guidelines for active tools
- `aider-polyglot`: Aider Polyglot 225 benchmark loader with unit test grading
- `multi-tier-retry`: Multi-tier retry with 60+ error classification patterns
- `overflow-detection`: Context overflow detection with 25+ provider patterns
- `harness-comparison`: Harness comparison table with correctness lift metrics

### Modified Capabilities
- `evaluation/benchmarking`: Add Polyglot benchmark suite alongside existing polyglot/tb2/smoke
- `tool-loop`: Integrate JSON repair into tool call parsing pipeline

## Impact

- `crates/kitty-harness/src/` — New modules: json_repair.rs, compaction.rs, skills.rs, overflow.rs
- `crates/kitty-cli/src/` — New module: polyglot.rs (Aider Polyglot loader)
- `crates/kitty-harness/src/tools.rs` — Integrate JSON repair into tool call parsing
- `crates/kitty-harness/src/agent.rs` — Integrate structured compaction, overflow detection, enhanced retry
- `crates/kitty-harness/src/context.rs` — Structured summary format
- `crates/kitty-harness/src/router.rs` — Tool-gated prompt sections
- `crates/kitty-cli/src/main.rs` — Polyglot bench command, harness comparison

# Structured Compaction Summaries

## Purpose
Replace free-form compaction summaries with a structured Goal/Constraints/Progress/Decisions/Next Steps format that preserves context better across compactions.

## Requirements

### R1: Structured Summary Format
- GIVEN a compaction is triggered (context near limit)
- WHEN the summary is generated
- THEN the output follows this structure:
  - `## Goal` — current objective
  - `## Constraints & Preferences` — user constraints, model preferences
  - `## Progress` — Done / In Progress / Blocked items
  - `## Key Decisions` — architectural and design choices made
  - `## Next Steps` — immediate next actions
  - `## Critical Context` — anything else essential for continuity

### R2: File Operation Tracking
- GIVEN messages contain tool results with file reads/writes
- WHEN compaction summary is generated
- THEN `<read-files>` and `<modified-files>` lists are appended
- AND files are listed by path with brief description

### R3: Iterative Summary Updates
- GIVEN a previous compaction summary exists
- WHEN a new compaction is triggered
- THEN the new summary is an UPDATE to the previous (preserves existing context)
- AND the model receives both the old summary and new messages

### R4: Cut Point Finding
- GIVEN context usage exceeds threshold
- WHEN finding where to cut
- THEN cut at a valid message boundary (user/assistant, never mid-tool-call)
- AND keep approximately `keep_recent_tokens` (default 20,000) of recent messages

### R5: Integration with ContextBudget
- GIVEN the `CompactionSnapshot` from F4.2
- WHEN compaction runs
- THEN snapshot data (recent tool results, active file, bench exercise) is included in the summary

### R6: Unit Tests
- GIVEN the compaction module
- WHEN tests are run
- THEN structured format is correct, file tracking works, iterative updates preserve context, cut points are valid

## Non-Goals
- No LLM-powered summarization (uses template-based approach for determinism)
- No cross-session compaction (session-scoped only)

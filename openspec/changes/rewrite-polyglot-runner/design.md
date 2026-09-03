## Context

Four benchmark runners have critical issues producing invalid or incomparable results. This rewrite aligns all runners with official methodologies.

## Goals / Non-Goals

**Goals:**
- Polyglot: Follow official Aider pattern (config.json, test hiding, retry, timeouts)
- GPQA: Align with official format (CoT, ANSWER: $LETTER, strict scoring)
- ClawEval: Fix model flag bug, remove dead code
- TB2: Add Harbor CLI integration for official Terminal-Bench 2.1

**Non-Goals:**
- No Docker sandbox for Polyglot (local execution)
- No changes to homemade Python runners (kept as internal tools)
- No GPQA dataset download (already materialized)

## Decisions

### Polyglot: config.json as single source of truth
Use `.meta/config.json` for file classification. Test files hidden during agent turn.

### GPQA: CoT + strict format
Add chain-of-thought prompt and `ANSWER: $LETTER` format. Strict scoring: no prefix = 0 points.

### ClawEval: delegation to official CLI
Fix model flag passthrough. Remove dead code. Keep wrapper architecture.

### TB2: Harbor subprocess wrapper
Delegate to `harbor run` CLI. Parse output for resolution rate.

## Risks

- [GPQA CoT] → CoT may increase latency for local models. Acceptable for accuracy.
- [ClawEval model flag] → Critical bug: results invalid without fix.
- [TB2 Harbor dependency] → Requires Docker + Harbor installed. Preflight checks.

# ADR: Release panic strategy (`panic = "abort"`)

- **Status:** Accepted (kept)
- **Date:** 2026-09-03
- **Deciders:** kitty-code maintainers
- **Scope:** `[profile.release]` in the workspace root `Cargo.toml`

## Context

The 2026-09 code audit flagged a design tension: the agent loop's stated
philosophy is fail-soft ("tool errors → failed result messages, never abort
session"), yet the release profile sets `panic = "abort"`, which kills the
whole process on any latent panic — with no unwinding, no telemetry hook, and
no per-tool isolation.

## Decision

**Keep `panic = "abort"` in release builds.** Rationale:

1. **Single-user local tool.** One agent, one session, one process. A panic
   aborts the session, not a shared service. The blast radius is the session
   the user is watching.
2. **No plugin boundary.** Panics cannot cross an FFI/plugin edge today;
   there is no embedded third-party runtime whose unwinding we must contain.
3. **Binary size and simplicity.** Abort drops the unwinding tables and
   landing pads (smaller, marginally faster); `catch_unwind`-based tool
   isolation would add a recovery layer that hides harness bugs instead of
   making them loud.
4. **Fail-fast semantics are already explicit.** Tool failures are mapped to
   `ToolResult::success = false` long before a panic would occur; panics are
   reserved for genuine invariant violations, where dying loudly is the
   correct contract.

## Consequences

- A latent panic in the loop terminates the session immediately. The user
  re-runs; state is bounded (episodic memory persists in SQLite, telemetry
  JSONL persists up to the last flushed turn).
- `catch_unwind`-based recovery cannot be added incrementally on top of
  abort; it requires the profile change below.

## Re-evaluation triggers

Switch to `panic = "unwind"` **and** add `catch_unwind` isolation at the
tool boundary (`ToolRegistry::execute`) when ANY of these hold:

- The MCP server (`kitty mcp`) serves multiple concurrent client sessions in
  one process — a panic must not take down other sessions.
- A plugin/extension boundary is introduced (third-party code runs
  in-process).
- The verification loop runs workspace tests in-process (currently it shells
  out through the sandbox; if that ever changes, panics in test compilation
  must not kill the agent).

## Notes

- Debug builds keep the default (unwind) so tests can assert panics.
- `panic = "abort"` also suppresses `catch_unwind`-based leak sanitizers;
  accepted trade-off documented above.

# Delta Spec: design/run-overrides

## ADDED Requirements

### Requirement: Per-run overrides are explicit, not ambient mutations

The `Agent` SHALL accept per-run overrides via an explicit
`RunOverrides` value passed to `run()` instead of exposing mutable
interior state through `config_mut()` / `budget_mut()`.

- `RunOverrides` SHALL carry: `tuning: Option<ModelTuning>`,
  `compact_threshold: Option<f32>`, `autorun_hops: Option<u8>` (más campos
  que el diseño identifique necesarios para los benches).
- `Agent::config_mut()` and `Agent::budget_mut()` SHALL be removed from the
  public API.
- The thinking-budget re-seed workaround at the top of `run()`
  (G2-FIX3) SHALL be replaced by override application at a single point.

#### Scenario: bench override reaches the live budget

- **GIVEN** a caller constructs an `Agent` and wants a different thinking
  cap for one run
- **WHEN** it calls `agent.run_with(prompt, RunOverrides { .. })`
- **THEN** the per-turn thinking cap for that run equals the override, and
  construction-time defaults govern subsequent `run()` calls

#### Scenario: no public mutable accessors

- **WHEN** the public API of `kitty-harness::agent` is inspected
- **THEN** `config_mut` and `budget_mut` no longer exist

### Requirement: ToolRegistry hashline is private

`ToolRegistry::hashline` SHALL become a private field. External code SHALL
interact with hashline editing only through `ToolRegistry` methods that
preserve the read-before-edit and file-changed invariants.

#### Scenario: no public field access

- **WHEN** code outside `kitty-harness::tools` references
  `registry.hashline`
- **THEN** it fails to compile (zero occurrences in the workspace)

### Requirement: RunOverrides is unrepresentable-invalid

`RunOverrides` SHALL be constructible only via builder/`Default` so that
absence means "use configured defaults" (Option fields), never sentinel
values.

#### Scenario: default equals legacy behavior

- **GIVEN** `RunOverrides::default()`
- **WHEN** passed to `run_with`
- **THEN** behavior is identical to the legacy `run()` path

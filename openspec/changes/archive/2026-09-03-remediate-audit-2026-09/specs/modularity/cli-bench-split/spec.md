# Delta Spec: modularity/cli-bench-split

## ADDED Requirements

### Requirement: Benchmark suites live in dedicated bench modules

The kitty-cli binary SHALL keep its `main` and CLI parsing thin: all
benchmark orchestration (`gpqa`, `claweval`, `polyglot`, `tb2`) and benchmark
statistics SHALL live in dedicated modules under `crates/kitty-cli/src/bench/`.

- `main.rs` SHALL NOT contain suite runner functions (`run_*_suite`,
  `run_*_bench`) — verified by absence of `async fn run_gpqa_suite` and
  siblings in main.rs.
- Each suite module SHALL own its loading, scoring, and output logic.
- `main.rs` production LOC (excluding inline `#[cfg(test)]`) SHALL be < 1,000.

#### Scenario: main.rs delegates to bench modules

- **WHEN** the CLI is invoked with `bench polyglot`, `bench tb2`,
  `bench gpqa`, or `bench claweval`
- **THEN** the command dispatches into the corresponding
  `crate::bench::<suite>` entry point and behavior is byte-identical to the
  pre-split implementation

#### Scenario: main.rs size gate

- **WHEN** production lines of `main.rs` are counted
  (excluding `#[cfg(test)]` blocks)
- **THEN** the count is below 1,000

### Requirement: Harness god files are split into cohesive modules

`guard.rs` SHALL be split into a `guard/` module directory with
pattern tables, shell parsing, and interpreter logic as separate files.
`agent.rs` SHALL be split into an `agent/` module directory separating the
outer run/verification loop from response processing.

- Module files SHALL keep all existing behavior and test coverage.
- No public API change: existing re-exports in `kitty-harness` `lib.rs`
  SHALL resolve identically.

#### Scenario: guard split preserves behavior

- **WHEN** the full harness test suite runs after the split
- **THEN** all pre-existing guard tests pass without modification of their
  assertions

#### Scenario: agent split preserves behavior

- **WHEN** the full harness test suite runs after the split
- **THEN** all pre-existing agent tests pass without modification of their
  assertions

### Requirement: Central typed configuration for KITTY_* environment

All `KITTY_*` environment variables SHALL be read through a single typed
configuration module (`kitty-harness::config` or `kitty-cli::config` as the
design decides). Duplicate readers of the same variable
(`KITTY_SANDBOX`, `KITTY_NO_THINK`, `KITTY_MODEL_TURN_CAPS`,
`KITTY_TELEMETRY`, `KITTY_NUM_CTX`) SHALL consume the shared resolution.

- Each variable SHALL be documented once (name, default, effect).
- Environment reads SHALL happen once per process at config construction;
  no scattered `std::env::var("KITTY_...")` calls outside the config module.

#### Scenario: no duplicate env reads

- **WHEN** the workspace is searched for
  `std::env::var("KITTY_` outside the central config module
- **THEN** zero occurrences remain (test child-process helpers excepted,
  each documented)

#### Scenario: config is testable

- **WHEN** the config module is constructed from an explicit set of values
- **THEN** unit tests can inject values without mutating process env

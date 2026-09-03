# Delta Spec: security/supply-chain-2026

## ADDED Requirements

### Requirement: CI uses maintained toolchain setup

The CI workflow SHALL NOT use archived actions. `actions-rs/toolchain`
SHALL be replaced with a maintained toolchain setup (rustup directly or an
actively maintained action), keeping pinned-SHA discipline.

#### Scenario: no archived actions

- **WHEN** `.github/workflows/*.yml` is searched for `actions-rs/`
- **THEN** zero occurrences remain and all actions remain SHA-pinned

### Requirement: CI gates on strict clippy and full-feature tests

CI SHALL run `cargo clippy --all-targets --all-features -- -D warnings` and
`cargo test --workspace --all-features` so test code and opt-in features
(embedding) are gated.

#### Scenario: clippy gap closed

- **GIVEN** the hardened CI workflow
- **WHEN** a change introduces a clippy violation in test-only code
- **THEN** CI fails (would pass under the previous
  `cargo clippy -- -D warnings` gate)

### Requirement: CI produces an SBOM

CI SHALL generate a CycloneDX SBOM (`cargo cyclonedx`) as a workflow
artifact on every run on main.

#### Scenario: SBOM artifact present

- **WHEN** a main-branch CI run completes
- **THEN** a CycloneDX JSON SBOM for the workspace is uploaded as an
  artifact

### Requirement: Scheduled dependency audit

CI SHALL run `cargo audit` (advisory database check) on a weekly schedule
and on pull requests touching `Cargo.lock`.

#### Scenario: weekly audit job

- **WHEN** the schedule triggers or `Cargo.lock` changes in a PR
- **THEN** `cargo audit` runs and fails the job on known advisories with
  fixes available

### Requirement: panic strategy does not silently kill the session (ADR)

The workspace SHALL document an ADR decision for the release panic strategy.
If `panic = "abort"` is kept, the ADR SHALL state why and its operational
consequence; if changed to unwind, tool execution SHALL isolate panics with
`catch_unwind` at the tool boundary.

#### Scenario: ADR exists

- **WHEN** `docs/` is inspected after this change
- **THEN** an ADR covering the release panic strategy exists with the
  decision, rationale, and consequence recorded

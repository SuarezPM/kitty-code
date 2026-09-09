# Delta Spec: security/release-signing

## ADDED Requirements

### Requirement: Releases are built by automation on tag

A `release.yml` workflow SHALL build `kitty-cli` in release mode on
version tags with SBOM attached, so no release binary is produced on a
maintainer laptop.

#### Scenario: tag produces signed release artifacts

- **GIVEN** a version tag is pushed
- **WHEN** the release workflow runs
- **THEN** it compiles with `cargo build --release -p kitty-cli`,
  attaches the CycloneDX SBOM, and publishes both as release assets

### Requirement: Release artifacts are keylessly signed with provenance

Every release asset SHALL carry a cosign keyless signature (Rekor
bundle) plus a build-provenance attestation, verifiable with the
documented commands in `SECURITY.md`.

#### Scenario: signature verifies out of band

- **GIVEN** a published release asset with its `.sig`, `.pem`/bundle,
  and attestation
- **WHEN** a user runs the documented verify commands
- **THEN** identity, Rekor inclusion, and provenance all check out
  without any shared secret

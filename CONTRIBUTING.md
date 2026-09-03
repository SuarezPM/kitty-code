# Contributing to Kitty Code

Thank you for considering a contribution to Kitty Code. This document defines the contribution workflow, coding standards, and governance required by [OpenSSF Best Practices](https://www.bestpractices.dev/).

## Developer Certificate of Origin (DCO)

This project uses the [Developer Certificate of Origin](https://developercertificate.org/) (DCO) instead of a CLA. Every commit **MUST** include a `Signed-off-by` trailer certifying that you have the right to submit the contribution under the project's MIT license and that it complies with the DCO v1.1.

```
Signed-off-by: Your Name <your.email@example.com>
```

By adding this line you certify the following (from https://developercertificate.org/):

> Developer Certificate of Origin
> Version 1.1 — https://developercertificate.org/
>
> By making a contribution to this project, I certify that:
> (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
> (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that contribution with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
> (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
> (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.

**How to sign off:**

```bash
git commit -s -m "feat(scope): description"
# or add manually to an existing commit:
git commit --amend -s --no-edit
```

All pull requests are checked for a `Signed-off-by` line. Commits without it will not be merged (required for OpenSSF Silver).

## Code of Conduct

Be respectful and constructive. By participating you agree to uphold a professional, harassment-free environment. Issues of conduct may be reported to the maintainers listed in [GOVERNANCE.md](GOVERNANCE.md) or, until that file is finalized, via a confidential issue to the repository owners.

## Coding Standards

### Formatting — rustfmt

- Formatting is enforced by `rustfmt` with the default `rustfmt.toml` (project uses `edition = "2021"`).
- Do not hand-format code. Run before pushing:

```bash
cargo fmt --all -- --check   # CI check
cargo fmt --all              # fix locally
```

The required toolchain is pinned in `rust-toolchain.toml` (`channel = "stable"`, components `rustfmt` + `clippy`) so `cargo fmt` is always available.

### Linting — Clippy

- All code **MUST** pass Clippy with warnings denied:

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

- CI runs the same command. No `#[allow(clippy::...)]` without a justifying comment and maintainer approval.

### Style Principles

- Minimal, surgical changes — touch only what the task requires.
- Strict types, `serde` + `thiserror`, no `unwrap()`/`expect()` in production paths (tests OK).
- Keep modules under ~250 LOC; prefer small, testable functions.

## Build

Reproducible, locked build:

```bash
cargo build --locked
cargo build --locked --release   # release artifact
```

`Cargo.lock` is committed. `--locked` guarantees the exact dependency set reviewed in the lockfile.

## Test

Run the full workspace test suite:

```bash
cargo test --workspace
```

For a single crate during development:

```bash
cargo test -p kitty-harness
cargo test -p kitty-runtime
cargo test -p kitty-cli
```

Additional checks run in CI:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo deny check   # advisories + licenses + bans
```

## Test Policy

- Every bug fix **MUST** include a regression test that fails without the fix and passes with it.
- Every new feature **MUST** include tests covering the happy path and at least one edge/error case.
- Tests live next to the code (`#[cfg(test)]` modules) or in `crates/*/tests/` for integration tests.

> **regression_tests_added50 MUST** — More than 50% of bug-fix pull requests in the last 12 months are expected to add a regression test. This is tracked as part of OpenSSF Best Practices. PRs fixing a bug without a test must justify why a test is not applicable, subject to maintainer approval. In practice, add the test.

## Pull Request Process

1. Fork and create a feature branch (`feat/...`, `fix/...`, `docs/...`).
2. Ensure `cargo fmt`, `cargo clippy -- -D warnings`, and `cargo test --workspace` pass locally.
3. Commit with `Signed-off-by` (`git commit -s`) and a conventional prefix (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`).
4. Push and open a PR against `main`. Fill in the template: what, why, how verified.
5. At least one maintainer review is required. CI must be green (format, clippy, tests, deny).
6. Squash or rebase is at maintainer discretion; history should be linear and bisectable.

## Governance

Project governance, roles, and decision-making are defined in [GOVERNANCE.md](GOVERNANCE.md).

Until `GOVERNANCE.md` is formally adopted (tracked in the OpenSSF roadmap), the interim governance is:

- **Maintainers** have merge rights and set technical direction.
- **Contributors** propose changes via PRs/issues.
- Decisions are by lazy consensus; escalations by maintainer vote (simple majority).

See `GOVERNANCE.md` for the authoritative version once published — this section links there to satisfy OpenSSF Best Practices without duplicating the source of truth.

## Roles

| Role | Responsibilities | How to become |
|------|----------------|---------------|
| **Contributor** | Opens issues/PRs, follows DCO and coding standards | Any signed-off PR |
| **Reviewer** | Reviews PRs, triages issues | Sustained quality contributions, nominated by maintainer |
| **Maintainer** | Merges PRs, cuts releases, owns `CODEOWNERS` paths, enforces DCO/clippy/test policy | Nominated by existing maintainers, listed in `CODEOWNERS` and `GOVERNANCE.md` |

Current maintainers are listed in `.github/CODEOWNERS`.

## Reporting Security Issues

Do not open a public issue for vulnerabilities. See [SECURITY.md](SECURITY.md) for the private disclosure process.

## License

By contributing you agree that your contributions are licensed under the same license as the project ([MIT](LICENSE)).

## Mutation testing (guard/hashline hardening)

The two highest-risk modules are pattern/logic heavy (`guard`, `hashline`).
Unit tests alone do not prove the assertions kill mutants; run mutation
scoring locally before merging changes to either module:

```bash
cargo install --locked cargo-mutants
cargo mutants --package kitty-harness --file crates/kitty-harness/src/guard/*.rs
cargo mutants --package kitty-harness --file crates/kitty-harness/src/hashline.rs
```

Target: caught-mutant ratio > 70% on both surfaces. New tests for a behavior
must be written against the mutant, not just the happy path.

## Property-based and contract tests

- `crates/kitty-harness/tests/property_tests.rs` fuzzes the tool-call parser,
  hashline edits, and guard invariants with `proptest` (never-panic +
  stale-edit-detection + interpreter-wrapped-payload-blocking).
- `crates/kitty-runtime/tests/ollama_contract.rs` pins the Ollama HTTP
  contract with `wiremock` (`truncate:false` + session `num_ctx`, typed
  overflow on 400, transport-timeout mapping).

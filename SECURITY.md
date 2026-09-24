# Security Policy

## Supported Versions

We actively maintain security fixes for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| < 1.0   | :x:                |

Only the latest release on the `main` branch receives vulnerability patches. Older tagged releases are not supported — please upgrade before reporting.

## Reporting a Vulnerability

If you discover a security vulnerability, please report it privately so we can fix it before public disclosure.

**Preferred channels (private):**

- **GitHub Private Advisory:** https://github.com/SuarezPM/kitty-code/security/advisories/new
- **Email:** security@suarezpm.dev

When reporting a vulnerability, please include:

1. Description of the vulnerability and potential impact
2. Steps to reproduce or proof-of-concept
3. Affected version / commit and environment
4. Suggested mitigation if known

Do not open a public issue for security reports. We will treat every vulnerability report as confidential.

## Response Timeline

We are committed to a timely response for every vulnerability report:

- **Initial acknowledgement:** within **48 hours** of receipt.
- **Triage and assessment:** within **14 days** — we will confirm the issue, determine severity, and share a remediation plan.
- **Fix and release:** we aim to ship a patch as soon as possible depending on complexity; critical issues are prioritized within days.
- **Coordinated disclosure:** we will agree on a disclosure date with you once a fix is ready.

If you do not receive an acknowledgement within 48 hours, please follow up via the alternate channel above.

## Disclosure Policy

We follow coordinated disclosure to protect users. Once a vulnerability is confirmed:

- We will keep the report confidential until a fix is available.
- We will notify affected users via a GitHub Security Advisory and release notes.
- We ask reporters to avoid public disclosure before we have published a fix and to allow at least **14 days** for remediation, or a mutually agreed disclosure timeline.
- After the fix is released, we publish an advisory describing the vulnerability, its impact, and the patched versions.

## Release verification

Every release is built by automation — never on a maintainer laptop. Pushing
a version tag runs `.github/workflows/release.yml`, which compiles
`kitty-cli` in release mode, attaches the CycloneDX SBOM, and publishes both
as release assets. Each asset ships with:

- A **keyless cosign signature** (`.sig`) plus its **Rekor bundle**
  (`.bundle`), produced with ambient OIDC identity — there are no long-lived
  signing keys or secrets to manage.
- A **build-provenance attestation** recording which workflow built the
  asset from which source.

To verify a downloaded asset (example: the `kitty-cli` binary with its
`kitty-cli.bundle` sidecar in the same directory):

1. Verify the keyless signature and its Rekor inclusion proof:

   ```bash
   cosign verify-blob \
     --bundle kitty-cli.bundle \
     --certificate-identity-regexp '^https://github.com/SuarezPM/kitty-code/.github/workflows/release.yml@refs/tags/' \
     --certificate-oidc-issuer https://token.actions.githubusercontent.com \
     kitty-cli
   ```

2. Verify the build provenance attestation:

   ```bash
   gh attestation verify kitty-cli --repo SuarezPM/kitty-code
   ```

Both commands must succeed. If either fails, do not run the binary — please
report it through the private channels above.

## Embedded serving bind scope

El serving embebido default-abierto es loopback-only por diseño
(`engine/mistralrs-core/src/serving/api.rs:219`, `check_auth` sin token
configurado): no exponer sin auth + rate-limit + bind 127.0.0.1.

## Credit

We believe in giving credit to security researchers. With your permission, we will publicly acknowledge your contribution in the advisory, release notes, and in this repository. If you prefer to remain anonymous, we will respect that choice.

Thank you for helping keep Kitty Code and its users safe.

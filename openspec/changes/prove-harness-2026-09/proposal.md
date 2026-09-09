# Proposal: prove-harness-2026-09

## Why

Five open fronts, one central risk. The risk: measuring wrong (proxy
grading presented as official, modified bench envs, cherry-picked
subsets, best-of-k reported as mean) would destroy the honesty capital
(`AUDIT.md`/`CNAS.md`). Research (2026-09-09) pinned the exact
replication protocols, so this change measures the harness the same way
little-coder and the ornith team measured — same suites, same grading,
same reporting — without touching any bench environment:

1. **Guard mutation**: 218 background mutants from 2026-09-03 were never
   collected (`/tmp/opencode/guard_mutants.log` gone); 1 known MISSED in
   `strip_heredoc_bodies` (`+=`→`*=` loop arithmetic) needs a
   heredoc-with-line-arithmetic killing test.
2. **Unmeasured levers**: all F7/F8 levers landed in code with zero rates;
   the A/B runbook (`benches/ab/`) is tooling-only, never executed.
3. **Official benches**: only B1 measured (221/225 homemade string-match
   98.2%, NOT comparable to official rates); B2 (TB2.1), B3 (GPQA/HLE),
   B4, B5 missing; GPQA/ClawEval `blocked-harness` (no loader artifacts);
   no Harbor custom adapter → no real Consumer rate.
4. **Release + hygiene**: no `release.yml`, zero cosign/sigstore plumbing
   (the 9→9.5 security gap); bench `EpisodeStore` recall/save still blocks
   the async executor (agent path already migrated).
5. **Branding (decided by user, 2026-09-09)**: name stays `kitty-code`
   provisionally (crates.io / GitHub distribution); tagline must be
   original and is invented later — "Honey, I Shrunk…" stays internal
   positioning only.

## What Changes

- **testing/mutation-guard**: relaunch + collect guard mutants, add the
  heredoc-arithmetic killing test, ratchet the documented >70% ratio.
- **security/release-signing**: `release.yml` (tag trigger) + keyless
  cosign signing + build-provenance attestation.
- **performance/async-bench-recall**: `spawn_blocking` for the bench
  single-shot `EpisodeStore` paths, copying the agent pattern.
- **benchmarking/official-methodology**: pinned per-suite methodology
  cards (versions, params, grading, reporting) + pre-registration +
  honesty labels as hard requirements.
- **benchmarking/suite-loaders**: TB2.1 dataset pin + oracle smoke;
  GPQA/ClawEval unblock criteria (artifacts + preflight).
- **Heavy executions last**: polyglot-225 ×2 runs → TB2.1 89×5 →
  GPQA-198 → SWE-bench-500, each pre-registered, each gated.

## Capabilities

### New capabilities

- `testing/mutation-guard`
- `security/release-signing`
- `performance/async-bench-recall`
- `benchmarking/official-methodology`
- `benchmarking/suite-loaders`

## Non-goals

- Renaming the crate; inventing the tagline (deferred creative task).
- Grammar GO (reversed; separate trigger-fix work, `docs/ADR-GBNF-GO.md`).
- Modifying any bench environment (tests, Dockerfiles, oracles, datasets).
- Ever placing string-match homemade rates in the same column as
  official rates; ever reporting a subset as a full-suite number.

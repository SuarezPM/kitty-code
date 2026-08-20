# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-19

Baseline release for OpenSSF Best Practices. First tagged version; prior history was pre-release development on `main`.

### Added

- Local coding agent harness (tool-call loop, safety guards, hashline edits, quality monitor, context budget, knowledge injection, episodic memory) — `crates/kitty-harness`, `crates/kitty-runtime`, `crates/kitty-cli`, `crates/kitty-memory` (feat(harness): P0 Fundación — `171819c`; feat(harness): P1 Cierre — `875bd6a`).
- MCP server (`kitty mcp`) with Read/Edit/Bash/Glob/Grep tools (fix(harness): Fase 0 adaptación — `3048272`).
- Benchmark suites: `bench polyglot` (225 exercises) and `bench tb2` with baseline `benches/baseline.json` (F5.1-F5.3 — `34bf0f0`; bench proxy `speed_proxy.js`, `stress.js`).
- Documentation: `docs/COMPETITIVE-INTEL.md`, `docs/STRATEGY.md`, `docs/FLOOR-BENCHMARK.md`, `docs/TECHNICAL_AUDIT_REPORT.md`, `codemap.md` (docs: audit + codemap — `65b065a`; docs: intel+strategy — `0e49921`).
- Security policy `SECURITY.md` with coordinated disclosure (SC-OPENSSF-7).
- Workspace setup: `Cargo.toml` workspace (`kitty-runtime`, `kitty-harness`, `kitty-memory`, `kitty-cli`), `Cargo.lock` committed, `LICENSE` (MIT), `README.md`, `AGENTS.md`.
- Project rename `Qwen3.8-27B → Kitty-Code` (`38297cd`).
- OpenSSF compliance scaffolding: `rust-toolchain.toml` (stable + rustfmt/clippy), `CONTRIBUTING.md` (DCO), `CHANGELOG.md` (this file) — T20 / SC-OPENSSF-7.

### Fixed

- Empty-response handling — eliminate orphaned LLM round-trips (`4a642e4`).
- Bench floor tuning Fase 1 + `docs/FLOOR-BENCHMARK.md` post Gate 2 (`861a7c4`).
- Parser hardening and args-aware dedup for harness (`3048272`).
- AI slop cleaner P1 — dead code, duplicates, test reinforcement (`d5f6375`).

### Changed

- N/A for baseline — see Git history for iterative changes before 0.1.0.

### Security

- No known vulnerabilities at baseline. See `SECURITY.md` for reporting.

[Unreleased]: https://github.com/SuarezPM/kitty-code/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SuarezPM/kitty-code/releases/tag/v0.1.0

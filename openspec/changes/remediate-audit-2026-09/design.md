# Design: remediate-audit-2026-09

## Context

Audit 2026-09 scored the workspace 6.8/10. The remediation must be
incremental, each phase verifiable in isolation, and must not touch
published benchmark numbers (AUDIT.md/CNAS.md). Constraint from the
project's memory: no references to external inspiration sources in
commits/docs.

## Goals / Non-Goals

- Goals: raise every audited dimension by restructuring, not rewriting;
  keep public behavior byte-identical; make CI enforce the standards the
  repo already documents.
- Non-Goals: FTS5 migration (only if BM25 cache proves insufficient),
  cosign release signing (needs CI secrets), TUI feature work, model/bench
  number changes.

## Decisions

### D1. Bench suites → `crates/kitty-cli/src/bench/` (module, not crate)
A new crate is over-engineering: the suites share `BenchConfig`,
`BenchResult`, and helpers in main.rs. A `bench` module tree keeps the
shared types in `crate::bench::common` and re-exports one entry point per
suite. `main.rs` shrinks below 1K prod LOC. Verified by clippy+tests.

### D2. Split order: main.rs first, guard, then agent
main.rs split is mechanical (moves + use statements). guard.rs split is
table/parse separation (patterns vs shell_parse vs interpreters). agent.rs
is the riskiest (state machine interdependencies) and goes last, after the
other two establish the pattern. Each split lands as its own commit with
the full test suite green.

### D3. Central config lives in kitty-harness (`config.rs`)
Harness owns `KITTY_SANDBOX`, `KITTY_NO_THINK`, `KITTY_MODEL_TURN_CAPS`,
`KITTY_TELEMETRY`, `KITTY_TRUNCATION_DIR`, `KITTY_VERIFY_LOOP`... — the
majority. CLI-only vars (`KITTY_CLOUD_API_KEY`, bench vars) resolve via
harness config where shared, or stay in cli if cli-only. The config struct
is built once from env at startup, and accepts explicit overrides for
tests (`from_values`). Scattered `std::env::var("KITTY_")` reads move into
the module. Exception: `KITTY_SANDBOX_CHILD_*` are read by the sandboxed
child process itself and stay as documented exceptions.

### D4. `RunOverrides` passed through `run_with`
```rust
#[derive(Default, Clone)]
pub struct RunOverrides {
    pub tuning: Option<ModelTuning>,
    pub compact_threshold: Option<f32>,
    pub autorun_hops: Option<u8>,
}
```
`run(prompt)` keeps its signature and delegates to
`run_with(prompt, RunOverrides::default())`. Override application happens
at one point after the preflight, replacing the G2-FIX3 re-seed hack.
Callers migrated: CLI flag handling and bench A/B axes call `run_with`.

### D5. EpisodeStore async: wrapper functions, not trait split
The store keeps its sync API (used by sync callers/tests). The async
boundary is at the agent: `recall_episodes`/`save_episode` call
`tokio::task::spawn_blocking(move || store.recall_bm25(...))`. The store
must be `Send` — it already is (owned `Connection`). A
`std::sync::Mutex<...>`-free design is preserved (agent owns the store
exclusively, so no locking is introduced).

### D6. BM25 cache inside `EpisodeStore`
`recall_bm25` builds a cached `Bm25Index` plus the candidate token lists;
`save`/`save_with_embedding`/`update_embedding` set a dirty flag.
Cache holds `(Vec<Vec<String>>, Bm25Index)` derived from
`recall_recent(MAX_RECALL_CANDIDATES)`. Invalidation on any write. If the
criterion benchmark later shows the rebuild is not the bottleneck, the
cache stays anyway (it is trivially correct and the spec demands it).

### D7. CI: rustup + strict clippy + SBOM + audit
Replace archived toolchain action with `rustup toolchain install` steps
(keeps SHA-pinning trivial: only third-party actions need pinning, and
rustup is a shell step). Add `cargo cyclonedx` (SBOM artifact on main) and
a weekly `cargo audit` job + on `Cargo.lock` changes. Clippy gets
`--all-targets --all-features`; tests get `--all-features`.

### D8. Panic strategy ADR, keep `panic = "abort"` for now
Write `docs/ADR-PANIC-STRATEGY.md`: keep abort (binary size, single-user
local tool, no plugin boundary), document the consequence (a panic kills
the session), and record the criteria that would flip it to unwind +
`catch_unwind` at the tool boundary (e.g., if the MCP server serves
multiple concurrent sessions). Re-evaluation trigger documented.

### D9. Proptest + wiremock as dev-dependencies
`proptest` (workspace dev-dep in harness) for parser/hashline/guard
invariants. `wiremock` for runtime client contract tests. `cargo-mutants`
runs locally (documented command in tasks), not in CI (too slow for CI
budget). `cargo llvm-cov` in CI with a 70% harness gate.

## Risks / Trade-offs

- [Split may reveal hidden coupling between agent loop and response
  processing] → land splits incrementally, run full suite per commit.
- [Removing `config_mut` changes bench plumbing] → migrate all callers in
  the same commit; grep-driven verification.
- [Coverage gate may fail initially] → measure first, fix easy gaps,
  set the gate at the achieved floor ≥70% and ratchet.
- [proptest may surface real parser bugs] → that is the point; fix bugs
  found, do not weaken generators silently.

## Migration Plan

Phases 1→6 land sequentially with green suite each time. Each phase maps
to tasks.md task groups. Rollback = revert the phase commit; no data
migrations, no format changes, no persisted-state changes.

## Open Questions

- None blocking. (FTS5 vs BM25-cache is measurement-gated by design.)

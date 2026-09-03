# async-memory Specification

## Purpose
Operaciones de EpisodeStore fuera del executor async (spawn_blocking) e índice BM25 cacheado con invalidación on-write.

## Requirements

### Requirement: EpisodeStore operations do not block the async runtime

`EpisodeStore` operations (`save`, `recall_recent`, `recall_similar`,
`recall_bm25`, `count`) invoked from async contexts SHALL run on the
blocking thread pool via `tokio::task::spawn_blocking` (pattern already
established in `tools.rs::cargo_check_lint`).

#### Scenario: recall does not park the executor

- **GIVEN** an agent with an episode store containing episodes
- **WHEN** `recall_episodes` runs during `Agent::run`
- **THEN** the SQLite work executes inside `spawn_blocking` (verifiable by
  code inspection and existing async tests passing under
  `#[tokio::test(flavor = "multi_thread")]` where applicable)

### Requirement: BM25 index is cached across recalls

The BM25 ranking SHALL be computed against a cached index that is rebuilt
only when episodes are written, not on every recall call.

- Cache invalidation SHALL be on write (`save`, `save_with_embedding`,
  `update_embedding`).
- Recall results SHALL be identical (modulo timestamp tie-breaks) to the
  uncached implementation.

#### Scenario: repeated recalls reuse the index

- **GIVEN** a store with N episodes and no writes between two recalls
- **WHEN** recall_bm25 is called twice
- **THEN** the second call does not reconstruct the index (cache hit),
  and both calls return the same top-limit ordering

#### Scenario: write invalidates the cache

- **GIVEN** a cached index after a recall
- **WHEN** a new episode is saved
- **WHEN** recall_bm25 runs again
- **THEN** the new episode is eligible for the result set

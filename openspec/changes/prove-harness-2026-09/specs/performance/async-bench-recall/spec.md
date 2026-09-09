# Delta Spec: performance/async-bench-recall

## ADDED Requirements

### Requirement: Bench EpisodeStore calls do not block the executor

The single-shot bench paths SHALL perform `EpisodeStore` open/recall/
save through `spawn_blocking` using the agent's take/restore shape, so
blocking rusqlite I/O never runs on an async worker.

#### Scenario: bench recall runs off-executor

- **GIVEN** a bench run with memory enabled
- **WHEN** it recalls similar episodes or saves the result episode
- **THEN** the rusqlite call executes on the blocking pool and the
  store is returned for the next use, with fallback on pool failure

#### Scenario: no concurrency change

- **GIVEN** the sequential bench loop
- **WHEN** the migration lands
- **THEN** episode ordering and single-ownership semantics are
  unchanged; suites that persist no `Episode` today (`tb2-official`,
  `claweval`) still persist none

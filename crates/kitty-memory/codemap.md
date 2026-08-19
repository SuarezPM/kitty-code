# crates/kitty-memory/

## Responsibility
Episodic memory store: persistent record of past agent episodes in SQLite, keyed
by a BLAKE3 id/hash.

## Design
- `Episode` (episodic.rs): goal, files_touched, test_verdict, outcome, hash,
  timestamp. Id = `blake3(goal::ts)`; content hash = `blake3(goal::outcome)`.
- `EpisodeStore` (episodic.rs): `rusqlite` (bundled) wrapper. `open` creates the
  `episodes` table + timestamp index. `save` = INSERT OR REPLACE.
- Retrieval: `recall_recent` (timestamp desc) and `recall_similar`, which ranks
  candidates by **lexical Jaccard token-overlap** (a real IR baseline, not the old
  substring `LIKE`). Falls back to recent when no token overlap. Dense embedding +
  RRF retrieval is a planned upgrade (blocked by rten/Gemma int8 incompatibility).

## Flow
Agent loop → `store.recall_similar(goal,3)` on run start → `store.save(ep)` on
task completion (best-effort `.ok()`).

## Integration
- Consumed by: `kitty-harness` (recall/save), `kitty-cli` (bench harness).
- Depends on: `rusqlite`, `blake3`, `serde_json`, `chrono`-free (uses `std::time`).

## Honest status
Persistence works; *semantic* recall does not. The `test_verdict` field exists but
is populated as `Some("pass"/"fail")` regardless of actual correctness (the loop
passes `true` unconditionally).

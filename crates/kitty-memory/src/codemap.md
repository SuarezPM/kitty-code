# crates/kitty-memory/src/

## Responsibility

One module per memory concern. `episodic.rs` owns the SQLite store plus all
retrieval math (lexical, dense, fused); `embedding.rs` owns the optional
onnx encoder; `error.rs` owns the error type; `lib.rs` only re-exports.
Nothing here knows about agents, prompts, or backends — episodes in,
ranked episodes out.

## Design

- `episodic.rs`: `Episode` (blake3 `compute_id`/`compute_hash`, `chrono_now`
  stamping) + `EpisodeStore` (`rusqlite` + WAL + 5s `busy_timeout`,
  `episodes` table with `embedding BLOB`, idempotent migration) +
  `Bm25Cache` (`RefCell`, invalidated on write) + tokenize/similarity +
  `Bm25Index` (k1 1.2, b 0.75) + cosine + `RRF_K = 60.0` fusion +
  `recall_recent` / `recall_similar` / `recall_bm25` / `recall_by_embedding` /
  `recall_hybrid` (`MAX_RECALL = 50` window, single-ranking fast-path,
  most-recent fallback).
- `embedding.rs`: `MODEL_ID` (Qdrant MiniLM-L6-v2-onnx), `MODEL_DIM` 384,
  `~/.kitty/models` cache, `EmbeddingStore` (`embedding` feature via
  `fastembed`/`ort`; stub without it), episode embed/save/recall helpers.
- `error.rs`: `MemoryError::{Sqlite, Json, Io, NotFound, Embedding}`.
- `lib.rs`: re-exports only.

## Flow

Save (`Episode::new` → `INSERT OR REPLACE`, optionally with embedding
bytes) → recall (recent-50 window → Jaccard/BM25/cosine/hybrid ranking →
empty-score fallback to most recent). Cache invalidation is write-driven;
reads never mutate.

## Integration

- Consumed by: `kitty-harness` (recall at run start, save at completion;
  auto-disabled for small-model bench runs).
- Depends on: `rusqlite` (bundled), `blake3`, `serde`/`serde_json`,
  `bm25`, `thiserror`; `fastembed`/`ort` behind `embedding`.

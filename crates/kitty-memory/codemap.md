# crates/kitty-memory/

## Responsibility
Episodic memory crate: persistent record of agent episodes in SQLite, keyed by
blake3 id/hash. Lexical (BM25) + dense (embedding cosine) retrieval fused via RRF.

## Design
- `Episode` (episodic.rs): `id`, `goal`, `files_touched`, `test_verdict`, `outcome`,
  `hash`, `timestamp`. `compute_id(goal, timestamp)` = `blake3(goal::ts)`;
  `compute_hash(goal, outcome)` = `blake3(goal::outcome)`; `new()` stamps time via `chrono_now()`.
- `EpisodeStore` (episodic.rs): `rusqlite::Connection` wrapper + `Bm25Cache`
  (`RefCell<Option<...>>`, invalidated on `save`/`save_with_embedding`/`update_embedding`).
  `open(path)` sets WAL + `busy_timeout 5000`, creates `episodes` table
  (incl. `embedding BLOB`) + `idx_episodes_ts`; idempotent `ALTER TABLE ... ADD COLUMN embedding BLOB`.
- Tokenization/ranking (episodic.rs): `tokenize` (lowercase alphanum runs);
  `similarity` Jaccard baseline; `KittyTokenizer` adapts `tokenize` to `bm25::Tokenizer`;
  `Bm25Index` (`k1=1.2, b=0.75`, `doc_freq`, `doc_term_freqs`, `avgdl`, `from_corpus`,
  `score`/`scores` with dedup query terms, `idf`); `MAX_RECALL = 50` (`MAX_RECALL_CANDIDATES`).
- Dense + fusion (episodic.rs): `encode_embedding`/`decode_embedding` (LE `f32` bytes),
  `cosine_similarity`, `RRF_K = 60.0`, `reciprocal_rank_fusion` / `reciprocal_rank_fusion_k60`.
  `save_with_embedding`, `update_embedding` (→ `MemoryError::NotFound`), `get_embedding`,
  `recall_by_embedding` (cosine desc + timestamp tie-break), `recall_hybrid(goal, Option<&[f32]>, limit)`
  (BM25 ranking + dense ranking → RRF; single-ranking fast-path; recent fallback); `count`.
- `embedding.rs`: `MODEL_ID = Qdrant/all-MiniLM-L6-v2-onnx`, `MODEL_DIM = 384`,
  `MODEL_FILE = model.onnx`, `default_cache_dir()` (`~/.kitty/models`), `normalize_embedding`.
  `EmbeddingStore` (feature `embedding` = `fastembed`/`ort`): `try_new` / `try_new_with_cache`
  (`InitOptions::new(AllMiniLML6V2)`), `embed` / `embed_one` / `embed_goal_outcome`,
  `cosine`, `save_episode_with_embedding`, `update_episode_embedding`, `recall`,
  `encode` / `decode`. Without feature: stub `EmbeddingStore` with `dim()` + `cosine`.
- `error.rs`: `MemoryError::{Sqlite, Json, Io, NotFound, Embedding}` + `Result<T>`.
- `lib.rs`: re-exports `Episode`, `EpisodeStore`, `Bm25Index`, `KittyTokenizer`,
  `MAX_RECALL`, `RRF_K`, `reciprocal_rank_fusion*`, `cosine_similarity`,
  `encode/decode_embedding`, `EmbeddingStore`, `MODEL_DIM/ID/FILE`, `MemoryError`.

## Flow
- Save: `Episode::new(goal, files_touched, test_verdict, outcome)` → `store.save(&ep)`
  (`INSERT OR REPLACE`) or `EmbeddingStore::save_episode_with_embedding` → `store.save_with_embedding`.
- Recall: run start pulls `recall_recent(MAX_RECALL)` window (50) → rank:
  `recall_similar` (Jaccard) / `recall_bm25` (cached `Bm25Cache`) /
  `recall_by_embedding` (cosine over `embedding BLOB`) / `recall_hybrid` (RRF K=60);
  empty-score fallback returns most recent.

## Integration
- Consumed by: `kitty-harness` agent loop (recall on run start, save on task completion).
- Depends on: `rusqlite` (bundled), `blake3`, `serde`/`serde_json`, `bm25`,
  `thiserror`; `fastembed` opt-in via `embedding` feature (model cached, not vendored).

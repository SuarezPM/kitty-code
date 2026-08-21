# crates/kitty-memory/

## Responsibility
Episodic memory store: persistent record of past agent episodes in SQLite, keyed
by a BLAKE3 id/hash. Retrieval is lexical (BM25) + dense (embedding) fused via RRF.

## Design
- `Episode` (episodic.rs: ~357L committed, ~790L with F6 tracks): goal, files_touched, test_verdict, outcome, hash,
  timestamp. Id = `blake3(goal::ts)`; content hash = `blake3(goal::outcome)`.
- `EpisodeStore` (episodic.rs): `rusqlite` (bundled) wrapper. `open` creates the
  `episodes` table + timestamp index + `embedding BLOB` column (idempotent `ALTER TABLE ... embedding BLOB` on `duplicate column` ignore). `save` = INSERT OR REPLACE; `save_with_embedding`, `update_embedding`, `get_embedding`, `ensure_embedding_column`.
- Retrieval:
  - `recall_recent(limit)` — timestamp desc, MRU fallback.
  - `recall_similar(goal, limit)` — **lexical Jaccard token-overlap** (real IR baseline, not old `LIKE`) — fallback to recent when no overlap.
  - `recall_bm25(goal, limit)` (F6 task 7, window `MAX_RECALL=50` alias `MAX_RECALL_CANDIDATES`): **BM25** `k1=1.2, b=0.75` with `KittyTokenizer` (lowercase alphanum, delegates to `tokenize`), IDF `ln(1+(N-df+0.5)/(df+0.5))` + TF `tf*(k1+1)/(tf+k1*(1-b+b*|D|/avgdl))`, `Bm25Index` `doc_freq` + `doc_term_freqs` + `avgdl`, scores dedup query terms. Superior to Jaccard on TF fixture (`bm25_supera_jaccard` tie 0.5 vs BM25 boost).
  - `recall_by_embedding(query_emb, limit)` (F6 task 8, `embedding` feature): cosine `dot/(norm_a*norm_b)` over `embedding BLOB` (1536B LE `f32*384`), filter `is_multiple_of(4)` + `chunks_exact`, sort by cosine desc + timestamp tie-break, fallback to recent.
  - `recall_hybrid(goal, Option<&[f32]>, limit)` (F6 task 9): **RRF K=60** (`reciprocal_rank_fusion(rankings,k)` `1/(k+rank)` Cormack, `RRF_K=60.0`, `reciprocal_rank_fusion_k60`) fusion of BM25 ranking + dense ranking → fused `HashMap<id, f32>` sort desc + timestamp; single-ranking fast-path (BM25-only) still beats Jaccard; `benches/ab_recall.py` strict 83.3% vs Jaccard 25% (synthetic 12 episodes, 12 queries, tie idéntico token set).
  - Encoding helpers: `encode_embedding` / `decode_embedding` LE bytes, `cosine_similarity`, `normalize_embedding`.
- `EmbeddingStore` (embedding.rs: 201L, F6 task 8): `fastembed 4` pure Rust ONNX via `ort` (download-binaries), `MODEL_ID Qdrant/all-MiniLM-L6-v2-onnx`, `MODEL_DIM 384`, `default_cache_dir ~/.kitty/models` (`~/.kitty/models/models--Qdrant--.../blobs + snapshots`, 87M `model.onnx`), `try_new` (`InitOptions::new(AllMiniLML6V2).with_cache_dir(...).with_show_download_progress(true)`), `embed`/`embed_one`/`embed_goal_outcome` (goal+outcome), `cosine`/`normalize_embedding`, `save_episode_with_embedding`/`update_episode_embedding`/`recall`. Feature `embedding = ["dep:fastembed"]` opt-in (pure Rust `ort`), `--no-memory` tiny still blocked (no penalty). NOTE `rten` int8 Gemma blocked (documented).
- `lib.rs` re-exports `Episode`, `EpisodeStore`, `Bm25Index`, `KittyTokenizer`, `MAX_RECALL`, `RRF_K`, `reciprocal_rank_fusion*`, `MODEL_ID/DIM` when features wired.

## Flow
Agent loop → `store.recall_similar` / `recall_bm25` / `recall_hybrid(Some(qemb))` on run start (with `store.recall_recent` window 50 candidates) → `store.save(ep)` + `save_with_embedding` on task completion (best-effort `.ok()`). Bench `run_one_bench` hybrid path via `--semantic-memory`.

## Integration
- Consumed by: `kitty-harness` (`recall_episodes` hybrid guard `semantic_memory && !is_tiny` + `#[cfg(feature="embedding")]` `EmbeddingStore::try_new` fallback to BM25-only), `kitty-cli` (bench harness past episodes `if semantic_memory { recall_hybrid } else recall_similar`).
- Depends on: `rusqlite`, `blake3`, `serde_json`, `chrono`-free (`std::time`), `bm25` (2.3, `default-features=false` + `KittyTokenizer`), `fastembed` (4, optional `embedding` feature via `ort`).

## Honest status
- Persistence works; lexical BM25 is real IR baseline (TF+IDF+length norm) superior to Jaccard set-only; dense + RRF K=60 hybrid strictly beats both.
- `test_verdict` field exists and `save_episode` records real `true/false` (exhaust → fail, not unconditional pass).
- `embedding` is opt-in feature — not bundled model in repo (`.gitignore` `*.onnx` + `.kitty/` + `models/`, cache `~/.kitty/models` 88M, `Cargo.toml` `fastembed` optional).

## Tests
10 lib tests on committed episodic.rs (Jaccard) + 8 BM25 + 7 embedding (cosine + migration + recall) = **18** on `cargo test -p kitty-memory` (without feature) → **25** with `--features embedding` (BM25 18 + embedding 7). `cargo clippy -p kitty-memory --features embedding -- -D warnings` clean. Full F6 hybrid adds 2 more RRF tests when `recall_hybrid` wired.

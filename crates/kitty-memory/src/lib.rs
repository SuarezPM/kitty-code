//! Kitty Memory — Episodic memory (blake3 + SQLite).
//!
//! Stores and retrieves session episodes for cross-session recall.

pub mod embedding;
pub mod episodic;
pub mod error;

pub use embedding::{
    default_cache_dir, normalize_embedding, EmbeddingStore, MODEL_DIM, MODEL_FILE, MODEL_ID,
};
pub use episodic::{
    cosine_similarity, decode_embedding, encode_embedding, reciprocal_rank_fusion,
    reciprocal_rank_fusion_k60, Bm25Index, Episode, EpisodeStore, KittyTokenizer, MAX_RECALL,
    RRF_K,
};
pub use error::MemoryError;

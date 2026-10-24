//! Kitty Memory — Episodic memory (blake3 + SQLite).
//!
//! Stores and retrieves session episodes for cross-session recall.

pub mod embedding;
pub mod episodic;
pub mod error;
pub mod lessons;

pub use embedding::{
    default_cache_dir, normalize_embedding, EmbeddingStore, MODEL_DIM, MODEL_FILE, MODEL_ID,
};
pub use episodic::{
    cosine_similarity, decode_embedding, encode_embedding, fuse_rankings_weighted,
    hybrid_recall_enabled, ndcg_at_k, phrase_boost, recall_at_k, recency_score, reciprocal_rank,
    reciprocal_rank_fusion, reciprocal_rank_fusion_k60, three_factor_score, Bm25Index,
    ChannelRankings, Episode, EpisodeStore, FuseChannel, KittyTokenizer, RecallMode, RrfParams,
    ScoringWeights, ENV_HYBRID_RECALL, MAX_RECALL, RECENCY_HALF_LIFE_SECS, RRF_K,
};
pub use error::MemoryError;
pub use lessons::{
    LessonError, LessonStore, MemoryItem, RunProvenance, LESSON_DEFAULT_TTL_SECS,
    MAX_LESSON_RECALL,
};

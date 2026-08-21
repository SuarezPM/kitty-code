use std::path::PathBuf;

#[cfg(not(feature = "embedding"))]
use crate::episodic::cosine_similarity;
#[cfg(feature = "embedding")]
use crate::episodic::{cosine_similarity, encode_embedding, Episode, EpisodeStore};
#[cfg(feature = "embedding")]
use crate::error::{MemoryError, Result};

/// NOTE: rten bloqueado — int8 Gemma / ONNX runtime rten no disponible para AllMiniLML6V2
/// en kitty-memory. fastembed 4 con ort (pure Rust ONNX) es el backend elegido:
/// descarga modelo ~80MB a ~/.kitty/models/ vía hf-hub, cosine en Rust, sin bundled
/// modelo en repo (gitignore *.onnx). Tiny mantiene --no-memory, embedding es opt-in
/// feature `embedding` y no bloquea tiny.
pub const MODEL_ID: &str = "Qdrant/all-MiniLM-L6-v2-onnx";
pub const MODEL_DIM: usize = 384;
pub const MODEL_FILE: &str = "model.onnx";

pub fn default_cache_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".kitty/models")
}

pub fn normalize_embedding(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(feature = "embedding")]
pub struct EmbeddingStore {
    model: fastembed::TextEmbedding,
    dim: usize,
}

#[cfg(feature = "embedding")]
impl EmbeddingStore {
    pub fn try_new() -> Result<Self> {
        Self::try_new_with_cache(default_cache_dir())
    }

    pub fn try_new_with_cache(cache_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&cache_dir).map_err(|e| MemoryError::Embedding(e.to_string()))?;
        let opts = fastembed::InitOptions::new(fastembed::EmbeddingModel::AllMiniLML6V2)
            .with_cache_dir(cache_dir)
            .with_show_download_progress(true);
        let model = fastembed::TextEmbedding::try_new(opts)
            .map_err(|e| MemoryError::Embedding(e.to_string()))?;
        Ok(Self {
            model,
            dim: MODEL_DIM,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        self.model
            .embed(texts, None)
            .map_err(|e| MemoryError::Embedding(e.to_string()))
    }

    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let v = self.embed(vec![text.to_string()])?;
        v.into_iter()
            .next()
            .ok_or_else(|| MemoryError::Embedding("empty embedding".into()))
    }

    pub fn embed_goal_outcome(&self, goal: &str, outcome: &str) -> Result<Vec<f32>> {
        let combined = format!("{goal} {outcome}");
        self.embed_one(&combined)
    }

    pub fn cosine(&self, a: &[f32], b: &[f32]) -> f32 {
        cosine_similarity(a, b)
    }

    pub fn save_episode_with_embedding(
        &self,
        store: &EpisodeStore,
        episode: &Episode,
    ) -> Result<()> {
        let emb = self.embed_goal_outcome(&episode.goal, &episode.outcome)?;
        store.save_with_embedding(episode, &emb)
    }

    pub fn update_episode_embedding(
        &self,
        store: &EpisodeStore,
        id: &str,
        goal: &str,
        outcome: &str,
    ) -> Result<()> {
        let emb = self.embed_goal_outcome(goal, outcome)?;
        store.update_embedding(id, &emb)
    }

    pub fn recall(&self, store: &EpisodeStore, query: &str, limit: usize) -> Result<Vec<Episode>> {
        let q = self.embed_one(query)?;
        store.recall_by_embedding(&q, limit)
    }

    pub fn encode(emb: &[f32]) -> Vec<u8> {
        encode_embedding(emb)
    }

    pub fn decode(bytes: &[u8]) -> Vec<f32> {
        crate::episodic::decode_embedding(bytes)
    }
}

#[cfg(not(feature = "embedding"))]
pub struct EmbeddingStore;

#[cfg(not(feature = "embedding"))]
impl EmbeddingStore {
    pub fn dim() -> usize {
        MODEL_DIM
    }
    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        cosine_similarity(a, b)
    }
}

#[cfg(feature = "embedding")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::episodic::{decode_embedding, encode_embedding, Episode, EpisodeStore};

    #[test]
    fn cosine_identical_is_one() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let s = cosine_similarity(&a, &b);
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_opposite_is_minus_one() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let v = vec![0.1_f32, -0.5, 1.41, 0.0];
        let blob = encode_embedding(&v);
        assert_eq!(blob.len(), v.len() * 4);
        let out = decode_embedding(&blob);
        assert_eq!(out.len(), v.len());
        for (a, b) in v.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn embedding_blob_migration_and_recall() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = EpisodeStore::open(tmp.path().to_str().unwrap()).unwrap();
        store.ensure_embedding_column().unwrap();
        let ep = Episode::new("hello world".into(), vec!["a.rs".into()], None, "ok".into());
        let emb = vec![0.1_f32; MODEL_DIM];
        store.save_with_embedding(&ep, &emb).unwrap();
        let fetched = store.get_embedding(&ep.id).unwrap().unwrap();
        assert_eq!(fetched.len(), MODEL_DIM);
        for (a, b) in emb.iter().zip(fetched.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        let q = vec![0.1_f32; MODEL_DIM];
        let res = store.recall_by_embedding(&q, 5).unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, ep.id);
    }

    #[test]
    fn cache_dir_default_is_kitty_models() {
        let dir = default_cache_dir();
        assert!(dir.to_string_lossy().ends_with(".kitty/models"));
    }

    #[test]
    fn model_constants() {
        assert_eq!(MODEL_DIM, 384);
        assert!(MODEL_ID.contains("MiniLM"));
    }
}

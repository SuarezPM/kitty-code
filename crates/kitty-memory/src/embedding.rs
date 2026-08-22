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

    /// Deterministic PRNG (splitmix64) so the synthetic 384-dim embeddings are
    /// reproducible without downloading the fastembed model (network-free test).
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn next_f32(&mut self) -> f32 {
            ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    fn random_unit_vec(dim: usize, rng: &mut SplitMix64) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
        normalize_embedding(&mut v);
        v
    }

    #[test]
    fn embedding_blob_migration_and_recall() {
        // e2e against a REAL sqlite file on disk with production semantics
        // (WAL + busy_timeout, set by EpisodeStore::open). NEVER touches the
        // live ~/.kitty/episodes.db — tempdir only.
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("episodes.db");
        let dim = MODEL_DIM;
        let mut rng = SplitMix64::new(42);

        // --- session 1: create schema, insert episodes WITH embedding BLOBs ---
        let store = EpisodeStore::open(db_path.to_str().unwrap()).unwrap();
        // production pragmas actually applied to the file
        {
            let probe = rusqlite::Connection::open(&db_path).unwrap();
            let mode: String = probe
                .query_row("PRAGMA journal_mode", [], |r| r.get(0))
                .unwrap();
            assert_eq!(mode.to_lowercase(), "wal", "DB must be in WAL mode");
        }

        // Target inserted FIRST (oldest) so recency cannot explain a top-1 hit.
        let target = Episode::new(
            "fix flaky websocket reconnect loop in gateway service".into(),
            vec!["gateway/ws.rs".into()],
            Some("passed".into()),
            "reconnect backoff fixed".into(),
        );
        let target_emb = random_unit_vec(dim, &mut rng);
        store.save_with_embedding(&target, &target_emb).unwrap();

        let distractor_goals = [
            "deploy kubernetes cluster to production",
            "python fibonacci memoization benchmark",
            "rust borrow checker fight in parser module",
            "css flexbox centering workaround safari",
            "bash backup rotation cron script",
        ];
        for goal in distractor_goals {
            let ep = Episode::new(goal.into(), vec!["x".into()], None, "done".into());
            let emb = random_unit_vec(dim, &mut rng);
            store.save_with_embedding(&ep, &emb).unwrap();
        }
        assert_eq!(store.count().unwrap(), distractor_goals.len() + 1);
        drop(store); // close: exercises clean WAL shutdown

        // --- session 2: close/reopen = migration path over existing table ---
        let store = EpisodeStore::open(db_path.to_str().unwrap()).unwrap();
        // duplicate-column tolerant migration no-op on the existing schema
        store.ensure_embedding_column().unwrap();
        assert_eq!(store.count().unwrap(), distractor_goals.len() + 1);

        // blake3 content hash survives reopen untouched (id/hash path; note:
        // episodic.rs has NO incremental blake3 delta path for embeddings —
        // none exists to cover).
        let reopened: Vec<Episode> = store.recall_recent(distractor_goals.len() + 1).unwrap();
        let t = reopened.iter().find(|e| e.id == target.id).unwrap();
        assert_eq!(t.hash, target.hash);
        assert_eq!(
            t.hash,
            Episode::compute_hash(&target.goal, &target.outcome)
        );

        // BLOB roundtrip after reopen
        let emb_back = store.get_embedding(&target.id).unwrap().unwrap();
        assert_eq!(emb_back.len(), dim);
        for (a, b) in target_emb.iter().zip(emb_back.iter()) {
            assert!((a - b).abs() < 1e-6);
        }

        // --- hybrid RRF recall with a paraphrased query ---
        // Query shares only rare tokens ("websocket", "gateway") with the
        // target and is semantically near it via the noisy embedding; hybrid
        // is NOT asserted to beat lexical in general — losing honestly is
        // acceptable. Here we only assert top-1 correctness on this fixture.
        let query_text = "websocket gateway drops connection intermittently";
        let mut query_emb = target_emb.clone();
        for x in query_emb.iter_mut() {
            *x += 0.05 * rng.next_f32();
        }
        normalize_embedding(&mut query_emb);

        let results = store
            .recall_hybrid(query_text, Some(&query_emb), 3)
            .unwrap();
        assert!(!results.is_empty());
        assert_eq!(
            results[0].id, target.id,
            "hybrid RRF top-1 must be the paraphrased target episode"
        );
        assert!(results[0].goal.contains("websocket"));
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

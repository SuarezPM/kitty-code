use std::collections::{HashMap, HashSet};

use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// An episode stored in memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: String,
    pub goal: String,
    pub files_touched: Vec<String>,
    pub test_verdict: Option<String>,
    pub outcome: String,
    pub hash: String,
    pub timestamp: i64,
}

impl Episode {
    /// Compute blake3 ID from goal + timestamp.
    pub fn compute_id(goal: &str, timestamp: i64) -> String {
        let input = format!("{}::{}", goal, timestamp);
        let hash = blake3::hash(input.as_bytes());
        hash.to_hex().to_string()
    }

    /// Compute blake3 content hash of goal + outcome.
    pub fn compute_hash(goal: &str, outcome: &str) -> String {
        let input = format!("{}::{}", goal, outcome);
        let hash = blake3::hash(input.as_bytes());
        hash.to_hex().to_string()
    }

    /// Create a new episode with computed id and hash.
    pub fn new(
        goal: String,
        files_touched: Vec<String>,
        test_verdict: Option<String>,
        outcome: String,
    ) -> Self {
        let timestamp = chrono_now();
        let id = Self::compute_id(&goal, timestamp);
        let hash = Self::compute_hash(&goal, &outcome);
        Self {
            id,
            goal,
            files_touched,
            test_verdict,
            outcome,
            hash,
            timestamp,
        }
    }
}

/// Simple monotonic timestamp (epoch seconds).
fn chrono_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Candidate window pulled for lexical ranking before returning top-k.
pub const MAX_RECALL: usize = 50;
const MAX_RECALL_CANDIDATES: usize = MAX_RECALL;

/// Split text into lowercase word tokens (alphanumeric runs).
fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

/// Jaccard similarity between two token sets: |A ∩ B| / |A ∪ B|.
/// Returns 0.0 when there is no overlap.
fn similarity(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let set_a: HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
    let set_b: HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
    let intersection = set_a.intersection(&set_b).count();
    if intersection == 0 {
        return 0.0;
    }
    let union = set_a.union(&set_b).count();
    intersection as f64 / union as f64
}

// ---------------------------------------------------------------------------
// BM25 lexical ranking
// ---------------------------------------------------------------------------

/// Tokenizer adapter that reuses the project's `tokenize` (lowercase alphanum)
/// so the external `bm25` crate and our `Bm25Index` share identical tokenization.
#[derive(Debug, Default, Clone)]
pub struct KittyTokenizer;

impl bm25::Tokenizer for KittyTokenizer {
    fn tokenize(&self, input_text: &str) -> Vec<String> {
        tokenize(input_text)
    }
}

/// BM25 index with IDF + TF and length normalization.
/// k1=1.2, b=0.75 are standard defaults (Robertson / Zaragoza).
#[derive(Debug, Clone)]
pub struct Bm25Index {
    k1: f32,
    b: f32,
    avgdl: f32,
    n: usize,
    doc_freq: HashMap<String, usize>,
    doc_lens: Vec<usize>,
    doc_term_freqs: Vec<HashMap<String, usize>>,
}

impl Bm25Index {
    /// Build index from pre-tokenized documents.
    pub fn new(docs: &[Vec<String>]) -> Self {
        let n = docs.len();
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let mut doc_lens = Vec::with_capacity(n);
        let mut doc_term_freqs = Vec::with_capacity(n);
        let mut total_len: usize = 0;

        for tokens in docs {
            let len = tokens.len();
            doc_lens.push(len);
            total_len = total_len.saturating_add(len);
            let mut tf: HashMap<String, usize> = HashMap::new();
            let mut seen: HashSet<String> = HashSet::new();
            for tok in tokens {
                let c = tf.entry(tok.clone()).or_insert(0);
                *c += 1;
                if seen.insert(tok.clone()) {
                    let e = doc_freq.entry(tok.clone()).or_insert(0);
                    *e += 1;
                }
            }
            doc_term_freqs.push(tf);
        }

        let avgdl = if n == 0 {
            0.0
        } else {
            total_len as f32 / n as f32
        };

        Self {
            k1: 1.2,
            b: 0.75,
            avgdl,
            n,
            doc_freq,
            doc_lens,
            doc_term_freqs,
        }
    }

    /// Convenience: build from raw strings using shared `tokenize`.
    pub fn from_corpus(corpus: &[String]) -> Self {
        let docs: Vec<Vec<String>> = corpus.iter().map(|s| tokenize(s)).collect();
        Self::new(&docs)
    }

    fn idf(&self, term: &str) -> f32 {
        let df = self.doc_freq.get(term).copied().unwrap_or(0) as f32;
        let n = self.n as f32;
        // Lucene / BM25 idf: ln(1 + (N - df + 0.5)/(df + 0.5))
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
    }

    /// BM25 score for a single document.
    pub fn score(&self, query: &[String], doc_idx: usize) -> f32 {
        if doc_idx >= self.doc_lens.len() || query.is_empty() {
            return 0.0;
        }
        let doc_len = self.doc_lens[doc_idx] as f32;
        let avgdl = if self.avgdl <= 0.0 { 1.0 } else { self.avgdl };
        let tf_map = match self.doc_term_freqs.get(doc_idx) {
            Some(m) => m,
            None => return 0.0,
        };
        let mut total = 0.0_f32;
        // dedup query terms so repeated query tokens don't double-count unintentionally
        let mut seen_q: HashSet<&String> = HashSet::new();
        for term in query {
            if !seen_q.insert(term) {
                continue;
            }
            let tf = tf_map.get(term).copied().unwrap_or(0) as f32;
            if tf == 0.0 {
                continue;
            }
            let idf = self.idf(term);
            let numerator = tf * (self.k1 + 1.0);
            let denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avgdl));
            total += idf * (numerator / denominator);
        }
        total
    }

    /// Score all documents; returns (doc_idx, score) sorted desc where score > 0.
    pub fn scores(&self, query: &[String]) -> Vec<(usize, f32)> {
        let mut out: Vec<(usize, f32)> = (0..self.n)
            .map(|i| (i, self.score(query, i)))
            .filter(|(_, s)| *s > 0.0)
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Expose avgdl for diagnostics.
    pub fn avgdl(&self) -> f32 {
        self.avgdl
    }

    /// Expose doc frequency.
    pub fn doc_freq(&self, term: &str) -> usize {
        self.doc_freq.get(term).copied().unwrap_or(0)
    }
}

/// Cached BM25 index over the recall candidates (Fase 3.2: rebuilt only
/// when a write invalidates it, never on every recall).
struct Bm25Cache {
    index: Bm25Index,
}

/// Store for episodic memory.
pub struct EpisodeStore {
    conn: Connection,
    /// Interior mutability is safe here: `EpisodeStore` is `Send` (owned
    /// `Connection`) and used from a single owner at a time; `!Sync` is
    /// inherited from `Connection` and preserved by `RefCell`.
    bm25_cache: std::cell::RefCell<Option<Bm25Cache>>,
}

impl EpisodeStore {
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        // Production DB semantics (Catalyst apohara-episodic/store.rs:68):
        // WAL so readers don't block the single writer; busy_timeout 5s so a
        // concurrent cross-session writer blocks instead of erroring.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "busy_timeout", 5000)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                files_touched TEXT NOT NULL,
                test_verdict TEXT,
                outcome TEXT NOT NULL,
                hash TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                embedding BLOB
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(timestamp DESC);",
        )?;
        let _ = conn.execute("ALTER TABLE episodes ADD COLUMN embedding BLOB", []);
        Ok(Self {
            conn,
            bm25_cache: std::cell::RefCell::new(None),
        })
    }

    /// Save an episode (insert or replace).
    pub fn save(&self, episode: &Episode) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO episodes (id, goal, files_touched, test_verdict, outcome, hash, timestamp)
              VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                episode.id,
                episode.goal,
                serde_json::to_string(&episode.files_touched)?,
                episode.test_verdict,
                episode.outcome,
                episode.hash,
                episode.timestamp,
            ],
        )?;
        self.bm25_cache.borrow_mut().take();
        Ok(())
    }

    /// Recall the N most recent episodes.
    pub fn recall_recent(&self, limit: usize) -> Result<Vec<Episode>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, goal, files_touched, test_verdict, outcome, hash, timestamp
              FROM episodes
              ORDER BY timestamp DESC
              LIMIT ?1",
        )?;
        let episodes = stmt
            .query_map(rusqlite::params![limit as i64], |row| {
                Ok(Episode {
                    id: row.get(0)?,
                    goal: row.get(1)?,
                    files_touched: serde_json::from_str(&row.get::<_, String>(2)?)
                        .unwrap_or_default(),
                    test_verdict: row.get(3)?,
                    outcome: row.get(4)?,
                    hash: row.get(5)?,
                    timestamp: row.get(6)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(episodes)
    }

    /// Recall the `limit` episodes most similar to `goal` (Jaccard baseline).
    pub fn recall_similar(&self, goal: &str, limit: usize) -> Result<Vec<Episode>> {
        let candidates = self.recall_recent(MAX_RECALL_CANDIDATES)?;
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let query_tokens = tokenize(goal);
        let mut scored: Vec<(f64, &Episode)> = candidates
            .iter()
            .map(|ep| (similarity(&query_tokens, &tokenize(&ep.goal)), ep))
            .filter(|(score, _)| *score > 0.0)
            .collect();

        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.1.timestamp.cmp(&a.1.timestamp))
        });

        let top: Vec<Episode> = scored
            .into_iter()
            .take(limit)
            .map(|(_, ep)| ep.clone())
            .collect();

        if top.is_empty() {
            return Ok(candidates.into_iter().take(limit).collect());
        }
        Ok(top)
    }

    /// Recall with BM25 ranking (IDF+TF, length-normalized).
    /// Fase 3.2: the index is cached and rebuilt only when a write
    /// invalidates it, so repeated recalls between writes stop paying the
    /// O(candidates) index construction on every turn.
    pub fn recall_bm25(&self, goal: &str, limit: usize) -> Result<Vec<Episode>> {
        let candidates = self.recall_recent(MAX_RECALL_CANDIDATES)?;
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let query_tokens = tokenize(goal);
        if query_tokens.is_empty() {
            return Ok(candidates.into_iter().take(limit).collect());
        }
        if self.bm25_cache.borrow().is_none() {
            let doc_tokens: Vec<Vec<String>> =
                candidates.iter().map(|ep| tokenize(&ep.goal)).collect();
            *self.bm25_cache.borrow_mut() = Some(Bm25Cache {
                index: Bm25Index::new(&doc_tokens),
            });
        }
        let cache = self.bm25_cache.borrow();
        let index = &cache.as_ref().expect("index rebuilt above").index;
        let mut scored: Vec<(f32, &Episode)> = candidates
            .iter()
            .enumerate()
            .map(|(idx, ep)| (index.score(&query_tokens, idx), ep))
            .filter(|(s, _)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.1.timestamp.cmp(&a.1.timestamp))
        });
        let top: Vec<Episode> = scored
            .into_iter()
            .take(limit)
            .map(|(_, ep)| ep.clone())
            .collect();
        if top.is_empty() {
            return Ok(candidates.into_iter().take(limit).collect());
        }
        Ok(top)
    }

    /// Count total episodes.
    pub fn count(&self) -> Result<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM episodes", [], |row| row.get(0))?;
        Ok(count as usize)
    }
}

pub const RRF_K: f32 = 60.0;
#[cfg(test)]
mod bm25_cache_tests {
    use super::*;

    fn ep(goal: &str, ts: i64) -> Episode {
        Episode {
            id: format!("id-{ts}-{goal}"),
            goal: goal.to_string(),
            files_touched: vec![],
            test_verdict: None,
            outcome: "pass".into(),
            hash: "h".into(),
            timestamp: ts,
        }
    }

    #[test]
    fn bm25_recall_is_stable_across_cached_calls() {
        let dir = tempfile::tempdir().unwrap();
        let store = EpisodeStore::open(dir.path().join("t.db").to_str().unwrap()).unwrap();
        store.save(&ep("write fibonacci python", 1)).unwrap();
        store.save(&ep("parse json rust", 2)).unwrap();
        let a = store.recall_bm25("fibonacci", 2).unwrap();
        let b = store.recall_bm25("fibonacci", 2).unwrap();
        assert_eq!(a.len(), b.len());
        assert_eq!(a[0].goal, b[0].goal);
    }

    #[test]
    fn bm25_write_invalidates_cache_so_new_episode_is_eligible() {
        let dir = tempfile::tempdir().unwrap();
        let store = EpisodeStore::open(dir.path().join("t.db").to_str().unwrap()).unwrap();
        store.save(&ep("unrelated first", 1)).unwrap();
        let before = store.recall_bm25("quicksort", 5).unwrap();
        assert!(before.is_empty() || !before.iter().any(|e| e.goal == "write quicksort"));
        store.save(&ep("write quicksort", 2)).unwrap();
        let after = store.recall_bm25("quicksort", 5).unwrap();
        assert!(after.iter().any(|e| e.goal == "write quicksort"));
    }
}

/// Reciprocal Rank Fusion (Cormack et al.). fuses multiple rankings.
/// Each ranking is Vec<String> of ids ordered best->worst (rank 1 = index 0).
pub fn reciprocal_rank_fusion(
    rankings: &[Vec<String>],
    k: f32,
) -> std::collections::HashMap<String, f32> {
    let mut fused: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    for ranking in rankings {
        for (rank, id) in ranking.iter().enumerate() {
            let score = 1.0 / (k + rank as f32 + 1.0);
            *fused.entry(id.clone()).or_insert(0.0) += score;
        }
    }
    fused
}

pub fn reciprocal_rank_fusion_k60(
    rankings: &[Vec<String>],
) -> std::collections::HashMap<String, f32> {
    reciprocal_rank_fusion(rankings, RRF_K)
}

pub fn encode_embedding(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

pub fn decode_embedding(bytes: &[u8]) -> Vec<f32> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Vec::new();
    }
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

impl EpisodeStore {
    pub fn ensure_embedding_column(&self) -> Result<()> {
        let res = self
            .conn
            .execute("ALTER TABLE episodes ADD COLUMN embedding BLOB", []);
        match res {
            Ok(_) => Ok(()),
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("duplicate column") {
                    Ok(())
                } else {
                    Err(e.into())
                }
            }
        }
    }

    pub fn save_with_embedding(&self, episode: &Episode, embedding: &[f32]) -> Result<()> {
        self.bm25_cache.borrow_mut().take();
        self.ensure_embedding_column()?;
        let blob = encode_embedding(embedding);
        self.conn.execute(
            "INSERT OR REPLACE INTO episodes (id, goal, files_touched, test_verdict, outcome, hash, timestamp, embedding) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                episode.id,
                episode.goal,
                serde_json::to_string(&episode.files_touched)?,
                episode.test_verdict,
                episode.outcome,
                episode.hash,
                episode.timestamp,
                blob,
            ],
        )?;
        Ok(())
    }

    pub fn update_embedding(&self, id: &str, embedding: &[f32]) -> Result<()> {
        self.bm25_cache.borrow_mut().take();
        self.ensure_embedding_column()?;
        let blob = encode_embedding(embedding);
        let n = self.conn.execute(
            "UPDATE episodes SET embedding = ?1 WHERE id = ?2",
            rusqlite::params![blob, id],
        )?;
        if n == 0 {
            return Err(crate::error::MemoryError::NotFound(id.to_string()));
        }
        Ok(())
    }

    pub fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>> {
        self.ensure_embedding_column()?;
        let mut stmt = self
            .conn
            .prepare("SELECT embedding FROM episodes WHERE id = ?1")?;
        let opt: Option<Vec<u8>> = stmt
            .query_row(rusqlite::params![id], |row| row.get(0))
            .unwrap_or(None);
        match opt {
            Some(bytes) if !bytes.is_empty() => Ok(Some(decode_embedding(&bytes))),
            _ => Ok(None),
        }
    }

    pub fn recall_by_embedding(&self, query: &[f32], limit: usize) -> Result<Vec<Episode>> {
        self.ensure_embedding_column()?;
        let candidates = self.recall_recent(MAX_RECALL_CANDIDATES)?;
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        // Pull embeddings for candidates
        let mut scored: Vec<(f32, Episode)> = Vec::new();
        for ep in &candidates {
            if let Ok(Some(emb)) = self.get_embedding(&ep.id) {
                if emb.len() != query.len() || emb.is_empty() {
                    continue;
                }
                let s = cosine_similarity(query, &emb);
                if s.is_finite() && s > 0.0 {
                    scored.push((s, ep.clone()));
                }
            }
        }
        if scored.is_empty() {
            return Ok(candidates.into_iter().take(limit).collect());
        }
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.1.timestamp.cmp(&a.1.timestamp))
        });
        Ok(scored.into_iter().take(limit).map(|(_, ep)| ep).collect())
    }

    pub fn recall_hybrid(
        &self,
        goal: &str,
        query_embedding: Option<&[f32]>,
        limit: usize,
    ) -> Result<Vec<Episode>> {
        let candidates = self.recall_recent(MAX_RECALL_CANDIDATES)?;
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        // BM25 ranking
        let query_tokens = tokenize(goal);
        let mut rankings: Vec<Vec<String>> = Vec::new();
        if !query_tokens.is_empty() {
            let doc_tokens: Vec<Vec<String>> =
                candidates.iter().map(|ep| tokenize(&ep.goal)).collect();
            let index = Bm25Index::new(&doc_tokens);
            let mut bm25_scored: Vec<(f32, usize)> = candidates
                .iter()
                .enumerate()
                .map(|(idx, _)| (index.score(&query_tokens, idx), idx))
                .filter(|(s, _)| *s > 0.0)
                .collect();
            bm25_scored.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(candidates[b.1].timestamp.cmp(&candidates[a.1].timestamp))
            });
            if !bm25_scored.is_empty() {
                let rank: Vec<String> = bm25_scored
                    .into_iter()
                    .map(|(_, idx)| candidates[idx].id.clone())
                    .collect();
                rankings.push(rank);
            }
        }
        // Dense ranking if embedding provided and column exists
        if let Some(qemb) = query_embedding {
            self.ensure_embedding_column()?;
            let mut dense_scored: Vec<(f32, String)> = Vec::new();
            for ep in &candidates {
                if let Ok(Some(emb)) = self.get_embedding(&ep.id) {
                    if emb.len() != qemb.len() {
                        continue;
                    }
                    let s = cosine_similarity(qemb, &emb);
                    if s.is_finite() && s > 0.0 {
                        dense_scored.push((s, ep.id.clone()));
                    }
                }
            }
            dense_scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            if !dense_scored.is_empty() {
                rankings.push(dense_scored.into_iter().map(|(_, id)| id).collect());
            }
        }
        if rankings.is_empty() {
            return Ok(candidates.into_iter().take(limit).collect());
        }
        if rankings.len() == 1 {
            let rank = &rankings[0];
            let map: std::collections::HashMap<String, &Episode> =
                candidates.iter().map(|ep| (ep.id.clone(), ep)).collect();
            let mut out: Vec<Episode> = Vec::new();
            for id in rank.iter().take(limit) {
                if let Some(ep) = map.get(id) {
                    out.push((*ep).clone());
                }
            }
            // fill rest with recent if not enough
            if out.len() < limit {
                for ep in &candidates {
                    if out.iter().any(|e| e.id == ep.id) {
                        continue;
                    }
                    out.push(ep.clone());
                    if out.len() >= limit {
                        break;
                    }
                }
            }
            return Ok(out);
        }
        let fused = reciprocal_rank_fusion(&rankings, RRF_K);
        let mut with_ts: Vec<(&Episode, f32)> = candidates
            .iter()
            .filter_map(|ep| fused.get(&ep.id).map(|s| (ep, *s)))
            .collect();
        with_ts.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.0.timestamp.cmp(&a.0.timestamp))
        });
        let mut out: Vec<Episode> = with_ts
            .into_iter()
            .take(limit)
            .map(|(ep, _)| ep.clone())
            .collect();
        if out.len() < limit {
            for ep in &candidates {
                if out.iter().any(|e| e.id == ep.id) {
                    continue;
                }
                out.push(ep.clone());
                if out.len() >= limit {
                    break;
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn episode_new_computes_id_and_hash() {
        let ep = Episode::new(
            "test goal".into(),
            vec!["file.rs".into()],
            None,
            "passed".into(),
        );
        assert!(!ep.id.is_empty());
        assert!(!ep.hash.is_empty());
        assert_eq!(ep.goal, "test goal");
    }

    #[test]
    fn episode_deterministic_id() {
        let id1 = Episode::compute_id("goal", 1000);
        let id2 = Episode::compute_id("goal", 1000);
        assert_eq!(id1, id2);
    }

    #[test]
    fn episode_different_goals_different_ids() {
        let id1 = Episode::compute_id("goal1", 1000);
        let id2 = Episode::compute_id("goal2", 1000);
        assert_ne!(id1, id2);
    }

    #[test]
    fn episode_store_save_and_recall() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = EpisodeStore::open(tmp.path().to_str().unwrap()).unwrap();

        let ep = Episode::new("test".into(), vec![], None, "ok".into());
        store.save(&ep).unwrap();

        let all = store.recall_recent(10).unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].goal, "test");
    }

    #[test]
    fn episode_store_count() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = EpisodeStore::open(tmp.path().to_str().unwrap()).unwrap();

        assert_eq!(store.count().unwrap(), 0);

        store
            .save(&Episode::new("a".into(), vec![], None, "ok".into()))
            .unwrap();
        store
            .save(&Episode::new("b".into(), vec![], None, "ok".into()))
            .unwrap();

        assert_eq!(store.count().unwrap(), 2);
    }

    #[test]
    fn recall_similar_keyword_match() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = EpisodeStore::open(tmp.path().to_str().unwrap()).unwrap();

        store
            .save(&Episode::new(
                "write a bash script".into(),
                vec!["backup.sh".into()],
                None,
                "created".into(),
            ))
            .unwrap();
        store
            .save(&Episode::new(
                "write a python script".into(),
                vec!["parse.py".into()],
                None,
                "created".into(),
            ))
            .unwrap();

        let results = store.recall_similar("bash", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].goal.contains("bash"));
    }

    #[test]
    fn recall_similar_ranks_by_relevance() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = EpisodeStore::open(tmp.path().to_str().unwrap()).unwrap();

        store
            .save(&Episode::new(
                "write a bash script to parse json".into(),
                vec!["parse.sh".into()],
                None,
                "created".into(),
            ))
            .unwrap();
        store
            .save(&Episode::new(
                "write a bash backup script for files".into(),
                vec!["backup.sh".into()],
                None,
                "created".into(),
            ))
            .unwrap();

        let results = store.recall_similar("bash backup script", 10).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].goal.contains("backup"));
    }

    #[test]
    fn recall_similar_falls_back_to_recent_when_no_overlap() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = EpisodeStore::open(tmp.path().to_str().unwrap()).unwrap();

        store
            .save(&Episode::new(
                "refactor the rust module".into(),
                vec!["lib.rs".into()],
                None,
                "done".into(),
            ))
            .unwrap();

        let results = store.recall_similar("make me a sandwich", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].goal.contains("rust"));
    }

    #[test]
    fn similarity_is_zero_for_disjoint_tokens() {
        assert_eq!(
            similarity(&tokenize("alpha beta"), &tokenize("gamma delta")),
            0.0
        );
    }

    #[test]
    fn similarity_is_one_for_identical_tokens() {
        assert_eq!(similarity(&tokenize("a b c"), &tokenize("a b c")), 1.0);
    }

    // BM25 tests

    #[test]
    fn bm25_index_scores_tf_higher() {
        // Doc with higher term frequency for "bash" should score higher under BM25,
        // while Jaccard would tie (both token sets are {bash, script}).
        let docs = vec![tokenize("bash script"), tokenize("bash bash bash script")];
        let index = Bm25Index::new(&docs);
        let query = tokenize("bash");
        let s0 = index.score(&query, 0);
        let s1 = index.score(&query, 1);
        assert!(s1 > s0, "high-TF doc should outrank low-TF: {s1} > {s0}");
    }

    #[test]
    fn bm25_supera_jaccard_en_fixture() {
        // Fixture where Jaccard ties but BM25 distinguishes by TF.
        // Both docs share same set {bash, script}, query tokenizes to {bash}.
        // Jaccard: both 0.5 (1/2). BM25: repetition wins.
        let docs = vec![tokenize("bash script"), tokenize("bash bash bash script")];
        let query = tokenize("bash");
        let j0 = similarity(&query, &docs[0]);
        let j1 = similarity(&query, &docs[1]);
        assert_eq!(j0, j1, "Jaccard should tie for this fixture");
        let index = Bm25Index::new(&docs);
        let b0 = index.score(&query, 0);
        let b1 = index.score(&query, 1);
        assert!(b1 > b0, "BM25 must break the Jaccard tie: {b1} > {b0}");
    }

    #[test]
    fn bm25_idf_rarity() {
        // Rare term "kubernetes" appears in only 1 of 3 docs, common "script" in all.
        // IDF(rare) > IDF(common) so doc containing rare term should outrank for rare query.
        let docs = vec![
            tokenize("write a script for deployment"),
            tokenize("write a script for testing"),
            tokenize("deploy kubernetes cluster script"),
        ];
        let index = Bm25Index::new(&docs);
        assert!(index.idf("kubernetes") > index.idf("script"));
        let query = tokenize("kubernetes");
        let scores = index.scores(&query);
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].0, 2);
    }

    #[test]
    fn recall_bm25_ranks_by_tf() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = EpisodeStore::open(tmp.path().to_str().unwrap()).unwrap();
        // Save low-TF first, high-TF second (more recent). To prove BM25 beats recency tie,
        // we need Jaccard tie scenario: but BM25 will still rank high-TF first even if we
        // reverse insertion order, we test that high-TF wins regardless.
        store
            .save(&Episode::new(
                "bash script".into(),
                vec!["a.sh".into()],
                None,
                "created".into(),
            ))
            .unwrap();
        store
            .save(&Episode::new(
                "bash bash bash script".into(),
                vec!["b.sh".into()],
                None,
                "created".into(),
            ))
            .unwrap();
        let results = store.recall_bm25("bash", 10).unwrap();
        assert_eq!(results.len(), 2);
        // BM25 should rank the repeated-bash doc first
        assert!(
            results[0].goal.contains("bash bash bash"),
            "BM25 should rank high-TF first, got {}",
            results[0].goal
        );
    }

    #[test]
    fn recall_bm25_fallback_when_no_overlap() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = EpisodeStore::open(tmp.path().to_str().unwrap()).unwrap();
        store
            .save(&Episode::new(
                "refactor the rust module".into(),
                vec!["lib.rs".into()],
                None,
                "done".into(),
            ))
            .unwrap();
        let results = store.recall_bm25("make me a sandwich", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].goal.contains("rust"));
    }

    #[test]
    fn tokenize_lowercase_alphanum() {
        // Must match spec: lowercase alphanum runs, lines 67-73 semantics
        assert_eq!(tokenize("Hello WORLD 123"), vec!["hello", "world", "123"]);
        assert_eq!(tokenize("foo-bar_baz"), vec!["foo", "bar", "baz"]);
        assert_eq!(tokenize(""), Vec::<String>::new());
    }

    #[test]
    fn kitty_tokenizer_matches_tokenize() {
        let kt = KittyTokenizer;
        let a = tokenize("Hello Rust-World 42");
        let b: Vec<String> = bm25::Tokenizer::tokenize(&kt, "Hello Rust-World 42");
        assert_eq!(a, b);
    }

    #[test]
    fn bm25_index_via_bm25_crate_embedder() {
        // Sanity that external bm25 crate works with our KittyTokenizer
        let corpus = [
            "bash script".to_string(),
            "bash bash bash script".to_string(),
        ];
        let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
        let embedder =
            bm25::EmbedderBuilder::<u32, KittyTokenizer>::with_tokenizer_and_fit_to_corpus(
                KittyTokenizer,
                &corpus_refs,
            )
            .build();
        let q_emb = embedder.embed("bash");
        assert!(!q_emb.is_empty());
    }
}

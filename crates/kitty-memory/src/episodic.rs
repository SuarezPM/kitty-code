use std::collections::{HashMap, HashSet};

use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// An episode stored in memory.
///
/// `importance` (T6.2, 3-factor scoring) is in-memory only with serde default
/// 0.0: old rows and old DBs read back as neutral without any migration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: String,
    pub goal: String,
    pub files_touched: Vec<String>,
    pub test_verdict: Option<String>,
    pub outcome: String,
    pub hash: String,
    pub timestamp: i64,
    #[serde(default)]
    pub importance: f32,
    /// W6: episodio verificado/cerrado. Con TTL obligatorio
    /// ([`SETTLED_TTL_SECS`]): estados permanentes prohibidos.
    /// `false` en filas viejas (serde default + DEFAULT 0 en migración).
    #[serde(default)]
    pub settled: bool,
    /// Epoch-secs hasta el que vale `settled`. 0 = sin TTL / no settled.
    #[serde(default)]
    pub settled_until: i64,
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
            importance: 0.0,
            settled: false,
            settled_until: 0,
        }
    }
}

/// W6: TTL del estado settled: una semana. Caducado ⇒ no settled
/// (re-verificable, nunca permanente).
pub const SETTLED_TTL_SECS: i64 = 7 * 24 * 3600;

impl Episode {
    /// Marca el episodio como settled con TTL desde `now` (epoch-secs).
    pub fn mark_settled(&mut self, now: i64) {
        self.settled = true;
        self.settled_until = now.saturating_add(SETTLED_TTL_SECS);
    }

    /// Settled vigente: flag + TTL no caducado.
    pub fn is_settled(&self, now: i64) -> bool {
        self.settled && self.settled_until > 0 && now <= self.settled_until
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
        // T6.2: importance as REAL with DEFAULT 0.0 — existing rows read as
        // neutral, no data migration. Errors ignored like the column above.
        let _ = conn.execute(
            "ALTER TABLE episodes ADD COLUMN importance REAL DEFAULT 0.0",
            [],
        );
        // W6: settled con TTL — filas viejas leen no-settled, sin migración.
        let _ = conn.execute(
            "ALTER TABLE episodes ADD COLUMN settled INTEGER DEFAULT 0",
            [],
        );
        let _ = conn.execute(
            "ALTER TABLE episodes ADD COLUMN settled_until INTEGER DEFAULT 0",
            [],
        );
        Ok(Self {
            conn,
            bm25_cache: std::cell::RefCell::new(None),
        })
    }

    /// Save an episode (insert or replace).
    pub fn save(&self, episode: &Episode) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO episodes (id, goal, files_touched, test_verdict, outcome, hash, timestamp, importance, settled, settled_until)
              VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            rusqlite::params![
                episode.id,
                episode.goal,
                serde_json::to_string(&episode.files_touched)?,
                episode.test_verdict,
                episode.outcome,
                episode.hash,
                episode.timestamp,
                episode.importance as f64,
                episode.settled as i64,
                episode.settled_until,
            ],
        )?;
        self.bm25_cache.borrow_mut().take();
        Ok(())
    }

    /// Recall the N most recent episodes.
    pub fn recall_recent(&self, limit: usize) -> Result<Vec<Episode>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, goal, files_touched, test_verdict, outcome, hash, timestamp, importance, settled, settled_until
              FROM episodes
              ORDER BY timestamp DESC
              LIMIT ?1",
        )?;
        let episodes = stmt
            .query_map(rusqlite::params![limit as i64], |row| {
                // Tolerant read: pre-T6.2 DBs (or a missing column) yield 0.0.
                let importance: f64 = row.get::<_, Option<f64>>(7).ok().flatten().unwrap_or(0.0);
                // W6 tolerant read: pre-W6 DBs yield no-settled.
                let settled: i64 = row.get::<_, Option<i64>>(8).ok().flatten().unwrap_or(0);
                let settled_until: i64 = row.get::<_, Option<i64>>(9).ok().flatten().unwrap_or(0);
                Ok(Episode {
                    id: row.get(0)?,
                    goal: row.get(1)?,
                    files_touched: serde_json::from_str(&row.get::<_, String>(2)?)
                        .unwrap_or_default(),
                    test_verdict: row.get(3)?,
                    outcome: row.get(4)?,
                    hash: row.get(5)?,
                    timestamp: row.get(6)?,
                    importance: importance as f32,
                    settled: settled != 0,
                    settled_until,
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

    /// Gated Jaccard arm (no `fill_recent`, no fallback).
    ///
    /// m06: sin `unwrap` en prod — los errores SQLite suben con `?`.
    fn recall_gated_jaccard(&self, goal: &str, limit: usize) -> Result<Vec<Episode>> {
        let candidates = self.recall_recent(MAX_RECALL_CANDIDATES)?;
        if candidates.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let query_tokens = tokenize(goal);
        if query_tokens.is_empty() {
            return Ok(Vec::new());
        }
        let mut scored: Vec<(f64, &Episode)> = candidates
            .iter()
            .map(|ep| (similarity(&query_tokens, &tokenize(&ep.goal)), ep))
            .filter(|(score, _)| *score > 0.0)
            .collect();
        // Determinista: score desc, timestamp desc, id asc.
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.1.timestamp.cmp(&a.1.timestamp))
                .then_with(|| a.1.id.cmp(&b.1.id))
        });
        Ok(scored
            .into_iter()
            .take(limit)
            .map(|(_, ep)| ep.clone())
            .collect())
    }

    /// Gated hybrid arm (BM25 léxico + coseno denso, fusión RRF).
    ///
    /// SIN `fill_recent` ni fallback: vacío si ningún canal puntúa > 0.
    /// NO delega en `recall_hybrid*` (todos terminan en `fill_recent`).
    /// m06: sin `unwrap` en prod — `get_embedding` tolera filas sin BLOB.
    fn recall_gated_hybrid(
        &self,
        goal: &str,
        query_embedding: Option<&[f32]>,
        limit: usize,
    ) -> Result<Vec<Episode>> {
        let candidates = self.recall_recent(MAX_RECALL_CANDIDATES)?;
        if candidates.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        // Canal léxico: índice BM25 fresco (no la caché: el path gated es
        // puro y no muta cachés).
        let mut rankings: Vec<Vec<String>> = Vec::new();
        let query_tokens = tokenize(goal);
        if !query_tokens.is_empty() {
            let doc_tokens: Vec<Vec<String>> =
                candidates.iter().map(|ep| tokenize(&ep.goal)).collect();
            let index = Bm25Index::new(&doc_tokens);
            let mut lex: Vec<(f32, usize)> = candidates
                .iter()
                .enumerate()
                .map(|(idx, _)| (index.score(&query_tokens, idx), idx))
                .filter(|(s, _)| *s > 0.0)
                .collect();
            lex.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(candidates[b.1].timestamp.cmp(&candidates[a.1].timestamp))
                    .then_with(|| candidates[a.1].id.cmp(&candidates[b.1].id))
            });
            if !lex.is_empty() {
                rankings.push(
                    lex.into_iter()
                        .map(|(_, idx)| candidates[idx].id.clone())
                        .collect(),
                );
            }
        }
        // Canal denso: solo si hay query embedding y columna disponible.
        if let Some(qemb) = query_embedding {
            if !qemb.is_empty() {
                // `ensure_embedding_column` crea la columna si falta; si
                // falla, el canal denso se omite (no se aborta el recall).
                let dense_ok = self.ensure_embedding_column().is_ok();
                if dense_ok {
                    let mut dense: Vec<(f32, String)> = Vec::new();
                    for ep in &candidates {
                        if let Ok(Some(emb)) = self.get_embedding(&ep.id) {
                            if emb.len() != qemb.len() || emb.is_empty() {
                                continue;
                            }
                            let s = cosine_similarity(qemb, &emb);
                            if s.is_finite() && s > 0.0 {
                                dense.push((s, ep.id.clone()));
                            }
                        }
                    }
                    dense.sort_by(|a, b| {
                        b.0.partial_cmp(&a.0)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| a.1.cmp(&b.1))
                    });
                    if !dense.is_empty() {
                        rankings.push(dense.into_iter().map(|(_, id)| id).collect());
                    }
                }
            }
        }
        if rankings.is_empty() {
            return Ok(Vec::new());
        }
        let by_id: std::collections::HashMap<&str, &Episode> =
            candidates.iter().map(|ep| (ep.id.as_str(), ep)).collect();
        if rankings.len() == 1 {
            let mut out: Vec<Episode> = Vec::new();
            for id in rankings[0].iter().take(limit) {
                if let Some(ep) = by_id.get(id.as_str()) {
                    out.push((*ep).clone());
                }
            }
            return Ok(out);
        }
        let fused = reciprocal_rank_fusion(&rankings, RRF_K);
        let mut scored: Vec<(&Episode, f32)> = candidates
            .iter()
            .filter_map(|ep| fused.get(&ep.id).map(|s| (ep, *s)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.0.timestamp.cmp(&a.0.timestamp))
                .then_with(|| a.0.id.cmp(&b.0.id))
        });
        Ok(scored
            .into_iter()
            .take(limit)
            .map(|(ep, _)| ep.clone())
            .collect())
    }

    /// Recall gated: Jaccard o híbrido SIN `fill_recent` ni fallback.
    ///
    /// - Vacío si `score <= 0` (o `limit == 0`, o sin candidatos).
    /// - Orden determinista: score desc, timestamp desc, id asc.
    /// - `Hybrid` detrás del selector default-off
    ///   ([`hybrid_recall_enabled`]): sin selector ⇒ degrada a Jaccard.
    /// - NO llama a `recall_hybrid*` (todos terminan en `fill_recent`).
    pub fn recall_gated(
        &self,
        goal: &str,
        query_embedding: Option<&[f32]>,
        limit: usize,
        mode: RecallMode,
        now: i64,
    ) -> Result<Vec<Episode>> {
        // `now` se conserva en la firma Oracle para futuros decaimientos de
        // recencia; el orden gated ya desempata por timestamp.
        let _ = now;
        if limit == 0 {
            return Ok(Vec::new());
        }
        let effective = match mode {
            RecallMode::Jaccard => RecallMode::Jaccard,
            RecallMode::Hybrid if hybrid_recall_enabled() => RecallMode::Hybrid,
            RecallMode::Hybrid => RecallMode::Jaccard,
        };
        match effective {
            RecallMode::Jaccard => self.recall_gated_jaccard(goal, limit),
            RecallMode::Hybrid => self.recall_gated_hybrid(goal, query_embedding, limit),
        }
    }
}

/// Modo de recall gated (firma Oracle exacta).
///
/// - `Jaccard`: similitud de tokens, sin fallback.
/// - `Hybrid`: BM25 + denso con fusión RRF, sin `fill_recent`; detrás del
///   selector default-off ([`hybrid_recall_enabled`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecallMode {
    Jaccard,
    Hybrid,
}

/// Selector del brazo híbrido (default-off).
///
/// Habilitado si `KITTY_HYBRID_RECALL=1` (o `true`, insensible a mayúsculas)
/// o si se compila con `--features hybrid-recall`. En cualquier otro caso
/// el recall gated degrada `Hybrid` ⇒ `Jaccard`.
pub const ENV_HYBRID_RECALL: &str = "KITTY_HYBRID_RECALL";

/// `true` cuando el brazo híbrido está habilitado (ver [`ENV_HYBRID_RECALL`]).
pub fn hybrid_recall_enabled() -> bool {
    if let Ok(v) = std::env::var(ENV_HYBRID_RECALL) {
        return v == "1" || v.eq_ignore_ascii_case("true");
    }
    #[cfg(feature = "hybrid-recall")]
    {
        return true;
    }
    #[cfg(not(feature = "hybrid-recall"))]
    {
        return false;
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
            importance: 0.0,
            settled: false,
            settled_until: 0,
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

// ---------------------------------------------------------------------------
// T6.1: parametrized fusion over the existing RRF (reenfocar, no reconstruir)
// ---------------------------------------------------------------------------

/// Channel rankings with raw scores: `(channel, [(doc_id, raw_score)])`
/// ordered best-first per channel. Raw scores feed stage-2; id order feeds
/// RRF fusion.
pub type ChannelRankings = Vec<(FuseChannel, Vec<(String, f32)>)>;

/// Retrieval channel feeding the fusion. Sparse (BGE-M3) is HELD: it needs a
/// multi-GB model download plus a sparse-vector storage schema, neither
/// available offline — the weight plumbing below already admits it when a
/// third channel lands (no fusion rewrite then).
///
/// F2b P0-b: `Tier` is the episode-tier channel (operator-settled /
/// starred episodes outrank routine ones pre-fusion). It rides the same
/// RRF summands as the other channels; the per-document boost lives in
/// [`tier_adjust`] and the tiered entry point is
/// [`fuse_rankings_weighted_tiered`]. The legacy [`fuse_rankings_weighted`]
/// path is unchanged for Lexical/Dense-only inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FuseChannel {
    Lexical,
    Dense,
    /// F2b P0-b: episode-tier channel (tiered ranking input).
    Tier,
}

/// Fusion knobs (plan `RrfParams{k=60, alpha, N=100}`).
/// - `k`: RRF smoothing constant (existing `RRF_K`).
/// - `alpha`: dense-channel weight in [0,1] (lexical gets `1-alpha`).
///   `0.5` reproduces the legacy equal-weight sum exactly.
/// - `n`: candidate window pulled before fusion (legacy path uses 50).
#[derive(Debug, Clone, Copy)]
pub struct RrfParams {
    pub k: f32,
    pub alpha: f32,
    pub n: usize,
}

impl Default for RrfParams {
    fn default() -> Self {
        Self {
            k: RRF_K,
            alpha: 0.5,
            n: 100,
        }
    }
}

/// F2b P0-b: default weight of the [`FuseChannel::Tier`] channel.
/// Shared with [`TierWeights::default`] so the legacy and tiered fusion
/// paths agree on the Tier channel mass.
pub const TIER_CHANNEL_DEFAULT_WEIGHT: f32 = 0.25;

fn channel_weight(channel: FuseChannel, alpha: f32) -> f32 {
    let a = alpha.clamp(0.0, 1.0);
    match channel {
        FuseChannel::Lexical => 1.0 - a,
        FuseChannel::Dense => a,
        FuseChannel::Tier => TIER_CHANNEL_DEFAULT_WEIGHT,
    }
}

fn channel_weight_with_tier(channel: FuseChannel, alpha: f32, tier: &TierWeights) -> f32 {
    match channel {
        FuseChannel::Tier => tier.channel.max(0.0),
        _ => channel_weight(channel, alpha),
    }
}

/// Channel-weighted RRF: same `1/(k+rank+1)` summands as
/// [`reciprocal_rank_fusion`], scaled by channel weight. With `alpha = 0.5`
/// the output equals the legacy unweighted sum.
pub fn fuse_rankings_weighted(
    rankings: &[(FuseChannel, Vec<String>)],
    params: &RrfParams,
) -> std::collections::HashMap<String, f32> {
    let mut fused: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    for (channel, ranking) in rankings {
        let w = channel_weight(*channel, params.alpha);
        if w <= 0.0 {
            continue;
        }
        for (rank, id) in ranking.iter().enumerate() {
            let score = w / (params.k + rank as f32 + 1.0);
            *fused.entry(id.clone()).or_insert(0.0) += score;
        }
    }
    fused
}

// ---------------------------------------------------------------------------
// F2b P0-b: tier_adjust pre-fusión (canal Tier).
// ---------------------------------------------------------------------------

/// Episode tier for pre-fusion adjustment: 0 = verified/starred
/// (settled vigente o importance alta), 1 = intermedio, 2 = rutina.
/// Pure y offline (el caller pasa los campos; sin reloj dentro).
pub fn episode_tier(importance: f32, settled: bool) -> u8 {
    if settled || importance >= 0.7 {
        0
    } else if importance >= 0.3 {
        1
    } else {
        2
    }
}

/// Knobs del ajuste por tier pre-fusión.
/// - `channel`: peso del canal [`FuseChannel::Tier`] en la fusión.
/// - `boost`: escala del boost multiplicativo por documento
///   (`tier_adjust`): tier 0 ×(1+boost), tier 1 ×(1+boost/2), tier 2 ×1.
///   `0.0` desactiva el ajuste (la fusión tiered iguala a la legacy).
#[derive(Debug, Clone, Copy)]
pub struct TierWeights {
    pub channel: f32,
    pub boost: f32,
}

impl Default for TierWeights {
    fn default() -> Self {
        Self {
            channel: TIER_CHANNEL_DEFAULT_WEIGHT,
            boost: TIER_CHANNEL_DEFAULT_WEIGHT,
        }
    }
}

/// Multiplicative pre-fusion boost for one RRF summand: top tier scales by
/// `1+boost`, mid tier by `1+boost/2`, routine tier is untouched.
/// Negative boosts clamp to 0 (never penaliza por debajo del base).
pub fn tier_adjust(base: f32, tier: u8, weights: &TierWeights) -> f32 {
    let boost = weights.boost.max(0.0);
    let factor = match tier {
        0 => 1.0,
        1 => 0.5,
        _ => 0.0,
    };
    base * (1.0 + boost * factor)
}

/// Tiered fusion: same channel-weighted RRF summands as
/// [`fuse_rankings_weighted`], but each summand is scaled pre-fusion by
/// [`tier_adjust`] from `doc_tiers` (missing ids ⇒ tier 2, sin boost).
/// With `TierWeights { channel: 0.25, boost: 0.0 }` and no Tier channel
/// the output equals [`fuse_rankings_weighted`] exactly.
pub fn fuse_rankings_weighted_tiered(
    rankings: &[(FuseChannel, Vec<String>)],
    params: &RrfParams,
    tier_weights: &TierWeights,
    doc_tiers: &std::collections::HashMap<String, u8>,
) -> std::collections::HashMap<String, f32> {
    let mut fused: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    for (channel, ranking) in rankings {
        let w = channel_weight_with_tier(*channel, params.alpha, tier_weights);
        if w <= 0.0 {
            continue;
        }
        for (rank, id) in ranking.iter().enumerate() {
            let base = w / (params.k + rank as f32 + 1.0);
            let tier = doc_tiers.get(id).copied().unwrap_or(2);
            let adj = tier_adjust(base, tier, tier_weights);
            *fused.entry(id.clone()).or_insert(0.0) += adj;
        }
    }
    fused
}

// ---------------------------------------------------------------------------
// T6.2: 3-factor scoring (recency + relevance + importance)
// ---------------------------------------------------------------------------

/// Half-life for recency decay: a week-old episode scores half of "now".
pub const RECENCY_HALF_LIFE_SECS: f64 = 7.0 * 24.0 * 3600.0;

/// Blend weights for [`three_factor_score`]. Default puts relevance first;
/// recency breaks ties, importance (operator-set, 0..1) breaks the rest.
#[derive(Debug, Clone, Copy)]
pub struct ScoringWeights {
    pub w_recency: f32,
    pub w_relevance: f32,
    pub w_importance: f32,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            w_recency: 0.2,
            w_relevance: 0.6,
            w_importance: 0.2,
        }
    }
}

/// Exponential recency in [0,1]: 1.0 at/after `now`, half per half-life back.
pub fn recency_score(timestamp: i64, now: i64, half_life_secs: f64) -> f32 {
    if timestamp >= now || half_life_secs <= 0.0 {
        return 1.0;
    }
    let age = (now - timestamp) as f64;
    0.5_f64.powf(age / half_life_secs) as f32
}

/// Linear 3-factor blend. All inputs are [0,1]; weights need not sum to 1
/// (relative proportions decide).
pub fn three_factor_score(
    recency: f32,
    relevance: f32,
    importance: f32,
    weights: &ScoringWeights,
) -> f32 {
    weights.w_recency * recency.clamp(0.0, 1.0)
        + weights.w_relevance * relevance.clamp(0.0, 1.0)
        + weights.w_importance * importance.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Stage-2 rerank (offline): exact-phrase boost over fused top candidates.
// Neural cross-encoder rerank (`fastembed::TextRerank`) is HELD-network — it
// needs a model download, so the second stage stays a pure, testable
// re-score until a vendored reranker exists.
// ---------------------------------------------------------------------------

/// Multiplicative boost when the episode goal contains the whole query
/// (case-insensitive). Pure and offline.
pub fn phrase_boost(query: &str, goal: &str, boost: f32) -> f32 {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return 1.0;
    }
    if goal.to_lowercase().contains(&q) {
        1.0 + boost.max(0.0)
    } else {
        1.0
    }
}

// ---------------------------------------------------------------------------
// T6.3: offline IR metrics over ranked id lists (binary relevance).
// ---------------------------------------------------------------------------

/// Fraction of relevant ids present in the first `k` ranks. 1.0 when
/// `relevant` is empty (nothing to miss).
pub fn recall_at_k(
    ranked: &[String],
    relevant: &std::collections::HashSet<String>,
    k: usize,
) -> f32 {
    if relevant.is_empty() || k == 0 {
        return 1.0;
    }
    let hits = ranked
        .iter()
        .take(k)
        .filter(|id| relevant.contains(*id))
        .count();
    hits as f32 / relevant.len() as f32
}

/// Reciprocal rank of the first relevant hit (0.0 when absent). Averaged over
/// queries it is the MRR.
pub fn reciprocal_rank(ranked: &[String], relevant: &std::collections::HashSet<String>) -> f32 {
    for (i, id) in ranked.iter().enumerate() {
        if relevant.contains(id) {
            return 1.0 / (i as f32 + 1.0);
        }
    }
    0.0
}

/// Binary nDCG@k: DCG = Σ 1/log2(rank+1) over relevant hits, normalized by
/// the ideal ranking (all relevant first).
pub fn ndcg_at_k(ranked: &[String], relevant: &std::collections::HashSet<String>, k: usize) -> f32 {
    if relevant.is_empty() || k == 0 {
        return 1.0;
    }
    let dcg: f32 = ranked
        .iter()
        .take(k)
        .enumerate()
        .filter(|(_, id)| relevant.contains(*id))
        .map(|(i, _)| 1.0 / ((i as f32 + 2.0).log2()))
        .sum();
    let ideal_hits = relevant.len().min(k);
    let idcg: f32 = (0..ideal_hits)
        .map(|i| 1.0 / ((i as f32 + 2.0).log2()))
        .sum();
    if idcg <= 0.0 {
        0.0
    } else {
        dcg / idcg
    }
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
            "INSERT OR REPLACE INTO episodes (id, goal, files_touched, test_verdict, outcome, hash, timestamp, embedding, importance, settled, settled_until) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            rusqlite::params![
                episode.id,
                episode.goal,
                serde_json::to_string(&episode.files_touched)?,
                episode.test_verdict,
                episode.outcome,
                episode.hash,
                episode.timestamp,
                blob,
                episode.importance as f64,
                episode.settled as i64,
                episode.settled_until,
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

    /// Shared channel-ranking builder (T6.1): BM25 lexical + dense cosine
    /// over the given candidates, with RAW channel scores attached.
    /// `recall_hybrid` keeps its inline copy frozen; the parametrized paths
    /// below build on this helper. Raw scores (not RRF ranks) feed stage-2
    /// relevance: RRF `1/(k+rank+1)` compresses magnitudes (~1.6%/rank at
    /// k=60), which would let recency dominate the blend.
    fn build_channel_rankings(
        &self,
        candidates: &[Episode],
        goal: &str,
        query_embedding: Option<&[f32]>,
    ) -> Result<ChannelRankings> {
        let mut rankings: Vec<(FuseChannel, Vec<(String, f32)>)> = Vec::new();
        let query_tokens = tokenize(goal);
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
                rankings.push((
                    FuseChannel::Lexical,
                    bm25_scored
                        .into_iter()
                        .map(|(s, idx)| (candidates[idx].id.clone(), s))
                        .collect(),
                ));
            }
        }
        if let Some(qemb) = query_embedding {
            self.ensure_embedding_column()?;
            let mut dense_scored: Vec<(f32, String)> = Vec::new();
            for ep in candidates {
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
                rankings.push((
                    FuseChannel::Dense,
                    dense_scored.into_iter().map(|(s, id)| (id, s)).collect(),
                ));
            }
        }
        Ok(rankings)
    }

    /// Min-max normalize raw channel scores over the fused candidate set.
    /// All-equal (or empty) maps to 1.0: no discrimination, no distortion.
    fn normalize_channel(
        raws: &[(String, f32)],
        fused_ids: &std::collections::HashSet<String>,
    ) -> std::collections::HashMap<String, f32> {
        let vals: Vec<f32> = raws
            .iter()
            .filter(|(id, _)| fused_ids.contains(id))
            .map(|(_, s)| *s)
            .collect();
        let (Some(&min), Some(&max)) = (
            vals.iter().min_by(|a, b| a.partial_cmp(b).unwrap()),
            vals.iter().max_by(|a, b| a.partial_cmp(b).unwrap()),
        ) else {
            return std::collections::HashMap::new();
        };
        let span = max - min;
        raws.iter()
            .filter(|(id, _)| fused_ids.contains(id))
            .map(|(id, s)| {
                let n = if span <= f32::EPSILON {
                    1.0
                } else {
                    (s - min) / span
                };
                (id.clone(), n)
            })
            .collect()
    }

    fn fill_recent(
        &self,
        mut out: Vec<Episode>,
        candidates: &[Episode],
        limit: usize,
    ) -> Vec<Episode> {
        if out.len() < limit {
            for ep in candidates {
                if out.iter().any(|e| e.id == ep.id) {
                    continue;
                }
                out.push(ep.clone());
                if out.len() >= limit {
                    break;
                }
            }
        }
        out
    }

    /// Hybrid recall with [`RrfParams`]: `n`-sized window + channel-weighted
    /// fusion. With default params and `n = 50` it matches `recall_hybrid`.
    pub fn recall_hybrid_params(
        &self,
        goal: &str,
        query_embedding: Option<&[f32]>,
        limit: usize,
        params: &RrfParams,
    ) -> Result<Vec<Episode>> {
        let candidates = self.recall_recent(params.n.max(1))?;
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let rankings = self.build_channel_rankings(&candidates, goal, query_embedding)?;
        if rankings.is_empty() {
            return Ok(candidates.into_iter().take(limit).collect());
        }
        // Fusion works on id order; raw scores stay attached for stage-2.
        let id_rankings: Vec<(FuseChannel, Vec<String>)> = rankings
            .iter()
            .map(|(ch, scored)| (*ch, scored.iter().map(|(id, _)| id.clone()).collect()))
            .collect();
        if id_rankings.len() == 1 {
            let rank = &id_rankings[0].1;
            let map: std::collections::HashMap<String, &Episode> =
                candidates.iter().map(|ep| (ep.id.clone(), ep)).collect();
            let mut out: Vec<Episode> = Vec::new();
            for id in rank.iter().take(limit) {
                if let Some(ep) = map.get(id) {
                    out.push((*ep).clone());
                }
            }
            return Ok(self.fill_recent(out, &candidates, limit));
        }
        let fused = fuse_rankings_weighted(&id_rankings, params);
        let mut with_ts: Vec<(&Episode, f32)> = candidates
            .iter()
            .filter_map(|ep| fused.get(&ep.id).map(|s| (ep, *s)))
            .collect();
        with_ts.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.0.timestamp.cmp(&a.0.timestamp))
        });
        let out: Vec<Episode> = with_ts
            .into_iter()
            .take(limit)
            .map(|(ep, _)| ep.clone())
            .collect();
        Ok(self.fill_recent(out, &candidates, limit))
    }

    /// Ranked recall (T6.2): stage-1 weighted fusion, then stage-2 rescoring
    /// with 3-factor blend + exact-phrase boost. `now` is a parameter so the
    /// recency term stays pure and testable.
    pub fn recall_hybrid_ranked(
        &self,
        goal: &str,
        query_embedding: Option<&[f32]>,
        limit: usize,
        params: &RrfParams,
        weights: &ScoringWeights,
        now: i64,
    ) -> Result<Vec<Episode>> {
        let candidates = self.recall_recent(params.n.max(1))?;
        if candidates.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let rankings = self.build_channel_rankings(&candidates, goal, query_embedding)?;
        if rankings.is_empty() {
            return Ok(candidates.into_iter().take(limit).collect());
        }
        let id_rankings: Vec<(FuseChannel, Vec<String>)> = rankings
            .iter()
            .map(|(ch, scored)| (*ch, scored.iter().map(|(id, _)| id.clone()).collect()))
            .collect();
        let fused = fuse_rankings_weighted(&id_rankings, params);
        // Stage-2 relevance from RAW channel scores (best channel wins), NOT
        // from RRF ranks — see the helper comment for why ranks mislead here.
        let fused_ids: std::collections::HashSet<String> = fused.keys().cloned().collect();
        let mut relevance: std::collections::HashMap<String, f32> =
            std::collections::HashMap::new();
        for (channel, scored) in &rankings {
            let _ = channel;
            for (id, n) in Self::normalize_channel(scored, &fused_ids) {
                let e = relevance.entry(id).or_insert(0.0);
                if n > *e {
                    *e = n;
                }
            }
        }
        let mut rescored: Vec<(f32, &Episode)> = candidates
            .iter()
            .filter_map(|ep| {
                let rel = relevance.get(&ep.id).copied().unwrap_or(0.0);
                if rel <= 0.0 {
                    return None;
                }
                let recency = recency_score(ep.timestamp, now, RECENCY_HALF_LIFE_SECS);
                let s = three_factor_score(recency, rel, ep.importance, weights)
                    * phrase_boost(goal, &ep.goal, 0.5);
                Some((s, ep))
            })
            .collect();
        rescored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.1.timestamp.cmp(&a.1.timestamp))
        });
        let out: Vec<Episode> = rescored
            .into_iter()
            .take(limit)
            .map(|(_, ep)| ep.clone())
            .collect();
        Ok(self.fill_recent(out, &candidates, limit))
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

    // ── T6.1: parametrized fusion ─────────────────────────────────────────

    #[test]
    fn rrf_params_default_matches_legacy() {
        let p = RrfParams::default();
        assert_eq!(p.k, RRF_K);
        assert_eq!(p.alpha, 0.5);
        assert_eq!(p.n, 100);
    }

    #[test]
    fn weighted_fusion_alpha_half_equals_legacy_sum() {
        let rankings = vec![
            (FuseChannel::Lexical, vec!["a".into(), "b".into()]),
            (FuseChannel::Dense, vec!["b".into(), "a".into()]),
        ];
        let legacy = reciprocal_rank_fusion(&[rankings[0].1.clone(), rankings[1].1.clone()], RRF_K);
        let weighted = fuse_rankings_weighted(&rankings, &RrfParams::default());
        // alpha=0.5 halves each channel: ratios (hence order) are preserved.
        for id in ["a", "b"] {
            let l = legacy[id] / 2.0;
            let w = weighted[id];
            assert!((l - w).abs() < 1e-6, "{id}: {l} vs {w}");
        }
    }

    #[test]
    fn fusion_k_alternative_changes_scores() {
        // Plan explicitly wants k alternatives 10-20 exercised.
        let rankings = vec![(FuseChannel::Lexical, vec!["a".into(), "b".into()])];
        let s10 = fuse_rankings_weighted(
            &rankings,
            &RrfParams {
                k: 10.0,
                ..RrfParams::default()
            },
        );
        let s20 = fuse_rankings_weighted(
            &rankings,
            &RrfParams {
                k: 20.0,
                ..RrfParams::default()
            },
        );
        assert!(s10["a"] > s20["a"], "smaller k concentrates top ranks");
        // Order preserved across k (RRF is rank-monotone).
        assert!(s10["a"] > s10["b"] && s20["a"] > s20["b"]);
    }

    #[test]
    fn fusion_alpha_extremes_isolate_channels() {
        let rankings = vec![
            (FuseChannel::Lexical, vec!["lex".into()]),
            (FuseChannel::Dense, vec!["dense".into()]),
        ];
        let lex_only = fuse_rankings_weighted(
            &rankings,
            &RrfParams {
                alpha: 0.0,
                ..RrfParams::default()
            },
        );
        assert!(lex_only.contains_key("lex"));
        assert!(!lex_only.contains_key("dense"));
        let dense_only = fuse_rankings_weighted(
            &rankings,
            &RrfParams {
                alpha: 1.0,
                ..RrfParams::default()
            },
        );
        assert!(dense_only.contains_key("dense"));
        assert!(!dense_only.contains_key("lex"));
    }

    // ── F2b P0-b: tier_adjust pre-fusión ──────────────────────────────────

    #[test]
    fn episode_tier_buckets_settled_and_importance() {
        assert_eq!(episode_tier(0.0, true), 0, "settled ⇒ top");
        assert_eq!(episode_tier(0.9, false), 0);
        assert_eq!(episode_tier(0.7, false), 0);
        assert_eq!(episode_tier(0.5, false), 1);
        assert_eq!(episode_tier(0.3, false), 1);
        assert_eq!(episode_tier(0.29, false), 2);
        assert_eq!(episode_tier(0.0, false), 2);
    }

    #[test]
    fn tier_adjust_boosts_top_only() {
        let w = TierWeights {
            channel: 0.25,
            boost: 0.4,
        };
        assert!((tier_adjust(1.0, 0, &w) - 1.4).abs() < 1e-6);
        assert!((tier_adjust(1.0, 1, &w) - 1.2).abs() < 1e-6);
        assert!((tier_adjust(1.0, 2, &w) - 1.0).abs() < 1e-6);
        assert!((tier_adjust(1.0, 9, &w) - 1.0).abs() < 1e-6);
        // Boost negativo o cero ⇒ sin ajuste (nunca penaliza).
        let off = TierWeights {
            channel: 0.0,
            boost: 0.0,
        };
        assert!((tier_adjust(2.0, 0, &off) - 2.0).abs() < 1e-6);
        let neg = TierWeights {
            channel: 0.25,
            boost: -1.0,
        };
        assert!((tier_adjust(2.0, 0, &neg) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn tiered_fusion_matches_legacy_when_boost_off_and_no_tier_channel() {
        let rankings = vec![
            (FuseChannel::Lexical, vec!["a".into(), "b".into()]),
            (FuseChannel::Dense, vec!["b".into(), "a".into()]),
        ];
        let params = RrfParams::default();
        let legacy = fuse_rankings_weighted(&rankings, &params);
        let off = TierWeights {
            channel: 0.0,
            boost: 0.0,
        };
        let tiered = fuse_rankings_weighted_tiered(&rankings, &params, &off, &Default::default());
        for id in ["a", "b"] {
            assert!(
                (legacy[id] - tiered[id]).abs() < 1e-6,
                "{id}: {} vs {}",
                legacy[id],
                tiered[id]
            );
        }
    }

    #[test]
    fn tiered_fusion_breaks_ties_toward_top_tier() {
        // Mismo ranking en ambos canales: empate legacy entre a y b.
        // Con a en tier 0 y b en rutina, el tiered debe ordenar a primero.
        let rankings = vec![
            (FuseChannel::Lexical, vec!["a".into(), "b".into()]),
            (FuseChannel::Dense, vec!["a".into(), "b".into()]),
        ];
        let params = RrfParams::default();
        let tiers: std::collections::HashMap<String, u8> =
            [("a".to_string(), 0_u8), ("b".to_string(), 2_u8)]
                .into_iter()
                .collect();
        let tiered =
            fuse_rankings_weighted_tiered(&rankings, &params, &TierWeights::default(), &tiers);
        assert!(
            tiered["a"] > tiered["b"],
            "top tier debe desempatar: a={} b={}",
            tiered["a"],
            tiered["b"]
        );
    }

    #[test]
    fn tier_channel_rides_fusion_with_default_weight() {
        // El canal Tier participa en la fusión legacy con el peso default.
        let rankings = vec![(FuseChannel::Tier, vec!["t".into(), "u".into()])];
        let fused = fuse_rankings_weighted(&rankings, &RrfParams::default());
        let w = TIER_CHANNEL_DEFAULT_WEIGHT;
        assert!((fused["t"] - w / (RRF_K + 1.0)).abs() < 1e-6);
        assert!(fused["t"] > fused["u"]);
    }

    // ── T6.2: 3-factor scoring ────────────────────────────────────────────

    #[test]
    fn recency_decay_half_life() {
        let now = 1_000_000;
        assert_eq!(recency_score(now, now, RECENCY_HALF_LIFE_SECS), 1.0);
        assert_eq!(recency_score(now + 50, now, RECENCY_HALF_LIFE_SECS), 1.0);
        let half = recency_score(
            (now as f64 - RECENCY_HALF_LIFE_SECS) as i64,
            now,
            RECENCY_HALF_LIFE_SECS,
        );
        assert!(
            (half - 0.5).abs() < 1e-6,
            "one half-life back = 0.5, got {half}"
        );
        let older = recency_score(now - 10_000_000_000, now, RECENCY_HALF_LIFE_SECS);
        assert!(older < half && older >= 0.0);
    }

    #[test]
    fn three_factor_relevance_dominates_by_default() {
        let w = ScoringWeights::default();
        // High relevance + stale beats fresh irrelevance under default weights.
        let relevant = three_factor_score(0.0, 1.0, 0.0, &w);
        let fresh = three_factor_score(1.0, 0.0, 0.0, &w);
        assert!(relevant > fresh);
    }

    #[test]
    fn three_factor_importance_breaks_ties() {
        let w = ScoringWeights::default();
        let plain = three_factor_score(0.5, 0.5, 0.0, &w);
        let starred = three_factor_score(0.5, 0.5, 1.0, &w);
        assert!(starred > plain);
        assert!((starred - plain - w.w_importance).abs() < 1e-6);
    }

    #[test]
    fn phrase_boost_only_on_substring() {
        assert_eq!(
            phrase_boost("websocket reconnect", "fix websocket reconnect loop", 0.5),
            1.5
        );
        assert_eq!(
            phrase_boost("websocket reconnect", "deploy kubernetes", 0.5),
            1.0
        );
        assert_eq!(phrase_boost("   ", "anything", 0.5), 1.0);
    }

    #[test]
    fn importance_roundtrips_without_migration() {
        // Old rows (no importance set) read as 0.0; new rows keep theirs.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = EpisodeStore::open(tmp.path().to_str().unwrap()).unwrap();
        let mut starred = Episode::new("critical deploy runbook".into(), vec![], None, "ok".into());
        starred.importance = 0.9;
        store.save(&starred).unwrap();
        store
            .save(&Episode::new(
                "routine lint fix".into(),
                vec![],
                None,
                "ok".into(),
            ))
            .unwrap();
        let all = store.recall_recent(10).unwrap();
        let back = all.iter().find(|e| e.id == starred.id).unwrap();
        assert!((back.importance - 0.9).abs() < 1e-6);
        let plain = all.iter().find(|e| e.goal == "routine lint fix").unwrap();
        assert_eq!(plain.importance, 0.0);
    }

    #[test]
    fn hybrid_params_matches_legacy_on_fixture() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = EpisodeStore::open(tmp.path().to_str().unwrap()).unwrap();
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
        let legacy = store.recall_hybrid("bash", None, 10).unwrap();
        let params = RrfParams {
            n: MAX_RECALL_CANDIDATES,
            ..RrfParams::default()
        };
        let paramed = store
            .recall_hybrid_params("bash", None, 10, &params)
            .unwrap();
        assert_eq!(legacy.len(), paramed.len());
        assert_eq!(legacy[0].goal, paramed[0].goal);
        assert!(paramed[0].goal.contains("bash bash bash"));
    }

    #[test]
    fn hybrid_ranked_promotes_important_match() {
        // Same tokens (BM25 tie) + older timestamp loses on recency, but an
        // operator-starred importance must still win the top-1.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = EpisodeStore::open(tmp.path().to_str().unwrap()).unwrap();
        let mut starred = Episode::new(
            "restart gateway service".into(),
            vec!["gateway/svc.rs".into()],
            None,
            "restarted".into(),
        );
        starred.timestamp -= 30 * 24 * 3600; // a month older
        starred.id = Episode::compute_id(&starred.goal, starred.timestamp);
        starred.importance = 1.0;
        store.save(&starred).unwrap();
        store
            .save(&Episode::new(
                "restart gateway service".into(),
                vec!["gateway/other.rs".into()],
                None,
                "restarted".into(),
            ))
            .unwrap();
        let now = chrono_now();
        let top = store
            .recall_hybrid_ranked(
                "restart gateway",
                None,
                2,
                &RrfParams {
                    n: MAX_RECALL_CANDIDATES,
                    ..RrfParams::default()
                },
                &ScoringWeights::default(),
                now,
            )
            .unwrap();
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].id, starred.id, "starred episode must win top-1");
    }

    #[test]
    fn ir_metrics_perfect_and_empty() {
        use std::collections::HashSet;
        let ranked = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let rel: HashSet<String> = ["a".into(), "c".into()].into_iter().collect();
        assert!((recall_at_k(&ranked, &rel, 3) - 1.0).abs() < 1e-6);
        assert!((recall_at_k(&ranked, &rel, 1) - 0.5).abs() < 1e-6);
        assert!((reciprocal_rank(&ranked, &rel) - 1.0).abs() < 1e-6);
        // Not ideal: relevant {a,c} ideally ranks [a,c] (DCG 1+1/log2(3));
        // here c sits at rank 3 → DCG 1+1/log2(4), nDCG ≈ 0.9197.
        let ndcg = ndcg_at_k(&ranked, &rel, 3);
        assert!((ndcg - 0.9197).abs() < 1e-3, "got {ndcg}");
        let missing: HashSet<String> = ["z".into()].into_iter().collect();
        assert_eq!(recall_at_k(&ranked, &missing, 3), 0.0);
        assert_eq!(reciprocal_rank(&ranked, &missing), 0.0);
        assert_eq!(ndcg_at_k(&ranked, &missing, 3), 0.0);
        let empty: HashSet<String> = HashSet::new();
        assert_eq!(recall_at_k(&ranked, &empty, 3), 1.0);
    }

    #[test]
    fn ir_metrics_partial_ranking_value() {
        use std::collections::HashSet;
        // Relevant at rank 3 (index 2): RR = 1/3; nDCG = (1/log2(4)) / (1/log2(2)) = 0.5.
        let ranked = vec!["x".to_string(), "y".to_string(), "a".to_string()];
        let rel: HashSet<String> = ["a".into()].into_iter().collect();
        assert!((reciprocal_rank(&ranked, &rel) - 1.0 / 3.0).abs() < 1e-6);
        assert!((ndcg_at_k(&ranked, &rel, 3) - 0.5).abs() < 1e-6);
        assert!((recall_at_k(&ranked, &rel, 10) - 1.0).abs() < 1e-6);
    }

    // ── W6: settled + TTL ────────────────────────────────────────────

    #[test]
    fn settled_nuevo_es_falso_sin_ttl() {
        let ep = Episode::new("x".into(), vec![], None, "ok".into());
        assert!(!ep.settled);
        assert_eq!(ep.settled_until, 0);
        assert!(!ep.is_settled(chrono_now()));
    }

    #[test]
    fn settled_vigente_y_caducado() {
        let mut ep = Episode::new("x".into(), vec![], None, "ok".into());
        let now = chrono_now();
        ep.mark_settled(now);
        assert!(ep.is_settled(now));
        assert!(ep.is_settled(now + SETTLED_TTL_SECS));
        // Caducado ⇒ no settled (re-verificable, nunca permanente).
        assert!(!ep.is_settled(now + SETTLED_TTL_SECS + 1));
    }

    #[test]
    fn settled_roundtrip_store() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let store = EpisodeStore::open(tmp.path().to_str().unwrap()).unwrap();
        let now = chrono_now();
        let mut ep = Episode::new("deploy".into(), vec![], None, "ok".into());
        ep.mark_settled(now);
        store.save(&ep).unwrap();
        store
            .save(&Episode::new("lint".into(), vec![], None, "ok".into()))
            .unwrap();
        let all = store.recall_recent(10).unwrap();
        let back = all.iter().find(|e| e.id == ep.id).unwrap();
        assert!(back.settled);
        assert!(back.is_settled(now));
        assert!(!back.is_settled(now + SETTLED_TTL_SECS + 1));
        let plain = all.iter().find(|e| e.goal == "lint").unwrap();
        assert!(!plain.settled);
    }

    // ── Pass 2 Lane B: recall_gated (firma Oracle) ──────────────────────

    fn gated_store_with(goals: &[(&str, i64)]) -> EpisodeStore {
        // Test-only: el TempDir se olvida a propósito para que el SQLite
        // sobreviva al helper (el SO limpia /tmp; producción nunca hace esto).
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("gated.db");
        let path_str = path.to_str().unwrap().to_string();
        std::mem::forget(dir);
        let store = EpisodeStore::open(&path_str).unwrap();
        for (goal, ts) in goals {
            let mut ep = Episode::new((*goal).into(), vec![], None, "ok".into());
            ep.timestamp = *ts;
            ep.id = Episode::compute_id(&ep.goal, ep.timestamp);
            store.save(&ep).unwrap();
        }
        store
    }

    #[test]
    fn gated_jaccard_sin_fill_devuelve_vacio_sin_overlap() {
        // Legacy `recall_similar` cae a recent aquí; el gated NO debe.
        let store = gated_store_with(&[("refactor the rust module", 100)]);
        let out = store
            .recall_gated("make me a sandwich", None, 10, RecallMode::Jaccard, 200)
            .unwrap();
        assert!(
            out.is_empty(),
            "gated sin overlap debe ser vacío (sin fill_recent), fue: {out:?}"
        );
        // Control: el legacy sí rellena (pin del contraste).
        let legacy = store.recall_similar("make me a sandwich", 10).unwrap();
        assert_eq!(legacy.len(), 1);
    }

    #[test]
    fn gated_50_goals_no_relacionados_cero() {
        let store = gated_store_with(&[("postgres index bloat autovacuum tuning", 100)]);
        const GOALS: [&str; 50] = [
            "bake sourdough bread at home",
            "grow tomatoes on balcony",
            "fix bicycle flat tire",
            "plan weekly meals cheap",
            "learn basic watercolor washes",
            "train puppy to sit",
            "organize garage shelves",
            "write wedding speech toast",
            "meditate ten minutes daily",
            "change car oil safely",
            "knit wool scarf pattern",
            "build birdhouse from pine",
            "study spanish verbs nightly",
            "clean oven glass door",
            "photograph moon with phone",
            "repair drywall small hole",
            "brew kombucha first batch",
            "sew button on coat",
            "prune roses in march",
            "fold fitted sheet neatly",
            "grill salmon fillets evenly",
            "unclog sink drain naturally",
            "paint bedroom accent wall",
            "potty train toddler tips",
            "sharpen kitchen knives whetstone",
            "make candles with soy wax",
            "tile backsplash grout lines",
            "compost kitchen scraps apartment",
            "draw cartoon faces simple",
            "jog first five kilometers",
            "hem curtains without sewing machine",
            "pickle cucumbers vinegar brine",
            "mount tv bracket studs",
            "whiten sneakers baking soda",
            "arrange flowers vase tall",
            "defrost freezer quickly safely",
            "wrap gifts with fabric",
            "calibrate guitar tuner ear",
            "remove rust from tools",
            "freeze herbs olive oil",
            "patch jeans knee denim",
            "steam dumplings bamboo basket",
            "refinish wooden chair sanding",
            "darn socks heel wool",
            "cure cast iron skillet",
            "hang curtains rod level",
            "mop hardwood floors vinegar",
            "fold origami crane paper",
            "descale kettle citric acid",
            "water orchids ice cubes",
        ];
        for g in GOALS {
            let out = store
                .recall_gated(g, None, 5, RecallMode::Jaccard, 200)
                .unwrap();
            assert!(
                out.is_empty(),
                "goal no-relacionado '{g}' debe dar 0, fue: {out:?}"
            );
        }
    }

    #[test]
    fn gated_orden_determinista_score_timestamp_id() {
        // Dos docs con el mismo score Jaccard: gana el más reciente; a igual
        // timestamp, gana el id menor (determinista entre runs/máquinas).
        let store = gated_store_with(&[
            ("bash script", 100),
            ("bash script", 200),
            ("bash script backup rotation", 200),
        ]);
        let a = store
            .recall_gated("bash script", None, 10, RecallMode::Jaccard, 300)
            .unwrap();
        let b = store
            .recall_gated("bash script", None, 10, RecallMode::Jaccard, 300)
            .unwrap();
        assert_eq!(a.len(), 3);
        assert_eq!(
            a.iter().map(|e| &e.id).collect::<Vec<_>>(),
            b.iter().map(|e| &e.id).collect::<Vec<_>>(),
            "el orden debe ser determinista entre llamadas"
        );
        // Score 1.0 ("bash script" exacto) antes que 2/3 (backup rotation).
        assert_eq!(a[0].timestamp, 200);
        assert_eq!(a[0].goal, "bash script");
        assert_eq!(a[1].goal, "bash script");
        assert!(
            a[0].id < a[1].id,
            "empate total ⇒ id asc: {:?} vs {:?}",
            a[0].id,
            a[1].id
        );
        // limit respeta el top sin rellenar.
        let top1 = store
            .recall_gated("bash script", None, 1, RecallMode::Jaccard, 300)
            .unwrap();
        assert_eq!(top1.len(), 1);
        assert_eq!(top1[0].id, a[0].id);
        // limit 0 ⇒ vacío.
        assert!(store
            .recall_gated("bash script", None, 0, RecallMode::Jaccard, 300)
            .unwrap()
            .is_empty());
    }

    /// Lock serializa los tests que mutan `KITTY_HYBRID_RECALL` (cargo corre
    /// hilos en paralelo en el mismo binario; sin lock dos `gated_hybrid_*`
    /// en paralelo se pisan ⇒ flaky). + guard con restore en `Drop`
    /// (panic-safe: un assert no deja el env sucio).
    static HYBRID_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn lock_hybrid_env() -> std::sync::MutexGuard<'static, ()> {
        HYBRID_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }
    struct HybridEnvGuard {
        prev: Option<String>,
    }
    impl HybridEnvGuard {
        fn set(value: &str) -> Self {
            let prev = std::env::var(ENV_HYBRID_RECALL).ok();
            // SAFETY: test-only bajo lock.
            unsafe {
                std::env::set_var(ENV_HYBRID_RECALL, value);
            }
            Self { prev }
        }
    }
    impl Drop for HybridEnvGuard {
        fn drop(&mut self) {
            // SAFETY: test-only restore.
            unsafe {
                match &self.prev {
                    Some(v) => std::env::set_var(ENV_HYBRID_RECALL, v),
                    None => std::env::remove_var(ENV_HYBRID_RECALL),
                }
            }
        }
    }

    #[test]
    fn gated_hybrid_default_off_degrada_a_jaccard() {
        // Sin `KITTY_HYBRID_RECALL=1` el modo Hybrid degrada a Jaccard:
        // sin overlap ⇒ vacío aunque haya candidatos (sin fill).
        // Hermético: lock + `"0"` explícito (fuerza Jaccard incluso con
        // feature `hybrid-recall`, donde `remove` seguiría habilitado).
        let _lock = lock_hybrid_env();
        let _guard = HybridEnvGuard::set("0");
        let store = gated_store_with(&[("refactor the rust module", 100)]);
        assert!(!hybrid_recall_enabled());
        let out = store
            .recall_gated("make me a sandwich", None, 5, RecallMode::Hybrid, 200)
            .unwrap();
        assert!(
            out.is_empty(),
            "hybrid default-off debe degradar a jaccard (vacío sin fill)"
        );
    }

    #[test]
    fn gated_hybrid_sin_overlap_vacio_aun_habilitado() {
        // Aun con el selector ON, el híbrido gated no rellena: sin canal con
        // score > 0 ⇒ vacío.
        // Hermético: lock + guard (restore en Drop, panic-safe).
        let _lock = lock_hybrid_env();
        let _guard = HybridEnvGuard::set("1");
        let store = gated_store_with(&[("refactor the rust module", 100)]);
        assert!(hybrid_recall_enabled());
        let out = store
            .recall_gated("xyzzy plugh qwerty asdf", None, 5, RecallMode::Hybrid, 200)
            .unwrap();
        assert!(
            out.is_empty(),
            "híbrido gated sin señal debe ser vacío (sin fill_recent), fue: {out:?}"
        );
    }
}

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
const MAX_RECALL_CANDIDATES: usize = 50;

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
    let set_a: std::collections::HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
    let set_b: std::collections::HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
    let intersection = set_a.intersection(&set_b).count();
    if intersection == 0 {
        return 0.0;
    }
    let union = set_a.union(&set_b).count();
    intersection as f64 / union as f64
}

/// Store for episodic memory.
pub struct EpisodeStore {
    conn: Connection,
}

impl EpisodeStore {
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                files_touched TEXT NOT NULL,
                test_verdict TEXT,
                outcome TEXT NOT NULL,
                hash TEXT NOT NULL,
                timestamp INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(timestamp DESC);",
        )?;
        Ok(Self { conn })
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

    /// Recall the `limit` episodes most similar to `goal`.
    ///
    /// Retrieval is lexical: candidate episodes are scored by token-overlap
    /// (Jaccard similarity) against the query and ranked. This is a genuine
    /// information-retrieval baseline — strictly better than the old substring
    /// `LIKE` match — and matches the consolidation/retrieval stage of the
    /// episodic-memory lifecycle (cf. EverMemOS, ACL 2026).
    ///
    /// NOTE: this is lexical similarity, not semantic. Dense embedding + RRF
    /// retrieval is a planned upgrade (currently blocked by the rten/Gemma
    /// int8 incompatibility from the embedding spike).
    pub fn recall_similar(&self, goal: &str, limit: usize) -> Result<Vec<Episode>> {
        // Pull a candidate window, then rank in Rust by similarity.
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

        // Highest similarity first; ties broken by recency (timestamp desc).
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

        // If nothing was similar, fall back to recent episodes.
        if top.is_empty() {
            return Ok(candidates.into_iter().take(limit).collect());
        }
        Ok(top)
    }

    /// Count total episodes.
    pub fn count(&self) -> Result<usize> {
        let count: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM episodes", [], |row| row.get(0))?;
        Ok(count as usize)
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

        // Two episodes share the token "bash" but only one is about "backup".
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

        // The more lexically similar goal should rank first.
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

        // Query shares no tokens with the stored episode → returns recent.
        let results = store.recall_similar("make me a sandwich", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].goal.contains("rust"));
    }

    #[test]
    fn similarity_is_zero_for_disjoint_tokens() {
        assert_eq!(similarity(&tokenize("alpha beta"), &tokenize("gamma delta")), 0.0);
    }

    #[test]
    fn similarity_is_one_for_identical_tokens() {
        assert_eq!(similarity(&tokenize("a b c"), &tokenize("a b c")), 1.0);
    }
}

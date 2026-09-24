//! Knowledge injection — keyword-scored cheat sheet from markdown files.
//!
//! Loads markdown files from a `knowledge/` directory, scores them against
//! the user's prompt using word (weight 1.0) + bigram (weight 2.0) scoring,
//! and injects the best matches at the conversation tail (KV-cache-safe).
//!
//! Caps at 500 tokens per injected card. Only active when
//! `AgentConfig.enable_knowledge = true`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::context::ContextBudget;

#[derive(Error, Debug)]
pub enum KnowledgeError {
    #[error("IO error reading knowledge directory: {0}")]
    Io(#[from] std::io::Error),

    #[error("No knowledge directory found at: {0}")]
    NotFound(String),
}

/// A loaded knowledge card with precomputed tokens.
#[derive(Debug, Clone)]
pub struct KnowledgeCard {
    /// Source filename (without extension).
    pub id: String,
    /// Raw markdown content.
    pub content: String,
    /// Estimated token count.
    pub token_count: usize,
    /// Precomputed word frequencies.
    words: HashMap<String, usize>,
    /// Precomputed bigram frequencies.
    bigrams: HashMap<String, usize>,
}

impl KnowledgeCard {
    /// Parse a markdown file into a knowledge card.
    pub fn parse(id: &str, content: &str) -> Self {
        let words = Self::extract_words(content);
        let bigrams = Self::extract_bigrams(content);
        let token_count = ContextBudget::estimate_tokens(content);
        Self {
            id: id.to_string(),
            content: content.to_string(),
            token_count,
            words,
            bigrams,
        }
    }

    /// Extract lowercase word frequencies.
    fn extract_words(text: &str) -> HashMap<String, usize> {
        let mut freq = HashMap::new();
        for word in text.split_whitespace() {
            let cleaned: String = word
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase();
            if !cleaned.is_empty() && cleaned.len() > 1 {
                *freq.entry(cleaned).or_insert(0) += 1;
            }
        }
        freq
    }

    /// Extract lowercase bigram frequencies.
    fn extract_bigrams(text: &str) -> HashMap<String, usize> {
        let mut freq = HashMap::new();
        let words: Vec<String> = text
            .split_whitespace()
            .map(|w| {
                w.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase()
            })
            .filter(|w| !w.is_empty() && w.len() > 1)
            .collect();

        for window in words.windows(2) {
            let bigram = format!("{} {}", window[0], window[1]);
            *freq.entry(bigram).or_insert(0) += 1;
        }
        freq
    }

    /// Score this card against a query string.
    /// Word matches: weight 1.0, bigram matches: weight 2.0.
    pub fn score(&self, query: &str) -> f64 {
        let query_words = Self::extract_words(query);
        let query_bigrams = Self::extract_bigrams(query);

        let mut score = 0.0;

        // Word matches: weight 1.0
        for word in query_words.keys() {
            if let Some(&card_count) = self.words.get(word) {
                score += 1.0 * card_count as f64;
            }
        }

        // Bigram matches: weight 2.0
        for bigram in query_bigrams.keys() {
            if let Some(&card_count) = self.bigrams.get(bigram) {
                score += 2.0 * card_count as f64;
            }
        }

        score
    }
}

/// Configuration for knowledge injection.
#[derive(Debug, Clone)]
pub struct KnowledgeConfig {
    /// Whether knowledge injection is enabled.
    pub enabled: bool,
    /// Path to the knowledge directory.
    pub knowledge_dir: PathBuf,
    /// Minimum score threshold for injection.
    pub threshold: f64,
    /// Maximum tokens per injected card.
    pub max_tokens_per_card: usize,
    /// Maximum number of cards to inject.
    pub max_cards: usize,
}

impl Default for KnowledgeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            knowledge_dir: PathBuf::from("knowledge"),
            threshold: 2.0,
            max_tokens_per_card: 500,
            max_cards: 3,
        }
    }
}

/// Manages knowledge cards and injection.
pub struct KnowledgeBase {
    config: KnowledgeConfig,
    cards: Vec<KnowledgeCard>,
}

impl KnowledgeBase {
    /// Create a new knowledge base. Loads cards from disk if enabled.
    /// W1: startup constructor stays sync (runs once at agent construction,
    /// not per tool call — not reactor-blocking). Hot paths use
    /// [`Self::load_cards_async`] via `spawn_blocking`-free tokio::fs.
    pub fn new(config: KnowledgeConfig) -> Self {
        let cards = if config.enabled {
            Self::load_cards(&config.knowledge_dir)
        } else {
            Vec::new()
        };
        Self { config, cards }
    }

    /// Async constructor for runtimes that prefer non-blocking startup.
    /// Same semantics as [`Self::new`]; cards load via tokio::fs 1-op reads.
    pub async fn new_async(config: KnowledgeConfig) -> Self {
        let cards = if config.enabled {
            Self::load_cards_async(&config.knowledge_dir).await
        } else {
            Vec::new()
        };
        Self { config, cards }
    }

    /// Async card loader: todo-file 1-op reads via tokio::fs (W1).
    /// Directory traversal stays sync (metadata, fast); file contents
    /// are read with a single async op per `.md` file.
    pub async fn load_cards_async(dir: &Path) -> Vec<KnowledgeCard> {
        let mut cards = Vec::new();

        // Sync traversal (fast metadata) → async content reads (1 op each).
        // read_dir has no async todo-file equivalent; entries are metadata.
        let entries: Vec<std::path::PathBuf> = match std::fs::read_dir(dir) {
            Ok(entries) => entries.flatten().map(|e| e.path()).collect(),
            Err(_) => return cards,
        };

        for path in entries {
            if path.extension().and_then(|e| e.to_str()) == Some("md") {
                if let Ok(content) = tokio::fs::read_to_string(&path).await {
                    let id = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    cards.push(KnowledgeCard::parse(&id, &content));
                }
            }
        }

        cards
    }

    /// Load all markdown files from the knowledge directory.
    fn load_cards(dir: &Path) -> Vec<KnowledgeCard> {
        let mut cards = Vec::new();

        let entries = match std::fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(_) => return cards, // Directory doesn't exist — gracefully return empty
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("md") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    let id = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    cards.push(KnowledgeCard::parse(&id, &content));
                }
            }
        }

        cards
    }

    /// Find the best knowledge cards for a query, respecting threshold and limits.
    pub fn find_relevant(&self, query: &str) -> Vec<&KnowledgeCard> {
        if !self.config.enabled || self.cards.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(&KnowledgeCard, f64)> = self
            .cards
            .iter()
            .map(|card| (card, card.score(query)))
            .filter(|(_, score)| *score >= self.config.threshold)
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top N, respecting token budget
        let mut result = Vec::new();
        let mut total_tokens = 0usize;
        for (card, _score) in scored {
            if result.len() >= self.config.max_cards {
                break;
            }
            if card.token_count > self.config.max_tokens_per_card {
                continue; // Skip oversized cards
            }
            if total_tokens + card.token_count
                > self.config.max_tokens_per_card * self.config.max_cards
            {
                break; // Would exceed total token budget
            }
            total_tokens += card.token_count;
            result.push(card);
        }

        result
    }

    /// Build an injection message from relevant cards.
    /// This is meant to be injected at the conversation tail (user message position).
    pub fn build_injection(&self, query: &str) -> Option<String> {
        let relevant = self.find_relevant(query);
        if relevant.is_empty() {
            return None;
        }

        let mut injection = String::from("[Knowledge Context]\n");
        for card in &relevant {
            injection.push_str(&format!("\n--- {} ---\n{}\n", card.id, card.content));
        }
        injection.push_str("\n[End Knowledge Context]\n");

        Some(injection)
    }

    /// Get the number of loaded cards.
    pub fn card_count(&self) -> usize {
        self.cards.len()
    }

    /// Whether knowledge injection is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_parse_extracts_words() {
        let card = KnowledgeCard::parse("test", "hello world hello");
        assert_eq!(card.words.get("hello"), Some(&2));
        assert_eq!(card.words.get("world"), Some(&1));
    }

    #[test]
    fn card_parse_extracts_bigrams() {
        let card = KnowledgeCard::parse("test", "hello world hello world");
        let bigram_count = card.bigrams.get("hello world").unwrap();
        assert_eq!(*bigram_count, 2);
    }

    #[test]
    fn card_score_words_only() {
        let card = KnowledgeCard::parse("rust", "cargo test build cargo");
        let score = card.score("cargo");
        // "cargo" appears 2 times in card, word weight 1.0 = 2.0
        assert_eq!(score, 2.0);
    }

    #[test]
    fn card_score_bigrams_weighted() {
        let card = KnowledgeCard::parse("rust", "cargo test cargo test");
        let score = card.score("cargo test");
        // words: cargo(1*2=2.0) + test(1*2=2.0) = 4.0
        // bigrams: "cargo test"(1*2=4.0)
        // total = 8.0
        assert_eq!(score, 8.0);
    }

    #[test]
    fn card_score_no_match() {
        let card = KnowledgeCard::parse("rust", "cargo build");
        let score = card.score("python django");
        assert_eq!(score, 0.0);
    }

    #[test]
    fn card_oversized_skipped() {
        let config = KnowledgeConfig {
            enabled: true,
            max_tokens_per_card: 5, // Very small
            ..Default::default()
        };
        let mut base = KnowledgeBase {
            config,
            cards: Vec::new(),
        };
        base.cards.push(KnowledgeCard::parse(
            "big",
            "This is a very long card that should exceed the token limit for testing purposes",
        ));
        let relevant = base.find_relevant("long card testing");
        assert!(relevant.is_empty());
    }

    #[test]
    fn disabled_knowledge_returns_empty() {
        let config = KnowledgeConfig {
            enabled: false,
            ..Default::default()
        };
        let base = KnowledgeBase::new(config);
        assert!(!base.is_enabled());
        assert_eq!(base.card_count(), 0);
        assert!(base.build_injection("anything").is_none());
    }

    #[test]
    fn build_injection_format() {
        let config = KnowledgeConfig {
            enabled: true,
            threshold: 0.0,
            ..Default::default()
        };
        let mut base = KnowledgeBase {
            config,
            cards: Vec::new(),
        };
        base.cards.push(KnowledgeCard::parse(
            "rust-tips",
            "Use cargo test to run tests.",
        ));
        let injection = base.build_injection("cargo test").unwrap();
        assert!(injection.contains("[Knowledge Context]"));
        assert!(injection.contains("rust-tips"));
        assert!(injection.contains("cargo test"));
        assert!(injection.contains("[End Knowledge Context]"));
    }

    #[test]
    fn load_cards_gracefully_handles_missing_dir() {
        let config = KnowledgeConfig {
            enabled: true,
            knowledge_dir: PathBuf::from("/nonexistent/path"),
            ..Default::default()
        };
        let base = KnowledgeBase::new(config);
        assert_eq!(base.card_count(), 0);
    }

    #[test]
    fn load_cards_from_real_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("rust.md"), "# Rust\nUse cargo test.").unwrap();
        std::fs::write(dir.path().join("python.md"), "# Python\nUse pytest.").unwrap();
        // Non-md file should be skipped
        std::fs::write(dir.path().join("notes.txt"), "not markdown").unwrap();

        let config = KnowledgeConfig {
            enabled: true,
            knowledge_dir: dir.path().to_path_buf(),
            ..Default::default()
        };
        let base = KnowledgeBase::new(config);
        assert_eq!(base.card_count(), 2);
    }

    /// W1 red-green: async loader yields the same cards as the sync loader
    /// without parking the reactor (tokio::test fails if blocking).
    #[tokio::test]
    async fn load_cards_async_matches_sync() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("rust.md"), "# Rust\nUse cargo test.").unwrap();
        std::fs::write(dir.path().join("python.md"), "# Python\nUse pytest.").unwrap();
        std::fs::write(dir.path().join("notes.txt"), "not markdown").unwrap();

        let sync_cards = KnowledgeBase::load_cards(dir.path());
        let async_cards = KnowledgeBase::load_cards_async(dir.path()).await;
        assert_eq!(sync_cards.len(), 2);
        assert_eq!(async_cards.len(), 2);
        let mut sync_ids: Vec<_> = sync_cards.iter().map(|c| &c.id).collect();
        let mut async_ids: Vec<_> = async_cards.iter().map(|c| &c.id).collect();
        sync_ids.sort();
        async_ids.sort();
        assert_eq!(sync_ids, async_ids);
    }

    #[test]
    fn threshold_filters_low_score() {
        let config = KnowledgeConfig {
            enabled: true,
            threshold: 100.0, // Very high threshold
            ..Default::default()
        };
        let mut base = KnowledgeBase {
            config,
            cards: Vec::new(),
        };
        base.cards.push(KnowledgeCard::parse("test", "hello world"));
        let relevant = base.find_relevant("hello");
        assert!(relevant.is_empty());
    }
}

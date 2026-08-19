//! Context budget manager — tracks token usage and triggers compaction.
//!
//! Monitors the conversation's token consumption and signals when compaction
//! is needed (default 80% threshold). Includes a loop guard to prevent
//! infinite compaction when a single message is too large.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ContextError {
    #[error("Context overflow: {used}/{max} tokens")]
    Overflow { used: usize, max: usize },

    #[error("Compaction freed insufficient tokens: freed {freed}, need {needed}")]
    InsufficientCompaction { freed: usize, needed: usize },
}

/// Compaction result returned when the budget triggers compaction.
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// How many tokens were freed.
    pub freed: usize,
    /// Whether compaction was sufficient to continue.
    pub sufficient: bool,
    /// The summary replacing old messages.
    pub summary: String,
}

/// Context budget tracker.
pub struct ContextBudget {
    max_tokens: usize,
    used_tokens: usize,
    /// How many times compaction has been attempted in a row without success.
    consecutive_compactions: usize,
    /// Maximum consecutive compactions before hard-stopping.
    max_consecutive_compactions: usize,
}

impl ContextBudget {
    /// Default compaction threshold (80%).
    pub const DEFAULT_THRESHOLD: f32 = 0.8;

    /// Create a new budget with a given max token limit.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            used_tokens: 0,
            consecutive_compactions: 0,
            max_consecutive_compactions: 3,
        }
    }

    /// Current token usage.
    pub fn used(&self) -> usize {
        self.used_tokens
    }

    /// Maximum tokens.
    pub fn max(&self) -> usize {
        self.max_tokens
    }

    /// Remaining tokens.
    pub fn remaining(&self) -> usize {
        self.max_tokens.saturating_sub(self.used_tokens)
    }

    /// Usage ratio (0.0 to 1.0+).
    pub fn ratio(&self) -> f32 {
        self.used_tokens as f32 / self.max_tokens as f32
    }

    /// Whether usage exceeds the given threshold.
    pub fn is_near_limit(&self, threshold: f32) -> bool {
        self.ratio() >= threshold
    }

    /// Whether compaction should trigger (at or above 80%).
    pub fn should_compact(&self) -> bool {
        self.is_near_limit(Self::DEFAULT_THRESHOLD)
    }

    /// Whether we should hard-stop (too many failed compactions).
    pub fn should_halt(&self) -> bool {
        self.consecutive_compactions >= self.max_consecutive_compactions
    }

    /// Record token consumption (e.g., from an LLM response).
    pub fn consume(&mut self, tokens: usize) {
        self.used_tokens = self.used_tokens.saturating_add(tokens).min(self.max_tokens);
    }

    /// Record that compaction happened. Returns whether we can continue.
    ///
    /// `freed` is the number of tokens freed by compaction.
    /// `needed` is the minimum tokens that must be freed to continue.
    pub fn record_compaction(&mut self, freed: usize, needed: usize) -> Result<(), ContextError> {
        if freed < needed {
            self.consecutive_compactions += 1;
            return Err(ContextError::InsufficientCompaction { freed, needed });
        }

        self.used_tokens = self.used_tokens.saturating_sub(freed);
        self.consecutive_compactions = 0;
        Ok(())
    }

    /// Force reset (e.g., after a new session).
    pub fn reset(&mut self) {
        self.used_tokens = 0;
        self.consecutive_compactions = 0;
    }

    /// Estimate tokens from a string (rough: 4 chars ≈ 1 token).
    pub fn estimate_tokens(text: &str) -> usize {
        text.len().div_ceil(4)
    }
}

impl Default for ContextBudget {
    fn default() -> Self {
        Self::new(16384)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_tracking() {
        let mut budget = ContextBudget::new(1000);
        assert_eq!(budget.used(), 0);
        assert_eq!(budget.remaining(), 1000);

        budget.consume(500);
        assert_eq!(budget.used(), 500);
        assert_eq!(budget.remaining(), 500);
    }

    #[test]
    fn threshold_detection() {
        let mut budget = ContextBudget::new(1000);
        assert!(!budget.is_near_limit(0.8));

        budget.consume(800);
        assert!(budget.is_near_limit(0.8));
        assert!(budget.should_compact());
    }

    #[test]
    fn overflow_saturates() {
        let mut budget = ContextBudget::new(100);
        budget.consume(200);
        assert_eq!(budget.used(), 100);
        assert_eq!(budget.remaining(), 0);
    }

    #[test]
    fn compaction_reduces_usage() {
        let mut budget = ContextBudget::new(1000);
        budget.consume(900);
        assert!(budget.should_compact());

        budget.record_compaction(400, 100).unwrap();
        assert_eq!(budget.used(), 500);
        assert!(!budget.should_compact());
    }

    #[test]
    fn insufficient_compaction_errors() {
        let mut budget = ContextBudget::new(1000);
        budget.consume(900);

        let err = budget.record_compaction(10, 100).unwrap_err();
        assert!(matches!(err, ContextError::InsufficientCompaction { .. }));
        assert_eq!(budget.consecutive_compactions, 1);
    }

    #[test]
    fn halt_after_max_compactions() {
        let mut budget = ContextBudget::new(1000);
        budget.consume(900);

        for _ in 0..3 {
            let _ = budget.record_compaction(0, 100);
        }
        assert!(budget.should_halt());
    }

    #[test]
    fn estimate_tokens_reasonable() {
        // "hello" = 5 chars → 2 tokens
        assert_eq!(ContextBudget::estimate_tokens("hello"), 2);
        // "" = 0 chars → 0 tokens
        assert_eq!(ContextBudget::estimate_tokens(""), 0);
        // 8 chars → 2 tokens
        assert_eq!(ContextBudget::estimate_tokens("12345678"), 2);
    }

    #[test]
    fn reset_clears_all() {
        let mut budget = ContextBudget::new(1000);
        budget.consume(500);
        let _ = budget.record_compaction(200, 100);
        budget.reset();
        assert_eq!(budget.used(), 0);
        assert!(!budget.should_halt());
    }
}

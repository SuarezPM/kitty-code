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

/// Context budget tracker.
pub struct ContextBudget {
    max_tokens: usize,
    used_tokens: usize,
    /// How many times compaction has been attempted in a row without success.
    consecutive_compactions: usize,
    /// Maximum consecutive compactions before hard-stopping.
    max_consecutive_compactions: usize,
    compact_threshold: f32,
}

impl ContextBudget {
    /// Default compaction threshold (80%).
    pub const DEFAULT_THRESHOLD: f32 = 0.8;

    /// AD5 floor for the derived compaction threshold: small windows compact
    /// early enough to leave working room, but never absurdly early.
    pub const DERIVED_THRESHOLD_FLOOR: f32 = 0.62;

    /// Token headroom reserved below the window when deriving a threshold:
    /// `(ctx - RESERVE)/ctx` leaves this much room for prompt+output.
    pub const DERIVED_RESERVE_TOKENS: usize = 2048;

    /// Derive an EFFECTIVE compaction threshold for the given context window
    /// (f8 AD5): `min(profile_threshold, max(0.62, (ctx - 2048) / ctx))`.
    ///
    /// Pure function applied wherever a profile threshold meets a possibly
    /// REDUCED window (nano-tier presets or CLI/env-reduced contexts). The
    /// `min` guarantees derivation NEVER raises above the profile threshold:
    /// when `(ctx-reserve)/ctx >= profile_threshold` (e.g. any ≥16k window
    /// against an 0.8 profile) the result equals the profile value unchanged.
    #[must_use]
    pub fn derive_compaction_threshold(profile_threshold: f32, context_window: usize) -> f32 {
        if context_window == 0 {
            return profile_threshold;
        }
        let usable = context_window.saturating_sub(Self::DERIVED_RESERVE_TOKENS) as f32;
        let ratio = usable / context_window as f32;
        ratio
            .max(Self::DERIVED_THRESHOLD_FLOOR)
            .min(profile_threshold)
    }

    /// Create a new budget with a given max token limit (uses default threshold 0.8).
    pub fn new(max_tokens: usize) -> Self {
        Self::new_with_threshold(max_tokens, Self::DEFAULT_THRESHOLD)
    }

    /// Create a new budget with explicit compaction threshold.
    pub fn new_with_threshold(max_tokens: usize, threshold: f32) -> Self {
        Self {
            max_tokens,
            used_tokens: 0,
            consecutive_compactions: 0,
            max_consecutive_compactions: 3,
            compact_threshold: threshold.clamp(0.1, 0.95),
        }
    }

    /// Update the compaction threshold (0.1–0.95, clamped).
    pub fn set_threshold(&mut self, threshold: f32) {
        self.compact_threshold = threshold.clamp(0.1, 0.95);
    }

    /// Update max tokens (e.g. when CLI overrides profile).
    pub fn set_max(&mut self, max: usize) {
        self.max_tokens = max;
    }

    /// Current threshold.
    pub fn threshold(&self) -> f32 {
        self.compact_threshold
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

    /// Whether compaction should trigger (at or above threshold, default 80%).
    pub fn should_compact(&self) -> bool {
        self.is_near_limit(self.compact_threshold)
    }

    /// Whether we should hard-stop (too many failed compactions).
    pub fn should_halt(&self) -> bool {
        self.consecutive_compactions >= self.max_consecutive_compactions
    }

    /// Consecutive unsuccessful compaction attempts (observability for the
    /// halt guard: halt engages when this reaches `max_consecutive_compactions`).
    pub fn consecutive_compactions(&self) -> usize {
        self.consecutive_compactions
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

    /// f8 T3.2 (AD5) exact expectation table for the derived threshold.
    #[test]
    fn derive_compaction_threshold_expectation_table() {
        let cases = [
            // (profile_threshold, context_window, expected)
            (0.7_f32, 4096, 0.62_f32), // tiny@4096: (2048/4096)=0.5 floored
            (0.8, 4096, 0.62),         // medium@4096
            (0.8, 16384, 0.80),        // medium@16384: 0.875 capped at profile
            (0.8, 32768, 0.80),        // large@32768: ~0.94 capped at profile
        ];
        for (profile, ctx, expected) in cases {
            let got = ContextBudget::derive_compaction_threshold(profile, ctx);
            assert!(
                (got - expected).abs() < 1e-6,
                "derive({profile}, {ctx}) = {got}, expected {expected}"
            );
        }
    }

    /// Derivation is INACTIVE (returns the profile value unchanged) whenever
    /// the formula would raise above it — and never raises, ever.
    #[test]
    fn derive_compaction_threshold_never_raises_and_inactive_at_defaults() {
        // At-default windows where the ratio exceeds the profile threshold.
        assert!((ContextBudget::derive_compaction_threshold(0.8, 16384) - 0.8).abs() < 1e-6);
        assert!((ContextBudget::derive_compaction_threshold(0.7, 8192) - 0.7).abs() < 1e-6);
        // Property: for arbitrary inputs the result never exceeds profile.
        for ctx in [1usize, 512, 2048, 4096, 8192, 16384, 1_000_000] {
            for profile in [0.62_f32, 0.7, 0.8, 0.95] {
                let got = ContextBudget::derive_compaction_threshold(profile, ctx);
                assert!(
                    got <= profile + 1e-6,
                    "raised above profile: {got} > {profile}"
                );
                assert!(got >= ContextBudget::DERIVED_THRESHOLD_FLOOR - 1e-6);
            }
        }
        // Degenerate zero window falls back to the profile value.
        assert!((ContextBudget::derive_compaction_threshold(0.8, 0) - 0.8).abs() < 1e-6);
    }
}

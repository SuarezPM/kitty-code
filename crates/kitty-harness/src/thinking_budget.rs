//! Thinking budget — caps reasoning tokens per turn and across the session.
//!
//! If thinking exceeds the per-turn cap, the agent retries the LLM call with
//! thinking disabled. Total session thinking tokens are tracked for observability.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ThinkingBudgetError {
    #[error("Thinking budget exceeded: {used}/{max} tokens this turn")]
    TurnExceeded { used: usize, max: usize },

    #[error("Session thinking budget exhausted: {used}/{max} tokens")]
    SessionExhausted { used: usize, max: usize },
}

/// Tracks thinking token usage per turn and across the session.
pub struct ThinkingBudget {
    /// Per-turn cap. None = no cap.
    per_turn_cap: Option<usize>,
    /// Session-wide cap. None = no cap.
    session_cap: Option<usize>,
    /// Tokens used in the current turn.
    turn_tokens: usize,
    /// Total tokens used across the session.
    session_tokens: usize,
}

impl ThinkingBudget {
    /// Create a new thinking budget tracker.
    pub fn new(per_turn_cap: Option<usize>, session_cap: Option<usize>) -> Self {
        Self {
            per_turn_cap,
            session_cap,
            turn_tokens: 0,
            session_tokens: 0,
        }
    }

    /// Record thinking tokens from an LLM response.
    /// Returns `Err` if the turn or session cap is exceeded.
    pub fn record(&mut self, tokens: usize) -> Result<(), ThinkingBudgetError> {
        self.turn_tokens = self.turn_tokens.saturating_add(tokens);
        self.session_tokens = self.session_tokens.saturating_add(tokens);

        if let Some(max) = self.per_turn_cap {
            if self.turn_tokens > max {
                return Err(ThinkingBudgetError::TurnExceeded {
                    used: self.turn_tokens,
                    max,
                });
            }
        }

        if let Some(max) = self.session_cap {
            if self.session_tokens > max {
                return Err(ThinkingBudgetError::SessionExhausted {
                    used: self.session_tokens,
                    max,
                });
            }
        }

        Ok(())
    }

    /// Check if the per-turn cap is exceeded (without recording).
    pub fn is_turn_exceeded(&self) -> bool {
        match self.per_turn_cap {
            Some(max) => self.turn_tokens > max,
            None => false,
        }
    }

    /// Check if the session cap is exceeded.
    pub fn is_session_exhausted(&self) -> bool {
        match self.session_cap {
            Some(max) => self.session_tokens > max,
            None => false,
        }
    }

    /// Reset the turn counter (call at the start of each new turn).
    pub fn reset_turn(&mut self) {
        self.turn_tokens = 0;
    }

    /// Get the current turn's thinking token count.
    pub fn turn_tokens(&self) -> usize {
        self.turn_tokens
    }

    /// Get the total session thinking token count.
    pub fn session_tokens(&self) -> usize {
        self.session_tokens
    }

    /// Whether thinking budget is enabled (any cap set).
    pub fn is_enabled(&self) -> bool {
        self.per_turn_cap.is_some() || self.session_cap.is_some()
    }
}

impl Default for ThinkingBudget {
    fn default() -> Self {
        Self::new(Some(2048), None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_has_per_turn_cap() {
        let budget = ThinkingBudget::default();
        assert!(budget.is_enabled());
        assert_eq!(budget.per_turn_cap, Some(2048));
    }

    #[test]
    fn record_within_budget() {
        let mut budget = ThinkingBudget::new(Some(100), None);
        budget.record(50).unwrap();
        assert_eq!(budget.turn_tokens(), 50);
        assert_eq!(budget.session_tokens(), 50);
    }

    #[test]
    fn record_exceeds_turn_cap() {
        let mut budget = ThinkingBudget::new(Some(100), None);
        budget.record(50).unwrap();
        let err = budget.record(60).unwrap_err();
        assert!(matches!(err, ThinkingBudgetError::TurnExceeded { .. }));
    }

    #[test]
    fn record_exceeds_session_cap() {
        let mut budget = ThinkingBudget::new(None, Some(100));
        budget.record(60).unwrap();
        let err = budget.record(50).unwrap_err();
        assert!(matches!(err, ThinkingBudgetError::SessionExhausted { .. }));
    }

    #[test]
    fn reset_turn_clears_turn_counter() {
        let mut budget = ThinkingBudget::new(Some(100), None);
        budget.record(80).unwrap();
        assert!(!budget.is_turn_exceeded());
        budget.reset_turn();
        assert_eq!(budget.turn_tokens(), 0);
        assert_eq!(budget.session_tokens(), 80);
    }

    #[test]
    fn no_cap_allows_unlimited() {
        let mut budget = ThinkingBudget::new(None, None);
        budget.record(10000).unwrap();
        assert!(!budget.is_turn_exceeded());
        assert!(!budget.is_session_exhausted());
    }

    #[test]
    fn both_caps_independent() {
        let mut budget = ThinkingBudget::new(Some(50), Some(100));
        budget.record(40).unwrap();
        assert!(!budget.is_turn_exceeded());
        assert!(!budget.is_session_exhausted());

        budget.reset_turn();
        budget.record(40).unwrap();
        // turn: 40, session: 80 — both fine
        assert!(!budget.is_turn_exceeded());
        assert!(!budget.is_session_exhausted());

        budget.reset_turn();
        let err = budget.record(30).unwrap_err();
        // session: 110 > 100
        assert!(matches!(err, ThinkingBudgetError::SessionExhausted { .. }));
    }
}

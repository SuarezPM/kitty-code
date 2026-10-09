//! Prompt-lookup-decoding n-gram predictor for tool calls (W5-code, SDD).
//!
//! Learns `P(next_tool | last order-1 tools)` from executed tool sequences
//! and predicts the most likely next tool name. The speculative path uses
//! the prediction to prefetch read-only calls (PASTE: never write-capable
//! tools). Pure observation + lookup — no model, fully deterministic
//! (frequency ties break lexicographically).
//!
//! Code-only in this lane: prediction quality is measured in the serial
//! window (tb2-8 latencies), never in unit tests with invented cells.

use std::collections::HashMap;

/// N-gram predictor over tool-call name sequences.
#[derive(Debug, Clone, Default)]
pub struct NgramPredictor {
    /// N-gram order (context length + 1). `2` = bigram (default).
    order: usize,
    /// `context → (next → count)`.
    counts: HashMap<Vec<String>, HashMap<String, usize>>,
}

impl NgramPredictor {
    /// Create with bigram order.
    pub fn new() -> Self {
        Self::with_order(2)
    }

    /// Create with an explicit order (≥1; 1 = unigram frequency).
    pub fn with_order(order: usize) -> Self {
        Self {
            order: order.max(1),
            counts: HashMap::new(),
        }
    }

    /// Observe one executed tool sequence (e.g. a turn's tool names).
    pub fn observe(&mut self, seq: &[String]) {
        let ctx_len = self.order - 1;
        if seq.len() < self.order {
            // Too short for a full n-gram: still count the unigram tail so
            // order-1 predictors degrade gracefully instead of going blind.
            if self.order == 1 {
                for name in seq {
                    *self
                        .counts
                        .entry(Vec::new())
                        .or_default()
                        .entry(name.clone())
                        .or_insert(0) += 1;
                }
            }
            return;
        }
        for window in seq.windows(self.order) {
            let (ctx, next) = window.split_at(ctx_len);
            *self
                .counts
                .entry(ctx.to_vec())
                .or_default()
                .entry(next[0].clone())
                .or_insert(0) += 1;
        }
    }

    /// Predict the most likely next tool after `ctx` (trailing names of the
    /// current sequence). `None` when the context was never observed.
    /// Frequency ties break toward the lexicographically smallest name, so
    /// the same table always predicts the same tool.
    pub fn predict(&self, ctx: &[String]) -> Option<String> {
        let ctx_len = self.order - 1;
        let key: Vec<String> = if ctx.len() >= ctx_len {
            ctx[ctx.len() - ctx_len..].to_vec()
        } else if ctx_len == 0 {
            Vec::new()
        } else {
            return None;
        };
        let nexts = self.counts.get(&key)?;
        nexts
            .iter()
            .max_by(|(a_name, a_n), (b_name, b_n)| a_n.cmp(b_n).then_with(|| b_name.cmp(a_name)))
            .map(|(name, _)| name.clone())
    }

    /// Number of distinct contexts learned (introspection/testing).
    pub fn context_count(&self) -> usize {
        self.counts.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn learns_bigram_continuation() {
        let mut p = NgramPredictor::new();
        p.observe(&names(&["Read", "Edit", "Bash"]));
        p.observe(&names(&["Read", "Edit", "Bash"]));
        p.observe(&names(&["Read", "Edit", "Read"]));
        // P(Bash | [Read, Edit]) = 2/3 → Bash.
        assert_eq!(p.predict(&names(&["Read", "Edit"])), Some("Bash".into()));
        // Unseen context → None (fail-open, never invent).
        assert_eq!(p.predict(&names(&["Bash", "Bash"])), None);
    }

    #[test]
    fn ties_break_lexicographically_smallest() {
        let mut p = NgramPredictor::new();
        p.observe(&names(&["Read", "Write"]));
        p.observe(&names(&["Read", "Glob"]));
        assert_eq!(p.predict(&names(&["Read"])), Some("Glob".into()));
        // Same table, second call → same answer (deterministic).
        assert_eq!(p.predict(&names(&["Read"])), Some("Glob".into()));
    }

    #[test]
    fn longer_context_uses_trailing_window() {
        // Bigram (order=2 ⇒ ctx_len=1): only the trailing name matters.
        let mut p = NgramPredictor::new();
        p.observe(&names(&["Glob", "Read", "Edit"]));
        assert_eq!(
            p.predict(&names(&["Bash", "Glob", "Read"])),
            Some("Edit".into())
        );
        assert_eq!(
            p.predict(&names(&["Bash", "Read", "Edit"])),
            None,
            "trailing [Edit] was never a context"
        );
        // Trigram (order=3 ⇒ ctx_len=2): trailing pair matters.
        let mut p3 = NgramPredictor::with_order(3);
        p3.observe(&names(&["Glob", "Read", "Edit"]));
        assert_eq!(
            p3.predict(&names(&["Bash", "Glob", "Read"])),
            Some("Edit".into())
        );
        // Same trailing pair behind a different prefix still predicts…
        assert_eq!(
            p3.predict(&names(&["X", "Glob", "Read"])),
            Some("Edit".into())
        );
        // …while a truly unseen pair abstains.
        assert_eq!(p3.predict(&names(&["X", "Y", "Read"])), None);
    }

    #[test]
    fn unigram_counts_frequencies() {
        let mut p = NgramPredictor::with_order(1);
        p.observe(&names(&["Read", "Bash", "Read"]));
        assert_eq!(p.predict(&names(&[])), Some("Read".into()));
    }
}

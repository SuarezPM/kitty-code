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

/// F2b P0-a: conversation segment kinds for per-segment budgets.
///
/// `SystemPreflight` (bootstrap/system prompt) and `UserTurn` are PROTECTED
/// (intocables): eviction/compaction must never trim them, and the emergency
/// scale does not apply. All other kinds share a fair slice of the window
/// with a floor of [`MIN_SEGMENT_TOKENS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentKind {
    /// Bootstrap/system preflight message (protected).
    SystemPreflight,
    /// User turn (protected).
    UserTurn,
    /// Assistant message.
    Assistant,
    /// Tool result.
    ToolResult,
    /// `[Memory]` block (lessons + gated episodes).
    Memory,
    /// Knowledge-base injection.
    Knowledge,
    /// `[Context Summary]` block from compaction.
    Summary,
}

impl SegmentKind {
    /// Number of variants (sizes the per-segment accounting array).
    pub const COUNT: usize = 7;

    /// All variants in index order (for tests/diagnostics).
    pub fn all() -> [SegmentKind; SegmentKind::COUNT] {
        use SegmentKind::*;
        [
            SystemPreflight,
            UserTurn,
            Assistant,
            ToolResult,
            Memory,
            Knowledge,
            Summary,
        ]
    }

    /// Array index for this variant.
    fn index(self) -> usize {
        match self {
            SegmentKind::SystemPreflight => 0,
            SegmentKind::UserTurn => 1,
            SegmentKind::Assistant => 2,
            SegmentKind::ToolResult => 3,
            SegmentKind::Memory => 4,
            SegmentKind::Knowledge => 5,
            SegmentKind::Summary => 6,
        }
    }

    /// Protected kinds are never trimmed and skip emergency scaling.
    pub fn protected(self) -> bool {
        matches!(self, SegmentKind::SystemPreflight | SegmentKind::UserTurn)
    }
}

/// F2b P0-a: minimum per-segment budget floor (tokens).
pub const MIN_SEGMENT_TOKENS: usize = 512;

/// F2b P0-a: emergency scale applied to NON-protected segment caps.
pub const EMERGENCY_SCALE: f32 = 0.8;

/// Lightweight snapshot saved before compaction for continuity.
#[derive(Debug, Clone, Default)]
pub struct CompactionSnapshot {
    /// Last N tool results (truncated to 200 chars each).
    pub recent_tool_results: Vec<String>,
    /// File currently being edited (if any).
    pub active_file: Option<String>,
    /// Current benchmark exercise (if any).
    pub bench_exercise: Option<String>,
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
    /// Lightweight snapshot saved before compaction for continuity.
    snapshot: Option<CompactionSnapshot>,
    /// F2b P0-a: cumulative tokens consumed per [`SegmentKind`]
    /// (observability; eviction attribution is P1 `evict_to_budget`).
    segment_used: [usize; SegmentKind::COUNT],
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
            snapshot: None,
            segment_used: [0; SegmentKind::COUNT],
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

    /// F2b P0-a: segment-aware consumption. Global accounting is identical
    /// to [`Self::consume`]; the per-kind counter is cumulative telemetry.
    /// Intended accounting point for `push_message(kind, …)`: callers pass
    /// the `count_tokens` real-token count (never `chars/4` here) — the
    /// estimate fallback already happened inside `count_tokens`.
    /// Wiring `push_message` to this is post-lane (agent/* untouched here).
    pub fn consume_as(&mut self, kind: SegmentKind, tokens: usize) {
        self.consume(tokens);
        let slot = &mut self.segment_used[kind.index()];
        *slot = slot.saturating_add(tokens);
    }

    /// Cumulative tokens consumed by one segment (telemetry, not a cap).
    pub fn segment_used(&self, kind: SegmentKind) -> usize {
        self.segment_used[kind.index()]
    }

    /// F2b P0-a budget table: protected kinds are intocables (whole window,
    /// no effective cap); every other kind gets a fair quarter slice floored
    /// at [`MIN_SEGMENT_TOKENS`] (clamped to the window on tiny budgets).
    pub fn segment_cap(&self, kind: SegmentKind) -> usize {
        if kind.protected() {
            return self.max_tokens;
        }
        let floor = MIN_SEGMENT_TOKENS.min(self.max_tokens);
        (self.max_tokens / 4).clamp(floor, self.max_tokens)
    }

    /// Emergency budget: protected kinds bypass scaling (intocables);
    /// the rest are cut to `cap × [`EMERGENCY_SCALE`]` (a recorte, so it may
    /// dip below the 512 floor by design).
    pub fn emergency_cap(&self, kind: SegmentKind) -> usize {
        let cap = self.segment_cap(kind);
        if kind.protected() {
            return cap;
        }
        ((cap as f32 * EMERGENCY_SCALE) as usize).max(1)
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
        self.segment_used = [0; SegmentKind::COUNT];
    }

    /// Save a snapshot before compaction.
    pub fn save_snapshot(&mut self, snapshot: CompactionSnapshot) {
        self.snapshot = Some(snapshot);
    }

    /// Take the snapshot (returns None if not set, clears it).
    pub fn take_snapshot(&mut self) -> Option<CompactionSnapshot> {
        self.snapshot.take()
    }

    /// Get a reference to the snapshot without consuming it.
    pub fn snapshot(&self) -> Option<&CompactionSnapshot> {
        self.snapshot.as_ref()
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

    /// F2b P0-a: per-segment accounting accumulates without changing the
    /// global total (same saturation semantics as `consume`).
    #[test]
    fn consume_as_tracks_segments_and_global() {
        use SegmentKind::*;
        let mut budget = ContextBudget::new(1000);
        budget.consume_as(UserTurn, 200);
        budget.consume_as(ToolResult, 300);
        budget.consume_as(UserTurn, 100);
        assert_eq!(budget.used(), 600);
        assert_eq!(budget.segment_used(UserTurn), 300);
        assert_eq!(budget.segment_used(ToolResult), 300);
        assert_eq!(budget.segment_used(Assistant), 0);
        // Saturation still caps the global total.
        budget.consume_as(ToolResult, 900);
        assert_eq!(budget.used(), 1000);
        // Reset clears both.
        budget.reset();
        assert_eq!(budget.used(), 0);
        assert_eq!(budget.segment_used(UserTurn), 0);
    }

    /// F2b P0-a: UserTurn y SystemPreflight intocables — protected, cap ==
    /// ventana completa y la emergencia NO los recorta.
    #[test]
    fn protected_kinds_userturn_and_preflight_intact() {
        use SegmentKind::*;
        let budget = ContextBudget::new(16384);
        for kind in [SystemPreflight, UserTurn] {
            assert!(kind.protected(), "{kind:?} debe ser protegido");
            assert_eq!(budget.segment_cap(kind), 16384);
            assert_eq!(budget.emergency_cap(kind), 16384);
        }
        for kind in [Assistant, ToolResult, Memory, Knowledge, Summary] {
            assert!(!kind.protected(), "{kind:?} no debe ser protegido");
            assert!(budget.segment_cap(kind) < budget.max());
            assert!(budget.emergency_cap(kind) < budget.segment_cap(kind));
        }
    }

    /// F2b P0-a: tabla de presupuestos — cuarto justo con suelo de 512 tok.
    #[test]
    fn segment_table_fair_quarter_with_512_floor() {
        use SegmentKind::*;
        let budget = ContextBudget::new(16384);
        assert_eq!(budget.segment_cap(ToolResult), 4096);
        assert!(budget.segment_cap(ToolResult) >= MIN_SEGMENT_TOKENS);
        // Ventana enana: el suelo se sujeta a la ventana, nunca la supera.
        let tiny = ContextBudget::new(100);
        assert_eq!(tiny.segment_cap(ToolResult), 100);
        assert_eq!(tiny.segment_cap(UserTurn), 100);
    }

    /// F2b P0-a: emergencia ×0.8 exacta en no-protegidos.
    #[test]
    fn emergency_cap_is_point_eight_on_unprotected() {
        use SegmentKind::*;
        let budget = ContextBudget::new(16384);
        let cap = budget.segment_cap(ToolResult);
        assert_eq!(cap, 4096);
        assert_eq!(
            budget.emergency_cap(ToolResult),
            (4096.0 * EMERGENCY_SCALE) as usize
        );
        assert_eq!(budget.emergency_cap(ToolResult), 3276);
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

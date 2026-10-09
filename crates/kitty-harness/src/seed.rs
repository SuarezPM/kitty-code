//! Deterministic seeding (W7-code, SDD estado-arte).
//!
//! A tiny self-contained SplitMix64 + FNV-1a mix — no new dependencies.
//! `None` seed = legacy exact behavior everywhere (no jitter, stable-sort
//! load order); `Some(seed)` = deterministic-but-seeded (same seed + same
//! inputs ⇒ same outputs, across runs and machines).
//!
//! Seeds flow from `Agent::with_seed` into retry backoff jitter and
//! knowledge tie-breaks. Goldens con modelo quedan para la ventana serial.

/// Deterministic 64-bit PRNG (SplitMix64).
#[derive(Debug, Clone)]
pub struct Seed {
    state: u64,
}

impl Seed {
    /// Create from an explicit seed value.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next pseudo-random `u64` (SplitMix64).
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Deterministic sub-stream for `salt` (does not advance `self`).
    #[must_use]
    pub fn fork(&self, salt: u64) -> Seed {
        Seed::new(
            self.state
                .wrapping_add(salt.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        )
    }
}

/// Deterministic 64-bit hash of `(seed, text)`: FNV-1a over the seed bytes +
/// text bytes, finished with one SplitMix64 pass. Used for seeded tie-breaks
/// (knowledge card order) — fully determined by inputs, no `RandomState`.
#[must_use]
pub fn mix_str(seed: u64, text: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01B3;
    let mut h = FNV_OFFSET;
    for b in seed
        .to_le_bytes()
        .iter()
        .chain(text.bytes().collect::<Vec<_>>().iter())
    {
        h ^= *b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    let mut s = Seed::new(h);
    s.next_u64()
}

/// LLM retry backoff in ms for `attempt` (0-based).
///
/// - `None` → legacy exact values `500 * 2^attempt` (compat: current
///   behavior bit-for-bit).
/// - `Some(seed)` → deterministic ±20% jitter derived from
///   `seed.fork(attempt)` (same seed ⇒ same backoffs; spreads synchronized
///   retries without nondeterminism).
#[must_use]
pub fn retry_backoff_ms(attempt: u32, seed: Option<u64>) -> u64 {
    let base = 500_u64.saturating_mul(2_u64.saturating_pow(attempt));
    match seed {
        None => base,
        Some(s) => {
            let jitter = Seed::new(s).fork(u64::from(attempt)).next_u64() % 41; // 0..=40
            base.saturating_mul(80 + jitter).saturating_div(100) // 80%..120%
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unseeded_backoff_is_legacy_exact() {
        assert_eq!(
            (0..4)
                .map(|a| retry_backoff_ms(a, None))
                .collect::<Vec<_>>(),
            vec![500, 1000, 2000, 4000]
        );
    }

    #[test]
    fn seeded_backoff_is_deterministic_within_band() {
        let a: Vec<u64> = (0..4).map(|a| retry_backoff_ms(a, Some(42))).collect();
        let b: Vec<u64> = (0..4).map(|a| retry_backoff_ms(a, Some(42))).collect();
        assert_eq!(a, b, "same seed must give same backoffs");
        for (attempt, ms) in a.iter().enumerate() {
            let base = 500_u64 * 2_u64.pow(attempt as u32);
            assert!(
                (base * 80 / 100..=base * 120 / 100).contains(ms),
                "attempt {attempt}: {ms}ms outside ±20% of {base}ms"
            );
        }
        // A different seed may differ (not asserted — only determinism is).
        let _ = (0..4)
            .map(|a| retry_backoff_ms(a, Some(7)))
            .collect::<Vec<_>>();
    }

    #[test]
    fn mix_str_is_deterministic_and_seed_sensitive() {
        assert_eq!(mix_str(1, "abc"), mix_str(1, "abc"));
        assert_ne!(mix_str(1, "abc"), mix_str(2, "abc"));
        assert_ne!(mix_str(1, "abc"), mix_str(1, "abd"));
    }

    #[test]
    fn seed_stream_is_deterministic() {
        let mut a = Seed::new(99);
        let mut b = Seed::new(99);
        for _ in 0..8 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
}

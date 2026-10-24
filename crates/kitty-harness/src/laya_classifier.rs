//! Laya classifier reimplemented without torch (W4).
//!
//! Buckets + ECE + temperature scaling, fitted ONLY on own traffic
//! (`(confidence, correct)` pairs). Zero dependencies (std only).
//!
//! Fail-open contract: until a fit on measured traffic exists, every
//! decision is [`LayaDecision::Proceed`]. There are NO hard gates here —
//! the halting lane may only gate once this module reports a fitted model
//! with measured accuracy. Without ≥ [`MIN_FIT_SAMPLES`] labeled pairs,
//! [`fit`] returns [`FitOutcome::HeldDatos`] with the exact missing count.

/// Labeled pairs required before any calibration is trusted.
pub const MIN_FIT_SAMPLES: usize = 200;

/// Default bucket count for ECE (equal-width over confidence).
pub const DEFAULT_BUCKETS: usize = 10;

/// One traffic observation: predicted confidence + ground-truth outcome.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayaSample {
    /// Predicted probability of being correct, 0..=1.
    pub confidence: f32,
    /// Whether the prediction was actually correct.
    pub correct: bool,
}

impl LayaSample {
    pub fn new(confidence: f32, correct: bool) -> Self {
        Self {
            confidence: confidence.clamp(0.0, 1.0),
            correct,
        }
    }
}

/// Calibration stats for one confidence bucket.
#[derive(Debug, Clone, PartialEq)]
pub struct BucketStat {
    /// Inclusive lower edge, exclusive upper (last bucket includes 1.0).
    pub lo: f32,
    /// Upper edge.
    pub hi: f32,
    /// Samples in this bucket.
    pub n: usize,
    /// Empirical accuracy in this bucket.
    pub accuracy: f32,
    /// Mean predicted confidence in this bucket.
    pub mean_confidence: f32,
}

/// Expected Calibration Error report (plus Max CE for the worst bucket).
#[derive(Debug, Clone, PartialEq)]
pub struct EceReport {
    pub ece: f32,
    pub mce: f32,
    pub n: usize,
    pub buckets: Vec<BucketStat>,
}

/// Equal-width ECE over `n_buckets` (0 → [`DEFAULT_BUCKETS`]). Empty input
/// yields a zeroed report (fail-open: no evidence, no error).
#[must_use]
pub fn expected_calibration_error(samples: &[LayaSample], n_buckets: usize) -> EceReport {
    let nb = if n_buckets == 0 {
        DEFAULT_BUCKETS
    } else {
        n_buckets
    };
    let mut buckets = Vec::with_capacity(nb);
    for i in 0..nb {
        buckets.push(BucketStat {
            lo: i as f32 / nb as f32,
            hi: (i + 1) as f32 / nb as f32,
            n: 0,
            accuracy: 0.0,
            mean_confidence: 0.0,
        });
    }
    let total = samples.len();
    if total == 0 {
        return EceReport {
            ece: 0.0,
            mce: 0.0,
            n: 0,
            buckets,
        };
    }
    let mut acc_sum = vec![0.0f32; nb];
    let mut conf_sum = vec![0.0f32; nb];
    for s in samples {
        let mut idx = (s.confidence * nb as f32).floor() as usize;
        if idx >= nb {
            idx = nb - 1;
        }
        buckets[idx].n += 1;
        acc_sum[idx] += if s.correct { 1.0 } else { 0.0 };
        conf_sum[idx] += s.confidence;
    }
    let mut ece = 0.0f32;
    let mut mce = 0.0f32;
    for (i, b) in buckets.iter_mut().enumerate() {
        if b.n == 0 {
            continue;
        }
        b.accuracy = acc_sum[i] / b.n as f32;
        b.mean_confidence = conf_sum[i] / b.n as f32;
        let gap = (b.accuracy - b.mean_confidence).abs();
        ece += gap * b.n as f32 / total as f32;
        mce = mce.max(gap);
    }
    EceReport {
        ece,
        mce,
        n: total,
        buckets,
    }
}

/// Clamp for logit stability (avoids ±inf at p ∈ {0, 1}).
const EPS: f32 = 1e-6;

fn clamp_p(p: f32) -> f32 {
    p.clamp(EPS, 1.0 - EPS)
}

fn logit(p: f32) -> f32 {
    let p = clamp_p(p);
    (p / (1.0 - p)).ln()
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Binary temperature scaling of a confidence: `sigmoid(logit(p) / t)`.
/// `t == 1` is the identity; `t > 1` softens overconfidence.
#[must_use]
pub fn scale_confidence(p: f32, temperature: f32) -> f32 {
    if !temperature.is_finite() || temperature <= 0.0 {
        return clamp_p(p);
    }
    sigmoid(logit(p) / temperature)
}

fn nll(samples: &[LayaSample], temperature: f32) -> f32 {
    let mut sum = 0.0f32;
    for s in samples {
        let p = scale_confidence(s.confidence, temperature);
        sum += if s.correct { -p.ln() } else { -(1.0 - p).ln() };
    }
    sum / samples.len().max(1) as f32
}

/// Temperature fit result (grid search minimizing binary NLL).
#[derive(Debug, Clone, PartialEq)]
pub struct TempFit {
    pub temperature: f32,
    pub nll_before: f32,
    pub nll_after: f32,
    pub ece_before: f32,
    pub ece_after: f32,
    pub n: usize,
}

/// Fit the temperature on labeled samples. Pure math — works at any `n`,
/// but callers must gate trust via [`fit`] ([`MIN_FIT_SAMPLES`]). Returns
/// `None` on empty input or single-class labels (no signal to fit).
#[must_use]
pub fn fit_temperature(samples: &[LayaSample]) -> Option<TempFit> {
    if samples.is_empty() {
        return None;
    }
    let positives = samples.iter().filter(|s| s.correct).count();
    if positives == 0 || positives == samples.len() {
        return None;
    }
    // Log-spaced grid T ∈ [0.05, 20]: covers sharpening and softening.
    let mut best_t = 1.0f32;
    let mut best_nll = nll(samples, 1.0);
    for i in 0..81 {
        let t = 10.0f32.powf(-1.3 + 2.6 * i as f32 / 80.0);
        let v = nll(samples, t);
        if v < best_nll {
            best_nll = v;
            best_t = t;
        }
    }
    let n_buckets = DEFAULT_BUCKETS;
    let ece_before = expected_calibration_error(samples, n_buckets).ece;
    let scaled: Vec<LayaSample> = samples
        .iter()
        .map(|s| LayaSample::new(scale_confidence(s.confidence, best_t), s.correct))
        .collect();
    let ece_after = expected_calibration_error(&scaled, n_buckets).ece;
    Some(TempFit {
        temperature: best_t,
        nll_before: nll(samples, 1.0),
        nll_after: best_nll,
        ece_before,
        ece_after,
        n: samples.len(),
    })
}

/// Fitted, calibrated classifier. Decisions stay fail-open (see
/// [`CalibratedLaya::decide`]); the numbers exist so a FUTURE lane can gate
/// on measured evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibratedLaya {
    pub temperature: f32,
    /// ECE after scaling, measured on the fit set.
    pub ece: f32,
    /// Fraction correct on the fit set (the "measured precision").
    pub accuracy: f32,
    pub n: usize,
}

impl CalibratedLaya {
    /// Calibrated confidence for a raw model confidence.
    #[must_use]
    pub fn calibrated(&self, p: f32) -> f32 {
        scale_confidence(p, self.temperature)
    }

    /// Fail-open decision: ALWAYS [`LayaDecision::Proceed`] in this lane.
    /// Hard gates need measured precision + their own lane's thresholds —
    /// zero gates without evidence.
    #[must_use]
    pub fn decide(&self, _p: f32) -> LayaDecision {
        LayaDecision::Proceed
    }
}

/// Outcome of [`fit`] on own traffic.
#[derive(Debug, Clone, PartialEq)]
pub enum FitOutcome {
    /// Calibrated model with measured accuracy — evidence exists.
    Fitted(CalibratedLaya),
    /// Not enough labeled `(confidence, correct)` pairs yet. `have` is the
    /// exact usable count observed; `need` is [`MIN_FIT_SAMPLES`].
    HeldDatos { have: usize, need: usize },
}

/// Fail-open decision type. Only `Proceed` exists in this lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayaDecision {
    Proceed,
}

/// Lane C gate: solo [`FitOutcome::Fitted`] habilita Laya; cualquier
/// [`FitOutcome::HeldDatos`] ⇒ Off/Noul (fail-open, decide el SLM).
/// Puro, sin I/O: no calibra nada, solo clasifica el outcome.
#[must_use]
pub fn is_fitted(outcome: &FitOutcome) -> bool {
    matches!(outcome, FitOutcome::Fitted(_))
}

/// Fit on own traffic. Usable pairs = finite confidence + any label.
/// Fewer than [`MIN_FIT_SAMPLES`] (or single-class) → [`FitOutcome::HeldDatos`].
#[must_use]
pub fn fit(samples: &[LayaSample]) -> FitOutcome {
    let usable: Vec<LayaSample> = samples
        .iter()
        .copied()
        .filter(|s| s.confidence.is_finite())
        .collect();
    if usable.len() < MIN_FIT_SAMPLES {
        return FitOutcome::HeldDatos {
            have: usable.len(),
            need: MIN_FIT_SAMPLES,
        };
    }
    match fit_temperature(&usable) {
        Some(t) => {
            let accuracy = usable.iter().filter(|s| s.correct).count() as f32 / usable.len() as f32;
            FitOutcome::Fitted(CalibratedLaya {
                temperature: t.temperature,
                ece: t.ece_after,
                accuracy,
                n: usable.len(),
            })
        }
        None => FitOutcome::HeldDatos {
            have: usable.len(),
            need: MIN_FIT_SAMPLES,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibrated_set(n: usize) -> Vec<LayaSample> {
        // Perfectly calibrated: confidence == empirical rate per level.
        let mut v = Vec::new();
        for i in 0..n {
            let p = 0.1 + 0.8 * (i % 10) as f32 / 9.0;
            // Deterministic pseudo-labels at exactly rate p.
            let correct = (i / 10) % 10 < (p * 10.0) as usize;
            v.push(LayaSample::new(p, correct));
        }
        v
    }

    #[test]
    fn ece_near_zero_when_calibrated() {
        let samples = calibrated_set(2000);
        let r = expected_calibration_error(&samples, 10);
        assert!(r.ece < 0.06, "ece={}", r.ece);
        assert_eq!(r.n, 2000);
        assert_eq!(r.buckets.len(), 10);
    }

    #[test]
    fn ece_large_when_overconfident() {
        // Always predicts 0.9, right only half the time.
        let samples: Vec<LayaSample> = (0..500).map(|i| LayaSample::new(0.9, i % 2 == 0)).collect();
        let r = expected_calibration_error(&samples, 10);
        assert!(r.ece > 0.3, "ece={}", r.ece);
        assert!(r.mce > 0.3, "mce={}", r.mce);
    }

    #[test]
    fn temp_fit_softens_overconfidence() {
        let samples: Vec<LayaSample> = (0..1000)
            .map(|i| LayaSample::new(0.9, i % 2 == 0))
            .collect();
        let t = fit_temperature(&samples).unwrap();
        assert!(t.temperature > 1.0, "T={}", t.temperature);
        assert!(t.nll_after < t.nll_before, "no NLL gain");
        assert!(t.ece_after < t.ece_before, "no ECE gain");
    }

    #[test]
    fn fit_held_when_few_labels() {
        // Own-traffic reality today: zero (confidence, correct) pairs.
        assert_eq!(fit(&[]), FitOutcome::HeldDatos { have: 0, need: 200 });
        let few: Vec<LayaSample> = (0..199).map(|i| LayaSample::new(0.7, i % 2 == 0)).collect();
        assert_eq!(
            fit(&few),
            FitOutcome::HeldDatos {
                have: 199,
                need: 200
            }
        );
    }

    #[test]
    fn fit_single_class_is_held() {
        let one_sided: Vec<LayaSample> = (0..250).map(|_| LayaSample::new(0.8, true)).collect();
        assert!(matches!(fit(&one_sided), FitOutcome::HeldDatos { .. }));
    }

    #[test]
    fn fitted_model_is_fail_open() {
        let samples = calibrated_set(400);
        let FitOutcome::Fitted(m) = fit(&samples) else {
            panic!("expected fit with 400 mixed samples");
        };
        assert_eq!(m.decide(0.01), LayaDecision::Proceed);
        assert_eq!(m.decide(0.99), LayaDecision::Proceed);
        assert!(m.temperature.is_finite() && m.temperature > 0.0);
        assert!((0.0..=1.0).contains(&m.calibrated(0.9)));
        assert!((0.0..=1.0).contains(&m.accuracy));
    }

    #[test]
    fn scale_identity_at_t1_and_clamped() {
        assert!((scale_confidence(0.7, 1.0) - 0.7).abs() < 1e-5);
        assert!((0.0..=1.0).contains(&scale_confidence(0.0, 2.0)));
        assert!((0.0..=1.0).contains(&scale_confidence(1.0, 0.5)));
        // Degenerate temperature falls back to clamped input.
        assert_eq!(scale_confidence(0.7, 0.0), 0.7);
    }
}

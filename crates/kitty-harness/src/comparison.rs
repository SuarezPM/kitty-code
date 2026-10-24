//! Harness comparison table with correctness lift metrics.
//!
//! Capa 0 / F2 (Lane B): SIGNAL/NO-SIGNAL con CI + p99 censored.
//! - `decide_signal`: "NO SIGNAL" si el lift no tiene CI que lo sostenga o
//!   si el run_id no está limpio (sin run_id limpio no hay señal publicable).
//! - `p99_ms_censored`: p99 sobre walls reales; si hubo timeouts el p99 es
//!   `censored=true` (los timeouts truncan la cola y el cuantil miente).

use serde::{Deserialize, Serialize};

/// A single exercise result for comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExerciseResult {
    pub id: String,
    pub lang: String,
    pub passed: bool,
    pub latency_ms: u64,
}

/// Comparison between baseline and candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessComparison {
    pub baseline_name: String,
    pub candidate_name: String,
    pub total_exercises: usize,
    pub baseline_pass_rate: f64,
    pub candidate_pass_rate: f64,
    pub correctness_lift: f64,
    pub baseline_wins: usize,
    pub candidate_wins: usize,
    pub ties: usize,
    pub avg_latency_delta_ms: f64,
}

/// Format comparison as a readable report.
pub fn format_comparison(comp: &HarnessComparison) -> String {
    format!(
        "Baseline  {}\n\
         Candidate  {} ({} exercises)\n\
         Pass rate  {:+.1} pp (candidate {:.1}%, baseline {:.1}%)\n\
         Wins       {} candidate, {} baseline, {} ties\n\
         Latency    {:+.0}ms avg delta",
        comp.baseline_name,
        comp.candidate_name,
        comp.total_exercises,
        comp.correctness_lift,
        comp.candidate_pass_rate,
        comp.baseline_pass_rate,
        comp.candidate_wins,
        comp.baseline_wins,
        comp.ties,
        comp.avg_latency_delta_ms,
    )
}

// ── SIGNAL / NO-SIGNAL + CI (F2) ─────────────────────────────────────

/// Veredicto de señal: solo `Signal` es publicable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Signal {
    Signal,
    NoSignal(String),
}

impl Signal {
    /// `true` solo para señal publicable.
    pub fn is_signal(&self) -> bool {
        matches!(self, Signal::Signal)
    }
}

impl std::fmt::Display for Signal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Signal::Signal => write!(f, "SIGNAL"),
            Signal::NoSignal(reason) => write!(f, "NO-SIGNAL ({reason})"),
        }
    }
}

/// Wilson 95% para una proporción (misma fórmula que `bench/ab.rs`).
pub fn wilson_ci(passed: usize, total: usize) -> (f64, f64) {
    if total == 0 {
        return (0.0, 0.0);
    }
    let n = total as f64;
    let p = passed as f64 / n;
    let z: f64 = 1.96;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let centre = p + z2 / (2.0 * n);
    let inner = (p * (1.0 - p) + z2 / (4.0 * n)) / n;
    let delta = z * inner.sqrt();
    let lo = (centre - delta) / denom;
    let hi = (centre + delta) / denom;
    (lo.clamp(0.0, 1.0), hi.clamp(0.0, 1.0))
}

/// Decide SIGNAL/NO-SIGNAL (F2 + B3):
/// - `ledger_clean == false` ⇒ NO-SIGNAL (sin run_id limpio no hay señal).
/// - `reps < 3` (k reps, NO n tareas) ⇒ siempre NO-SIGNAL: con k=1 Wilson no
///   discrimina aunque n≥3 y el lift sea total (0/8 vs 8/8 con k=1 ⇒
///   NO-SIGNAL, nunca publicable). `n` (totales por brazo) y `k` (reps) son
///   ejes distintos: no pasar tareas como reps.
/// - n<3 por brazo ⇒ NO-SIGNAL (Wilson no discrimina).
/// - Victoria SOLO si el intervalo Wilson del candidato queda estrictamente
///   por encima del baseline (sin solape por arriba); cualquier otro caso ⇒
///   NO-SIGNAL (mismo criterio que `compact_ab_verdict` en `bench/ab.rs`).
pub fn decide_signal(
    baseline_passed: usize,
    baseline_total: usize,
    candidate_passed: usize,
    candidate_total: usize,
    ledger_clean: bool,
    reps: usize,
) -> Signal {
    if !ledger_clean {
        return Signal::NoSignal("run_id dirty: sin ledger limpio no hay señal".to_string());
    }
    // B3: k reps manda sobre n tareas. k=1 ⇒ siempre NoSignal + exit 2.
    if reps < 3 {
        return Signal::NoSignal(
            "k<3 por brazo: Wilson no discrimina (k=1, sin reps)".to_string(),
        );
    }
    if baseline_total < 3 || candidate_total < 3 {
        return Signal::NoSignal("n<3 por brazo: Wilson no discrimina".to_string());
    }
    let (_, baseline_hi) = wilson_ci(baseline_passed, baseline_total);
    let (candidate_lo, _) = wilson_ci(candidate_passed, candidate_total);
    if candidate_lo > baseline_hi {
        Signal::Signal
    } else {
        Signal::NoSignal("CI solapa: lift sin significación Wilson 95%".to_string())
    }
}

// ── p99 censored ─────────────────────────────────────────────────────

/// p99 sobre walls reales + flag censored (F2).
/// `timed_out > 0` ⇒ `censored=true`: el timeout trunca la cola (la celda
/// reporta el cap, no el wall real) y el cuantil no es publicable.
/// Vacío ⇒ `(0.0, censored según timeouts)`.
pub fn p99_ms_censored(mut latencies: Vec<f64>, timed_out: usize) -> (f64, bool) {
    let censored = timed_out > 0;
    if latencies.is_empty() {
        return (0.0, censored);
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((latencies.len() as f64 * 0.99).ceil() as usize)
        .saturating_sub(1)
        .min(latencies.len() - 1);
    (latencies[idx], censored)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_report() {
        let comp = HarnessComparison {
            baseline_name: "v1".into(),
            candidate_name: "v2".into(),
            total_exercises: 100,
            baseline_pass_rate: 50.0,
            candidate_pass_rate: 60.0,
            correctness_lift: 10.0,
            baseline_wins: 5,
            candidate_wins: 15,
            ties: 80,
            avg_latency_delta_ms: -50.0,
        };
        let report = format_comparison(&comp);
        assert!(report.contains("+10.0 pp"));
        assert!(report.contains("15 candidate"));
    }

    #[test]
    fn dirty_ledger_is_no_signal() {
        assert_eq!(
            decide_signal(0, 8, 8, 8, false, 3),
            Signal::NoSignal("run_id dirty: sin ledger limpio no hay señal".to_string())
        );
    }

    #[test]
    fn under_k3_is_no_signal() {
        // reps<3 (k) ⇒ NoSignal aunque n cumpla; n<3 ⇒ NoSignal aunque k cumpla.
        assert!(matches!(decide_signal(2, 2, 2, 2, true, 3), Signal::NoSignal(_)));
        assert!(matches!(decide_signal(0, 0, 0, 0, true, 3), Signal::NoSignal(_)));
        assert!(matches!(decide_signal(2, 3, 2, 3, true, 1), Signal::NoSignal(_)));
    }

    #[test]
    fn overlapping_ci_is_no_signal() {
        // 2/3 vs 2/3 idéntico ⇒ solape total (k=3, n=3).
        assert!(matches!(decide_signal(2, 3, 2, 3, true, 3), Signal::NoSignal(_)));
        // Mejora aparente sin separación (3/3 vs 2/3 con k=3,n=3 se solapa).
        assert!(matches!(decide_signal(2, 3, 3, 3, true, 3), Signal::NoSignal(_)));
    }

    #[test]
    fn clear_win_is_signal() {
        assert_eq!(decide_signal(0, 8, 8, 8, true, 3), Signal::Signal);
        assert!(decide_signal(0, 8, 8, 8, true, 3).is_signal());
        assert_eq!(decide_signal(0, 8, 8, 8, true, 3).to_string(), "SIGNAL");
    }

    // === B3: k=1 siempre NoSignal (n tareas ≠ k reps) ===

    #[test]
    fn k1_siempre_nosignal_aunque_lift_total() {
        // Golden B3: 3/3 vs 0/3 y 0/8 vs 8/8 con k=1 ⇒ NoSignal (exit 2, no 0/1).
        // Sin el param reps estos dos serían 0/1 por Wilson; con k=1 son
        // inconclusos por falta de reps, nunca publicables.
        let s1 = decide_signal(0, 3, 3, 3, true, 1);
        assert!(
            matches!(s1, Signal::NoSignal(_)),
            "3/3 vs 0/3 con k=1 debe ser NoSignal, fue {s1}"
        );
        assert!(s1.to_string().contains("k<3"), "razón k<3, fue {s1}");
        let s2 = decide_signal(0, 8, 8, 8, true, 1);
        assert!(
            matches!(s2, Signal::NoSignal(_)),
            "0/8 vs 8/8 con k=1 debe ser NoSignal, fue {s2}"
        );
        assert!(s2.to_string().contains("k<3"), "razón k<3, fue {s2}");
        // Controles: mismo lift con k=3 sí discrimina por Wilson.
        assert_eq!(decide_signal(0, 8, 8, 8, true, 3), Signal::Signal);
        assert!(matches!(decide_signal(2, 3, 3, 3, true, 3), Signal::NoSignal(_)));
    }

    #[test]
    fn p99_empty_and_single() {
        assert_eq!(p99_ms_censored(vec![], 0), (0.0, false));
        assert_eq!(p99_ms_censored(vec![], 2), (0.0, true));
        assert_eq!(p99_ms_censored(vec![1234.0], 0), (1234.0, false));
    }

    #[test]
    fn p99_golden_1_to_100() {
        let walls: Vec<f64> = (1..=100).map(|v| v as f64).collect();
        assert_eq!(p99_ms_censored(walls, 0), (99.0, false));
    }

    #[test]
    fn p99_with_timeouts_is_censored() {
        let walls: Vec<f64> = (1..=100).map(|v| v as f64).collect();
        let (p99, censored) = p99_ms_censored(walls, 1);
        assert_eq!(p99, 99.0);
        assert!(censored, "con timeouts el p99 debe marcarse censored");
    }
}

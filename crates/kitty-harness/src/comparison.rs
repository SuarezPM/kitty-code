//! Harness comparison table with correctness lift metrics.

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
}

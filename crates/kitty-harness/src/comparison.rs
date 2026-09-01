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

/// Compare two harness results.
pub fn compare_harnesses(
    baseline_name: &str,
    candidate_name: &str,
    baseline: &[ExerciseResult],
    candidate: &[ExerciseResult],
) -> HarnessComparison {
    let baseline_passed = baseline.iter().filter(|r| r.passed).count();
    let candidate_passed = candidate.iter().filter(|r| r.passed).count();

    let baseline_rate = if baseline.is_empty() {
        0.0
    } else {
        baseline_passed as f64 / baseline.len() as f64 * 100.0
    };
    let candidate_rate = if candidate.is_empty() {
        0.0
    } else {
        candidate_passed as f64 / candidate.len() as f64 * 100.0
    };

    // Pair results by exercise ID
    let mut baseline_wins = 0;
    let mut candidate_wins = 0;
    let mut ties = 0;
    let mut latency_delta_sum = 0.0;
    let mut paired_count = 0;

    for b in baseline {
        if let Some(c) = candidate.iter().find(|c| c.id == b.id) {
            paired_count += 1;
            latency_delta_sum += c.latency_ms as f64 - b.latency_ms as f64;

            match (b.passed, c.passed) {
                (true, false) => baseline_wins += 1,
                (false, true) => candidate_wins += 1,
                (true, true) | (false, false) => ties += 1,
            }
        }
    }

    let avg_latency_delta = if paired_count > 0 {
        latency_delta_sum / paired_count as f64
    } else {
        0.0
    };

    HarnessComparison {
        baseline_name: baseline_name.to_string(),
        candidate_name: candidate_name.to_string(),
        total_exercises: baseline.len().max(candidate.len()),
        baseline_pass_rate: baseline_rate,
        candidate_pass_rate: candidate_rate,
        correctness_lift: candidate_rate - baseline_rate,
        baseline_wins,
        candidate_wins,
        ties,
        avg_latency_delta_ms: avg_latency_delta,
    }
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
    fn compare_equal_results() {
        let results = vec![
            ExerciseResult {
                id: "1".into(),
                lang: "python".into(),
                passed: true,
                latency_ms: 100,
            },
            ExerciseResult {
                id: "2".into(),
                lang: "python".into(),
                passed: false,
                latency_ms: 200,
            },
        ];
        let comp = compare_harnesses("baseline", "candidate", &results, &results);
        assert_eq!(comp.correctness_lift, 0.0);
        assert_eq!(comp.ties, 2);
    }

    #[test]
    fn compare_candidate_wins() {
        let baseline = vec![ExerciseResult {
            id: "1".into(),
            lang: "python".into(),
            passed: false,
            latency_ms: 100,
        }];
        let candidate = vec![ExerciseResult {
            id: "1".into(),
            lang: "python".into(),
            passed: true,
            latency_ms: 80,
        }];
        let comp = compare_harnesses("baseline", "candidate", &baseline, &candidate);
        assert_eq!(comp.correctness_lift, 100.0);
        assert_eq!(comp.candidate_wins, 1);
    }

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

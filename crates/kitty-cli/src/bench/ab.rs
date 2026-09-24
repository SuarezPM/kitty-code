//! A/B thinking-budget benchmark helpers: budget parsing, interleaved plan
//! building (deterministic seed 42), per-budget summaries, Wilson CI, and
//! median latency.

pub(crate) use crate::bench::common::BenchResult;
use anyhow::Context;

pub(crate) fn parse_thinking_budgets(s: &str) -> anyhow::Result<Vec<usize>> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        anyhow::bail!("thinking budgets string empty");
    }
    let mut out = Vec::new();
    for part in trimmed.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        let v: usize = p
            .parse()
            .with_context(|| format!("invalid thinking budget: {p}"))?;
        if v == 0 {
            anyhow::bail!("thinking budget must be > 0: {v}");
        }
        out.push(v);
    }
    if out.is_empty() {
        anyhow::bail!("no valid budgets parsed from: {s}");
    }
    Ok(out)
}

pub(crate) fn build_ab_plan(
    limit: usize,
    budgets: &[usize],
    reps: usize,
    interleave: bool,
) -> Vec<(usize, usize)> {
    let mut plan = Vec::new();
    let reps = reps.max(1);
    for &budget in budgets {
        for task_idx in 0..limit {
            for _ in 0..reps {
                plan.push((task_idx, budget));
            }
        }
    }
    if interleave {
        use rand::rngs::StdRng;
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);
        plan.shuffle(&mut rng);
    }
    plan
}

/// Aggregate one budget arm's results into its `per_budget` JSON entry
/// (G4-FIX1: the runbook's central errored-trials gate needs serialized
/// error counts, which the output previously lacked).
///
/// Verdict policy, documented: `"error"` and `"timeout"` are counted in
/// SEPARATE keys — `errored` (in-run failures) vs `timed_out` (task-level
/// harness-limit outcomes, a different failure class). The README gate
/// applies to `errored`; consumers apply their own timeout policy. All
/// pre-existing fields are preserved untouched for backward compatibility.
pub(crate) fn budget_summary(budget: usize, results: &[BenchResult]) -> serde_json::Value {
    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();
    let rate = if total > 0 {
        100.0 * passed as f64 / total as f64
    } else {
        0.0
    };
    let (lo, hi) = wilson_ci(passed, total);
    let p50 = p50_ms(results.iter().map(|r| r.latency_ms).collect());
    let errored = results.iter().filter(|r| r.verdict == "error").count();
    let timed_out = results.iter().filter(|r| r.verdict == "timeout").count();
    serde_json::json!({
        "budget": budget,
        "passed": passed,
        "total": total,
        "rate": rate,
        "wilson_ci": [lo, hi],
        "p50_ms": p50,
        "errored": errored,
        "timed_out": timed_out,
    })
}

pub(crate) fn wilson_ci(passed: usize, total: usize) -> (f64, f64) {
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

pub(crate) fn p50_ms(mut latencies: Vec<f64>) -> f64 {
    if latencies.is_empty() {
        return 0.0;
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = latencies.len();
    if n % 2 == 1 {
        latencies[n / 2]
    } else {
        (latencies[n / 2 - 1] + latencies[n / 2]) / 2.0
    }
}
#[cfg(test)]
mod bench_ab_tests {
    use super::super::common::resolve_exercise_timeout_secs;
    use super::BenchResult;
    use super::{budget_summary, build_ab_plan, parse_thinking_budgets, wilson_ci};
    use crate::{Cli, Commands};
    use clap::Parser;
    use kitty_harness::agent::{Agent, AgentConfig};
    use kitty_harness::ModelProfile;

    #[test]
    fn test_bench_ab_parses() {
        let cli = Cli::try_parse_from([
            "kitty",
            "bench-ab",
            "--thinking",
            "1024,2048,4096",
            "--limit",
            "50",
        ])
        .expect("parse bench-ab");
        match cli.command {
            Commands::BenchAb {
                thinking, limit, ..
            } => {
                assert_eq!(limit, 50);
                let budgets = parse_thinking_budgets(&thinking).expect("parse budgets");
                assert_eq!(budgets, vec![1024, 2048, 4096]);
            }
            _ => panic!("expected BenchAb"),
        }
    }

    /// G3-FIX1 AC (Fase 2.1 form): the bench threshold override derives from
    /// the agent's EFFECTIVE window via the RunOverrides path — a
    /// tiny-profile run WITHOUT an explicit --context-window keeps AD5's
    /// small-window protection (0.62 exactly).
    fn dummy_response(content: &str) -> kitty_runtime::ChatResponse {
        kitty_runtime::ChatResponse {
            model: "test-model".into(),
            message: kitty_runtime::Message::assistant(content),
            done: true,
            eval_count: None,
            eval_duration: None,
            prompt_eval_count: None,
        }
    }

    #[tokio::test]
    async fn bench_threshold_override_derives_from_effective_window() {
        use kitty_harness::agent::RunOverrides;
        use kitty_runtime::MockClient;

        // Tiny preset ⇒ effective window 4096; NO --context-window override.
        let mut agent = Agent::new(
            MockClient::new(vec![dummy_response("ok"), dummy_response("final")]),
            AgentConfig::default().with_profile("tiny"),
        );
        assert_eq!(agent.config().context_window, 4096);
        agent
            .run_with(
                "probe",
                RunOverrides {
                    max_iterations: Some(1),
                    compact_threshold: Some(ModelProfile::tiny().compact_threshold),
                    ..RunOverrides::default()
                },
            )
            .await
            .expect("mock run completes");
        assert!(
            (agent.budget().threshold() - 0.62).abs() < 1e-6,
            "tiny bench agent must keep derived 0.62, got {}",
            agent.budget().threshold()
        );

        // Control: medium at its default window stays the profile value.
        let mut agent = Agent::new(
            MockClient::new(vec![dummy_response("ok"), dummy_response("final")]),
            AgentConfig::default().with_profile("medium"),
        );
        agent
            .run_with(
                "probe",
                RunOverrides {
                    max_iterations: Some(1),
                    compact_threshold: Some(ModelProfile::medium().compact_threshold),
                    ..RunOverrides::default()
                },
            )
            .await
            .expect("mock run completes");
        assert!((agent.budget().threshold() - 0.8).abs() < 1e-6);

        // Explicit reduced override derives against THAT window.
        let mut agent = Agent::new(
            MockClient::new(vec![dummy_response("ok"), dummy_response("final")]),
            AgentConfig::default().with_profile("medium"),
        );
        agent
            .run_with(
                "probe",
                RunOverrides {
                    max_iterations: Some(1),
                    context_window: Some(4096),
                    compact_threshold: Some(ModelProfile::medium().compact_threshold),
                    ..RunOverrides::default()
                },
            )
            .await
            .expect("mock run completes");
        assert!((agent.budget().threshold() - 0.62).abs() < 1e-6);
    }

    /// f9 calibration AC: per-exercise timeout resolution honors
    /// flag > env > default, with invalid env falling back to the default.
    #[test]
    fn resolve_exercise_timeout_flag_wins_over_env_over_default() {
        const DEFAULT: u64 = crate::bench::common::DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS;

        // Explicit flag wins over ANY env (valid or garbage).
        let (got, invalid) = resolve_exercise_timeout_secs(Some(900), Some("60"));
        assert_eq!(got, 900);
        assert!(!invalid);
        let (got, invalid) = resolve_exercise_timeout_secs(Some(900), Some("garbage"));
        assert_eq!(got, 900);
        assert!(!invalid);

        // No flag + valid env (trimmed) ⇒ env value.
        let (got, invalid) = resolve_exercise_timeout_secs(None, Some("900"));
        assert_eq!(got, 900);
        assert!(!invalid);
        let (got, invalid) = resolve_exercise_timeout_secs(None, Some(" 600 "));
        assert_eq!(got, 600);
        assert!(!invalid);

        // No flag + INVALID env (unparsable, empty, zero, negative) ⇒
        // default + invalid marker so the caller warns once.
        for bad in ["abc", "", "   ", "0", "-5", "12.5"] {
            let (got, invalid) = resolve_exercise_timeout_secs(None, Some(bad));
            assert_eq!(got, DEFAULT, "bad env {bad:?} must fall back to default");
            assert!(invalid, "bad env {bad:?} must be flagged invalid");
        }

        // Neither flag nor env ⇒ default, not invalid.
        let (got, invalid) = resolve_exercise_timeout_secs(None, None);
        assert_eq!(got, DEFAULT);
        assert!(!invalid);
    }

    /// f8 T1.3 AC: `--hops` propagates; absent flag ⇒ 0.
    #[test]
    fn test_bench_hops_flag_propagation() {
        // Explicit --hops 3 propagates.
        let cli = Cli::try_parse_from(["kitty", "bench-ab", "--hops", "3"])
            .expect("parse bench-ab --hops 3");
        match cli.command {
            Commands::BenchAb { hops, .. } => assert_eq!(hops, 3),
            _ => panic!("expected BenchAb"),
        }
        // Absent flag defaults to 0 (byte-identical legacy behavior).
        let cli = Cli::try_parse_from(["kitty", "bench-ab"]).expect("parse bench-ab default");
        match cli.command {
            Commands::BenchAb { hops, .. } => assert_eq!(hops, 0),
            _ => panic!("expected BenchAb"),
        }
        // Same on the Bench subcommand.
        let cli =
            Cli::try_parse_from(["kitty", "bench", "--hops", "2"]).expect("parse bench --hops 2");
        match cli.command {
            Commands::Bench { hops, .. } => assert_eq!(hops, 2),
            _ => panic!("expected Bench"),
        }
    }

    #[test]
    fn test_bench_ab_interleave() {
        let budgets = vec![1024, 2048];
        let plan = build_ab_plan(2, &budgets, 1, true);
        assert_eq!(plan.len(), 4);
        let plan_no_shuffle = build_ab_plan(2, &budgets, 1, false);
        assert_ne!(plan, plan_no_shuffle, "interleaved should be shuffled");
        let plan2 = build_ab_plan(2, &budgets, 1, true);
        assert_eq!(plan, plan2, "shuffled with seed 42 should be deterministic");
        let t4096 = wilson_ci(25, 50);
        assert!(
            t4096.0 < 0.5 && t4096.1 > 0.5,
            "wilson ci should surround p=0.5"
        );
    }

    /// Golden de rand 0.10; StdRng NO es estable entre majors — este test
    /// detecta drift, no corrige historia (si cambia el major, actualizar el
    /// golden tras revisión consciente, nunca a ciegas).
    #[test]
    fn seed_42_interleave_golden() {
        let plan = build_ab_plan(2, &[1024, 2048], 1, true);
        assert_eq!(
            plan,
            vec![(1, 1024), (0, 1024), (1, 2048), (0, 2048)],
            "seed-42 interleave drift: revisar bump de rand antes de tocar el golden"
        );
    }

    /// G4-FIX1: mixed-verdict results must serialize `errored` and
    /// `timed_out` counts per budget (the runbook gate reads these), with
    /// all pre-existing fields preserved untouched.
    #[test]
    fn budget_summary_serializes_errored_and_timed_out_counts() {
        fn mk(id: &str, passed: bool, verdict: &str) -> BenchResult {
            BenchResult {
                id: id.into(),
                lang: "python".into(),
                passed,
                verdict: verdict.into(),
                latency_ms: 100.0,
            }
        }
        // Mixed fixture: 2 pass + 1 fail + 2 error + 1 timeout.
        let results = vec![
            mk("a", true, "pass"),
            mk("b", true, "pass"),
            mk("c", false, "fail"),
            mk("d", false, "error"),
            mk("e", false, "error"),
            mk("f", false, "timeout"),
        ];
        let entry = budget_summary(2048, &results);

        // New counters: error and timeout counted SEPARATELY.
        assert_eq!(entry["errored"], serde_json::json!(2));
        assert_eq!(entry["timed_out"], serde_json::json!(1));

        // Backward compatibility: existing fields untouched.
        assert_eq!(entry["budget"], serde_json::json!(2048));
        assert_eq!(entry["passed"], serde_json::json!(2));
        assert_eq!(entry["total"], serde_json::json!(6));
        assert!((entry["rate"].as_f64().unwrap() - 100.0 * 2.0 / 6.0).abs() < 1e-9);
        assert!(entry.get("wilson_ci").is_some());
        assert!(entry.get("p50_ms").is_some());

        // Empty arm: counters are zero, not absent.
        let empty = budget_summary(1024, &[]);
        assert_eq!(empty["errored"], serde_json::json!(0));
        assert_eq!(empty["timed_out"], serde_json::json!(0));
    }
}

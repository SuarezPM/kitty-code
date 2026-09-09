//! Benchmark command dispatch — the `bench` and `bench-ab` subcommand
//! implementations live here so `main.rs` stays a thin CLI parsing layer
//! (audit 2026-09 Fase 1.1).

pub mod ab;
pub mod claweval;
pub mod common;
pub mod gpqa;
pub mod polyglot;
pub mod run;
pub mod tb2;

pub(crate) use ab::{budget_summary, build_ab_plan, parse_thinking_budgets};
pub(crate) use common::{
    bench_exit_code, exercises, is_small_model, json_to_bench_item, load_json_exercises,
    resolve_timeout_from_flag, tally_by_language, BenchConfig, BenchItem, BenchResult,
    DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS,
};
pub(crate) use gpqa::run_gpqa_suite;
pub(crate) use run::run_bench;

use crate::default_db_path;
use anyhow::{Context, Result};
use kitty_harness::agent::clamp_autorun_hops;
use kitty_harness::ModelProfile;
use kitty_runtime::OllamaClient;
use std::path::PathBuf;
use tracing::warn;

/// CLI-level settings the bench commands need, cloned field-by-field from
/// `Cli` before the subcommand payload is destructured (avoids borrowing the
/// whole `Cli` after a partial move).
pub struct BenchCliCtx {
    pub url: String,
    pub model: String,
    pub thinking_budget: usize,
    pub knowledge: bool,
    pub knowledge_dir: String,
    pub context_window: usize,
    pub max_iterations: usize,
    pub bash_timeout: u64,
    pub profile: String,
}

/// Destructured `Commands::Bench` arguments.
pub struct BenchArgs {
    pub suite: String,
    pub limit: usize,
    pub hops: usize,
    pub timeout_secs: u64,
    pub out: PathBuf,
    pub exercises: Option<PathBuf>,
    pub pre_registration: Option<PathBuf>,
    pub no_memory: bool,
    pub claweval_trials: usize,
    pub claweval_parallel: usize,
    pub claweval_sandbox: bool,
}

/// Destructured `Commands::BenchAb` arguments.
pub struct BenchAbArgs {
    pub thinking: String,
    pub limit: usize,
    pub hops: usize,
    pub timeout_secs: u64,
    pub reps: usize,
    pub suite: String,
    pub out: PathBuf,
    pub interleave: bool,
}

/// `Commands::Bench` implementation (moved verbatim from `main.rs`).
pub async fn bench_command(cli: BenchCliCtx, client: OllamaClient, args: BenchArgs) -> Result<()> {
    let BenchArgs {
        suite,
        limit,
        hops,
        timeout_secs,
        out,
        exercises: exercises_path,
        pre_registration,
        no_memory,
        claweval_trials,
        claweval_parallel,
        claweval_sandbox,
    } = args;

    // Pre-bench CWD contamination guard (Gate 2 hardening):
    // leftover solution.py in CWD contaminates grading (agent may write to CWD
    // instead of tmpdir). Warn and remove stale files if present.
    for stale in ["./solution.py", "./solution.js", "./solution.sh"] {
        if std::path::Path::new(stale).exists() {
            warn!("{stale} in CWD will contaminate bench tmpdir; removing stale file");
            let _ = std::fs::remove_file(stale);
        }
    }

    // Ensure memory DB directory exists
    if !no_memory {
        let db_path = default_db_path();
        if let Some(db_dir) = std::path::Path::new(&db_path).parent() {
            std::fs::create_dir_all(db_dir).ok();
        }
    }

    // Gate 2 tuning: bench uses max_iterations=5 for sub-20s smoke (default 20 remains for `run`).
    // If CLI left default 20, bench overrides to 5; explicit CLI value is respected.
    let bench_max_iterations = if cli.max_iterations == 20 {
        5
    } else {
        cli.max_iterations
    };
    let small_model = is_small_model(&cli.profile, &cli.model);
    let effective_no_memory = no_memory || small_model;
    if small_model && !no_memory {
        warn!(model = %cli.model, "small model detected — auto-disabling episodic memory for bench (use explicit --no-memory to silence)");
    }
    // f9 calibration: per-exercise timeout — flag > env > default.
    let (exercise_timeout_secs, env_invalid) = resolve_timeout_from_flag(timeout_secs);
    if env_invalid {
        warn!("per-exercise timeout falling back to {DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS}s");
    }

    let cfg = BenchConfig {
        client,
        model: cli.model.clone(),
        limit,
        exercise_timeout_secs,
        use_memory: !effective_no_memory,
        thinking_budget: cli.thinking_budget,
        context_window: cli.context_window,
        max_iterations: bench_max_iterations,
        autorun_hops: clamp_autorun_hops(hops),
        bash_timeout: cli.bash_timeout,
        profile: ModelProfile::from_cli(&cli.profile, &cli.model),
        enable_knowledge: cli.knowledge,
        knowledge_dir: PathBuf::from(cli.knowledge_dir.clone()),
    };

    // --- GPQA Diamond local approximation (F1: offline-first, gated, pre-registro explícito G1) ---
    if suite == "gpqa" || suite == "gpqa_diamond" {
        return run_gpqa_suite(&cfg, &out, limit, pre_registration).await;
    }

    // --- ClawEval integration (F2: offline-first, local approximation) ---
    if suite == "claweval" {
        return claweval::run_claweval_suite(
            &cli.model,
            &out,
            claweval_trials,
            claweval_parallel,
            claweval_sandbox,
        )
        .await;
    }

    // --- TB2 Official (Harbor CLI wrapper for terminal-bench-2.1) ---
    if suite == "tb2-official" {
        return tb2::run_tb2_official(&cli.model, &cli.url, &out, limit).await;
    }

    // --- Polyglot unit test grading (Aider-style 225 Exercism exercises) ---
    if suite == "polyglot" {
        return polyglot::run_polyglot_bench(&cfg, &out, limit).await;
    }

    println!("Kitty Code Benchmark");
    println!("  Model: {}", cli.model);
    println!("  Suite: {} (limit: {})", suite, limit);
    println!(
        "  Memory: {}",
        if effective_no_memory { "off" } else { "on" }
    );
    println!(
        "  Knowledge: {}",
        if cli.knowledge {
            format!("on ({})", cli.knowledge_dir)
        } else {
            "off (use --knowledge to enable, 5 cards)".to_string()
        }
    );
    if bench_max_iterations != cli.max_iterations {
        println!(
            "  Max iterations: {} (bench override, CLI default 20)",
            bench_max_iterations
        );
    } else {
        println!("  Max iterations: {}", bench_max_iterations);
    }
    println!("  Thinking budget: {}", cli.thinking_budget);
    println!("  Context window: {}", cli.context_window);
    if let Some(ref path) = exercises_path {
        println!("  Exercises JSON: {}", path.display());
    }
    println!();

    // If --exercises provided, load from JSON; otherwise use built-in
    let results = if let Some(ref path) = exercises_path {
        let json_exercises = load_json_exercises(path)?;
        let bench_items: Vec<BenchItem> = json_exercises
            .iter()
            .filter_map(json_to_bench_item)
            .collect();
        if bench_items.is_empty() {
            anyhow::bail!("No valid exercises found in {}", path.display());
        }
        println!("  Loaded {} exercises from JSON", bench_items.len());
        run_bench(&bench_items, &cfg).await?
    } else {
        match suite.as_str() {
            "tb2" => run_bench(&run::tb2_exercises(), &cfg).await?,
            _ => run_bench(&exercises(), &cfg).await?,
        }
    };

    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();
    let rate = if total > 0 {
        100.0 * passed as f64 / total as f64
    } else {
        0.0
    };

    println!();
    println!("RESULTS");
    println!("Passed: {}/{} ({:.1}%)", passed, total, rate);

    // Per-language breakdown: language parsed from polyglot task-name
    // prefixes (polyglot_python_* → python), label fallback otherwise.
    let by_lang = tally_by_language(&results);
    for (lang, t) in &by_lang {
        let lang_rate = if t.attempted > 0 {
            100.0 * t.passed as f64 / t.attempted as f64
        } else {
            0.0
        };
        println!(
            "  {lang}: attempted={} passed={} failed={} errored={} ({:.1}%)",
            t.attempted, t.passed, t.failed, t.errored, lang_rate
        );
    }

    let gate = 50.0;
    if rate >= gate {
        println!();
        println!("GATE PASSED: {:.1}% >= {:.0}%", rate, gate);
    } else {
        println!();
        println!("GATE FAILED: {:.1}% < {:.0}%", rate, gate);
    }

    let output = serde_json::json!({
        "model": cli.model,
        "suite": suite,
        "total": total,
        "passed": passed,
        "rate_pct": rate,
        "memory": cfg.use_memory,
        "by_language": by_lang.iter().map(|(lang, t)| (lang.clone(), serde_json::json!({
            "attempted": t.attempted,
            "passed": t.passed,
            "failed": t.failed,
            "errored": t.errored,
        }))).collect::<std::collections::BTreeMap<String, serde_json::Value>>(),
        "details": results.iter().map(|r| serde_json::json!({
            "id": r.id,
            "lang": r.lang,
            "passed": r.passed,
            "verdict": r.verdict,
            "latency_ms": r.latency_ms,
        })).collect::<Vec<_>>(),
    });
    std::fs::write(&out, serde_json::to_string_pretty(&output)?)?;
    println!("Results saved to {}", out.display());
    // bench-reproducibility: exit-code 0=pass/improved, 1=gate failed, 2=noise (no significant change)
    let baseline_rate: Option<f64> = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../benches/baseline.json"),
    )
    .ok()
    .and_then(|c| serde_json::from_str::<serde_json::Value>(&c).ok())
    .and_then(|v| v.get("rate_pct").and_then(|r| r.as_f64()));
    let exit_code = bench_exit_code(rate, gate, baseline_rate);
    std::process::exit(exit_code);
}

/// `Commands::BenchAb` implementation (moved verbatim from `main.rs`).
pub async fn bench_ab_command(
    cli: BenchCliCtx,
    client: OllamaClient,
    args: BenchAbArgs,
) -> Result<()> {
    let BenchAbArgs {
        thinking,
        limit,
        hops,
        timeout_secs,
        reps,
        suite,
        out,
        interleave,
    } = args;

    let budgets = parse_thinking_budgets(&thinking)?;
    let plan = build_ab_plan(limit, &budgets, reps, interleave);

    // Same tuning as `bench`: 5 iterations for sub-20s smoke unless CLI overrides.
    let bench_max_iterations = if cli.max_iterations == 20 {
        5
    } else {
        cli.max_iterations
    };
    let small_model = is_small_model(&cli.profile, &cli.model);
    let effective_no_memory = small_model;
    if small_model {
        warn!(model = %cli.model, "small model detected — auto-disabling episodic memory for bench-ab");
    }
    // f9 calibration: per-exercise timeout — flag > env > default.
    let (exercise_timeout_secs, env_invalid) = resolve_timeout_from_flag(timeout_secs);
    if env_invalid {
        warn!("per-exercise timeout falling back to {DEFAULT_BENCH_EXERCISE_TIMEOUT_SECS}s");
    }

    let base_cfg = BenchConfig {
        client,
        model: cli.model.clone(),
        limit,
        exercise_timeout_secs,
        use_memory: !effective_no_memory,
        thinking_budget: 2048, // overridden per run below
        context_window: cli.context_window,
        max_iterations: bench_max_iterations,
        autorun_hops: clamp_autorun_hops(hops),
        bash_timeout: cli.bash_timeout,
        profile: ModelProfile::from_cli(&cli.profile, &cli.model),
        enable_knowledge: cli.knowledge,
        knowledge_dir: PathBuf::from(cli.knowledge_dir.clone()),
    };

    let items: Vec<BenchItem> = match suite.as_str() {
        "tb2" => run::tb2_exercises(),
        _ => exercises(),
    };

    let mut store = if base_cfg.use_memory {
        // Blocking rusqlite open runs on the blocking pool so the async
        // executor never parks on SQLite.
        let path = default_db_path();
        Some(
            tokio::task::spawn_blocking(move || kitty_memory::EpisodeStore::open(&path))
                .await
                .context("episode store open task failed")??,
        )
    } else {
        None
    };

    println!("Kitty Code A/B Benchmark");
    println!("  Model: {}", cli.model);
    println!("  Suite: {} (limit: {}, reps: {})", suite, limit, reps);
    println!("  Budgets: {:?}", budgets);
    println!("  Interleave: {}", interleave);
    println!("  Runs: {}", plan.len());
    println!();

    let mut per_budget: std::collections::HashMap<usize, Vec<BenchResult>> =
        std::collections::HashMap::new();
    for (i, (task_idx, budget)) in plan.iter().enumerate() {
        let ex = &items[*task_idx % items.len()];
        let mut cfg = base_cfg.clone();
        cfg.thinking_budget = *budget;
        print!(
            "[{:2}/{:2}] budget={:4} task={:2} ",
            i + 1,
            plan.len(),
            budget,
            task_idx
        );
        let r = run::run_one_bench(ex, &cfg, &mut store).await?;
        per_budget.entry(*budget).or_default().push(r);
    }

    let mut budgets_sorted: Vec<usize> = per_budget.keys().copied().collect();
    budgets_sorted.sort_unstable();

    let mut per_budget_out = Vec::new();
    for budget in &budgets_sorted {
        per_budget_out.push(budget_summary(*budget, &per_budget[budget]));
    }

    let mut pairwise = Vec::new();
    for i in 0..budgets_sorted.len() {
        for j in (i + 1)..budgets_sorted.len() {
            let a = &per_budget_out[i];
            let b = &per_budget_out[j];
            let a_rate = a["rate"].as_f64().unwrap_or(0.0);
            let b_rate = b["rate"].as_f64().unwrap_or(0.0);
            let winner = if a_rate >= b_rate {
                a["budget"].clone()
            } else {
                b["budget"].clone()
            };
            pairwise.push(serde_json::json!({
                "a": a["budget"],
                "b": b["budget"],
                "a_rate": a_rate,
                "b_rate": b_rate,
                "winner": winner,
            }));
        }
    }

    println!();
    println!("RESULTS");
    for entry in &per_budget_out {
        let b = entry["budget"].as_u64().unwrap_or(0);
        println!(
            "  budget={:4}  {}/{} ({:.1}%)  wilson95=[{:.1}%, {:.1}%]  p50={:.0}ms",
            b,
            entry["passed"].as_u64().unwrap_or(0),
            entry["total"].as_u64().unwrap_or(0),
            entry["rate"].as_f64().unwrap_or(0.0),
            entry["wilson_ci"][0].as_f64().unwrap_or(0.0) * 100.0,
            entry["wilson_ci"][1].as_f64().unwrap_or(0.0) * 100.0,
            entry["p50_ms"].as_f64().unwrap_or(0.0),
        );
    }

    let output = serde_json::json!({
        "model": cli.model,
        "suite": suite,
        "limit": limit,
        "reps": reps,
        "interleave": interleave,
        "budgets": budgets_sorted,
        "per_budget": per_budget_out,
        "pairwise": pairwise,
    });
    std::fs::write(&out, serde_json::to_string_pretty(&output)?)?;
    println!("Results saved to {}", out.display());
    Ok(())
}

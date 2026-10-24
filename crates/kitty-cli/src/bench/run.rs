//! Benchmark drivers: shared exercise runner, per-exercise execution, and
//! the built-in TB2 exercise set.

use crate::bench::common::{
    bench_file, file_contains, file_is_executable, list_files, BenchConfig, BenchItem, BenchResult,
};
use crate::default_db_path;
use anyhow::{Context, Result};
use kitty_harness::agent::{AgentConfig, RunOverrides};
use kitty_harness::knowledge::KnowledgeConfig;
use kitty_harness::ThinkingMode;
use kitty_memory::{Episode, EpisodeStore};
use std::time::Instant;

/// Shared benchmark driver used by every exercise set.
pub(crate) async fn run_bench(items: &[BenchItem], cfg: &BenchConfig) -> Result<Vec<BenchResult>> {
    let mut results = Vec::new();

    let mut store = if cfg.use_memory {
        // Blocking rusqlite open runs on the blocking pool so the async
        // executor never parks on SQLite.
        let path = default_db_path();
        Some(
            tokio::task::spawn_blocking(move || EpisodeStore::open(&path))
                .await
                .context("episode store open task failed")??,
        )
    } else {
        None
    };

    for (i, ex) in items.iter().take(cfg.limit).enumerate() {
        print!("[{:2}/{:2}] ", i + 1, cfg.limit.min(items.len()));
        results.push(run_one_bench(ex, cfg, &mut store).await?);
    }

    Ok(results)
}

pub(crate) async fn run_one_bench(
    ex: &BenchItem,
    cfg: &BenchConfig,
    store: &mut Option<EpisodeStore>,
) -> Result<BenchResult> {
    print!("{:20}...", ex.id);

    // Recall on the blocking pool (take/restore, best-effort).
    // Pool failure ⇒ no-memory-for-this-exercise.
    if let Some(mem) = store.take() {
        let goal = ex.prompt;
        let (back, recalled) = tokio::task::spawn_blocking(move || {
            let result = mem.recall_similar(goal, 3);
            (Some(mem), result)
        })
        .await
        .unwrap_or((None, Ok(Vec::new())));
        if let Some(mem) = back {
            *store = Some(mem);
        }
        let past = recalled.unwrap_or_default();
        if !past.is_empty() {
            print!(" ({} past)", past.len());
        }
    }

    let tmpdir = tempfile::tempdir().context("Failed to create temp dir")?;
    let tmpdir_str = tmpdir.path().to_string_lossy().to_string();

    // Prompt hardening: use real check_file per exercise (fix .py forced for js/ts/rust/go)
    // Gate 2 absolute path prevents CWD contamination.
    let target_file = if ex.check_file.is_empty() {
        "solution.py"
    } else {
        ex.check_file
    };
    let prompt = format!(
        "{}\n\nWork in directory: {} — Save to {}/{} (use absolute path)",
        ex.prompt, tmpdir_str, tmpdir_str, target_file
    );

    let start = Instant::now();
    // Bench determinism gate: exercises are single-file throwaway workspaces
    // graded via `ex.check`; if one gained a Cargo.toml/pyproject.toml
    // mid-task, the agent verification loop would run workspace tests plus
    // revision cycles against the 30s exercise cap. Bench never wants that.
    std::env::set_var(kitty_harness::agent::ENV_VERIFY_LOOP, "off");

    let base_config = AgentConfig {
        model: cfg.model.clone(),
        profile: cfg.profile,
        enable_knowledge: cfg.enable_knowledge,
        knowledge_config: KnowledgeConfig {
            enabled: cfg.enable_knowledge,
            knowledge_dir: cfg.knowledge_dir.clone(),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut agent =
        crate::bench::common::build_bench_agent(cfg.client.clone(), base_config).await?;
    // Per-run overrides (Fase 2.1): bench tuning flows through the single
    // auditable override path instead of post-construction mutations.
    // f8 T1.3: bench autorun is FLAG-ONLY — set explicitly (default 0) so an
    // ambient KITTY_AUTORUN_HOPS can never leak into a benchmark run.
    let mut overrides = RunOverrides {
        max_iterations: Some(cfg.max_iterations),
        autorun_hops: Some(cfg.autorun_hops),
        // Ensure compact threshold follows profile (tiny 0.7, medium 0.8,
        // large 0.8); G3-FIX1 derives from the agent's EFFECTIVE window.
        compact_threshold: Some(cfg.profile.compact_threshold),
        ..RunOverrides::default()
    };
    if cfg.context_window != 16384 {
        overrides.context_window = Some(cfg.context_window);
    }
    if cfg.thinking_budget != 2048 {
        // tuning.thinking is the LIVE thinking axis — this is what makes
        // bench-ab A/B arms actually differ.
        let mut tuning = agent.config().tuning.clone();
        tuning.thinking = ThinkingMode::Budget(cfg.thinking_budget as u32);
        overrides.tuning = Some(tuning);
    }
    if cfg.bash_timeout != 60 {
        overrides.bash_timeout = Some(cfg.bash_timeout);
    }
    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(cfg.exercise_timeout_secs),
        agent.run_with(&prompt, overrides),
    )
    .await;

    // Determine pass/fail and persist an episode for memory.
    let (passed, verdict, latency_ms) = match outcome {
        Ok(Ok(_)) => {
            let passed = (ex.check)(&tmpdir_str);
            (
                passed,
                if passed { "pass" } else { "fail" },
                start.elapsed().as_millis() as f64,
            )
        }
        Ok(Err(e)) => {
            println!(" error: {}", e);
            (false, "error", start.elapsed().as_millis() as f64)
        }
        Err(_) => {
            println!(" timeout");
            (false, "timeout", 30000.0)
        }
    };

    let status = if passed { "PASS" } else { "FAIL" };
    println!("{} ({:.0}ms)", status, latency_ms);

    // Persist the episode on the blocking pool (take/restore, best-effort).
    if let Some(mem) = store.take() {
        let files_touched = if verdict == "pass" || verdict == "fail" {
            list_files(&tmpdir_str)
        } else {
            vec![]
        };
        let ep = Episode::new(
            ex.prompt.to_string(),
            files_touched,
            Some(verdict.into()),
            format!("{verdict} (latency {latency_ms:.0}ms)"),
        );
        let (back, saved) = tokio::task::spawn_blocking(move || {
            let result = mem.save(&ep);
            (Some(mem), result)
        })
        .await
        .unwrap_or((None, Ok(())));
        if let Some(mem) = back {
            *store = Some(mem);
        }
        let _ = saved;
    }

    // W1B: esfuerzo real del run (siempre disponible: el loop publica
    // contadores cada turno, también en error/timeout parcial).
    // F1b: ejes reales del loop, nunca "pending": terminal_reason del
    // agent, invalid_tool_calls acumulado, y available_context_tokens del
    // engine (kitty-infer vía run_context) o del harness (ollama).
    let effort = agent.last_run_effort();
    let terminal_reason = agent
        .last_terminal_reason()
        .map(|r| r.as_str().to_string())
        .unwrap_or_else(|| verdict.to_string());
    let invalid_tool_calls = agent.invalid_tool_calls();
    let available_context_tokens = cfg
        .client
        .infer_run_context()
        .map(|c| c.available_context_tokens)
        .unwrap_or_else(|| agent.available_context_tokens());
    Ok(BenchResult {
        id: ex.id.to_string(),
        lang: ex.label.to_string(),
        passed,
        verdict: verdict.to_string(),
        latency_ms,
        hops: Some(effort.hops_used),
        iters: Some(effort.iters_used),
        margin: None, // sin logprobs en esta lane (W3)
        effort_off_in_bench: effort.router_off,
        terminal_reason,
        invalid_tool_calls,
        available_context_tokens,
    })
}
// ── TB2 homemade: 8 ejercicios locales (NO oficial) ────────────────
// T8.1: `tb2` suite = homemade; `tb2-official` = Harbor oficial (ver tb2.rs).

pub(crate) fn tb2_homemade_exercises() -> Vec<BenchItem> {
    vec![
        bench_file("tb2-bash-count", "bash", "Create a bash script called count_lines.sh that counts the number of lines in all .rs files in the current directory recursively. Save it to count_lines.sh and make it executable.", "count_lines.sh", |d| file_is_executable(d, "count_lines.sh") && file_contains(d, "count_lines.sh", "find")),
        bench_file("tb2-py-csv", "python", "Write a Python script called csv_to_json.py that reads a CSV file (input path as first argument) and outputs JSON to stdout. Save it to csv_to_json.py.", "csv_to_json.py", |d| file_contains(d, "csv_to_json.py", "csv") && file_contains(d, "csv_to_json.py", "json")),
        bench_file("tb2-makefile", "makefile", "Create a Makefile with three targets: test (runs cargo test), build (runs cargo build), and clean (removes target directory). Save to Makefile.", "Makefile", |d| file_contains(d, "Makefile", "test:") && file_contains(d, "Makefile", "build:") && file_contains(d, "Makefile", "clean:")),
        bench_file("tb2-dockerfile", "docker", "Write a Dockerfile for a Rust project that uses multi-stage builds: builder stage compiles, runtime stage copies binary. Save to Dockerfile.", "Dockerfile", |d| file_contains(d, "Dockerfile", "FROM") && file_contains(d, "Dockerfile", "COPY")),
        bench_file("tb2-gh-actions", "ci", "Create a GitHub Actions CI workflow at .github/workflows/ci.yml that runs cargo test and cargo build on push/PR. Save to .github/workflows/ci.yml.", ".github/workflows/ci.yml", |d| {
            let path = std::path::Path::new(d).join(".github/workflows/ci.yml");
            match std::fs::read_to_string(&path) {
                Ok(c) => c.contains("cargo test") || c.contains("cargo build"),
                Err(_) => false,
            }
        }),
        bench_file("tb2-sh-find-modified", "bash", "Write a bash script called find_recent.sh that finds all files modified in the last 24 hours in a given directory (first argument). Save to find_recent.sh.", "find_recent.sh", |d| file_is_executable(d, "find_recent.sh") && file_contains(d, "find_recent.sh", "find")),
        bench_file("tb2-py-dirsize", "python", "Write a Python script called dirsize.py that calculates and prints the total size of each subdirectory in the current directory. Save to dirsize.py.", "dirsize.py", |d| file_contains(d, "dirsize.py", "os") && file_contains(d, "dirsize.py", "path")),
        bench_file("tb2-sh-backup", "bash", "Write a bash script called backup.sh that backs up files from a source directory (first argument) to a destination with a timestamp suffix (e.g. backup_20260817_120000). Save to backup.sh.", "backup.sh", |d| file_is_executable(d, "backup.sh") && file_contains(d, "backup.sh", "cp")),
    ]
}

//! Aider Polyglot benchmark runner (official pattern): load exercises from
//! `.meta/config.json`, copy solution trees to a tempdir, run the agent,
//! materialize tests, run unit tests with retries feeding test output back.

use crate::bench::common::BenchConfig;
use crate::default_db_path;
use anyhow::{Context, Result};
use kitty_harness::agent::{Agent, AgentConfig, RunOverrides};
use kitty_harness::knowledge::KnowledgeConfig;
use kitty_harness::ThinkingMode;
use kitty_memory::{Episode, EpisodeStore};
use std::path::PathBuf;
use std::time::Instant;

// ── Polyglot (Aider-style unit test grading) ──────────────────────

/// Run the Aider Polyglot benchmark — official pattern:
/// 1. Load exercises from .meta/config.json
/// 2. Copy ONLY solution files to tempdir (hide test files)
/// 3. Build prompt from .docs/ + file list
/// 4. Run agent (edits solution files in tempdir)
/// 5. Materialize test files AFTER agent finishes
/// 6. Run unit tests (180s timeout)
/// 7. If fail AND retries remain → feed test output back, retry
pub(crate) async fn run_polyglot_bench(
    cfg: &BenchConfig,
    out: &PathBuf,
    limit: usize,
) -> Result<()> {
    let base_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/official/polyglot-benchmark");
    if !base_dir.exists() {
        anyhow::bail!(
            "Polyglot benchmark directory not found: {}",
            base_dir.display()
        );
    }

    let exercises = crate::polyglot::load_polyglot_exercises(&base_dir);
    let take = limit.min(exercises.len());

    // Disable write-guard so small models can overwrite stubs via Write
    std::env::set_var(kitty_harness::config::ENV_BENCH_MODE, "1");

    println!("Kitty Code Polyglot (unit test grading)");
    println!("  Model: {}", cfg.model);
    println!(
        "  Exercises: {} available, running {}",
        exercises.len(),
        take
    );

    // Blocking rusqlite open runs on the blocking pool so the async
    // executor never parks on SQLite.
    let mut store = if cfg.use_memory {
        let path = default_db_path();
        Some(
            tokio::task::spawn_blocking(move || EpisodeStore::open(&path))
                .await
                .context("episode store open task failed")??,
        )
    } else {
        None
    };

    let mut results: Vec<crate::polyglot::PolyglotResult> = Vec::with_capacity(take);
    let agent_timeout_secs = 900u64;
    let test_timeout_secs = 180u64;
    let max_attempts = 2usize;

    for (i, ex) in exercises.iter().take(take).enumerate() {
        print!("[{:2}/{:2}] {} ...", i + 1, take, ex.id);

        let start = Instant::now();
        let mut attempts = 0usize;
        let mut passed = false;
        let mut last_test_output = String::new();
        let mut last_prompt = String::new();
        let mut error_msg: Option<String> = None;

        // Create tempdir ONCE before the retry loop so the agent's work
        // is preserved across retries (not recreated from original stubs).
        let tmp_dir = match tempfile::tempdir() {
            Ok(d) => d,
            Err(e) => {
                error_msg = Some(format!("tempdir failed: {e}"));
                // Skip to result processing
                let latency_ms = start.elapsed().as_millis() as u64;
                let verdict = "error";
                let status_char = "FAIL";
                println!("{status_char} ({latency_ms:.0}ms, {attempts} attempt(s))");
                results.push(crate::polyglot::PolyglotResult {
                    id: ex.id.clone(),
                    lang: ex.lang.clone(),
                    passed: false,
                    latency_ms,
                    attempts,
                    verdict: verdict.to_string(),
                    error: error_msg,
                });
                continue;
            }
        };

        // Copy ENTIRE exercise tree (NO .meta/) — test files visible as reference
        // like little-coder. Agent sees stubs + tests, but prompt says "DO NOT edit tests".
        if let Err(e) = crate::polyglot::copy_exercise_tree(&ex.exercise_dir, tmp_dir.path()) {
            error_msg = Some(format!("copy solution files failed: {e}"));
            let latency_ms = start.elapsed().as_millis() as u64;
            let verdict = "error";
            let status_char = "FAIL";
            println!("{status_char} ({latency_ms:.0}ms, {attempts} attempt(s))");
            results.push(crate::polyglot::PolyglotResult {
                id: ex.id.clone(),
                lang: ex.lang.clone(),
                passed: false,
                latency_ms,
                attempts,
                verdict: verdict.to_string(),
                error: error_msg,
            });
            continue;
        }

        while attempts < max_attempts {
            attempts += 1;

            // Step 3: Build prompt — always include original instructions (with tempdir absolute paths)
            let prompt = if attempts == 1 {
                crate::polyglot::build_prompt(&ex.exercise_dir, &ex.config, tmp_dir.path())
            } else {
                // Retry: original instructions + test output from previous attempt
                format!(
                    "{}\n\n{}",
                    crate::polyglot::build_prompt(&ex.exercise_dir, &ex.config, tmp_dir.path()),
                    crate::polyglot::retry_prompt(&last_test_output)
                )
            };
            last_prompt = prompt.clone();

            // Step 4: Run agent (workspace = tempdir)
            std::env::set_var(kitty_harness::agent::ENV_VERIFY_LOOP, "off");

            let base_config = AgentConfig {
                model: cfg.model.clone(),
                profile: cfg.profile,
                enable_knowledge: cfg.enable_knowledge,
                workspace: tmp_dir.path().to_path_buf(),
                knowledge_config: KnowledgeConfig {
                    enabled: cfg.enable_knowledge,
                    knowledge_dir: cfg.knowledge_dir.clone(),
                    ..Default::default()
                },
                ..Default::default()
            };
            let mut agent = Agent::new(cfg.client.clone(), base_config);
            let mut overrides = RunOverrides {
                max_iterations: Some(cfg.max_iterations.max(20)),
                autorun_hops: Some(cfg.autorun_hops),
                compact_threshold: Some(cfg.profile.compact_threshold),
                ..RunOverrides::default()
            };
            if cfg.context_window != 16384 {
                overrides.context_window = Some(cfg.context_window);
            }
            if cfg.thinking_budget != 2048 {
                let mut tuning = agent.config().tuning.clone();
                tuning.thinking = ThinkingMode::Budget(cfg.thinking_budget as u32);
                overrides.tuning = Some(tuning);
            }
            if cfg.bash_timeout != 60 {
                overrides.bash_timeout = Some(cfg.bash_timeout);
            }

            let agent_result = tokio::time::timeout(
                std::time::Duration::from_secs(agent_timeout_secs),
                agent.run_with(&prompt, overrides),
            )
            .await;

            match agent_result {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => {
                    error_msg = Some(format!("agent error: {e}"));
                    break;
                }
                Err(_) => {
                    error_msg = Some(format!("agent timed out after {agent_timeout_secs}s"));
                    break;
                }
            }

            // Step 5: Materialize test files AFTER agent finishes
            if let Err(e) =
                crate::polyglot::materialize_tests(&ex.exercise_dir, tmp_dir.path(), &ex.config)
            {
                error_msg = Some(format!("materialize tests failed: {e}"));
                break;
            }

            // Step 6: Run unit tests
            let (test_passed, test_output) =
                crate::polyglot::run_test_with_timeout(tmp_dir.path(), &ex.lang, test_timeout_secs)
                    .await;

            last_test_output = test_output;

            if test_passed {
                passed = true;
                break;
            }

            // Step 7: If fail AND retries remain → loop back for retry
            // (the while condition handles this)
        }

        let latency_ms = start.elapsed().as_millis() as u64;
        let verdict = if passed {
            "pass"
        } else if error_msg.is_some() {
            "error"
        } else {
            "fail"
        };

        let status_char = if passed { "PASS" } else { "FAIL" };
        println!(
            "{} ({:.0}ms, {attempts} attempt(s))",
            status_char, latency_ms
        );

        let result = crate::polyglot::PolyglotResult {
            id: ex.id.clone(),
            lang: ex.lang.clone(),
            passed,
            latency_ms,
            attempts,
            verdict: verdict.to_string(),
            error: error_msg,
        };

        // Persist episode for memory on the blocking pool (take/restore
        // shape from `agent/response.rs`: sequential loop, single ownership,
        // best effort with fallback on pool failure).
        if let Some(s) = store.take() {
            let ep = Episode::new(
                last_prompt,
                vec![],
                Some(result.verdict.clone()),
                format!(
                    "{} {} (latency {:.0}ms, {attempts} attempt(s))",
                    ex.id, result.verdict, result.latency_ms
                ),
            );
            let (back, saved) = tokio::task::spawn_blocking(move || {
                let result = s.save(&ep);
                (Some(s), result)
            })
            .await
            .unwrap_or((None, Ok(())));
            if let Some(s) = back {
                store = Some(s);
            }
            let _ = saved;
        }

        results.push(result);
    }

    // Print summary
    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let rate = if total > 0 {
        100.0 * passed as f64 / total as f64
    } else {
        0.0
    };

    println!();
    println!("RESULTS (Polyglot unit test grading)");
    println!("Passed: {}/{} ({:.1}%)", passed, total, rate);

    // Per-language breakdown
    let mut by_lang: std::collections::BTreeMap<String, (usize, usize)> =
        std::collections::BTreeMap::new();
    for r in &results {
        let entry = by_lang.entry(r.lang.clone()).or_insert((0, 0));
        entry.0 += 1;
        if r.passed {
            entry.1 += 1;
        }
    }
    for (lang, (attempted, lang_passed)) in &by_lang {
        let lang_rate = 100.0 * *lang_passed as f64 / *attempted as f64;
        println!(
            "  {lang}: attempted={} passed={} ({:.1}%)",
            attempted, lang_passed, lang_rate
        );
    }

    // Generate Aider-compatible report
    let report = crate::polyglot::generate_polyglot_report(&results);
    let output = serde_json::json!({
        "model": cfg.model,
        "suite": "polyglot",
        "total": total,
        "passed": passed,
        "rate_pct": rate,
        "by_language": by_lang.iter().map(|(lang, (attempted, lang_passed))| {
            (lang.clone(), serde_json::json!({
                "attempted": attempted,
                "passed": lang_passed,
                "failed": attempted - lang_passed,
            }))
        }).collect::<std::collections::BTreeMap<String, serde_json::Value>>(),
        "details": results.iter().map(|r| serde_json::json!({
            "id": r.id,
            "lang": r.lang,
            "passed": r.passed,
            "verdict": r.verdict,
            "latency_ms": r.latency_ms,
            "attempts": r.attempts,
            "error": r.error,
        })).collect::<Vec<_>>(),
        "report": report,
    });
    std::fs::write(out, serde_json::to_string_pretty(&output)?)?;
    println!("Results saved to {}", out.display());
    Ok(())
}

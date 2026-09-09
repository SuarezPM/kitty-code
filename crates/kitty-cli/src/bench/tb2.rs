//! Terminal-Bench 2.1 official runner via the Harbor CLI wrapper (official
//! dataset, official harness, Docker-managed).

use anyhow::Result;
use std::path::PathBuf;

// ── TB2 Official (Harbor CLI wrapper) ────────────────────────────────

/// Run Terminal-Bench 2.1 via Harbor (official dataset, official harness).
/// This delegates to the `harbor` CLI which manages Docker containers,
/// dataset download, and result aggregation.
pub(crate) async fn run_tb2_official(
    model: &str,
    api_base: &str,
    out: &PathBuf,
    limit: usize,
) -> Result<()> {
    println!("Kitty Code TB2 Official (Harbor wrapper)");
    println!("  Model: {model}");

    // Limit footgun: `--limit 0` selects no tasks; bail before preflight
    // so `-l 0` never reaches Harbor.
    if limit == 0 {
        let blocked = serde_json::json!({
            "suite": "tb2_official",
            "status": "blocked",
            "reason": "TB2 Official blocked: --limit 0 selects no tasks (use 1..=88 for smoke, 89+ for the full evidence set)",
            "label": "SMOKE-ONLY blocked: --limit 0 (never evidence-grade)",
            "grade": "smoke-only",
            "model": model,
            "dataset": crate::tb2::tb2_dataset_pinned(),
            "dataset_version": crate::tb2::TB2_DATASET_VERSION,
            "full_task_count": crate::tb2::TB2_FULL_TASK_COUNT,
            "resolved_task_count": 0,
            "requested_limit": limit,
            "evidence_grade": false,
        });
        let _ = std::fs::write(
            out,
            serde_json::to_string_pretty(&blocked).unwrap_or_default(),
        );
        println!("Results (blocked) saved to {}", out.display());
        anyhow::bail!("TB2 Official blocked: --limit 0 selects no tasks");
    }

    // Resolved selection: bounded N is maximum N after filters
    // (order-dependent smoke-only selection), full set is evidence-grade.
    // Record both counts in every manifest; a Hub rev bump changing the
    // count voids prior baselines (void-and-re-baseline, never auto-pass).
    let task_limit = crate::tb2::task_limit_for(limit);
    let resolved_count = crate::tb2::resolved_task_count(task_limit);
    let is_smoke = task_limit.is_some();
    let grade_label = if is_smoke {
        "SMOKE-ONLY official terminal-bench-2.1 via Harbor (not evidence-grade)"
    } else {
        "official terminal-bench-2.1 via Harbor"
    };
    let grade = if is_smoke { "smoke-only" } else { "evidence" };

    // Preflight
    let pf = crate::tb2::preflight();
    if !crate::tb2::is_ready(&pf) {
        let reasons = crate::tb2::blocked_reasons(&pf);
        eprintln!("\nTB2 preflight FAILED:");
        for r in &reasons {
            eprintln!("  - {r}");
        }
        eprintln!(
            "\nNota: TB2 Official requiere 'harbor' CLI (uv tool install harbor) y Docker corriendo."
        );

        let blocked = serde_json::json!({
            "suite": "tb2_official",
            "status": "blocked",
            "reasons": reasons,
            "label": grade_label,
            "grade": grade,
            "model": model,
            "dataset": crate::tb2::tb2_dataset_pinned(),
            "dataset_version": crate::tb2::TB2_DATASET_VERSION,
            "full_task_count": crate::tb2::TB2_FULL_TASK_COUNT,
            "resolved_task_count": resolved_count,
            "requested_limit": limit,
            "evidence_grade": !is_smoke,
        });
        let _ = std::fs::write(
            out,
            serde_json::to_string_pretty(&blocked).unwrap_or_default(),
        );
        println!("Results (blocked) saved to {}", out.display());
        anyhow::bail!("TB2 Official blocked: preflight failed");
    }

    let version_str = pf.harbor_version.as_deref().unwrap_or("unknown");
    println!("  Harbor version: {version_str}");

    let tb2_config = crate::tb2::Tb2Config {
        model: model.to_string(),
        api_base: api_base.to_string(),
        trials: 1,
        parallel: 1,
        timeout_multiplier: 3,
        max_turns: 50,
        // `--limit` selects the task count: bounded N for smokes
        // (maximum N after filters, order-dependent, never evidence-grade),
        // full 89-task set (no `-l` flag) for evidence runs.
        task_limit,
    };

    match tb2_config.task_limit {
        Some(n) => println!(
            "  Tasks: SMOKE-ONLY — maximum {n} after filters of {} (via -l; order-dependent, never evidence-grade)",
            crate::tb2::TB2_FULL_TASK_COUNT
        ),
        None => println!(
            "  Tasks: full {}-task set (evidence run)",
            crate::tb2::TB2_FULL_TASK_COUNT
        ),
    }
    println!(
        "  Dataset: {} (full {}; resolved {})",
        crate::tb2::tb2_dataset_pinned(),
        crate::tb2::TB2_FULL_TASK_COUNT,
        resolved_count
    );

    println!("  Running: harbor run (terminal-bench-2.1 dataset)");
    println!();

    match crate::tb2::run_tb2(&tb2_config).await {
        Ok(result) => {
            let rate = result.resolution_rate;
            println!();
            println!("RESULTS (TB2 Official)");
            println!("  Model: {}", result.model);
            println!(
                "  Resolved: {}/{} ({:.1}%)",
                result.resolved, result.total_tasks, rate
            );
            println!("  Label: {grade_label}");
            if is_smoke {
                println!(
                    "  SMOKE-ONLY: maximum {resolved_count} after filters (never evidence-grade)"
                );
            }

            let output = serde_json::json!({
                "suite": "tb2_official",
                "status": "executed",
                "label": grade_label,
                "grade": grade,
                "model": result.model,
                "dataset": crate::tb2::tb2_dataset_pinned(),
                "dataset_version": crate::tb2::TB2_DATASET_VERSION,
                "full_task_count": crate::tb2::TB2_FULL_TASK_COUNT,
                "resolved_task_count": resolved_count,
                "requested_limit": limit,
                "evidence_grade": !is_smoke,
                "total_tasks": result.total_tasks,
                "resolved": result.resolved,
                "resolution_rate": result.resolution_rate,
                "per_task": result.per_task,
                "note": if is_smoke {
                    "SMOKE-ONLY: maximum N after filters via -l (order-dependent), not evidence-grade. Resultado ejecutado con harbor CLI (official harness)."
                } else {
                    "Resultado ejecutado con harbor CLI (official harness)."
                },
            });
            std::fs::write(out, serde_json::to_string_pretty(&output)?)?;
            println!("Results saved to {}", out.display());

            let gate = 50.0;
            if rate >= gate {
                println!();
                println!("GATE PASSED: {:.1}% >= {:.0}%", rate, gate);
            } else {
                println!();
                println!("GATE FAILED: {:.1}% < {:.0}%", rate, gate);
            }

            let exit_code = if rate < gate { 1 } else { 0 };
            std::process::exit(exit_code);
        }
        Err(e) => {
            let blocked = serde_json::json!({
                "suite": "tb2_official",
                "status": "blocked",
                "reason": e.to_string(),
                "label": grade_label,
                "grade": grade,
                "model": model,
                "dataset": crate::tb2::tb2_dataset_pinned(),
                "dataset_version": crate::tb2::TB2_DATASET_VERSION,
                "full_task_count": crate::tb2::TB2_FULL_TASK_COUNT,
                "resolved_task_count": resolved_count,
                "requested_limit": limit,
                "evidence_grade": !is_smoke,
            });
            let _ = std::fs::write(
                out,
                serde_json::to_string_pretty(&blocked).unwrap_or_default(),
            );
            println!("Results (blocked) saved to {}", out.display());
            anyhow::bail!("TB2 Official failed: {e}");
        }
    }
}

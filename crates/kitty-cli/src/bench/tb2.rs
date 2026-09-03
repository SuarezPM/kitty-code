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
    _limit: usize,
) -> Result<()> {
    println!("Kitty Code TB2 Official (Harbor wrapper)");
    println!("  Model: {model}");

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
            "label": "official terminal-bench-2.1 via Harbor",
            "model": model,
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
    };

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
            println!("  Label: official terminal-bench-2.1 via Harbor");

            let output = serde_json::json!({
                "suite": "tb2_official",
                "status": "executed",
                "label": "official terminal-bench-2.1 via Harbor",
                "model": result.model,
                "total_tasks": result.total_tasks,
                "resolved": result.resolved,
                "resolution_rate": result.resolution_rate,
                "per_task": result.per_task,
                "note": "Resultado ejecutado con harbor CLI (official harness).",
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
                "label": "official terminal-bench-2.1 via Harbor",
                "model": model,
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

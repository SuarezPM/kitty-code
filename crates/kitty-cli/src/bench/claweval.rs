//! ClawEval suite runner: offline-first preflight, wraps `claw-eval batch`.
//! Results are tagged as "local approximation — not official ClawEval
//! leaderboard".

use anyhow::Result;
use std::path::PathBuf;

// ── ClawEval integration (F2: offline-first, local approximation) ────

/// ClawEval suite: offline-first preflight, wraps `claw-eval batch`.
/// Results tagged as "local approximation — not official ClawEval leaderboard".
pub(crate) async fn run_claweval_suite(
    model: &str,
    out: &PathBuf,
    trials: usize,
    parallel: usize,
    sandbox: bool,
) -> Result<()> {
    println!("Kitty Code ClawEval (local approximation)");
    println!("  Trials per task: {trials}");
    println!("  Parallel: {parallel}");
    println!("  Sandbox: {sandbox}");

    // Preflight
    let pf = crate::claweval::preflight_claweval();
    if !crate::claweval::is_ready(&pf) {
        let reasons = crate::claweval::blocked_reasons(&pf);
        eprintln!("\nClawEval preflight FAILED:");
        for r in &reasons {
            eprintln!("  - {r}");
        }
        eprintln!(
            "\nNota: ClawEval requiere 'claw-eval' CLI (pip install claw-eval) y OPENROUTER_API_KEY."
        );
        eprintln!("El resultado ClawEval siempre se etiqueta como \"aproximación local no oficial\" (no leaderboard).");

        let blocked = serde_json::json!({
            "suite": "claweval_local",
            "status": "blocked",
            "reasons": reasons,
            "label": "aproximación local no oficial — no leaderboard oficial de ClawEval",
            "wrapper_version": crate::claweval::CLAWEVAL_WRAPPER_VERSION,
            "compatibility": crate::claweval::CLAWEVAL_COMPATIBILITY,
        });
        let _ = std::fs::write(
            out,
            serde_json::to_string_pretty(&blocked).unwrap_or_default(),
        );
        println!("Results (blocked) saved to {}", out.display());
        anyhow::bail!("ClawEval blocked: preflight failed");
    }

    let version_str = pf.version.as_deref().unwrap_or("unknown");
    println!("  ClawEval version: {version_str}");

    let api_key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
    let model = model.to_string();

    let output_dir = {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        PathBuf::from(home).join(".cache/kitty/claweval")
    };

    let cfg = crate::claweval::ClawEvalRunConfig {
        model: model.clone(),
        api_key,
        base_url: None,
        trials,
        parallel,
        sandbox,
        task_ids: None,
        config_path: None,
        output_dir,
    };

    println!("  Running: claw-eval batch ...");

    match crate::claweval::run_claweval_batch(&cfg).await {
        Ok(result) => {
            let status = crate::claweval::ClawEvalStatus::Executed {
                result: result.clone(),
                official: false,
            };
            let output = serde_json::json!({
                "suite": "claweval_local",
                "status": status,
                "label": "aproximación local no oficial — no leaderboard oficial de ClawEval",
                "wrapper_version": crate::claweval::CLAWEVAL_WRAPPER_VERSION,
                "compatibility": crate::claweval::CLAWEVAL_COMPATIBILITY,
                "model": result.model,
                "total_tasks": result.total_tasks,
                "trials_per_task": result.trials_per_task,
                "average_score": result.average_score,
                "pass_at_k": result.pass_at_k,
                "pass_power_k": result.pass_power_k,
                "per_task": result.per_task,
                "note": "Resultado local aproximado; no comparable a leaderboard oficial de ClawEval (requiere harness oficial).",
            });
            std::fs::write(out, serde_json::to_string_pretty(&output)?)?;
            println!();
            println!("RESULTS (ClawEval local approximation)");
            println!("  Model: {}", result.model);
            println!("  Tasks: {}", result.total_tasks);
            println!("  Trials/task: {}", result.trials_per_task);
            println!("  Average score: {:.3}", result.average_score);
            println!("  pass@k: {:.3}", result.pass_at_k);
            println!("  pass^k: {:.3}", result.pass_power_k);
            println!("  Label: aproximación local no oficial — no leaderboard oficial de ClawEval");
            println!("Results saved to {}", out.display());
            Ok(())
        }
        Err(e) => {
            let reasons = match &e {
                crate::claweval::ClawEvalError::Blocked { reasons } => reasons.clone(),
                _ => vec![e.to_string()],
            };
            eprintln!("\nClawEval execution FAILED:\n  {e}\n");
            let blocked = serde_json::json!({
                "suite": "claweval_local",
                "status": "blocked",
                "reasons": reasons,
                "label": "aproximación local no oficial — no leaderboard oficial de ClawEval",
                "wrapper_version": crate::claweval::CLAWEVAL_WRAPPER_VERSION,
            });
            let _ = std::fs::write(
                out,
                serde_json::to_string_pretty(&blocked).unwrap_or_default(),
            );
            println!("Results (blocked) saved to {}", out.display());
            anyhow::bail!("ClawEval failed: {e}");
        }
    }
}

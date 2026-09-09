//! GPQA Diamond local suite runner: offline-first, gated, explicit
//! pre-registration (G1). Blocks without invoking the model when the
//! pre-registration manifest is missing or invalid.

use crate::bench::common::{BenchConfig, BenchResult};
use crate::default_db_path;
use anyhow::{Context, Result};
use kitty_harness::agent::{Agent, AgentConfig, RunOverrides};
use kitty_harness::knowledge::KnowledgeConfig;
use kitty_harness::ThinkingMode;
use kitty_memory::{Episode, EpisodeStore};
use std::path::PathBuf;
use std::time::Instant;

// ── GPQA Diamond local approximation (F1 + G1) ─────────────────

/// GPQA Diamond local suite: offline-first, gated, pre-registro explícito (G1).
/// Carga solo materializado validado contra pre-registro; no hace red, no registra prompts, no usa HF_TOKEN.
/// Resultado etiquetado como aproximación local no oficial con proveniencia completa.
/// Si falta/inválido el pre-registro ⇒ salida `blocked` y nunca invoca el modelo.
pub(crate) async fn run_gpqa_suite(
    cfg: &BenchConfig,
    out: &PathBuf,
    limit: usize,
    pre_registration: Option<PathBuf>,
) -> Result<()> {
    println!("Kitty Code GPQA Diamond (local approximation)");
    println!("  Model: {}", cfg.model);

    // Preflight con pre-registro explícito (G1): bloqueado sin invocar modelo si falta/inválido
    let (pre, manifest) = match &pre_registration {
        None => {
            let help = crate::gpqa::pre_registration_help_text();
            eprintln!("\nGPQA preflight FAILED (offline-first, gated):\nPre-registration required but not provided.\n{help}\n");
            eprintln!("Nota: GPQA Diamond es gated; este harness nunca descarga automáticamente ni usa HF_TOKEN.");
            // Escribir salida blocked para trazabilidad
            let blocked = serde_json::json!({
                "suite": "gpqa_diamond_local",
                "status": "blocked",
                "reason": "missing --pre-registration",
                "help": help,
                "label": "aproximación local no oficial — no leaderboard oficial",
                "model": cfg.model,
                "limit": limit,
            });
            let _ = std::fs::write(
                out,
                serde_json::to_string_pretty(&blocked).unwrap_or_default(),
            );
            println!("Results (blocked) saved to {}", out.display());
            anyhow::bail!("GPQA blocked: missing --pre-registration");
        }
        Some(p) => match crate::gpqa::preflight_gpqa_with_prereg(p) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("\nGPQA preflight FAILED (offline-first, gated):\n{e}\n");
                eprintln!("Nota: GPQA Diamond es gated; este harness nunca descarga automáticamente ni usa HF_TOKEN.");
                eprintln!("El resultado GPQA siempre se etiqueta como \"aproximación local no oficial\" (no leaderboard).");
                let blocked = serde_json::json!({
                    "suite": "gpqa_diamond_local",
                    "status": "blocked",
                    "reason": e.to_string(),
                    "label": "aproximación local no oficial — no leaderboard oficial",
                    "model": cfg.model,
                    "limit": limit,
                    "pre_registration": p.display().to_string(),
                });
                let _ = std::fs::write(
                    out,
                    serde_json::to_string_pretty(&blocked).unwrap_or_default(),
                );
                println!("Results (blocked) saved to {}", out.display());
                anyhow::bail!("GPQA preflight blocked: {e}");
            }
        },
    };

    let cache_path = crate::gpqa::gpqa_cache_path();

    // G1.1: GPQA solo puede ejecutar las 198 preguntas; bloquea antes de crear Agent si limit !=198
    if limit != crate::gpqa::GPQA_NUM_QUESTIONS {
        let msg = format!(
            "GPQA blocked: --limit must be {} for GPQA Diamond (got {}). Usa --limit 198. Pre-registro n/limits.limit también debe ser {}.",
            crate::gpqa::GPQA_NUM_QUESTIONS, limit, crate::gpqa::GPQA_NUM_QUESTIONS
        );
        eprintln!("\n{msg}\n");
        let blocked = serde_json::json!({
            "suite": "gpqa_diamond_local",
            "status": "blocked",
            "reason": msg,
            "help": "Ejecuta: kitty bench --suite gpqa --pre-registration <PATH> --limit 198",
            "label": "aproximación local no oficial — no leaderboard oficial",
            "model": cfg.model,
            "limit": limit,
            "pre_registration": pre_registration.as_ref().map(|p| p.display().to_string()).unwrap_or_default(),
        });
        let _ = std::fs::write(
            out,
            serde_json::to_string_pretty(&blocked).unwrap_or_default(),
        );
        println!("Results (blocked) saved to {}", out.display());
        anyhow::bail!(msg);
    }
    if pre.limits.limit != crate::gpqa::GPQA_NUM_QUESTIONS {
        let msg = format!(
            "GPQA blocked: pre-registration limits.limit must be {} (got {}). Corrige el pre-registro y usa --limit 198.",
            crate::gpqa::GPQA_NUM_QUESTIONS, pre.limits.limit
        );
        eprintln!("\n{msg}\n");
        let blocked = serde_json::json!({
            "suite": "gpqa_diamond_local",
            "status": "blocked",
            "reason": msg,
            "label": "aproximación local no oficial — no leaderboard oficial",
            "model": cfg.model,
            "limit": limit,
        });
        let _ = std::fs::write(
            out,
            serde_json::to_string_pretty(&blocked).unwrap_or_default(),
        );
        anyhow::bail!(msg);
    }

    // G1.2: pre-registro debe ser vinculante: model y límites relevantes deben coincidir con BenchConfig efectivos
    let mut binding_errors: Vec<String> = Vec::new();
    if pre.model != cfg.model {
        binding_errors.push(format!(
            "model mismatch: pre-registration '{}' != bench '{}'",
            pre.model, cfg.model
        ));
    }
    if let Some(v) = pre.limits.timeout_secs {
        if v != cfg.exercise_timeout_secs {
            binding_errors.push(format!(
                "timeout_secs mismatch: pre-registration {} != bench {}",
                v, cfg.exercise_timeout_secs
            ));
        }
    }
    if let Some(v) = pre.limits.max_iterations {
        if v != cfg.max_iterations {
            binding_errors.push(format!(
                "max_iterations mismatch: pre-registration {} != bench {}",
                v, cfg.max_iterations
            ));
        }
    }
    if let Some(v) = pre.limits.thinking_budget {
        if v != cfg.thinking_budget {
            binding_errors.push(format!(
                "thinking_budget mismatch: pre-registration {} != bench {}",
                v, cfg.thinking_budget
            ));
        }
    }
    if let Some(v) = pre.limits.context_window {
        if v != cfg.context_window {
            binding_errors.push(format!(
                "context_window mismatch: pre-registration {} != bench {}",
                v, cfg.context_window
            ));
        }
    }
    if !binding_errors.is_empty() {
        let msg = format!(
            "GPQA blocked: pre-registration no vinculante: {}",
            binding_errors.join("; ")
        );
        eprintln!("\n{msg}\nPre-registro debe fijar model y límites efectivos de la ejecución. Corrige el JSON o los flags CLI.\n");
        let blocked = serde_json::json!({
            "suite": "gpqa_diamond_local",
            "status": "blocked",
            "reason": msg,
            "label": "aproximación local no oficial — no leaderboard oficial",
            "model": cfg.model,
            "limit": limit,
            "pre_registration": pre_registration.as_ref().map(|p| p.display().to_string()).unwrap_or_default(),
        });
        let _ = std::fs::write(
            out,
            serde_json::to_string_pretty(&blocked).unwrap_or_default(),
        );
        println!("Results (blocked) saved to {}", out.display());
        anyhow::bail!(msg);
    }

    let questions = crate::gpqa::load_questions(&cache_path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to load GPQA questions from {}: {e}",
            cache_path.display()
        )
    })?;

    let total_available = questions.len();
    // load_questions ya garantiza 198, pero mantenemos chequeo defensivo
    if total_available != crate::gpqa::GPQA_NUM_QUESTIONS {
        let msg = format!(
            "GPQA blocked: cache must contain exactly {} questions, found {}. Usa --limit 198 y materializa la revisión correcta.",
            crate::gpqa::GPQA_NUM_QUESTIONS, total_available
        );
        eprintln!("\n{msg}\n");
        let blocked = serde_json::json!({
            "suite": "gpqa_diamond_local",
            "status": "blocked",
            "reason": msg,
            "label": "aproximación local no oficial — no leaderboard oficial",
        });
        let _ = std::fs::write(
            out,
            serde_json::to_string_pretty(&blocked).unwrap_or_default(),
        );
        anyhow::bail!(msg);
    }

    let take = crate::gpqa::GPQA_NUM_QUESTIONS;
    println!(
        "  Dataset: {} rev {} ({} questions, limit {})",
        manifest.dataset_id,
        &manifest.revision[..8.min(manifest.revision.len())],
        total_available,
        take
    );
    println!(
        "  Grading: {} | Permutation: {} | Template: {}",
        manifest.grading_method, manifest.permutation, manifest.template
    );
    println!(
        "  Cache: {} (outside repo, sha256 {}...)",
        cache_path.display(),
        &manifest.sha256[..8.min(manifest.sha256.len())]
    );
    println!("  Label: aproximación local no oficial — no leaderboard oficial");
    println!();

    let mut results: Vec<BenchResult> = Vec::with_capacity(take);
    let mut correct = 0usize;

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

    for (idx, q) in questions.iter().take(take).enumerate() {
        // Permutación determinista de 4 opciones (correcta + 3 distractores) con revisión validada
        let distractors = q.distractors.clone();
        let (permuted, correct_idx) =
            crate::gpqa::permute_four(&q.id, &pre.revision, q.correct_answer.clone(), distractors)
                .map_err(|e| anyhow::anyhow!("permute failed for {}: {e}", q.id))?;
        let correct_letter =
            crate::gpqa::idx_to_letter(correct_idx).map_err(|e| anyhow::anyhow!("{e}"))?;
        let prompt = crate::gpqa::format_prompt(&q.question, &permuted)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        print!("[{:3}/{:3}] {} ...", idx + 1, take, q.id);

        let start = Instant::now();
        let outcome = {
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
            let mut agent = Agent::new(cfg.client.clone(), base_config);
            let mut overrides = RunOverrides {
                max_iterations: Some(cfg.max_iterations),
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
            std::env::set_var(kitty_harness::config::ENV_VERIFY_LOOP, "off");
            tokio::time::timeout(
                std::time::Duration::from_secs(cfg.exercise_timeout_secs),
                agent.run_with(&prompt, overrides),
            )
            .await
        };

        let (passed, verdict, latency_ms, normalized) = match outcome {
            Ok(Ok(response)) => {
                let norm = crate::gpqa::extract_answer_strict(&response);
                let ok = crate::gpqa::grade_answer(norm, correct_letter);
                // No logueamos el prompt ni la respuesta cruda más allá de la letra normalizada
                (
                    ok,
                    if ok { "pass" } else { "fail" },
                    start.elapsed().as_millis() as f64,
                    norm,
                )
            }
            Ok(Err(e)) => {
                println!(" error: {}", e);
                (false, "error", start.elapsed().as_millis() as f64, None)
            }
            Err(_) => {
                println!(" timeout");
                (false, "timeout", 30000.0, None)
            }
        };

        if passed {
            correct += 1;
        }
        let status = if passed { "PASS" } else { "FAIL" };
        // Solo mostramos letra normalizada, nunca el prompt
        match normalized {
            Some(l) => println!("{} ({:.0}ms) -> {}", status, latency_ms, l),
            None => println!("{} ({:.0}ms) -> ?", status, latency_ms),
        }

        // Episode sin prompt (no registrar prompts GPQA), persistido en el
        // blocking pool (take/restore: bucle secuencial, propiedad única,
        // best effort con fallback si falla el pool).
        if let Some(s) = store.take() {
            let ep = Episode::new(
                format!("gpqa:{}", q.id),
                vec![],
                Some(verdict.to_string()),
                format!("gpqa {} {} (latency {:.0}ms)", q.id, verdict, latency_ms),
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

        results.push(BenchResult {
            id: q.id.clone(),
            lang: "gpqa".to_string(),
            passed,
            verdict: verdict.to_string(),
            latency_ms,
        });
    }

    let rate = if take > 0 {
        100.0 * correct as f64 / take as f64
    } else {
        0.0
    };
    println!();
    println!("RESULTS (GPQA Diamond local approximation)");
    println!("Passed: {}/{} ({:.1}%)", correct, take, rate);
    println!("Label: aproximación local no oficial — no leaderboard oficial");
    println!(
        "Provenance: dataset={} rev={} sha256={}... grading={} permutation={}",
        manifest.dataset_id,
        &manifest.revision[..8.min(manifest.revision.len())],
        &manifest.sha256[..8.min(manifest.sha256.len())],
        manifest.grading_method,
        manifest.permutation
    );

    let pre_path = pre_registration
        .as_ref()
        .expect("pre-registration validated");
    let pre_sha = crate::gpqa::sha256_file(pre_path).unwrap_or_else(|_| "unknown".to_string());
    let gpqa_result = crate::gpqa::GpqaResult::new_with_prereg(
        &manifest, &pre, pre_path, &pre_sha, take, correct,
    );
    let output = serde_json::json!({
        "model": cfg.model,
        "suite": gpqa_result.suite,
        "version": gpqa_result.version,
        "dataset_id": gpqa_result.dataset_id,
        "revision": gpqa_result.revision,
        "sha256": gpqa_result.sha256,
        "license": gpqa_result.license,
        "attribution": gpqa_result.attribution,
        "grading_method": gpqa_result.grading_method,
        "compatibility": gpqa_result.compatibility,
        "permutation": gpqa_result.permutation,
        "template": gpqa_result.template,
        "label": gpqa_result.label,
        "preconditions": gpqa_result.preconditions,
        "pre_registration_path": gpqa_result.pre_registration_path,
        "pre_registration_sha256": gpqa_result.pre_registration_sha256,
        "limits": gpqa_result.limits,
        "expected_artifacts": gpqa_result.expected_artifacts,
        "note": gpqa_result.note,
        "total": gpqa_result.total,
        "correct": gpqa_result.correct,
        "accuracy": gpqa_result.accuracy,
        "rate_pct": rate,
        "details": results.iter().map(|r| serde_json::json!({
            "id": r.id,
            "passed": r.passed,
            "verdict": r.verdict,
            "latency_ms": r.latency_ms,
        })).collect::<Vec<_>>(),
    });
    std::fs::write(out, serde_json::to_string_pretty(&output)?)?;
    println!("Results saved to {}", out.display());
    println!("Nota: este es un resultado local aproximado, no comparable al leaderboard oficial de GPQA.");
    Ok(())
}

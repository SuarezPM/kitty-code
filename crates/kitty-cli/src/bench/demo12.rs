//! demo-12 — suite fija versionada con justicia embebida (W8, lane cli).
//!
//! `kitty bench demo-12` corre los 12 ejercicios de
//! `bench/demo-12/demo-12.json` (envelope versionado con objeto `fairness`)
//! con justicia forzada: pineos (modelo+digest+versiones en el manifiesto),
//! `k>=3` validado en el envelope (la serie hace 3 invocaciones, una por rep),
//! `gates_off` forzado (autorun_hops 0, sin knowledge/memory, router en
//! pass-through registrado por celda), timeout 60s y max_turns 20.
//!
//! TRAMPA CONOCIDA: `kitty bench --exercises <json>` NO ejecuta demo-12
//! (`bench/mod.rs`: `suite == "polyglot"` retorna antes del manejo de
//! `--exercises`, y además espera un array pelado `Vec<JsonExercise>`, no el
//! envelope versionado). Este runner parsea el envelope embebido y NUNCA usa
//! `--exercises`; `bench_command` lo ignora con warn para esta suite.

use crate::bench::common::{
    json_to_bench_item, tally_by_language, BenchConfig, BenchItem, BenchResult, JsonExercise,
};
use crate::bench::run::run_one_bench;
use crate::bench::BenchCliCtx;
use anyhow::{Context, Result};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{info, warn};

/// Versión pineada de la suite (el envelope manda; el runner la exige exacta).
pub(crate) const DEMO12_VERSION: &str = "1.0.0";
/// Número fijo de ejercicios (el envelope manda).
pub(crate) const DEMO12_N_EXERCISES: usize = 12;
/// Timeout por ejercicio de la justicia (fairness + celda ornith-ollama).
pub(crate) const DEMO12_TIMEOUT_SECS: u64 = 60;
/// Turnos máximos de la justicia (ambos lados; fija la asimetría tb2 previa).
pub(crate) const DEMO12_MAX_TURNS: usize = 20;
/// Reps mínimas exigidas por el protocolo (la serie las hace por invocación).
pub(crate) const DEMO12_K_MIN: usize = 3;

const EMBEDDED_DEMO12_JSON: &str = include_str!("../../../../bench/demo-12/demo-12.json");

/// Modelo pineado al correr (id + digest + nota del envelope).
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub(crate) struct Demo12Model {
    #[serde(default)]
    pub(crate) id: String,
    #[serde(default)]
    pub(crate) digest: String,
    #[serde(default)]
    pub(crate) note: String,
}

/// Versiones pineadas por manifiesto.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub(crate) struct Demo12Pinned {
    #[serde(default)]
    pub(crate) kitty: String,
    #[serde(default)]
    pub(crate) ollama: String,
    #[serde(default)]
    pub(crate) rival: String,
}

/// Gates apagados declarados (trampa conocida: el bench aísla latencia base).
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub(crate) struct Demo12GatesOff {
    #[serde(default)]
    pub(crate) autorun_hops: usize,
    #[serde(default)]
    pub(crate) knowledge: bool,
    #[serde(default)]
    pub(crate) memory: bool,
    #[serde(default)]
    pub(crate) effort_router: String,
    #[serde(default)]
    pub(crate) note: String,
}

/// Objeto fairness embebido (el JSON manda; el runner lo valida + lo copia).
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub(crate) struct Demo12Fairness {
    #[serde(default)]
    pub(crate) protocol: String,
    #[serde(default)]
    pub(crate) model: Demo12Model,
    #[serde(default)]
    pub(crate) backend: String,
    #[serde(default)]
    pub(crate) pinned_versions: Demo12Pinned,
    #[serde(default)]
    pub(crate) max_turns: usize,
    #[serde(default)]
    pub(crate) timeout_secs_per_exercise: u64,
    #[serde(default)]
    pub(crate) k: usize,
    #[serde(default)]
    pub(crate) grade: String,
    #[serde(default)]
    pub(crate) gates_off: Demo12GatesOff,
    #[serde(default)]
    pub(crate) exclusivity: Vec<String>,
    #[serde(default)]
    pub(crate) declared_asymmetries: Vec<String>,
    #[serde(default)]
    pub(crate) verdicts: Vec<String>,
    #[serde(default)]
    pub(crate) stats: String,
}

/// Envelope versionado demo-12 (fairness + 12 ejercicios esquema JsonExercise).
#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct Demo12Suite {
    pub(crate) suite: String,
    pub(crate) version: String,
    pub(crate) fairness: Demo12Fairness,
    pub(crate) exercises: Vec<JsonExercise>,
}

/// Parse del envelope embebido (sin IO: justicia disponible offline).
pub(crate) fn load_demo12_suite() -> Result<Demo12Suite> {
    let suite: Demo12Suite = serde_json::from_str(EMBEDDED_DEMO12_JSON)
        .context("demo-12 embebido no parsea (envelope versionado roto)")?;
    validate_demo12_suite(&suite)?;
    Ok(suite)
}

/// Validación espejo de `bench/demo-12/validate.py` (envelope + justicia +
/// un-estilo-de-check por ejercicio). El runner la exige antes de correr.
pub(crate) fn validate_demo12_suite(suite: &Demo12Suite) -> Result<()> {
    if suite.suite != "demo-12" {
        anyhow::bail!("demo-12: suite != 'demo-12' (got {:?})", suite.suite);
    }
    if suite.version != DEMO12_VERSION {
        anyhow::bail!(
            "demo-12: version debe ser {:?} pineada (got {:?})",
            DEMO12_VERSION,
            suite.version
        );
    }
    if suite.fairness.k < DEMO12_K_MIN {
        anyhow::bail!(
            "demo-12: fairness.k debe ser >= {} (got {})",
            DEMO12_K_MIN,
            suite.fairness.k
        );
    }
    if suite.fairness.timeout_secs_per_exercise != DEMO12_TIMEOUT_SECS {
        anyhow::bail!(
            "demo-12: fairness.timeout debe ser {}s (got {}s)",
            DEMO12_TIMEOUT_SECS,
            suite.fairness.timeout_secs_per_exercise
        );
    }
    if suite.fairness.max_turns != DEMO12_MAX_TURNS {
        anyhow::bail!(
            "demo-12: fairness.max_turns debe ser {} (got {})",
            DEMO12_MAX_TURNS,
            suite.fairness.max_turns
        );
    }
    // Justicia completa espejo de validate.py (lee todos los campos para que
    // el envelope no pueda perder protocolo por drift silencioso).
    if suite.fairness.protocol.is_empty() {
        anyhow::bail!("demo-12: fairness.protocol vacío");
    }
    if suite.fairness.backend != "ollama" {
        anyhow::bail!(
            "demo-12: fairness.backend debe ser ollama (got {:?})",
            suite.fairness.backend
        );
    }
    if suite.fairness.model.id.is_empty()
        || suite.fairness.model.digest.is_empty()
        || suite.fairness.model.note.is_empty()
    {
        anyhow::bail!("demo-12: fairness.model.{{id,digest,note}} incompleto");
    }
    if suite.fairness.pinned_versions.kitty.is_empty()
        || suite.fairness.pinned_versions.ollama.is_empty()
        || suite.fairness.pinned_versions.rival.is_empty()
    {
        anyhow::bail!("demo-12: fairness.pinned_versions incompleto");
    }
    if suite.fairness.grade.is_empty() {
        anyhow::bail!("demo-12: fairness.grade vacío");
    }
    if suite.fairness.gates_off.autorun_hops != 0
        || suite.fairness.gates_off.knowledge
        || suite.fairness.gates_off.memory
        || suite.fairness.gates_off.effort_router.is_empty()
        || suite.fairness.gates_off.note.is_empty()
    {
        anyhow::bail!(
            "demo-12: fairness.gates_off debe ser hops=0, sin knowledge/memory, router declarado"
        );
    }
    if suite.fairness.exclusivity.is_empty() {
        anyhow::bail!("demo-12: fairness.exclusivity vacío");
    }
    if suite.fairness.declared_asymmetries.is_empty() {
        anyhow::bail!("demo-12: fairness.declared_asymmetries vacío");
    }
    for verdict in ["pass", "fail", "error", "timeout"] {
        if !suite.fairness.verdicts.iter().any(|v| v == verdict) {
            anyhow::bail!("demo-12: fairness.verdicts sin {verdict:?}");
        }
    }
    if suite.fairness.stats.is_empty() {
        anyhow::bail!("demo-12: fairness.stats vacío");
    }
    if suite.exercises.len() != DEMO12_N_EXERCISES {
        anyhow::bail!(
            "demo-12: exercises debe tener {} fijos (got {})",
            DEMO12_N_EXERCISES,
            suite.exercises.len()
        );
    }
    let mut ids = std::collections::HashSet::new();
    for ex in &suite.exercises {
        if ex.id.is_empty() || !ids.insert(ex.id.clone()) {
            anyhow::bail!("demo-12: ids deben ser 12 únicas no vacías");
        }
        if ex.lang.is_empty() || ex.prompt.is_empty() {
            anyhow::bail!("demo-12: {}: lang/prompt vacío", ex.id);
        }
        if ex.prompt.contains("/tmp/") || ex.prompt.starts_with('/') {
            anyhow::bail!("demo-12: {}: prompt con path absoluto", ex.id);
        }
        let polyglot = !ex.check_str.is_empty() && !ex.check_file.is_empty();
        let tb2 = !ex.check_type.is_empty()
            && (!ex.check_path.is_empty() || ex.check_type == "always_pass");
        if polyglot == tb2 {
            anyhow::bail!(
                "demo-12: {}: exactamente un estilo de check (polyglot XOR tb2)",
                ex.id
            );
        }
        if tb2
            && !matches!(
                ex.check_type.as_str(),
                "file_exists"
                    | "file_not_exists"
                    | "file_executable"
                    | "file_content_exact"
                    | "always_pass"
            )
        {
            anyhow::bail!("demo-12: {}: check_type desconocido", ex.id);
        }
    }
    Ok(())
}

/// Convierte los 12 ejercicios al `BenchItem` del harness (falla si alguno no
/// tiene check válido — nunca corre media suite en silencio).
pub(crate) fn demo12_to_bench_items(exercises: &[JsonExercise]) -> Result<Vec<BenchItem>> {
    let mut items = Vec::with_capacity(exercises.len());
    for ex in exercises {
        match json_to_bench_item(ex) {
            Some(item) => items.push(item),
            None => anyhow::bail!("demo-12: {}: sin check válido", ex.id),
        }
    }
    Ok(items)
}

/// p50 sobre walls reales (reutiliza la semántica de `ab::p50_ms`).
pub(crate) fn p50_wall(mut walls: Vec<f64>) -> f64 {
    crate::bench::ab::p50_ms(std::mem::take(&mut walls))
}

/// p99 sobre walls reales (estadística fairness: p50/p99 por celda).
pub(crate) fn p99_wall(mut walls: Vec<f64>) -> f64 {
    if walls.is_empty() {
        return 0.0;
    }
    walls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((walls.len() as f64 * 0.99).ceil() as usize)
        .saturating_sub(1)
        .min(walls.len() - 1);
    walls[idx]
}

/// Wall real para timeout: `run_one_bench` reporta 30000.0 hardcodeado en
/// timeout (bug documentado en la serie tb2); la justicia exige p50/p99 sobre
/// wall real (60000 en celda 60s), nunca ese display. Puro y testeado.
pub(crate) fn wall_for_verdict(verdict: &str, reported_ms: f64, wall_ms: f64) -> f64 {
    if verdict == "timeout" {
        wall_ms
    } else {
        reported_ms
    }
}

/// Versión kitty pineada (Cargo, en tiempo de compilación).
pub(crate) fn kitty_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Versión ollama best-effort (`ollama --version`; nunca falla el bench).
pub(crate) fn ollama_version() -> String {
    match std::process::Command::new("ollama")
        .arg("--version")
        .output()
    {
        Ok(out) => {
            let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if text.is_empty() {
                "unknown".to_string()
            } else {
                text.split_whitespace()
                    .last()
                    .unwrap_or("unknown")
                    .to_string()
            }
        }
        Err(_) => "unknown".to_string(),
    }
}

/// Digest del modelo best-effort (`ollama show <model> --modelfile`, línea
/// `FROM .../sha256-...`; nunca falla el bench, queda "unknown" + warn).
pub(crate) fn model_digest(model: &str) -> String {
    match std::process::Command::new("ollama")
        .arg("show")
        .arg(model)
        .arg("--modelfile")
        .output()
    {
        Ok(out) => {
            let text = String::from_utf8_lossy(&out.stdout);
            for line in text.lines() {
                let trimmed = line.trim();
                if let Some(rest) = trimmed.strip_prefix("FROM ") {
                    if let Some(sha) = rest.split("sha256-").nth(1) {
                        let digest: String =
                            sha.split_whitespace().next().unwrap_or("").to_string();
                        if !digest.is_empty() {
                            return format!("sha256-{digest}");
                        }
                    }
                    let fallback = rest.trim().to_string();
                    if !fallback.is_empty() {
                        return fallback;
                    }
                }
            }
            "unknown".to_string()
        }
        Err(_) => "unknown".to_string(),
    }
}

/// HEAD git best-effort (serie congela HEAD; el manifiesto lo fija).
pub(crate) fn git_head() -> String {
    match std::process::Command::new("git")
        .arg("rev-parse")
        .arg("HEAD")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
    {
        Ok(out) => {
            let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if text.is_empty() {
                "unknown".to_string()
            } else {
                text
            }
        }
        Err(_) => "unknown".to_string(),
    }
}

/// Timestamp UTC best-effort (`date -u`; nunca falla el bench).
pub(crate) fn utc_timestamp() -> String {
    match std::process::Command::new("date")
        .arg("-u")
        .arg("+%Y-%m-%dT%H:%M:%SZ")
        .output()
    {
        Ok(out) => {
            let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if text.is_empty() {
                "unknown".to_string()
            } else {
                text
            }
        }
        Err(_) => "unknown".to_string(),
    }
}

/// Serie exclusiva: `ollama ps` vacío antes de la rep (warn, no fail; la
/// disciplina la hace el llamador con `ollama stop` entre reps).
pub(crate) fn ollama_ps_status() -> (bool, String) {
    match std::process::Command::new("ollama").arg("ps").output() {
        Ok(out) => {
            let text = String::from_utf8_lossy(&out.stdout).to_string();
            let lines: Vec<&str> = text.lines().collect();
            // Header + 0 filas de modelo = vacío.
            let empty = lines.len() <= 1;
            let summary = if empty {
                "empty".to_string()
            } else {
                format!("{} model row(s)", lines.len() - 1)
            };
            (empty, summary)
        }
        Err(_) => (false, "ollama ps failed".to_string()),
    }
}

/// Runner demo-12: 12 ejercicios, UNA rep por invocación (la serie hace k=3
/// invocaciones con distintos `--out`). Justicia forzada + manifiesto por run
/// con desglose wall por ejercicio (alimenta Fase F).
pub(crate) async fn run_demo12_bench(
    cli: &BenchCliCtx,
    cfg: &BenchConfig,
    out: &PathBuf,
) -> Result<()> {
    let suite = load_demo12_suite()?;
    let fairness_value: serde_json::Value =
        serde_json::from_str(EMBEDDED_DEMO12_JSON).context("demo-12 fairness copy")?;
    let fairness_json = fairness_value
        .get("fairness")
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    // Trampa --exercises: esta suite NUNCA la usa (el llamador avisa); si el
    // bench se invocó con --exercises + demo-12, el mod ya advirtió. Aquí solo
    // corremos la suite propia embebida.
    let items = demo12_to_bench_items(&suite.exercises)?;

    // Justicia forzada (warn si el invocador pidió otra cosa).
    if cli.backend != "ollama" {
        warn!(
            backend = %cli.backend,
            "demo-12 fija backend ollama (fairness); se registra el backend real en el manifiesto"
        );
    }
    if cfg.max_iterations != DEMO12_MAX_TURNS {
        warn!(
            got = cfg.max_iterations,
            "demo-12 fuerza max_turns=20 (fairness); se ignora el override bench-5"
        );
    }
    if cfg.autorun_hops != 0 {
        warn!("demo-12 fuerza autorun_hops=0 (gates_off)");
    }
    if cfg.enable_knowledge {
        warn!("demo-12 fuerza knowledge off (gates_off)");
    }
    if cfg.use_memory {
        warn!("demo-12 fuerza memory off (gates_off; usar --no-memory para silenciar)");
    }
    if cfg.exercise_timeout_secs != DEMO12_TIMEOUT_SECS {
        warn!(
            got = cfg.exercise_timeout_secs,
            "demo-12 fuerza timeout 60s/celda (fairness)"
        );
    }
    if cfg.limit != DEMO12_N_EXERCISES {
        info!(
            got = cfg.limit,
            "demo-12 ignora --limit (siempre 12 fijos; --limit solo smoke e invalida evidencia)"
        );
    }

    let mut demo_cfg = cfg.clone();
    demo_cfg.max_iterations = DEMO12_MAX_TURNS;
    demo_cfg.autorun_hops = 0;
    demo_cfg.exercise_timeout_secs = DEMO12_TIMEOUT_SECS;
    demo_cfg.use_memory = false;
    demo_cfg.enable_knowledge = false;

    let (ps_empty_before, ps_summary_before) = ollama_ps_status();
    if !ps_empty_before {
        warn!(
            ps = %ps_summary_before,
            "demo-12: ollama ps no vacío antes de la rep (serie exige 1 modelo residente)"
        );
    }

    println!("Kitty Code demo-12 (justicia embebida)");
    println!(
        "  Suite: demo-12 v{} (12 fijos, 1 rep/invocación, serie k>=3)",
        suite.version
    );
    println!("  Model: {}", cfg.model);
    println!("  Backend: {}", cli.backend);
    println!("  Max turns: {DEMO12_MAX_TURNS} (fairness, ambos lados)");
    println!("  Timeout: {DEMO12_TIMEOUT_SECS}s/celda (fairness, wall real)");
    println!("  Gates: autorun_hops=0, knowledge off, memory off, router pass-through");
    println!("  ollama ps antes: {ps_summary_before}");
    println!();

    let mut store: Option<kitty_memory::EpisodeStore> = None;
    let mut results: Vec<BenchResult> = Vec::with_capacity(DEMO12_N_EXERCISES);
    for (i, ex) in items.iter().enumerate() {
        print!("[{:2}/{:2}] ", i + 1, DEMO12_N_EXERCISES);
        let wall_start = Instant::now();
        let r = run_one_bench(ex, &demo_cfg, &mut store).await?;
        let wall_ms = wall_start.elapsed().as_millis() as f64;
        let latency_ms = wall_for_verdict(&r.verdict, r.latency_ms, wall_ms);
        results.push(BenchResult { latency_ms, ..r });
    }

    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();
    let rate = if total > 0 {
        100.0 * passed as f64 / total as f64
    } else {
        0.0
    };
    let walls: Vec<f64> = results.iter().map(|r| r.latency_ms).collect();
    let p50 = p50_wall(walls.clone());
    let p99 = p99_wall(walls.clone());
    let rango = if walls.is_empty() {
        (0.0, 0.0)
    } else {
        (
            walls.iter().cloned().fold(f64::INFINITY, f64::min),
            walls.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        )
    };
    let by_lang = tally_by_language(&results);
    let (ps_empty_after, ps_summary_after) = ollama_ps_status();

    println!();
    println!("RESULTS (demo-12 evidencia, 1 rep)");
    println!("Passed: {passed}/{total} ({rate:.1}%)");
    for (lang, t) in &by_lang {
        let lang_rate = if t.attempted > 0 {
            100.0 * t.passed as f64 / t.attempted as f64
        } else {
            0.0
        };
        println!(
            "  {lang}: attempted={} passed={} failed={} errored={} ({lang_rate:.1}%)",
            t.attempted, t.passed, t.failed, t.errored
        );
    }
    println!(
        "  wall p50={p50:.0}ms p99={p99:.0}ms rango=[{:.0},{:.0}]ms",
        rango.0, rango.1
    );
    println!("  ollama ps después: {ps_summary_after} (hacer `ollama stop` entre reps)");

    let digest = model_digest(&cfg.model);
    if digest == "unknown" {
        warn!("no se pudo pinejar el digest del modelo (ollama show falló)");
    }
    let output = serde_json::json!({
        "suite": "demo-12",
        "version": suite.version,
        "fairness": fairness_json,
        "model": cfg.model,
        "model_digest": digest,
        "backend": cli.backend,
        "url": cli.url,
        "pinned_versions": {
            "kitty": kitty_version(),
            "ollama": ollama_version(),
            "rival": "NO_CORRIDA-en-este-run (rival se mide aparte; ver manifiesto comparativo)",
        },
        "max_turns": DEMO12_MAX_TURNS,
        "timeout_secs_per_exercise": DEMO12_TIMEOUT_SECS,
        "k_required": suite.fairness.k,
        "k_note": "este run = 1 rep de 12; la serie exige k>=3 invocaciones con distintos --out",
        "gates_off_actual": {
            "autorun_hops": 0,
            "knowledge": false,
            "memory": false,
            "effort_router": "pass-through (effort_off_in_bench por celda)",
        },
        "exclusivity": {
            "ps_empty_before": ps_empty_before,
            "ps_before": ps_summary_before,
            "ps_empty_after": ps_empty_after,
            "ps_after": ps_summary_after,
            "discipline": "1 modelo residente; `ollama stop` entre TODAS las reps, ps vacío verificado",
        },
        "context": {
            "context_window": cfg.context_window,
            "thinking_budget": cfg.thinking_budget,
            "profile": cfg.profile.name,
            "bash_timeout": cfg.bash_timeout,
            "ollama_server_ctx": std::env::var("OLLAMA_CONTEXT_LENGTH").unwrap_or("unset (server default; servicio fija 16384)".to_string()),
            "kitty_num_ctx_env": std::env::var(kitty_runtime::client::ENV_NUM_CTX).unwrap_or("unset (probe nativo con techo 32768)".to_string()),
        },
        "git_head": git_head(),
        "timestamp_utc": utc_timestamp(),
        "total": total,
        "passed": passed,
        "rate_pct": rate,
        "wall_p50_ms": p50,
        "wall_p99_ms": p99,
        "wall_rango_ms": [rango.0, rango.1],
        "memory": false,
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
            "wall_ms": r.latency_ms,
            "hops": r.hops,
            "iters": r.iters,
            "effort_off_in_bench": r.effort_off_in_bench,
        })).collect::<Vec<_>>(),
    });
    std::fs::write(out, serde_json::to_string_pretty(&output)?)?;
    println!("Results saved to {}", out.display());
    Ok(())
}

#[cfg(test)]
mod demo12_tests {
    use super::{
        demo12_to_bench_items, load_demo12_suite, p50_wall, p99_wall, validate_demo12_suite,
        wall_for_verdict, DEMO12_N_EXERCISES, DEMO12_VERSION,
    };

    #[test]
    fn embedded_envelope_validates() {
        let suite = load_demo12_suite().expect("envelope embebido válido");
        assert_eq!(suite.version, DEMO12_VERSION);
        assert_eq!(suite.exercises.len(), DEMO12_N_EXERCISES);
        assert!(suite.fairness.k >= 3);
    }

    #[test]
    fn all_12_convert_to_bench_items() {
        let suite = load_demo12_suite().unwrap();
        let items = demo12_to_bench_items(&suite.exercises).unwrap();
        assert_eq!(items.len(), DEMO12_N_EXERCISES);
        let mut ids: Vec<&str> = items.iter().map(|e| e.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), DEMO12_N_EXERCISES);
    }

    #[test]
    fn wall_percentiles_golden() {
        // Fixture ordenado 1..=100: p50 = 50.5, p99 = 99.
        let walls: Vec<f64> = (1..=100).map(|v| v as f64).collect();
        assert!((p50_wall(walls.clone()) - 50.5).abs() < 1e-9);
        assert!((p99_wall(walls.clone()) - 99.0).abs() < 1e-9);
        assert_eq!(p50_wall(vec![]), 0.0);
        assert_eq!(p99_wall(vec![]), 0.0);
        // Un solo wall: p50 == p99 == ese wall.
        assert_eq!(p50_wall(vec![1234.0]), 1234.0);
        assert_eq!(p99_wall(vec![1234.0]), 1234.0);
    }

    #[test]
    fn timeout_uses_real_wall_others_keep_reported() {
        assert_eq!(wall_for_verdict("timeout", 30000.0, 60012.0), 60012.0);
        assert_eq!(wall_for_verdict("pass", 1234.0, 1250.0), 1234.0);
        assert_eq!(wall_for_verdict("fail", 2000.0, 2010.0), 2000.0);
        assert_eq!(wall_for_verdict("error", 500.0, 510.0), 500.0);
    }

    #[test]
    fn validate_rejects_short_suite() {
        let mut suite = load_demo12_suite().unwrap();
        suite.exercises.truncate(11);
        assert!(validate_demo12_suite(&suite).is_err());
    }

    #[test]
    fn validate_rejects_weak_k() {
        let mut suite = load_demo12_suite().unwrap();
        suite.fairness.k = 1;
        assert!(validate_demo12_suite(&suite).is_err());
    }

    #[test]
    fn validate_rejects_version_drift() {
        let mut suite = load_demo12_suite().unwrap();
        suite.version = "9.9.9".to_string();
        assert!(validate_demo12_suite(&suite).is_err());
    }
}

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

/// Dirty git best-effort (`git status --porcelain` vacío = limpio).
pub(crate) fn git_dirty() -> bool {
    match std::process::Command::new("git")
        .arg("status")
        .arg("--porcelain")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
    {
        Ok(out) => !String::from_utf8_lossy(&out.stdout).trim().is_empty(),
        Err(_) => true,
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

    // Runmark Capa 0 (Lane B): fingerprint de celda + rechazo de rerun ANTES
    // del primer LLM. scored/offline: memory=off, systemone=off, seed fijo.
    // `rep` = 0 siempre (1 rep/invocación, k=1; hueco k=2 futuro, ppc k=1).
    let runmark = crate::bench::runmark::Runmark {
        model: cfg.model.clone(),
        digest: model_digest(&cfg.model),
        backend: cli.backend.to_string(),
        profile: cfg.profile.name.to_string(),
        ctx: cfg.context_window as u32,
        max_iter: DEMO12_MAX_TURNS,
        hops: 0,
        suite: "demo-12".to_string(),
        dataset_sha: crate::bench::runmark::sha_of_str(EMBEDDED_DEMO12_JSON),
        timeout_cell: DEMO12_TIMEOUT_SECS,
        seed: 42,
        rep: 0,
        git_head: git_head(),
        dirty: git_dirty(),
        memory: false,
        systemone: crate::bench::common::bench_systemone_label(),
        toolset: crate::bench::common::bench_toolset_label(),
    };
    // Ledger SQLite (única fuente de verdad): pre-chequeo + `begin` que
    // reserva la celda ANTES del primer LLM y rechaza el duplicado
    // (fingerprint,arm,rep) con exit 3.
    if let Err(e) = crate::bench::runmark::check_rerun_rejected(out, &runmark.fingerprint()) {
        eprintln!("integridad: {e:#}");
        std::process::exit(crate::bench::runmark::EXIT_INTEGRITY);
    }
    // B1: tripwire pre-LLM ANTES de `begin` (como ppc.rs:279): si el run ya
    // viene marcado con leak, integridad sin reservar fila — una fila con
    // finished_at NULL y dirty/leaked=0 pasaría `is_clean` (residual gate).
    if kitty_harness::guard::is_run_leaked() {
        eprintln!("integridad: run marcado con leak (tripwire): demo-12 no corre");
        std::process::exit(crate::bench::runmark::EXIT_INTEGRITY);
    }
    let ledger_path = crate::bench::ledger::ledger_db_path(out);
    let ledger = match crate::bench::ledger::Ledger::open(&ledger_path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("integridad: no se pudo abrir el ledger {}: {e:#}", ledger_path.display());
            std::process::exit(crate::bench::runmark::EXIT_INTEGRITY);
        }
    };
    let run_id = match ledger.begin(&crate::bench::ledger::LedgerStamp {
        fingerprint: runmark.fingerprint(),
        suite: "demo-12".to_string(),
        arm: "default".to_string(),
        rep: runmark.rep as i64,
        git_head: runmark.git_head.clone(),
        dirty: runmark.dirty,
        started_at: utc_timestamp(),
    }) {
        Ok(id) => id,
        Err(e) => {
            eprintln!("integridad: {e:#}");
            std::process::exit(crate::bench::runmark::EXIT_INTEGRITY);
        }
    };

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
    // F2: p99 censored — con timeouts la cola está truncada (cap, no wall).
    let timed_out = results.iter().filter(|r| r.verdict == "timeout").count();
    let (_p99_check, p99_censored) =
        kitty_harness::comparison::p99_ms_censored(walls.clone(), timed_out);
    debug_assert!((_p99_check - p99).abs() < 1e-9, "p99 local == comparison");
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
        "  wall p50={p50:.0}ms p99={p99:.0}ms rango=[{:.0},{:.0}]ms{}",
        rango.0, rango.1,
        if p99_censored { " (p99 censored: hubo timeouts)" } else { "" }
    );
    println!("  ollama ps después: {ps_summary_after} (hacer `ollama stop` entre reps)");

    // B1: tripwire intra-run — deny durante los probes ⇒ integridad (exit 3).
    if kitty_harness::guard::is_run_leaked() {
        let _ = ledger.mark_leaked(&run_id);
        let _ = ledger.finish(
            &run_id,
            &utc_timestamp(),
            crate::bench::runmark::EXIT_INTEGRITY,
        );
        eprintln!("integridad: run marcado con leak durante probes (tripwire): sin resume");
        std::process::exit(crate::bench::runmark::EXIT_INTEGRITY);
    }
    // B2: SIGNAL/NO-SIGNAL también en demo-12 (hoy solo PPC lo llamaba).
    // 1 rep/invocación (k=1) ⇒ siempre NoSignal + exit 2 (inconcluso, nunca
    // publicable): la serie k>=3 decide fuera de este proceso.
    // Ledger limpio = dirty==0 && leaked==0 (único gate publicable).
    let ledger_clean = ledger.is_clean(&run_id).unwrap_or(false);
    let signal = kitty_harness::comparison::decide_signal(passed, total, passed, total, ledger_clean, 1);
    let exit_code = match &signal {
        kitty_harness::comparison::Signal::Signal => crate::bench::runmark::EXIT_PASS,
        kitty_harness::comparison::Signal::NoSignal(_) => {
            crate::bench::runmark::EXIT_INCONCLUSIVE
        }
    };
    debug_assert!(
        matches!(signal, kitty_harness::comparison::Signal::NoSignal(_)),
        "demo12 1 rep siempre NoSignal (k=1)"
    );

    let digest = runmark.digest.clone();
    if digest == "unknown" {
        warn!("no se pudo pinejar el digest del modelo (ollama show falló)");
    }
    // Hoisted fuera del `json!` gigante (límite de recursión del macro):
    // se construyen como `Value` y se inyectan como variables.
    let by_language_json: std::collections::BTreeMap<String, serde_json::Value> = by_lang
        .iter()
        .map(|(lang, t)| {
            (
                lang.clone(),
                serde_json::json!({
                    "attempted": t.attempted,
                    "passed": t.passed,
                    "failed": t.failed,
                    "errored": t.errored,
                }),
            )
        })
        .collect();
    let details_json: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "id": r.id,
                "lang": r.lang,
                "passed": r.passed,
                "verdict": r.verdict,
                "latency_ms": r.latency_ms,
                "wall_ms": r.latency_ms,
                "hops": r.hops,
                "iters": r.iters,
                "effort_off_in_bench": r.effort_off_in_bench,
            })
        })
        .collect();
    let ledger_db_str = ledger_path.display().to_string();
    let fingerprint_str = runmark.fingerprint();
    let git_head_str = runmark.git_head.clone();
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
        "signal": signal.to_string(),
        "exit_code": exit_code,
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
        "git_head": git_head_str,
        "git_dirty": runmark.dirty,
        "run_id": run_id,
        "ledger_db": ledger_db_str,
        "runmark_fingerprint": fingerprint_str,
        "timestamp_utc": utc_timestamp(),
        "total": total,
        "passed": passed,
        "rate_pct": rate,
        "wall_p50_ms": p50,
        "wall_p99_ms": p99,
        "wall_p99_censored": p99_censored,
        "wall_rango_ms": [rango.0, rango.1],
        "memory": false,
        "by_language": by_language_json,
        "details": details_json,
    });
    std::fs::write(out, serde_json::to_string_pretty(&output)?)?;
    println!("Results saved to {}", out.display());
    // Sidecar runmark tmp+fsync+rename (caché legible; la decisión vive en el
    // ledger). Ledger finish con el exit decidido.
    crate::bench::runmark::write_sidecar_atomic(out, &runmark)?;
    if let Err(e) = ledger.finish(&run_id, &utc_timestamp(), exit_code) {
        warn!("ledger finish falló (best-effort): {e:#}");
    }
    println!(
        "Runmark saved to {} (fingerprint {})",
        crate::bench::runmark::sidecar_path(out).display(),
        runmark.fingerprint()
    );
    // B2: 1 rep ⇒ exit 2 (inconcluso), no Ok 0. La serie k>=3 decide fuera.
    println!("demo-12 {signal} (exit {exit_code})");
    std::process::exit(exit_code)
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

    // === B2: 1 rep ⇒ exit 2 (inconcluso), no Ok 0 ===

    #[test]
    fn single_rep_siempre_exit_2_inconcluso() {
        // Una invocación demo-12 = 1 rep (k=1<3): Wilson no discrimina aunque
        // sea 12/12. El runner debe reportar NoSignal + exit 2, nunca 0.
        let signal = kitty_harness::comparison::decide_signal(12, 12, 12, 12, true, 1);
        assert!(
            matches!(signal, kitty_harness::comparison::Signal::NoSignal(_)),
            "1 rep ⇒ NoSignal, fue {signal}"
        );
        assert_eq!(
            crate::bench::runmark::EXIT_INCONCLUSIVE,
            2,
            "exit 2 = inconcluso (contrato unificado B2)"
        );
        // Con k=3 y separación real sí habría señal (control Wilson).
        let sig3 = kitty_harness::comparison::decide_signal(0, 8, 8, 8, true, 3);
        assert!(sig3.is_signal());
    }

    // === Fix begin/leak: el tripwire pre-LLM va ANTES de `begin` (ppc.rs:279)
    // y el intra-run cierra con mark_leaked+finish antes del exit 3 ===

    #[test]
    fn leaked_run_cerrado_no_pasa_is_clean() {
        // Contrato que el fix exige: un run con leak queda CERRADO
        // (finished_at NOT NULL, exit 3) e `is_clean` es false — nunca una
        // fila residual con finished_at NULL y dirty/leaked=0.
        let tmp = tempfile::tempdir().expect("tmp ledger demo12");
        let db = tmp.path().join("leak.ledger.db");
        let l = crate::bench::ledger::Ledger::open(&db).expect("open ledger");
        let id = l
            .begin(&crate::bench::ledger::LedgerStamp {
                fingerprint: "fp-demo12-leak-fix".to_string(),
                suite: "demo-12".to_string(),
                arm: "default".to_string(),
                rep: 0,
                git_head: "test".to_string(),
                dirty: false,
                started_at: "2026-10-24T00:00:00Z".to_string(),
            })
            .expect("begin reserva la celda");
        // Camino tripwire intra-run (mark + finish antes del exit 3).
        l.mark_leaked(&id).expect("mark_leaked");
        l.finish(
            &id,
            "2026-10-24T00:01:00Z",
            crate::bench::runmark::EXIT_INTEGRITY,
        )
        .expect("finish cierra la fila");
        assert!(
            !l.is_clean(&id).expect("is_clean debe responder"),
            "run con leak ⇒ is_clean false (gate cerrado)"
        );
        drop(l);
        // Fila cerrada en disco: finished_at NOT NULL + exit 3.
        let conn = rusqlite::Connection::open(&db).expect("reopen ledger");
        let (fin, code): (Option<String>, Option<i64>) = conn
            .query_row(
                "SELECT finished_at, exit_code FROM runs WHERE run_id = ?1",
                rusqlite::params![id],
                |row| Ok((row.get(0).unwrap_or(None), row.get(1).unwrap_or(None))),
            )
            .expect("fila presente");
        assert!(fin.is_some(), "fila cerrada: finished_at NOT NULL");
        assert_eq!(
            code,
            Some(crate::bench::runmark::EXIT_INTEGRITY as i64),
            "exit de integridad (3)"
        );
    }
}

//! `bench ppc` — Harness² offline secuencial k=1 (Lane D implementa).
//!
//! Lane B (este fichero, Pass 1 Sketch): solo cablea el dispatch en
//! `bench/mod.rs` y deja este stub compilable. La implementación real
//! (proposer = mismo SLM triage temp baja, candidatos deltas config,
//! compose determinista por verifier, tabla `memory_items` con gate,
//! recall sin `fill_recent`, DB aislada, memory=off, runmark+ledger)
//! la provee Lane D sustituyendo este stub. — HECHO abajo (Lane D).
//!
//! Contrato que Lane D debe cumplir (Capa 0/2):
//! - scored/offline: `use_memory=false` + DB aislada (nunca `default_db_path`)
//! - runmark BLAKE3 + sidecar `<out>.runmark.json` tmp+fsync+rename
//! - `check_rerun_rejected` ANTES del primer LLM
//! - BenchPolicy ya deniega bench/** pre-dispatch (harness)
//! - exit 0 pass / 1 gate-fail / 2 inconcluso (k=1, sin CI) / 3 integridad
//! - SIGNAL solo con Wilson separado + ledger limpio (`comparison::decide_signal`)
//! - p99 con `timed_out>0` ⇒ censored (`comparison::p99_ms_censored`)
//!
//! Recortes Oracle aplicados por Lane D (fijan lo que el contrato deja abierto):
//! - **k = 1 fijo** (`PPC_K`); k=2 se rechaza explícito aunque el schema lo
//!   soporte — ver [`ppc_k_or_reject`]. (El "k=1→k=2" del contrato general es
//!   roadmap v0.2: exige serie +2 tareas CI>0 y wall/tokens ≤2.5x, no este lane.)
//! - **`limit ≤ 3`** tareas (`PPC_MAX_LIMIT`); más ⇒ rechazo.
//! - **Secuencial, un residente**: bucles `for`, sin `spawn`, sin paralelo.
//! - **Cap `≤ 3×tareas+1` llamadas** al residente; cada probe cuenta 1.
//! - **Candidatos = deltas de config sobre los knobs que `BenchConfig` expone**
//!   (`thinking_budget`, `max_iterations`, `autorun_hops`). prompt style /
//!   tool subset / temp quedan FIJOS: mapearlos exige `router.rs`, vetado en
//!   este lane (sin tocar `router.rs`); el espacio aplicable se amplía cuando
//!   el router exponga deltas.
//! - **Compose determinista por verifier** ([`score_of`] + [`compose_winner`]):
//!   gana el mejor `pass-rate`, desempates por timeouts → latencia → nombre.
//!   **Sin LLM judge** y sin red en el compose.
//! - `run_id`/`runmark`: los provee Lane B vía (`cli`, `cfg`, `out`); el
//!   `run_id` es `KITTY_RUN_ID` o el fingerprint corto, y el ledger limpio es
//!   best-effort (`!git_dirty`) hasta que el ledger de Lane A aterrice.
//! - `systemone_status()` (Lane C) entra SOLO como campo volcado
//!   (`systemone_status_dump`): no hay ningún `if` sobre este valor en el
//!   fichero — verificable por inspección.
//! - La lección del ganador se guarda en [`LessonStore`] SOLO con
//!   `evidence_ref` (`ppc:{run_id}:{ganador}`); el gate de `save` (sin
//!   evidence_ref o run con leak/ledger sucio ⇒ rechazo) aplica sin cambios.
//!   En run sucio NO se guarda lección y se reporta (no es error fatal).
//! - Sin tocar `bench/mod.rs` (Lane B) ni `router.rs`.

use super::common::{
    bench_toolset_label, homemade_exercises, BenchConfig, BenchItem, BenchResult,
};
use super::run::run_one_bench;
use super::runmark::{
    check_rerun_rejected, sha_of_str, write_sidecar_atomic, Runmark, EXIT_GATE_FAIL,
    EXIT_INCONCLUSIVE, EXIT_INTEGRITY, EXIT_PASS,
};
use super::BenchCliCtx;
use anyhow::{Context, Result};
use kitty_harness::comparison::{decide_signal, p99_ms_censored, Signal};
use kitty_memory::lessons::{LessonStore, MemoryItem, RunProvenance};
use std::path::{Path, PathBuf};

/// k fijo del recorte Oracle (noise-floor k=1). Cualquier otro k ⇒ rechazo.
pub(crate) const PPC_K: u32 = 1;
/// Máximo de tareas por corrida PPC.
pub(crate) const PPC_MAX_LIMIT: usize = 3;
/// Presupuesto de llamadas al residente por tarea candidata.
pub(crate) const PPC_CALL_BUDGET_PER_TASK: usize = 3;

/// Error de integridad (exit 3): rerun de la misma celda, leak tripwire,
/// celda con memory=on. Se distingue de `anyhow` genérico por downcast.
#[derive(Debug)]
struct IntegrityFail(String);

impl std::fmt::Display for IntegrityFail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PPC integridad: {}", self.0)
    }
}

impl std::error::Error for IntegrityFail {}

/// Suite PPC (stub Lane B → implementación Lane D).
///
/// Mantiene la firma que Lane B cablea en `bench/mod.rs`. Escribe el
/// manifiesto en `out`, el sidecar runmark y la DB aislada de lecciones, y
/// termina con `process::exit` según el contrato (0/1/2/3).
pub(crate) async fn run_ppc_suite(
    cli: &BenchCliCtx,
    cfg: &BenchConfig,
    out: &PathBuf,
) -> Result<()> {
    match run_ppc_inner(cli, cfg, out).await {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            if e.downcast_ref::<IntegrityFail>().is_some() {
                eprintln!("error: {e:?}");
                std::process::exit(EXIT_INTEGRITY);
            }
            Err(e)
        }
    }
}

/// Candidato PPC: delta de config aplicable vía `BenchConfig`.
#[derive(Debug, Clone)]
pub(crate) struct PpcCandidate {
    pub(crate) name: String,
    pub(crate) thinking_budget: usize,
    pub(crate) max_iterations: usize,
    pub(crate) autorun_hops: usize,
}

impl PpcCandidate {
    fn apply(&self, base: &BenchConfig) -> BenchConfig {
        let mut cfg = base.clone();
        cfg.thinking_budget = self.thinking_budget;
        cfg.max_iterations = self.max_iterations;
        cfg.autorun_hops = self.autorun_hops;
        cfg.use_memory = false;
        cfg
    }
}

/// 3 candidatos = 3 deltas de config (igual al presupuesto por tarea: con
/// `limit` tareas el coste es `3×limit ≤ cap`). Nombres ordenables para que
/// el desempate del compose sea determinista.
pub(crate) fn default_candidates(base: &BenchConfig) -> Vec<(PpcCandidate, BenchConfig)> {
    let mk = |name: &str, thinking_budget: usize, max_iterations: usize, autorun_hops: usize| {
        let cand = PpcCandidate {
            name: name.to_string(),
            thinking_budget,
            max_iterations,
            autorun_hops,
        };
        let cfg = cand.apply(base);
        (cand, cfg)
    };
    vec![
        mk("baseline", base.thinking_budget, base.max_iterations, base.autorun_hops),
        mk(
            "iter-cap-bajo",
            (base.thinking_budget / 2).max(256),
            base.max_iterations.min(3),
            base.autorun_hops,
        ),
        mk("hops-cero", base.thinking_budget, base.max_iterations.min(3), 0),
    ]
}

/// Score abstracto de un probe (lo que el verifier determinista consume).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ProbeScore {
    pub(crate) passed: bool,
    pub(crate) latency_ms: f64,
    pub(crate) timed_out: bool,
}

/// Verifier determinista: `pass ⇒ 1.0`, resto `⇒ 0.0`. Sin LLM, sin red.
pub(crate) fn score_of(r: &BenchResult) -> ProbeScore {
    ProbeScore {
        passed: r.passed,
        latency_ms: r.latency_ms,
        timed_out: r.verdict == "timeout",
    }
}

/// Compose determinista por verifier: mejor pass-rate medio; desempates por
/// menos timeouts → menor latencia total → nombre (estable). Sin LLM judge.
/// Devuelve `(ganador, score_medio)`.
pub(crate) fn compose_winner(scored: &[(String, Vec<ProbeScore>)]) -> (String, f32) {
    let mut ranked: Vec<(String, f32, usize, f64)> = scored
        .iter()
        .map(|(name, v)| {
            let n = v.len().max(1) as f32;
            let avg = v.iter().map(|s| if s.passed { 1.0 } else { 0.0 }).sum::<f32>() / n;
            let timeouts = v.iter().filter(|s| s.timed_out).count();
            let lat: f64 = v.iter().map(|s| s.latency_ms).sum();
            (name.clone(), avg, timeouts, lat)
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2.cmp(&b.2))
            .then_with(|| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| a.0.cmp(&b.0))
    });
    let (name, avg, _, _) = ranked[0].clone();
    (name, avg)
}

/// Gate k: **k=2 se rechaza** aunque el schema lo soporte (recorte Oracle).
pub(crate) fn ppc_k_or_reject(k: u32) -> Result<()> {
    if k != PPC_K {
        anyhow::bail!(
            "PPC recorte Oracle: k={k} rechazado (solo k=1; k=2 exige serie +2 tareas CI>0 y wall/tokens ≤2.5x, no este lane)"
        );
    }
    Ok(())
}

/// Valida `limit` (recorte Oracle `≤ 3`) y lo recorta al dataset disponible.
pub(crate) fn validate_limit(limit: usize, available: usize) -> Result<usize> {
    if limit == 0 {
        anyhow::bail!("PPC: limit debe ser ≥ 1");
    }
    if limit > PPC_MAX_LIMIT {
        anyhow::bail!("PPC recorte Oracle: limit={limit} rechazado (máximo {PPC_MAX_LIMIT})");
    }
    if available == 0 {
        anyhow::bail!("PPC: dataset vacío");
    }
    Ok(limit.min(available))
}

/// Cap de llamadas al residente: `≤ 3×tareas+1`.
pub(crate) fn call_cap(tasks: usize) -> usize {
    PPC_CALL_BUDGET_PER_TASK * tasks + 1
}

/// Exit unificado (B2/B3, contrato runmark): 0 pass / 1 gate-fail /
/// 2 inconcluso (k<3 por falta de reps o n<3 tareas o CI solapa con
/// evidencia insuficiente; nunca publicable) / 3 integridad.
/// B3: k=1 ⇒ siempre NoSignal + exit 2 aunque n≥3 y el lift sea total
/// (golden 3/3 vs 0/3 y 0/8 vs 8/8 con k=1 → 2, no 0/1): la razón `k<3`
/// domina sobre `tasks`.
pub(crate) fn decide_exit(signal: &Signal, tasks: usize) -> i32 {
    match signal {
        Signal::Signal => EXIT_PASS,
        // k<3 (falta de reps) ⇒ inconcluso aunque tasks≥3 (B3 golden).
        Signal::NoSignal(reason) if reason.contains("k<3") => EXIT_INCONCLUSIVE,
        Signal::NoSignal(_) if tasks < 3 => EXIT_INCONCLUSIVE,
        Signal::NoSignal(_) => EXIT_GATE_FAIL,
    }
}

/// DB aislada de lecciones (nunca `default_db_path`): `KITTY_LESSONS_DB` o
/// `<out>.lessons.db` junto al manifiesto.
pub(crate) fn ppc_lessons_db_path(out: &Path) -> PathBuf {
    if let Ok(p) = std::env::var("KITTY_LESSONS_DB") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    let mut s = out.as_os_str().to_owned();
    s.push(".lessons.db");
    PathBuf::from(s)
}

fn git_head() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn git_dirty() -> bool {
    std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| !o.stdout.iter().all(|b| b.is_ascii_whitespace()))
        .unwrap_or(false)
}

fn env_flag(key: &str) -> bool {
    matches!(std::env::var(key).as_deref(), Ok("1") | Ok("true") | Ok("TRUE"))
}

async fn run_ppc_inner(cli: &BenchCliCtx, cfg: &BenchConfig, out: &PathBuf) -> Result<i32> {
    ppc_k_or_reject(PPC_K)?;
    if cfg.use_memory {
        return Err(IntegrityFail("memory=on en PPC (scored/offline exige use_memory=false)".to_string()).into());
    }
    if env_flag("KITTY_RUN_LEAKED") {
        return Err(IntegrityFail("run marcado con leak (tripwire): PPC no corre".to_string()).into());
    }

    let all = homemade_exercises();
    let n = validate_limit(cfg.limit, all.len())?;
    let tasks: Vec<&BenchItem> = all.iter().take(n).collect();
    let dataset_sha = sha_of_str(&tasks.iter().map(|t| t.id).collect::<Vec<_>>().join(","));

    // systemone_status (Lane C) SOLO volcado: se copia al runmark y al
    // manifiesto; ninguna decisión del fichero lo lee.
    let (_s1key, s1mode) = kitty_harness::systemone_status();
    let systemone_dump = s1mode.to_string();

    let runmark = Runmark {
        model: cfg.model.clone(),
        digest: "unknown".to_string(),
        backend: cli.backend.to_string(),
        profile: cli.profile.clone(),
        ctx: cfg.context_window as u32,
        max_iter: cfg.max_iterations,
        hops: cfg.autorun_hops,
        suite: "ppc".to_string(),
        dataset_sha,
        timeout_cell: cfg.exercise_timeout_secs,
        seed: 42,
        git_head: git_head(),
        dirty: git_dirty(),
        memory: false,
        systemone: systemone_dump.clone(),
        toolset: bench_toolset_label(),
    };
    let fingerprint = runmark.fingerprint();
    // ANTES del primer LLM: rerun de la misma celda se rechaza aquí.
    if let Err(e) = check_rerun_rejected(out, &fingerprint) {
        return Err(IntegrityFail(e.to_string()).into());
    }

    // run_id best-effort (Lane A aterriza el ledger): explícito o fingerprint.
    let run_id = std::env::var("KITTY_RUN_ID")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| fingerprint.chars().take(16).collect());
    // Ledger limpio best-effort: árbol limpio. (Sin ledger de Lane A no hay
    // run_id blind/arm que comprobar; esto se endurece cuando aterrice.)
    let ledger_clean = !runmark.dirty;

    let candidates = default_candidates(cfg);
    let cap = call_cap(n);
    let mut calls: usize = 0;
    // Secuencial, un residente: sin spawn/thread.
    let mut per_candidate: Vec<(String, Vec<ProbeScore>)> = Vec::with_capacity(candidates.len());
    let mut latencies: Vec<f64> = Vec::with_capacity(candidates.len() * n);
    let mut timed_out: usize = 0;
    let mut no_store: Option<kitty_memory::EpisodeStore> = None;
    for (cand, cand_cfg) in &candidates {
        let mut scores = Vec::with_capacity(n);
        for ex in &tasks {
            if calls >= cap {
                anyhow::bail!("PPC cap excedido: {calls} llamadas ≥ cap {cap} (3×tareas+1)");
            }
            let r = run_one_bench(ex, cand_cfg, &mut no_store).await?;
            calls += 1;
            let s = score_of(&r);
            latencies.push(s.latency_ms);
            if s.timed_out {
                timed_out += 1;
            }
            scores.push(s);
        }
        per_candidate.push((cand.name.clone(), scores));
    }
    debug_assert!(calls <= cap, "cap 3×tareas+1 violado");

    let (winner, winner_avg) = compose_winner(&per_candidate);
    let passed_of = |name: &str| {
        per_candidate
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.iter().filter(|s| s.passed).count())
            .unwrap_or(0)
    };
    let baseline_passed = passed_of("baseline");
    let winner_passed = passed_of(&winner);
    // B1+B3: tripwire intra-run — si BenchPolicy denegó durante los probes
    // (KITTY_RUN_LEAKED=1 vía guard::flag_run_leaked) el run es integridad
    // (exit 3), sin lección ni resume, aunque el pre-check inicial pasara.
    // B3: SIGNAL solo con Wilson separado + ledger limpio + k≥3 reps.
    // k=1 (PPC_K) ⇒ siempre NoSignal + exit 2 (n tareas ≠ k reps).
    if kitty_harness::guard::is_run_leaked() || env_flag("KITTY_RUN_LEAKED") {
        return Err(
            IntegrityFail("run marcado con leak durante probes (tripwire): sin lección ni resume".to_string())
                .into(),
        );
    }
    let signal = decide_signal(
        baseline_passed,
        n,
        winner_passed,
        n,
        ledger_clean,
        PPC_K as usize,
    );
    let exit_code = decide_exit(&signal, n);
    let (p99_ms, p99_censored) = p99_ms_censored(latencies.clone(), timed_out);

    // Lección del ganador: SOLO con evidence_ref; run sucio ⇒ sin lección
    // (reportado, no fatal). DB aislada abierta en blocking pool.
    let evidence = serde_json::to_string(&serde_json::json!({
        "k": PPC_K,
        "tasks": tasks.iter().map(|t| t.id).collect::<Vec<_>>(),
        "scores": per_candidate.iter().map(|(nm, v)| serde_json::json!({
            "candidate": nm,
            "passed": v.iter().filter(|s| s.passed).count(),
            "total": v.len(),
        })).collect::<Vec<_>>(),
        "llm_calls": calls,
        "cap": cap,
        "signal": signal.to_string(),
    }))?;
    let evidence_ref = format!("ppc:{run_id}:{winner}");
    let db_path = ppc_lessons_db_path(out);
    let lesson_input = MemoryItem::new(
        format!("ppc harness config delta ({n})"),
        format!("usar candidato '{winner}' (compose determinista k=1)"),
        evidence,
        "ppc".to_string(),
        run_id.clone(),
        evidence_ref.clone(),
        winner_avg.clamp(0.0, 1.0),
        0,
    );
    // B1: proveniencia real — `leaked` refleja el tripwire
    // (KITTY_RUN_LEAKED lo escribe guard::bench_policy_check al denegar),
    // no siempre `false`. Con leak ⇒ LessonStore::save rechaza (DirtyRun) ⇒
    // sin lección ni resume (lesson_saved=false, evidence_ref vacío).
    let leaked_now = kitty_harness::guard::is_run_leaked() || env_flag("KITTY_RUN_LEAKED");
    let gate = RunProvenance {
        run_id: run_id.clone(),
        leaked: leaked_now,
        ledger_dirty: !ledger_clean,
    };
    let lesson_id: (String, bool) = tokio::task::spawn_blocking(move || {
        let store = LessonStore::open(db_path.to_str().unwrap_or(":memory:"))?;
        match store.save(lesson_input, &gate) {
            Ok(id) => Ok::<_, anyhow::Error>((id, true)),
            Err(kitty_memory::lessons::LessonError::MissingEvidenceRef)
            | Err(kitty_memory::lessons::LessonError::DirtyRun(_)) => {
                Ok((String::new(), false))
            }
            Err(e) => Err(e.into()),
        }
    })
    .await
    .context("lessons store task failed")??;

    let report = serde_json::json!({
        "suite": "ppc",
        "model": cfg.model,
        "backend": cli.backend,
        "k": PPC_K,
        "limit": n,
        "tasks": tasks.iter().map(|t| t.id).collect::<Vec<_>>(),
        "candidates": candidates.iter().map(|(c, _)| &c.name).collect::<Vec<_>>(),
        "winner": winner,
        "winner_avg_score": winner_avg,
        "baseline_passed": baseline_passed,
        "winner_passed": winner_passed,
        "llm_calls": calls,
        "llm_call_cap": cap,
        "signal": signal.to_string(),
        "exit_code": exit_code,
        "p99_ms": p99_ms,
        "p99_censored": p99_censored,
        "run_id": run_id,
        "runmark_fingerprint": fingerprint,
        "systemone_status_dump": systemone_dump,
        "lesson_id": lesson_id.0,
        "lesson_evidence_ref": if lesson_id.1 { evidence_ref } else { String::new() },
        "lesson_saved": lesson_id.1,
        "ledger_clean": ledger_clean,
    });
    std::fs::write(out, serde_json::to_string_pretty(&report)?)
        .with_context(|| format!("escribir manifiesto {}", out.display()))?;
    println!("Results saved to {}", out.display());
    println!("PPC {signal} (exit {exit_code})");
    write_sidecar_atomic(out, &runmark)?;
    Ok(exit_code)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores(passed: usize, total: usize, lat: f64) -> Vec<ProbeScore> {
        (0..total)
            .map(|i| ProbeScore {
                passed: i < passed,
                latency_ms: lat,
                timed_out: false,
            })
            .collect()
    }

    #[test]
    fn limit_mayor_3_rechazado() {
        let err = validate_limit(4, 15).unwrap_err();
        assert!(err.to_string().contains("máximo"), "limit 4 debe rechazar: {err}");
        assert!(validate_limit(3, 15).unwrap() == 3);
        assert!(validate_limit(3, 2).unwrap() == 2);
        assert!(validate_limit(0, 15).is_err());
    }

    #[test]
    fn k2_explicitamente_rechazado() {
        ppc_k_or_reject(1).unwrap();
        let err = ppc_k_or_reject(2).unwrap_err();
        assert!(err.to_string().contains("k=2"), "k=2 debe rechazarse explícito: {err}");
    }

    #[test]
    fn cap_llamadas_3t_mas_1() {
        assert_eq!(call_cap(1), 4);
        assert_eq!(call_cap(2), 7);
        assert_eq!(call_cap(3), 10);
        // 3 candidatos × n tareas siempre cabe: 3n ≤ 3n+1.
        for n in 1..=3 {
            assert!(3 * n <= call_cap(n));
        }
    }

    #[test]
    fn compose_determinista_sin_llm() {
        let table = vec![
            ("baseline".to_string(), scores(1, 2, 100.0)),
            ("iter-cap-bajo".to_string(), scores(2, 2, 200.0)),
            ("hops-cero".to_string(), scores(2, 2, 100.0)),
        ];
        // hops-cero y iter-cap-bajo empatan en pass-rate; gana menor latencia.
        assert_eq!(compose_winner(&table).0, "hops-cero");
        // Empate total ⇒ nombre menor (estable).
        let tie = vec![
            ("b".to_string(), scores(1, 1, 50.0)),
            ("a".to_string(), scores(1, 1, 50.0)),
        ];
        assert_eq!(compose_winner(&tie).0, "a");
        assert_eq!(compose_winner(&table), compose_winner(&table));
    }

    #[test]
    fn verifier_pass_1_resto_0() {
        let mk = |passed: bool, verdict: &str| BenchResult {
            id: "x".to_string(),
            lang: "python".to_string(),
            passed,
            verdict: verdict.to_string(),
            latency_ms: 10.0,
            hops: None,
            iters: None,
            margin: None,
            effort_off_in_bench: false,
        };
        let p = score_of(&mk(true, "pass"));
        assert!(p.passed && !p.timed_out);
        let t = score_of(&mk(false, "timeout"));
        assert!(!t.passed && t.timed_out);
        assert!(!score_of(&mk(false, "fail")).passed);
    }

    #[test]
    fn decide_exit_sigue_contrato() {
        assert_eq!(decide_exit(&Signal::Signal, 3), EXIT_PASS);
        // k=1 sin CI (brazos <3) ⇒ inconcluso, aunque haya NO-SIGNAL.
        assert_eq!(
            decide_exit(&Signal::NoSignal("k<3 por brazo: Wilson no discrimina".into()), 1),
            EXIT_INCONCLUSIVE
        );
        assert_eq!(
            decide_exit(&Signal::NoSignal("CI solapa".into()), 3),
            EXIT_GATE_FAIL
        );
    }

    // === B2/B3: exit 2 k<3 + k=1 NoSignal aunque n≥3 ===

    #[test]
    fn exit_2_k_menor_3_aunque_n_completo() {
        // B2: demo12/PPC con 1 rep ⇒ exit 2 (inconcluso), hoy Ok 0 en demo12.
        // B3: golden 3/3 vs 0/3 y 0/8 vs 8/8 con k=1 → exit 2, no 0/1.
        let s_k1_n3 = decide_signal(0, 3, 3, 3, true, 1);
        assert!(matches!(s_k1_n3, Signal::NoSignal(_)), "k=1 ⇒ NoSignal, fue {s_k1_n3}");
        assert_eq!(decide_exit(&s_k1_n3, 3), EXIT_INCONCLUSIVE);
        let s_k1_n8 = decide_signal(0, 8, 8, 8, true, 1);
        assert!(matches!(s_k1_n8, Signal::NoSignal(_)), "k=1 ⇒ NoSignal, fue {s_k1_n8}");
        assert_eq!(
            decide_exit(&s_k1_n8, 8),
            EXIT_INCONCLUSIVE,
            "0/8 vs 8/8 con k=1 ⇒ exit 2, no 0/1"
        );
        // Mismo lift con k=3 sí es señal publicable (exit 0).
        let s_k3_n8 = decide_signal(0, 8, 8, 8, true, 3);
        assert_eq!(s_k3_n8, Signal::Signal);
        assert_eq!(decide_exit(&s_k3_n8, 8), EXIT_PASS);
    }

    #[test]
    fn run_con_leak_no_guarda_leccion() {
        // B1 gate lessons: run con leak ⇒ sin lección (DirtyRun), aunque el
        // evidence_ref sea válido. `leaked:true` real, no siempre false.
        let store = LessonStore::open_in_memory().unwrap();
        let leaked = RunProvenance {
            run_id: "run-leak-1".into(),
            leaked: true,
            ledger_dirty: false,
        };
        let item = MemoryItem::new(
            "ppc harness config delta (1)".into(),
            "usar candidato 'x'".into(),
            "{\"k\":1}".into(),
            "ppc".into(),
            "run-leak-1".into(),
            "ppc:run-leak-1:x".into(),
            0.9,
            1_790_000_000,
        );
        let err = store.save(item, &leaked).unwrap_err();
        assert!(
            matches!(err, kitty_memory::lessons::LessonError::DirtyRun(_)),
            "leak debe rechazar lección, fue: {err}"
        );
        assert_eq!(store.count().unwrap(), 0, "con leak no hay resume ni lección");
    }

    #[test]
    fn limit_1_guarda_leccion_solo_con_evidence_ref() {
        // Núcleo offline: compose + gate de lessons con DB en memoria.
        let store = LessonStore::open_in_memory().unwrap();
        let table = vec![
            ("baseline".to_string(), scores(0, 1, 100.0)),
            ("iter-cap-bajo".to_string(), scores(1, 1, 90.0)),
        ];
        let (winner, avg) = compose_winner(&table);
        assert_eq!(winner, "iter-cap-bajo");
        let prov = RunProvenance::clean("run-test-1");
        let item = MemoryItem::new(
            "ppc harness config delta (1)".into(),
            format!("usar candidato '{winner}' (compose determinista k=1)"),
            "{\"k\":1}".into(),
            "ppc".into(),
            "run-test-1".into(),
            format!("ppc:run-test-1:{winner}"),
            avg,
            1_790_000_000,
        );
        let id = store.save(item, &prov).unwrap();
        assert!(!id.is_empty());
        let got = store.get(&id).unwrap().expect("lección guardada");
        assert!(got.evidence_ref.starts_with("ppc:"));
        // Sin evidence_ref jamás se habría guardado:
        let bad = MemoryItem::new(
            "x".into(), "y".into(), "{}".into(), "ppc".into(), "run-test-1".into(),
            String::new(), 0.5, 1_790_000_000,
        );
        assert!(store.save(bad, &prov).is_err());
        // Run sucio tampoco:
        let dirty = RunProvenance { run_id: "run-test-1".into(), leaked: false, ledger_dirty: true };
        let bad2 = MemoryItem::new(
            "x".into(), "y".into(), "{}".into(), "ppc".into(), "run-test-1".into(),
            "ppc:run-test-1:x".into(), 0.5, 1_790_000_001,
        );
        assert!(store.save(bad2, &dirty).is_err());
    }

    #[test]
    fn systemone_solo_volcado_nunca_gate() {
        // El volcado copia el modo tal cual; ninguna rama decide sobre él.
        // (Garantía estructural: `grep -n systemone` en este fichero solo
        //  aparece en construcción de Runmark/reporte, nunca en `if`.)
        let (_k, mode) = kitty_harness::systemone_status();
        let dump = mode.to_string();
        let report = serde_json::json!({ "systemone_status_dump": dump });
        assert_eq!(report["systemone_status_dump"], mode);
    }

    #[test]
    fn lessons_db_aislada_nunca_default() {
        std::env::remove_var("KITTY_LESSONS_DB");
        let out = PathBuf::from("/tmp/x/bench-ppc.json");
        let p = ppc_lessons_db_path(&out);
        assert_eq!(p, PathBuf::from("/tmp/x/bench-ppc.json.lessons.db"));
        assert!(!p.to_string_lossy().contains("kitty") || p.extension().unwrap() == "db");
        std::env::set_var("KITTY_LESSONS_DB", "/tmp/custom.db");
        assert_eq!(ppc_lessons_db_path(&out), PathBuf::from("/tmp/custom.db"));
        std::env::remove_var("KITTY_LESSONS_DB");
    }
}

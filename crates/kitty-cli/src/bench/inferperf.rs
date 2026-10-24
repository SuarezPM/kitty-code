//! inferperf — suite steady-state tok/s kitty-infer vs Ollama (F1b, Lane D).
//!
//! Metodología DeepInfer R3 adaptada a kitty (SDD-KITTY-VS-LITTLE-REALBENCH §F1b):
//!
//! - **Steady-state**: warmup fijo descartado + ventana limpia de N samples.
//! - **Ventana limpia**: se descartan intervalos contaminados (flag explícito,
//!   contadores/duraciones inválidas, o wall que excede
//!   `KITTY_INFERPERF_MAX_WALL_SECS` — GC/GPU contención sospechada). Si la
//!   ventana queda vacía, las tasas se reportan como `"N/A"` machine-readable
//!   (string, nunca `null`), no como 0.
//! - **Métricas**: prefill tok/s + decode tok/s + TTFT medio, siempre con las
//!   cantidades CRUDAS (`window_steps`, `window_*_tokens`, `window_*_secs`)
//!   al lado de cada tasa para auditoría.
//! - **JSONL append-only**: una línea por celda (backend) con la config
//!   COMPLETA (backend, modelo, ctx, features, seeds, speculative) + `load_duration`
//!   separado de la ventana. Comparar backends = correr la suite una vez por
//!   backend con MISMO modelo y MISMO `--out` (las líneas se anexan; el
//!   análisis compara líneas, nunca se mezclan ventanas).
//! - **Logging del motor APAGADO durante la ventana**: [`EngineLogGuard`]
//!   fija `KITTY_INFERPERF_ENGINE_LOG=off` (+ legacy `KITTY_ENGINE_LOG=off`)
//!   en warmup+ventana y restaura al salir; el registro declara
//!   `engine_logging: "off"|"on"`.
//! - **`load_duration` separado**: el arranque (probe de carga) se mide ANTES
//!   del warmup y NUNCA entra en los `*_secs` de la ventana.
//! - **Seeds fijos**: prompts sintéticos deterministas ([`synthetic_prompt`],
//!   seed [`INFERPERF_SEED`]); reproducibles byte a byte.
//! - **Speculative declarado**: cada registro lleva
//!   `speculative: {enabled, draft_model, tokens_per_step}`; sin draft el
//!   valor es `"1 token/step"` explícito, nunca implícito.
//!
//! Alcance del lane (NO tocar): `engine/*`, `agent/*`, `memory/*`,
//! `runmark.rs`, `demo12.rs`, `bench/mod.rs` (Lane B cablea).
//!
//! Firma que Lane B cablea en `bench/mod.rs`:
//!
//! ```ignore
//! pub(crate) mod inferperf; // junto al resto de `pub(crate) mod ...`
//! // ...
//! if suite == "inferperf" {
//!     return inferperf::run_inferperf_suite(&cli, &cfg, &out).await;
//! }
//! ```
//!
//! Honestidad de timings (importante): hoy ningún backend puebla
//! `ChatResponse.eval_duration`, así que el path en vivo opera en modo
//! `"wall-only-decode"` (decode tok/s = gen/wall; prefill y TTFT = `"N/A"` con
//! `timing_mode` declarado). El split prefill/decode/TTFT está implementado y
//! probado vía mocks ([`StepSample::ok`]); cuando el engine exponga el split,
//! el wrapper lo usa automáticamente (rama `eval_duration`, mismo record).
//! Sin GPU/LLM en vivo en tests: [`ScriptedBackend`] (mock) + funciones puras.

use super::{BenchCliCtx, BenchConfig};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

// ── Constantes del protocolo (fijas; cambiarlas cambia la serie) ──────

/// Versión de la suite (va en cada línea JSONL).
pub const INFERPERF_VERSION: &str = "1.0.0";
/// Nombre de suite (va en cada línea JSONL + futuro arm runmark).
pub const INFERPERF_SUITE: &str = "inferperf";
/// Warmup fijo: samples descartados antes de la ventana (no cuentan ni como
/// descartados: son warmup por diseño).
pub const INFERPERF_WARMUP_SAMPLES: usize = 2;
/// Ventana limpia: N samples steady-state por celda.
pub const INFERPERF_WINDOW_SAMPLES: usize = 10;
/// Seed fijo de los prompts sintéticos (reproducibilidad).
pub const INFERPERF_SEED: u64 = 42;
/// Tamaño objetivo del prompt sintético (palabras aprox; ver
/// [`synthetic_prompt`]).
pub const INFERPERF_PROMPT_TOKENS: usize = 512;
/// Tokens a generar por sample en vivo (`num_predict`-like del sampler).
pub const INFERPERF_GEN_TOKENS: usize = 128;
/// Flag/env que apaga el logging del motor durante la ventana.
pub const ENGINE_LOG_ENV: &str = "KITTY_INFERPERF_ENGINE_LOG";
/// Env legacy que el engine también respeta (se fija igual por si acaso).
pub const ENGINE_LOG_LEGACY_ENV: &str = "KITTY_ENGINE_LOG";
/// Cap opcional de wall por sample (`KITTY_INFERPERF_MAX_WALL_SECS`, segundos,
/// `>0`): excederlo marca el sample contaminado (GC/GPU contención sospechada).
/// Ausente/inválido = sin cap (cada sample vale por su flag + validez).
pub const WALL_CAP_ENV: &str = "KITTY_INFERPERF_MAX_WALL_SECS";
/// Opt-in speculative en vivo (`KITTY_INFERPERF_SPECULATIVE=1`) + draft
/// (`KITTY_INFERPERF_DRAFT_MODEL`, opcional).
pub const SPECULATIVE_ENV: &str = "KITTY_INFERPERF_SPECULATIVE";
/// Modelo draft para la contabilidad speculative (solo informativo).
pub const SPECULATIVE_DRAFT_ENV: &str = "KITTY_INFERPERF_DRAFT_MODEL";
/// Marcador machine-readable de ventana vacía (string, nunca null ni 0).
pub const NA: &str = "N/A";

// ── Config ──────────────────────────────────────────────────────────────

/// Features del backend DECLARADAS por celda (SDD F1b: `kv_fp8`,
/// `response_cache`, `kv_retention`, `paged_mem_config`).
///
/// Son declaradas, no sondadas: este lane no toca el engine; el probeo real
/// de flags del engine pertenece a F1a/engine. El record las vuelca tal cual
/// para que la comparación de líneas sea auditable.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InferperfFeatures {
    pub kv_fp8: bool,
    pub response_cache: bool,
    pub kv_retention: String,
    pub paged_mem_config: String,
}

impl InferperfFeatures {
    /// Todo apagado (base honesta hasta que F1a pruebe lo contrario).
    pub fn declared_off() -> Self {
        Self {
            kv_fp8: false,
            response_cache: false,
            kv_retention: "off".to_string(),
            paged_mem_config: "n/a".to_string(),
        }
    }

    /// Defaults declarados por backend (el bench corre con gates off).
    pub fn for_backend(backend: &str) -> Self {
        match backend {
            "kitty-infer" => Self {
                paged_mem_config: "paged (engine default; ver manifiesto F1a)".to_string(),
                ..Self::declared_off()
            },
            _ => Self {
                paged_mem_config: "n/a (ollama)".to_string(),
                ..Self::declared_off()
            },
        }
    }
}

/// Config de una celda inferperf (un backend, un modelo, una ventana).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferperfConfig {
    pub backend: String,
    pub model: String,
    pub ctx: usize,
    pub prompt_tokens: usize,
    pub gen_tokens: usize,
    pub seed: u64,
    pub warmup_samples: usize,
    pub window_samples: usize,
    pub features: InferperfFeatures,
    pub speculative_enabled: bool,
    pub speculative_draft_model: Option<String>,
}

impl Default for InferperfConfig {
    fn default() -> Self {
        Self {
            backend: "ollama".to_string(),
            model: "ornith:9b".to_string(),
            ctx: 16384,
            prompt_tokens: INFERPERF_PROMPT_TOKENS,
            gen_tokens: INFERPERF_GEN_TOKENS,
            seed: INFERPERF_SEED,
            warmup_samples: INFERPERF_WARMUP_SAMPLES,
            window_samples: INFERPERF_WINDOW_SAMPLES,
            features: InferperfFeatures::for_backend("ollama"),
            speculative_enabled: false,
            speculative_draft_model: None,
        }
    }
}

impl InferperfConfig {
    /// Celda desde el contexto bench (MISMO modelo/ctx en ambos backends; la
    /// comparación válida es misma config cambiando solo `backend`).
    pub fn from_cli_parts(backend: &str, model: &str, ctx: usize) -> Self {
        let (speculative_enabled, speculative_draft_model) = speculative_from_env();
        Self {
            backend: backend.to_string(),
            model: model.to_string(),
            ctx,
            features: InferperfFeatures::for_backend(backend),
            speculative_enabled,
            speculative_draft_model,
            ..Self::default()
        }
    }
}

/// Lee el opt-in speculative del env (default: off, sin draft).
pub fn speculative_from_env() -> (bool, Option<String>) {
    let enabled = std::env::var(SPECULATIVE_ENV)
        .map(|v| v.trim() == "1" || v.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let draft = std::env::var(SPECULATIVE_DRAFT_ENV)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    (enabled, draft)
}

// ── Samples y ventana limpia ────────────────────────────────────────────

/// Un sample steady-state. Los tres tiempos son `Option`: el backend en vivo
/// puede no exponer el split (ver docs del módulo); el mock los puebla todos.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StepSample {
    pub prompt_tokens: u64,
    pub gen_tokens: u64,
    pub prefill_secs: Option<f64>,
    pub decode_secs: Option<f64>,
    pub ttft_secs: Option<f64>,
    pub contaminated: bool,
    pub contamination_reason: Option<String>,
    /// Tokens/step medidos con speculative (None = sin draft o no medido).
    pub tokens_per_step_measured: Option<f64>,
}

impl StepSample {
    /// Sample con split completo (mock o backend con `eval_duration`).
    pub fn ok(
        prompt_tokens: u64,
        gen_tokens: u64,
        prefill_secs: f64,
        decode_secs: f64,
        ttft_secs: f64,
    ) -> Self {
        Self {
            prompt_tokens,
            gen_tokens,
            prefill_secs: Some(prefill_secs),
            decode_secs: Some(decode_secs),
            ttft_secs: Some(ttft_secs),
            contaminated: false,
            contamination_reason: None,
            tokens_per_step_measured: None,
        }
    }

    /// Sample wall-only (path en vivo actual): decode = wall; prefill/TTFT
    /// ausentes ⇒ no entra a la ventana limpia, sí al subset wall-only.
    pub fn wall_only(prompt_tokens: u64, gen_tokens: u64, wall_secs: f64) -> Self {
        Self {
            prompt_tokens,
            gen_tokens,
            prefill_secs: None,
            decode_secs: Some(wall_secs),
            ttft_secs: None,
            contaminated: false,
            contamination_reason: None,
            tokens_per_step_measured: None,
        }
    }

    /// Sample envenenado (error de chat, split imposible, cap excedido…).
    pub fn poisoned(reason: impl Into<String>) -> Self {
        Self {
            prompt_tokens: 0,
            gen_tokens: 0,
            prefill_secs: None,
            decode_secs: None,
            ttft_secs: None,
            contaminated: true,
            contamination_reason: Some(reason.into()),
            tokens_per_step_measured: None,
        }
    }
}

fn positive_finite(v: Option<f64>) -> bool {
    matches!(v, Some(x) if x.is_finite() && x > 0.0)
}

/// Ventana limpia estricta: sin flag + contadores `>0` + los tres tiempos
/// presentes, finitos y `>0`.
pub fn sample_is_clean(s: &StepSample) -> bool {
    !s.contaminated
        && s.contamination_reason.is_none()
        && s.prompt_tokens > 0
        && s.gen_tokens > 0
        && positive_finite(s.prefill_secs)
        && positive_finite(s.decode_secs)
        && positive_finite(s.ttft_secs)
}

/// Subset wall-only: sin flag + contadores `>0` + decode presente, pero sin
/// split completo. Aporta `wall_decode_tok_s` best-effort, no prefill/TTFT.
pub fn sample_is_wall_only(s: &StepSample) -> bool {
    !sample_is_clean(s)
        && !s.contaminated
        && s.contamination_reason.is_none()
        && s.prompt_tokens > 0
        && s.gen_tokens > 0
        && positive_finite(s.decode_secs)
}

/// Agregado de ventana con cantidades CRUDAS + tasas (`None` = `"N/A"`).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct WindowStats {
    /// Steps en la ventana limpia (split completo).
    pub steps: usize,
    pub prompt_tokens: u64,
    pub gen_tokens: u64,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    /// Media de TTFT sobre los steps limpios (`None` si vacía).
    pub ttft_mean_secs: Option<f64>,
    pub prefill_tok_s: Option<f64>,
    pub decode_tok_s: Option<f64>,
    /// Samples descartados (flag, inválidos o cap excedido).
    pub discarded: usize,
    /// Subset wall-only (decode best-effort del path en vivo actual).
    pub wall_only_steps: usize,
    pub wall_only_gen_tokens: u64,
    pub wall_only_decode_secs: f64,
    pub wall_decode_tok_s: Option<f64>,
    #[serde(skip)]
    ttft_sum: f64,
}

/// Filtra la ventana limpia (referencias a los samples válidos).
pub fn clean_window<'a>(samples: &'a [StepSample]) -> Vec<&'a StepSample> {
    samples.iter().filter(|s| sample_is_clean(s)).collect()
}

/// Agrega ventana limpia + subset wall-only + descarte.
pub fn window_stats(samples: &[StepSample]) -> WindowStats {
    let mut st = WindowStats::default();
    for s in samples {
        if sample_is_clean(s) {
            st.steps += 1;
            st.prompt_tokens += s.prompt_tokens;
            st.gen_tokens += s.gen_tokens;
            // Limpio ⇒ los tres son Some finitos >0 (garantizado por el
            // predicado; `unwrap_or(0.0)` inalcanzable pero total).
            st.prefill_secs += s.prefill_secs.unwrap_or(0.0);
            st.decode_secs += s.decode_secs.unwrap_or(0.0);
            st.ttft_sum += s.ttft_secs.unwrap_or(0.0);
        } else if sample_is_wall_only(s) {
            st.wall_only_steps += 1;
            st.wall_only_gen_tokens += s.gen_tokens;
            st.wall_only_decode_secs += s.decode_secs.unwrap_or(0.0);
        } else {
            st.discarded += 1;
        }
    }
    if st.steps > 0 {
        st.prefill_tok_s = Some(st.prompt_tokens as f64 / st.prefill_secs);
        st.decode_tok_s = Some(st.gen_tokens as f64 / st.decode_secs);
        st.ttft_mean_secs = Some(st.ttft_sum / st.steps as f64);
    }
    if st.wall_only_steps > 0 {
        st.wall_decode_tok_s = Some(st.wall_only_gen_tokens as f64 / st.wall_only_decode_secs);
    }
    st
}

/// Tasa machine-readable: número o string `"N/A"` (nunca null).
pub fn rate_json(v: Option<f64>) -> serde_json::Value {
    match v {
        Some(x) if x.is_finite() => serde_json::json!(x),
        _ => serde_json::Value::String(NA.to_string()),
    }
}

/// Contabilidad speculative por registro: sin draft, `"1 token/step"`
/// explícito (nunca implícito ni 0).
pub fn tokens_per_step_label(speculative_enabled: bool, measured: Option<f64>) -> String {
    if !speculative_enabled {
        return "1 token/step".to_string();
    }
    match measured {
        Some(x) if x.is_finite() && x > 0.0 => format!("{x:.2} tokens/step"),
        _ => "1 token/step (draft enabled, no per-step count reported)".to_string(),
    }
}

// ── Prompts sintéticos (seeds fijos) ────────────────────────────────────

const PROBE_WORDS: &[&str] = &[
    "alpha", "bravo", "casa", "delta", "eco", "fosil", "gato", "hotel", "isla", "junta",
    "kilo", "luna", "mesa", "nube", "obra", "perro", "queso", "rama", "silla", "taza",
    "union", "vela", "yunta", "zorro", "campo", "duende", "faro", "globo", "huerto",
    "inca", "jungla", "koala",
];

/// Prompt sintético determinista: LCG sembrado con `seed` + `step`.
/// Mismos args ⇒ mismos bytes; distinto step/seed ⇒ distinto texto.
pub fn synthetic_prompt(seed: u64, step: usize, target_tokens: usize) -> String {
    let mut state = seed.wrapping_add(
        0x9E3779B97F4A7C15u64.wrapping_mul(step as u64 + 1),
    );
    let mut out = String::from("Inferperf synthetic probe. Reproduce exactly: ");
    for _ in 0..target_tokens.max(8) {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.push_str(PROBE_WORDS[(state >> 33) as usize % PROBE_WORDS.len()]);
        out.push(' ');
    }
    out.push_str(&format!("step={step} seed={seed}."));
    out
}

/// Heurística de conteo (fallback SOLO cuando el backend omite contadores;
/// `len/4`, mínimo 1). Nunca se llama dentro de la ventana a endpoints extra
/// (una llamada auxiliar por sample contencionaría la GPU bajo medición).
pub fn estimate_tokens(text: &str) -> u64 {
    ((text.len().div_ceil(4)) as u64).max(1)
}

// ── Logging del motor apagado en ventana ────────────────────────────────

/// `true` si el logging del motor está apagado por flag/env.
pub fn engine_logging_suppressed() -> bool {
    [ENGINE_LOG_ENV, ENGINE_LOG_LEGACY_ENV].iter().any(|key| {
        std::env::var(key)
            .map(|v| v.trim().eq_ignore_ascii_case("off"))
            .unwrap_or(false)
    })
}

/// RAII: apaga el logging del motor (warmup+ventana) y restaura al salir.
pub struct EngineLogGuard {
    prev_main: Option<String>,
    prev_legacy: Option<String>,
}

impl EngineLogGuard {
    pub fn suppress() -> Self {
        let prev_main = std::env::var(ENGINE_LOG_ENV).ok();
        let prev_legacy = std::env::var(ENGINE_LOG_LEGACY_ENV).ok();
        std::env::set_var(ENGINE_LOG_ENV, "off");
        std::env::set_var(ENGINE_LOG_LEGACY_ENV, "off");
        Self {
            prev_main,
            prev_legacy,
        }
    }
}

impl Drop for EngineLogGuard {
    fn drop(&mut self) {
        match self.prev_main.take() {
            Some(v) => std::env::set_var(ENGINE_LOG_ENV, v),
            None => std::env::remove_var(ENGINE_LOG_ENV),
        }
        match self.prev_legacy.take() {
            Some(v) => std::env::set_var(ENGINE_LOG_LEGACY_ENV, v),
            None => std::env::remove_var(ENGINE_LOG_LEGACY_ENV),
        }
    }
}

/// Cap de wall por sample desde el env (`None` = sin cap).
pub fn wall_cap_secs() -> Option<f64> {
    std::env::var(WALL_CAP_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
}

/// Marca contaminado el sample si su wall total excede el cap (GC/GPU
/// contención sospechada). Sin cap o sin tiempos, no toca nada.
pub fn flag_wall_cap(sample: &mut StepSample, cap: Option<f64>) {
    let Some(cap) = cap else { return };
    if sample.contaminated {
        return;
    }
    let total = sample.prefill_secs.unwrap_or(0.0) + sample.decode_secs.unwrap_or(0.0);
    if total > 0.0 && total > cap {
        sample.contaminated = true;
        sample.contamination_reason =
            Some(format!("wall {total:.3}s > cap {cap:.3}s (GC/GPU contencion sospechada)"));
    }
}

// ── Registro JSONL ──────────────────────────────────────────────────────

/// Una línea JSONL por celda: config completa + crudos + tasas (`"N/A"` si la
/// ventana quedó vacía) + speculative declarado + F1a
/// (`mem_mode`/`decision_reason`/`available_context_tokens` del engine en
/// kitty-infer, `n/a (ollama)` fuera de él). `timestamp_epoch_secs` via
/// `SystemTime` (sin deps extra).
#[allow(clippy::too_many_arguments)]
pub fn cell_record(
    cfg: &InferperfConfig,
    load_duration_secs: f64,
    load_kind: &str,
    timing_mode: &str,
    stats: &WindowStats,
    engine_logging_off: bool,
    mem_mode: &str,
    decision_reason: &str,
    available_context_tokens: u32,
) -> serde_json::Value {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    serde_json::json!({
        "suite": INFERPERF_SUITE,
        "version": INFERPERF_VERSION,
        "backend": cfg.backend,
        "model": cfg.model,
        "ctx": cfg.ctx,
        "prompt_tokens": cfg.prompt_tokens,
        "gen_tokens": cfg.gen_tokens,
        "seed": cfg.seed,
        "warmup_samples": cfg.warmup_samples,
        "window_samples": cfg.window_samples,
        "features": {
            "kv_fp8": cfg.features.kv_fp8,
            "response_cache": cfg.features.response_cache,
            "kv_retention": cfg.features.kv_retention,
            "paged_mem_config": cfg.features.paged_mem_config,
        },
        "speculative": {
            "enabled": cfg.speculative_enabled,
            "draft_model": cfg.speculative_draft_model,
            "tokens_per_step": tokens_per_step_label(cfg.speculative_enabled, None),
        },
        "engine_logging": if engine_logging_off { "off" } else { "on" },
        "timing_mode": timing_mode,
        "load_kind": load_kind,
        "load_duration_secs": load_duration_secs,
        "mem_mode": mem_mode,
        "decision_reason": decision_reason,
        "available_context_tokens": available_context_tokens,
        "window": {
            "steps": stats.steps,
            "prompt_tokens": stats.prompt_tokens,
            "gen_tokens": stats.gen_tokens,
            "prefill_secs": stats.prefill_secs,
            "decode_secs": stats.decode_secs,
            "ttft_mean_secs": stats.ttft_mean_secs.unwrap_or(f64::NAN),
            "discarded": stats.discarded,
            "wall_only": {
                "steps": stats.wall_only_steps,
                "gen_tokens": stats.wall_only_gen_tokens,
                "decode_secs": stats.wall_only_decode_secs,
            },
        },
        "metrics": {
            "prefill_tok_s": rate_json(stats.prefill_tok_s),
            "decode_tok_s": rate_json(stats.decode_tok_s),
            "ttft_secs_mean": rate_json(stats.ttft_mean_secs),
            "wall_decode_tok_s": rate_json(stats.wall_decode_tok_s),
        },
        "timestamp_epoch_secs": now,
    })
}

/// Append-only: crea dirs padre, anexa UNA línea + `\n`, fsync. Nunca trunca
/// (comparar backends = anexar celdas al mismo `--out`).
pub fn append_jsonl_record(out: &Path, record: &serde_json::Value) -> Result<()> {
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("inferperf: crear dir {}", parent.display()))?;
    }
    use std::io::Write as _;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out)
        .with_context(|| format!("inferperf: abrir {}", out.display()))?;
    let mut line = serde_json::to_string(record).context("inferperf: serializar record")?;
    line.push('\n');
    f.write_all(line.as_bytes())
        .with_context(|| format!("inferperf: escribir {}", out.display()))?;
    f.sync_all()
        .with_context(|| format!("inferperf: fsync {}", out.display()))?;
    Ok(())
}

/// Exit para el report: ventana con señal ⇒ pass (0); vacía ⇒
/// inconcluso (2, nunca publicable — misma tabla `runmark::EXIT_*` B2).
/// La ventana limpia (`steps`) y el subset wall-only (`wall_only_steps`)
/// cuentan como señal: el path en vivo actual solo produce wall-only
/// (decode = wall, prefill/TTFT N/A) y su `wall_decode_tok_s` es la
/// métrica publicable de ese modo.
pub fn exit_code_for_stats(stats: &WindowStats) -> i32 {
    if stats.steps > 0 || stats.wall_only_steps > 0 {
        super::runmark::EXIT_PASS
    } else {
        super::runmark::EXIT_INCONCLUSIVE
    }
}

// ── BackendJSampler (núcleo offline testeable) ──────────────────────────

/// Fuente de samples (el wrapper en vivo implementa el loop con `await`;
/// los tests y el offline usan [`ScriptedBackend`]).
pub trait InferBackend {
    fn backend_name(&self) -> &str;
    /// Arranque medido ANTES del warmup (separado de la ventana).
    fn load_duration_secs(&mut self) -> f64;
    fn sample_step(&mut self, step: usize, prompt: &str) -> StepSample;
}

/// Mock scripted: devuelve el guion en orden cíclico (offline, sin GPU/LLM).
pub struct ScriptedBackend {
    pub name: String,
    pub load_secs: f64,
    pub script: Vec<StepSample>,
    pub calls: usize,
}

impl ScriptedBackend {
    pub fn new(
        name: impl Into<String>,
        load_secs: f64,
        script: Vec<StepSample>,
    ) -> Self {
        Self {
            name: name.into(),
            load_secs,
            script,
            calls: 0,
        }
    }
}

impl InferBackend for ScriptedBackend {
    fn backend_name(&self) -> &str {
        &self.name
    }

    fn load_duration_secs(&mut self) -> f64 {
        self.load_secs
    }

    fn sample_step(&mut self, step: usize, _prompt: &str) -> StepSample {
        self.calls += 1;
        self.script[step % self.script.len()].clone()
    }
}

/// Reporte de una celda (config + arranque + ventana).
pub struct InferperfReport {
    pub backend: String,
    pub load_duration_secs: f64,
    pub load_kind: String,
    pub timing_mode: String,
    pub stats: WindowStats,
    pub total_steps: usize,
}

/// Corre UNA celda: load separado → warmup descartado → ventana limpia con
/// logging apagado → stats. Puro/offline salvo el `backend` inyectado.
pub fn run_cell<B: InferBackend>(
    backend: &mut B,
    cfg: &InferperfConfig,
    load_kind: &str,
    timing_mode: &str,
) -> InferperfReport {
    let _guard = EngineLogGuard::suppress();
    let load_duration_secs = backend.load_duration_secs();
    // Warmup fijo: se corre (calienta el backend) y se descarta.
    for i in 0..cfg.warmup_samples {
        let prompt = synthetic_prompt(cfg.seed, i, cfg.prompt_tokens);
        let _ = backend.sample_step(i, &prompt);
    }
    let cap = wall_cap_secs();
    let mut samples = Vec::with_capacity(cfg.window_samples);
    for k in 0..cfg.window_samples {
        let step = cfg.warmup_samples + k;
        let prompt = synthetic_prompt(cfg.seed, step, cfg.prompt_tokens);
        let mut s = backend.sample_step(step, &prompt);
        flag_wall_cap(&mut s, cap);
        samples.push(s);
    }
    let stats = window_stats(&samples);
    InferperfReport {
        backend: backend.backend_name().to_string(),
        load_duration_secs,
        load_kind: load_kind.to_string(),
        timing_mode: timing_mode.to_string(),
        stats,
        total_steps: samples.len(),
    }
}

// ── Entrada que Lane B cablea ───────────────────────────────────────────

/// Suite inferperf (firma para `bench/mod.rs`, Lane B): UNA celda del backend
/// bajo medición (`cli.backend`, MISMO modelo/ctx que la celda rival). La
/// comparación kitty-infer vs Ollama = dos invocaciones con mismo modelo y
/// mismo `--out` (JSONL append-only; cada una anexa su línea).
///
/// No hace `process::exit`: devuelve `Ok` tras anexar; el caller mapea con
/// [`exit_code_for_stats`] (ventana vacía ⇒ 2 inconcluso).
pub async fn run_inferperf_suite(
    cli: &BenchCliCtx,
    cfg: &BenchConfig,
    out: &PathBuf,
) -> Result<()> {
    use kitty_runtime::{ChatClient, Message};

    let iconfig = InferperfConfig::from_cli_parts(cli.backend, &cfg.model, cfg.context_window);

    // Arranque SEPARADO de la ventana: probe sin carga de modelo (el wall de
    // carga real del engine lo declara el manifiesto F1a, no este probe).
    let t_load = std::time::Instant::now();
    cfg.client
        .health_check()
        .await
        .context("inferperf: backend no responde (health_check)")?;
    let load_duration_secs = t_load.elapsed().as_secs_f64();
    let load_kind = "health-probe (sin carga de modelo; wall de carga real en manifiesto F1a)";

    // Ventana con logging del motor apagado (flag/env, restaurado al salir).
    let _guard = EngineLogGuard::suppress();

    // Generación acotada F1b: INFERPERF_GEN_TOKENS vía `num_predict`
    // (antes `chat` sin límite = hasta `max_model_len`, ventana impagable).
    let gen_opts = Some(kitty_runtime::client::RequestOptions {
        num_predict: Some(INFERPERF_GEN_TOKENS as i32),
        ..Default::default()
    });
    // Warmup fijo descartado (calienta KV/caches; sin llamadas auxiliares
    // dentro de la ventana para no contencionar la GPU bajo medición).
    for i in 0..iconfig.warmup_samples {
        let prompt = synthetic_prompt(iconfig.seed, i, iconfig.prompt_tokens);
        cfg.client
            .chat_with_options(
                &iconfig.model,
                vec![Message::user(prompt)],
                None,
                gen_opts.clone(),
            )
            .await
            .with_context(|| format!("inferperf: warmup {i} falló"))?;
    }

    // Ventana limpia de N samples.
    let cap = wall_cap_secs();
    let mut samples = Vec::with_capacity(iconfig.window_samples);
    let mut saw_split = false;
    for k in 0..iconfig.window_samples {
        let step = iconfig.warmup_samples + k;
        let prompt = synthetic_prompt(iconfig.seed, step, iconfig.prompt_tokens);
        let t0 = std::time::Instant::now();
        let sample = match cfg
            .client
            .chat_with_options(
                &iconfig.model,
                vec![Message::user(prompt.clone())],
                None,
                gen_opts.clone(),
            )
            .await
        {
            Err(e) => StepSample::poisoned(format!("chat error en step {step}: {e:#}")),
            Ok(resp) => {
                let wall = t0.elapsed().as_secs_f64();
                let text = resp.message.content.as_deref().unwrap_or("");
                let pt = resp
                    .prompt_eval_count
                    .and_then(|c| u64::try_from(c).ok())
                    .filter(|c| *c > 0)
                    .unwrap_or_else(|| estimate_tokens(&prompt));
                let gt = resp
                    .eval_count
                    .and_then(|c| u64::try_from(c).ok())
                    .filter(|c| *c > 0)
                    .unwrap_or_else(|| estimate_tokens(text));
                match resp.eval_duration {
                    // Split declarado por el backend (hoy ningún backend lo
                    // puebla; rama lista para cuando el engine lo exponga).
                    Some(ns) => {
                        let decode = ns as f64 / 1e9;
                        let prefill = wall - decode;
                        if decode > 0.0 && prefill > 0.0 {
                            saw_split = true;
                            StepSample::ok(pt, gt, prefill, decode, prefill)
                        } else {
                            StepSample::poisoned(format!(
                                "split no positivo en step {step} (wall {wall:.3}s, decode {decode:.3}s)"
                            ))
                        }
                    }
                    // Modo actual: decode = wall; prefill/TTFT = N/A.
                    None => StepSample::wall_only(pt, gt, wall),
                }
            }
        };
        let mut sample = sample;
        flag_wall_cap(&mut sample, cap);
        samples.push(sample);
    }

    let timing_mode = if saw_split {
        "backend-eval-duration-split"
    } else {
        "wall-only-decode (prefill/TTFT N/A hasta split del engine)"
    };
    let stats = window_stats(&samples);
    debug_assert_eq!(
        clean_window(&samples).len(),
        stats.steps,
        "clean_window y window_stats deben coincidir en la ventana limpia"
    );
    let suggested_exit = exit_code_for_stats(&stats);
    // F1a: contexto real del engine (kitty-infer) o n/a honesto (ollama).
    let infer_ctx = cfg.client.infer_run_context();
    let (mem_mode, decision_reason, agg_available) = match &infer_ctx {
        Some(c) => (
            c.mem_mode.to_string(),
            c.decision_reason.to_string(),
            c.available_context_tokens as u32,
        ),
        None => (
            "n/a (ollama)".to_string(),
            "n/a (ollama)".to_string(),
            iconfig.ctx as u32,
        ),
    };
    let record = cell_record(
        &iconfig,
        load_duration_secs,
        load_kind,
        timing_mode,
        &stats,
        engine_logging_suppressed(),
        &mem_mode,
        &decision_reason,
        agg_available,
    );
    append_jsonl_record(out, &record)?;

    println!("inferperf {}", iconfig.backend);
    println!("  model: {}  ctx: {}", iconfig.model, iconfig.ctx);
    println!("  load_duration: {load_duration_secs:.3}s ({load_kind})");
    println!(
        "  window: {}/{} limpios, {} descartados (warmup {} aparte)",
        stats.steps,
        iconfig.window_samples,
        stats.discarded,
        iconfig.warmup_samples,
    );
    let show = |v: Option<f64>| match v {
        Some(x) => format!("{x:.1}"),
        None => NA.to_string(),
    };
    println!(
        "  prefill: {} tok/s  decode: {} tok/s  TTFT medio: {} s  (wall-only decode: {} tok/s)",
        show(stats.prefill_tok_s),
        show(stats.decode_tok_s),
        show(stats.ttft_mean_secs),
        show(stats.wall_decode_tok_s),
    );
    println!("  speculative: {}", tokens_per_step_label(iconfig.speculative_enabled, None));
    println!("  engine_logging en ventana: off");
    println!("Results appended to {}", out.display());
    if stats.steps == 0 {
        println!("INCONCLUSO: ventana vacía (tasas N/A) — exit sugerido 2");
    }
    println!("  exit sugerido: {suggested_exit}");
    Ok(())
}

// ── Tests offline (sin GPU/LLM: mocks + funciones puras) ────────────────

#[cfg(test)]
mod inferperf_tests {
    use super::*;

    fn ok_step(pt: u64, gt: u64, p: f64, d: f64, t: f64) -> StepSample {
        StepSample::ok(pt, gt, p, d, t)
    }

    #[test]
    fn clean_window_keeps_only_strict_clean() {
        let samples = vec![
            ok_step(100, 20, 0.5, 1.0, 0.5),
            StepSample::wall_only(100, 20, 1.5),
            StepSample::poisoned("boom"),
            StepSample {
                prompt_tokens: 0,
                ..ok_step(0, 20, 0.5, 1.0, 0.5)
            },
            StepSample {
                decode_secs: Some(-1.0),
                ..ok_step(100, 20, 0.5, -1.0, 0.5)
            },
        ];
        let clean = clean_window(&samples);
        assert_eq!(clean.len(), 1, "solo el split completo válido entra");
        let st = window_stats(&samples);
        assert_eq!(st.steps, 1);
        assert_eq!(st.wall_only_steps, 1, "wall-only va al subset, no a limpia");
        assert_eq!(st.discarded, 3);
    }

    #[test]
    fn window_stats_rates_from_raw_counts() {
        let samples = vec![
            ok_step(1000, 100, 2.0, 5.0, 2.0),
            ok_step(1000, 100, 2.0, 5.0, 4.0),
            ok_step(2000, 200, 4.0, 10.0, 4.0),
            StepSample::poisoned("gc pause"),
        ];
        let st = window_stats(&samples);
        assert_eq!(st.steps, 3);
        assert_eq!(st.prompt_tokens, 4000);
        assert_eq!(st.gen_tokens, 400);
        assert!((st.prefill_secs - 8.0).abs() < 1e-9);
        assert!((st.decode_secs - 20.0).abs() < 1e-9);
        assert!((st.prefill_tok_s.unwrap() - 500.0).abs() < 1e-9);
        assert!((st.decode_tok_s.unwrap() - 20.0).abs() < 1e-9);
        assert!((st.ttft_mean_secs.unwrap() - (10.0 / 3.0)).abs() < 1e-9);
        assert_eq!(st.discarded, 1);
    }

    #[test]
    fn wall_cap_flags_contaminated_samples() {
        let mut s = ok_step(100, 20, 0.4, 0.4, 0.4); // total 0.8s
        flag_wall_cap(&mut s, Some(1.0));
        assert!(!s.contaminated, "bajo el cap no se toca");
        flag_wall_cap(&mut s, Some(0.5));
        assert!(s.contaminated, "sobre el cap se descarta");
        assert!(s
            .contamination_reason
            .as_deref()
            .unwrap_or("")
            .contains("contencion"));
        // Sin cap: intacto aunque sea lento.
        let mut slow = ok_step(100, 20, 5.0, 5.0, 5.0);
        flag_wall_cap(&mut slow, None);
        assert!(!slow.contaminated);
        // Inválido en env ⇒ None (cubierto en test de parseo aparte).
        assert!(wall_cap_secs().is_none());
    }

    #[test]
    fn empty_window_yields_na_machine_readable() {
        let samples = vec![
            StepSample::poisoned("gc"),
            StepSample::poisoned("gpu contenida"),
        ];
        let st = window_stats(&samples);
        assert_eq!(st.steps, 0);
        assert!(st.prefill_tok_s.is_none());
        assert!(st.decode_tok_s.is_none());
        assert!(st.ttft_mean_secs.is_none());
        assert_eq!(rate_json(st.prefill_tok_s), serde_json::json!("N/A"));
        assert_eq!(rate_json(st.decode_tok_s), serde_json::json!("N/A"));
        assert_eq!(rate_json(st.ttft_mean_secs), serde_json::json!("N/A"));
        let cfg = InferperfConfig::default();
        let rec = cell_record(
            &cfg,
            1.25,
            "mock-load",
            "mock",
            &st,
            true,
            "context-size",
            "automatic-mapping",
            cfg.ctx as u32,
        );
        assert_eq!(rec["metrics"]["prefill_tok_s"], serde_json::json!("N/A"));
        assert_eq!(rec["metrics"]["decode_tok_s"], serde_json::json!("N/A"));
        assert_eq!(rec["metrics"]["ttft_secs_mean"], serde_json::json!("N/A"));
        // …pero los crudos siguen presentes (ceros auditables).
        assert_eq!(rec["window"]["steps"], serde_json::json!(0));
        assert_eq!(rec["window"]["discarded"], serde_json::json!(2));
        assert_eq!(rec["load_duration_secs"], serde_json::json!(1.25));
    }

    #[test]
    fn jsonl_appends_raw_counts_and_full_config() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("sub").join("inferperf.jsonl");
        let mut cfg = InferperfConfig::default();
        cfg.backend = "ollama".to_string();
        cfg.model = "minicpm5:2b".to_string();
        // Ventana de 4 con 1 contaminado: crudos = 3 limpios.
        let script = vec![
            ok_step(1000, 100, 2.0, 5.0, 2.0),
            StepSample::poisoned("gc"),
            ok_step(1000, 100, 2.0, 5.0, 2.0),
        ];
        let mut be = ScriptedBackend::new("ollama", 3.5, script);
        let mut cell_cfg = cfg.clone();
        cell_cfg.window_samples = 4;
        cell_cfg.warmup_samples = 1;
        let rep = run_cell(&mut be, &cell_cfg, "mock-load", "mock-breakdown");
        // 1 warmup + 4 ventana = 5 llamadas al backend. Guion cíclico
        // [ok, poisoned, ok] en steps 1..=4 ⇒ limpios en 2,3 (2 limpios).
        assert_eq!(be.calls, 5);
        assert_eq!(rep.total_steps, 4);
        assert_eq!(rep.stats.steps, 2, "2 limpios de 4 (guion cíclico)");
        assert_eq!(rep.stats.discarded, 2);
        assert_eq!(rep.load_duration_secs, 3.5, "load separado de la ventana");
        let rec = cell_record(
            &cell_cfg,
            rep.load_duration_secs,
            &rep.load_kind,
            &rep.timing_mode,
            &rep.stats,
            true,
            "context-size",
            "automatic-mapping",
            cell_cfg.ctx as u32,
        );
        // Config completa por línea.
        for key in [
            "suite",
            "version",
            "backend",
            "model",
            "ctx",
            "features",
            "seed",
            "speculative",
            "engine_logging",
            "timing_mode",
            "load_duration_secs",
            "mem_mode",
            "decision_reason",
            "available_context_tokens",
        ] {
            assert!(rec.get(key).is_some(), "falta clave {key}");
        }
        assert_eq!(rec["backend"], serde_json::json!("ollama"));
        assert_eq!(rec["model"], serde_json::json!("minicpm5:2b"));
        assert_eq!(rec["features"]["kv_fp8"], serde_json::json!(false));
        assert!(rec["features"].get("paged_mem_config").is_some());
        // Speculative sin draft: "1 token/step" explícito.
        assert_eq!(
            rec["speculative"]["tokens_per_step"],
            serde_json::json!("1 token/step")
        );
        // Crudos auditan las tasas (2 limpios × 1000pt/100gt/2.0s/5.0s).
        assert_eq!(rec["window"]["steps"], serde_json::json!(2));
        assert_eq!(rec["window"]["prompt_tokens"], serde_json::json!(2000));
        assert_eq!(rec["window"]["gen_tokens"], serde_json::json!(200));
        assert!((rec["window"]["prefill_secs"].as_f64().unwrap() - 4.0).abs() < 1e-9);
        assert!((rec["window"]["decode_secs"].as_f64().unwrap() - 10.0).abs() < 1e-9);
        assert!((rec["metrics"]["prefill_tok_s"].as_f64().unwrap() - 500.0).abs() < 1e-9);
        assert!((rec["metrics"]["decode_tok_s"].as_f64().unwrap() - 20.0).abs() < 1e-9);
        // Append-only: dos celdas ⇒ dos líneas, ambas parseables.
        append_jsonl_record(&out, &rec).unwrap();
        let mut cfg2 = cell_cfg.clone();
        cfg2.backend = "kitty-infer".to_string();
        let rec2 = cell_record(
            &cfg2,
            9.0,
            "mock-load",
            "mock-breakdown",
            &rep.stats,
            true,
            "context-size",
            "automatic-mapping",
            cfg2.ctx as u32,
        );
        append_jsonl_record(&out, &rec2).unwrap();
        let text = std::fs::read_to_string(&out).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2, "append-only: una línea por celda");
        let v0: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        let v1: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(v0["backend"], serde_json::json!("ollama"));
        assert_eq!(v1["backend"], serde_json::json!("kitty-infer"));
        assert_eq!(v0["load_duration_secs"], serde_json::json!(3.5));
        assert_eq!(v1["load_duration_secs"], serde_json::json!(9.0));
    }

    #[test]
    fn wall_only_subset_reports_decode_only() {
        // Forma del path en vivo actual: sin split ⇒ prefill/TTFT N/A,
        // decode best-effort en su subset.
        let samples = vec![
            StepSample::wall_only(500, 50, 2.5),
            StepSample::wall_only(500, 50, 2.5),
        ];
        let st = window_stats(&samples);
        assert_eq!(st.steps, 0);
        assert_eq!(st.wall_only_steps, 2);
        assert!((st.wall_decode_tok_s.unwrap() - 20.0).abs() < 1e-9);
        assert_eq!(rate_json(st.prefill_tok_s), serde_json::json!("N/A"));
        assert_eq!(
            rate_json(st.wall_decode_tok_s),
            serde_json::json!(20.0),
            "wall-only sí reporta decode"
        );
    }

    #[test]
    fn synthetic_prompts_deterministic_with_fixed_seeds() {
        let a = synthetic_prompt(INFERPERF_SEED, 3, 64);
        let b = synthetic_prompt(INFERPERF_SEED, 3, 64);
        assert_eq!(a, b, "mismos args ⇒ mismos bytes");
        assert_ne!(
            a,
            synthetic_prompt(INFERPERF_SEED, 4, 64),
            "distinto step ⇒ distinto texto"
        );
        assert_ne!(
            a,
            synthetic_prompt(INFERPERF_SEED + 1, 3, 64),
            "distinta seed ⇒ distinto texto"
        );
        assert!(a.contains("step=3"), "trazabilidad del step");
        assert!(!a.is_empty());
    }

    #[test]
    fn speculative_accounting_explicit_without_draft() {
        assert_eq!(tokens_per_step_label(false, None), "1 token/step");
        assert_eq!(
            tokens_per_step_label(false, Some(3.5)),
            "1 token/step",
            "sin speculative el medido no aplica"
        );
        assert_eq!(
            tokens_per_step_label(true, Some(2.5)),
            "2.50 tokens/step"
        );
        let unmeasured = tokens_per_step_label(true, None);
        assert!(
            unmeasured.starts_with("1 token/step"),
            "draft sin medición sigue explícito: {unmeasured}"
        );
    }

    // Tests con env global: serializados con lock local (mismo patrón que
    // ledger `test_ledger_env_lock`: el env es global del proceso).
    static ENV_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn engine_logging_guard_suppresses_and_restores() {
        let _lock = ENV_SERIAL.lock().unwrap();
        // Caso base: sin env previo ⇒ suppressed false, guard off, drop limpia.
        std::env::remove_var(ENGINE_LOG_ENV);
        std::env::remove_var(ENGINE_LOG_LEGACY_ENV);
        assert!(!engine_logging_suppressed());
        {
            let _g = EngineLogGuard::suppress();
            assert!(engine_logging_suppressed(), "ventana con logging off");
            assert_eq!(std::env::var(ENGINE_LOG_ENV).unwrap(), "off");
        }
        assert!(!engine_logging_suppressed(), "drop restaura (sin previo)");
        assert!(std::env::var(ENGINE_LOG_ENV).is_err());
        // Con valor previo: se restaura exacto.
        std::env::set_var(ENGINE_LOG_ENV, "debug");
        {
            let _g = EngineLogGuard::suppress();
            assert!(engine_logging_suppressed());
        }
        assert_eq!(std::env::var(ENGINE_LOG_ENV).unwrap(), "debug");
        std::env::remove_var(ENGINE_LOG_ENV);
        std::env::remove_var(ENGINE_LOG_LEGACY_ENV);
    }

    #[test]
    fn exit_code_empty_window_is_inconclusive() {
        let _lock = ENV_SERIAL.lock().unwrap();
        let full = window_stats(&[ok_step(100, 10, 1.0, 1.0, 1.0)]);
        let empty = window_stats(&[StepSample::poisoned("x")]);
        assert_eq!(exit_code_for_stats(&full), 0);
        assert_eq!(exit_code_for_stats(&empty), 2, "N/A nunca publicable");
    }

    #[test]
    fn exit_code_wall_only_counts_as_valid_decode_window() {
        // F1b: el path en vivo solo produce wall-only (decode = wall);
        // su `wall_decode_tok_s` es publicable, así que cuenta como
        // ventana válida (exit 0), no como vacía (exit 2).
        let wall_only = window_stats(&[
            StepSample::wall_only(500, 50, 2.5),
            StepSample::wall_only(500, 50, 2.5),
        ]);
        assert_eq!(wall_only.wall_only_steps, 2);
        assert!(wall_only.wall_decode_tok_s.is_some());
        assert_eq!(exit_code_for_stats(&wall_only), 0);
        // Vacía de verdad (solo poisoned) sigue inconclusa.
        let empty = window_stats(&[StepSample::poisoned("x")]);
        assert_eq!(exit_code_for_stats(&empty), 2);
    }

    #[test]
    fn gen_tokens_bounded_via_num_predict() {
        // F1b: INFERPERF_GEN_TOKENS acota la generación vía
        // `chat_with_options(num_predict)` — nunca `chat` sin límite.
        assert_eq!(INFERPERF_GEN_TOKENS, 128);
        let cfg = InferperfConfig::default();
        assert_eq!(cfg.gen_tokens, INFERPERF_GEN_TOKENS);
        let opts = kitty_runtime::client::RequestOptions {
            num_predict: Some(INFERPERF_GEN_TOKENS as i32),
            ..Default::default()
        };
        assert_eq!(opts.num_predict, Some(128));
    }

    #[test]
    fn cell_record_carries_mem_mode_and_available_tokens() {
        let cfg = InferperfConfig::default();
        let st = window_stats(&[ok_step(1000, 100, 2.0, 5.0, 2.0)]);
        let rec = cell_record(
            &cfg,
            1.0,
            "mock-load",
            "mock",
            &st,
            true,
            "context-size",
            "automatic-mapping",
            16_384,
        );
        assert_eq!(rec["mem_mode"], serde_json::json!("context-size"));
        assert_eq!(
            rec["decision_reason"],
            serde_json::json!("automatic-mapping")
        );
        assert_eq!(rec["available_context_tokens"], serde_json::json!(16_384));
    }
}

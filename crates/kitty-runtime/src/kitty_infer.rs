//! Embedded kitty-infer GGUF backend (`ChatClient`).
//!
//! In-process inference via the sibling `kitty-infer` repo
//! (`mistralrs::GgufModelBuilder`): `GgufLoadPlan` pieces → single-resident
//! registry (`global_registry`, evict-before-load + OOM retry) → generate.
//! Selected with `KITTY_BACKEND=kitty-infer` + `KITTY_INFER_MODEL=<path.gguf>`.
//!
//! Policy (lives in kitty-infer `serving::infer_policy`, shared with the
//! daemon path):
//! - dtype default F16 (`Auto`) + F32 override for `qwen2.5-coder*` (isolated
//!   F16-CUDA NaN on sm_75, humo F1) + `KITTY_INFER_F32_PREFIXES` extras;
//! - MiniCPM5 `tokenizer.json` sidecar-or-honest-error
//!   (`KITTY_INFER_TOKENIZER_DIR`, reference layout
//!   `<dir>/MiniCPM5-2B/tokenizer.json`).
//!
//! Explicit backend ⇒ explicit errors: a missing/unloadable GGUF fails
//! honestly, it NEVER silently falls back to Ollama (that would hide the
//! operator's choice). No auth in-process (nothing to authenticate).

// Build sin feature `kitty-infer`: los helpers puros quedan sin uso porque el
// cliente real vive tras el feature (Gate C) — silenciar ese ruido sin tocar
// el build con feature, donde siguen advirtiendo si mueren de verdad.
#![cfg_attr(not(feature = "kitty-infer"), allow(dead_code))]

use std::path::PathBuf;

use crate::error::{Result, RuntimeError};

#[cfg(feature = "kitty-infer")]
use crate::client::{ChatClient, ChatResponse, Message};
#[cfg(feature = "kitty-infer")]
use crate::tools::ToolDefinition;
#[cfg(feature = "kitty-infer")]
use std::future::Future;
#[cfg(feature = "kitty-infer")]
use std::pin::Pin;
#[cfg(feature = "kitty-infer")]
use std::sync::Arc;

#[cfg(feature = "kitty-infer")]
use mistralrs::{
    serving as ki, Constraint, GgufModelBuilder, Model, PagedAttentionMetaBuilder, ReasoningEffort,
    RequestBuilder, TextMessageRole,
};

/// Env switch selecting this backend in `kitty-cli`.
pub const ENV_BACKEND: &str = "KITTY_BACKEND";
/// Value of [`ENV_BACKEND`] that selects the embedded backend.
pub const BACKEND_KITTY_INFER: &str = "kitty-infer";
/// GGUF path for the embedded backend.
pub const ENV_MODEL: &str = "KITTY_INFER_MODEL";
/// Sidecar dir holding `<...>/MiniCPM5-2B/tokenizer.json`.
pub const ENV_TOKENIZER_DIR: &str = "KITTY_INFER_TOKENIZER_DIR";
/// Context length override (default 4096 — honest for 8GB consumer GPUs).
pub const ENV_CTX: &str = "KITTY_INFER_CTX";
/// Extra comma-separated F32 filename prefixes.
pub const ENV_F32_PREFIXES: &str = "KITTY_INFER_F32_PREFIXES";
/// GPU layer count override (e.g. `12` keeps 12 layers on cuda[0], spilling
/// the rest to CPU). Unset = automatic mapping. Needed for >8GB models
/// (K2 9.5GB) where auto packs too tight to survive prefill transients.
pub const ENV_GPU_LAYERS: &str = "KITTY_INFER_GPU_LAYERS";

const DEFAULT_CTX: usize = 4096;

/// Returns true when the operator selected the embedded backend. Pure and
/// unit-testable (env read stays at the call site).
pub fn backend_selected(raw: Option<&str>) -> bool {
    raw.map(|v| v.trim() == BACKEND_KITTY_INFER)
        .unwrap_or(false)
}

/// Parse the GPU-layer spill override. `None` = unset/invalid = automatic
/// mapping. Pure and unit-testable.
fn parse_gpu_layers(raw: Option<&str>) -> Option<usize> {
    raw?.trim().parse::<usize>().ok().filter(|v| *v > 0)
}

/// Validate a spill count against the model's own layer total. `n > total`
/// is always nonsense (asking for more GPU layers than exist) and bails.
/// Unknown total allows through: the absolute refuses it replaced couldn't
/// see per-model size either, and the byte pre-check stays fail-closed
/// below. Pure and unit-testable.
fn validate_gpu_layers(n: usize, total: Option<usize>) -> Result<()> {
    if let Some(t) = total {
        if n > t {
            return Err(RuntimeError::InvalidArgument(format!(
                "KITTY_INFER_GPU_LAYERS={n} exceeds the model's {t} layers — \
                 check the value (internal knob documented ≤12)"
            )));
        }
    }
    Ok(())
}

/// Layered GPU-byte estimate mirroring `vram_precheck` shape with spill:
/// `file × (n/total) × fraction + kv(ctx)`. Unknown total (or no spill)
/// falls back to the legacy full-file shape. Pure and unit-testable.
fn spill_estimate_bytes(
    file_len: u64,
    layers: Option<usize>,
    total: Option<usize>,
    fraction: f32,
    ctx_len: usize,
) -> u64 {
    let share = match (layers, total) {
        (Some(n), Some(t)) if t > 0 => n.min(t) as f64 / t as f64,
        _ => 1.0,
    };
    let model = file_len as f64 * share * fraction.clamp(0.0, 1.0) as f64;
    let kv = ctx_len as u64 * 256 * 1024;
    model as u64 + kv
}

/// Headroom under which a spill estimate warns. Measured: spill-20's
/// estimate sat ~1% under free yet OOMed on prefill transients/graphs, so
/// "under free" alone is not safe on 8GB — warn above 90%.
const SPILL_WARN_HEADROOM: f64 = 0.90;

/// Read the GGUF's own `block_count` for its declared architecture from the
/// file header (no model load). Returns `None` on any parse miss — callers
/// treat unknown as unvalidated, never as zero. Pure file IO, no new deps.
fn gguf_read_u32(f: &mut std::fs::File) -> Option<u32> {
    use std::io::Read;
    let mut b = [0u8; 4];
    f.read_exact(&mut b).ok()?;
    Some(u32::from_le_bytes(b))
}

fn gguf_read_u64(f: &mut std::fs::File) -> Option<u64> {
    use std::io::Read;
    let mut b = [0u8; 8];
    f.read_exact(&mut b).ok()?;
    Some(u64::from_le_bytes(b))
}

fn gguf_read_str(f: &mut std::fs::File) -> Option<String> {
    use std::io::Read;
    let len = gguf_read_u64(f)? as usize;
    if len > 100_000_000 {
        return None;
    }
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf).ok()?;
    String::from_utf8(buf).ok()
}

fn gguf_block_count(path: &std::path::Path) -> Option<usize> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let read_u32 = gguf_read_u32;
    let read_u64 = gguf_read_u64;
    let read_str = gguf_read_str;
    if &{
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic).ok()?;
        magic
    } != b"GGUF"
    {
        return None;
    }
    read_u32(&mut f)?; // version
    let tensor_count = read_u64(&mut f)?;
    let kv_count = read_u64(&mut f)?;
    if tensor_count > 1_000_000 || kv_count > 1_000_000 {
        return None;
    }
    // Order-independent: remember any `.block_count` candidate and resolve
    // against the architecture once known (writers usually put general.*
    // first, but nothing guarantees it).
    let mut arch: Option<String> = None;
    let mut candidate: Option<(String, u64)> = None;
    for _ in 0..kv_count {
        let key = read_str(&mut f)?;
        let ty = read_u32(&mut f)?;
        if key == "general.architecture" && ty == 8 {
            arch = Some(read_str(&mut f)?);
        } else if ty == 4 {
            let v = read_u32(&mut f)?;
            if candidate.is_none() && key.ends_with(".block_count") {
                candidate = Some((key, v as u64));
            }
        } else if ty == 10 {
            let mut b8 = [0u8; 8];
            f.read_exact(&mut b8).ok()?;
            let v = u64::from_le_bytes(b8);
            if candidate.is_none() && key.ends_with(".block_count") {
                candidate = Some((key, v));
            }
        } else {
            skip_gguf_value(&mut f, ty)?;
        }
    }
    let (key, count) = candidate?;
    let arch = arch?;
    if key != format!("{arch}.block_count") {
        return None;
    }
    usize::try_from(count).ok().filter(|v| *v > 0)
}

/// GGUF architectures the embedded engine can serve. Mirror of
/// `engine/mistralrs-core/src/gguf/normal_registry.rs` `as_str`; if the
/// registry gains an arch, add its string here.
const SUPPORTED_GGUF_ARCHITECTURES: &[&str] = &[
    "llama",
    "mistral3",
    "gemma",
    "gemma2",
    "phi2",
    "phi3",
    "phimoe",
    "qwen2",
    "qwen3",
    "qwen3moe",
    "qwen3next",
    "qwen35",
    "qwen35moe",
    "starcoder2",
    "deepseek2",
    "glm4",
    "glm4moe",
    "smollm3",
    "granite",
    "granitemoe",
    "granitehybrid",
    "gpt-oss",
    "hunyuan-dense",
    "hunyuan-moe",
    "lfm2",
    "lfm2moe",
    "k2-horizon",
];

/// Read `general.architecture` from a GGUF header. `None` = unreadable
/// header (fail-open: the loader gives the final verdict).
fn gguf_architecture(path: &std::path::Path) -> Option<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }
    gguf_read_u32(&mut f)?; // version
    let tensor_count = gguf_read_u64(&mut f)?;
    let kv_count = gguf_read_u64(&mut f)?;
    if tensor_count > 1_000_000 || kv_count > 1_000_000 {
        return None;
    }
    for _ in 0..kv_count {
        let key = gguf_read_str(&mut f)?;
        let ty = gguf_read_u32(&mut f)?;
        if key == "general.architecture" && ty == 8 {
            return gguf_read_str(&mut f);
        }
        skip_gguf_value(&mut f, ty)?;
    }
    None
}

fn is_supported_gguf_arch(arch: &str) -> bool {
    SUPPORTED_GGUF_ARCHITECTURES.contains(&arch)
}

/// Early arch gate: fail fast with an actionable message instead of a late
/// loader error. Unreadable headers fail OPEN (the loader decides).
fn check_gguf_arch(path: &std::path::Path) -> Result<()> {
    match gguf_architecture(path) {
        None => Ok(()),
        Some(arch) if is_supported_gguf_arch(&arch) => Ok(()),
        Some(arch) => Err(RuntimeError::InvalidArgument(format!(
            "GGUF architecture `{arch}` is not served by the embedded backend — \
             check docs/SUPPORTED-MODELS.md for the supported list, or run it via Ollama",
        ))),
    }
}

/// Skip one GGUF metadata value of the given type tag.
fn skip_gguf_value(f: &mut std::fs::File, ty: u32) -> Option<()> {
    use std::io::{Read, Seek, SeekFrom};
    match ty {
        0 | 1 => {
            f.seek(SeekFrom::Current(1)).ok()?;
        }
        2 | 3 => {
            f.seek(SeekFrom::Current(2)).ok()?;
        }
        4..=6 => {
            f.seek(SeekFrom::Current(4)).ok()?;
        }
        7 => {
            f.seek(SeekFrom::Current(1)).ok()?;
        }
        8 => {
            let mut len = [0u8; 8];
            f.read_exact(&mut len).ok()?;
            f.seek(SeekFrom::Current(u64::from_le_bytes(len) as i64))
                .ok()?;
        }
        9 => {
            // array: element type + count, then elements (assume no nesting)
            let mut b4 = [0u8; 4];
            f.read_exact(&mut b4).ok()?;
            let elem = u32::from_le_bytes(b4);
            let mut b8 = [0u8; 8];
            f.read_exact(&mut b8).ok()?;
            let n = u64::from_le_bytes(b8);
            if n > 10_000_000 {
                return None;
            }
            for _ in 0..n {
                skip_gguf_value(f, elem)?;
            }
        }
        10..=12 => {
            f.seek(SeekFrom::Current(8)).ok()?;
        }
        _ => return None,
    }
    Some(())
}

/// Thinking toggle for reasoning templates (K2 opens `<ifm|think>` at high
/// effort by default — fine on full GPU, far too slow on CPU-spilled
/// residents). `KITTY_INFER_NO_THINK=1` renders the empty think block.
/// Pure and unit-testable; anything else (unset/invalid) keeps thinking on.
pub const ENV_NO_THINK: &str = "KITTY_INFER_NO_THINK";
/// Template tool-call format override (`json` | `xml` | `xml_typed`;
/// unset/unknown = template default, `xml` for K2). Read by `to_request`.
pub const ENV_TOOL_FORMAT: &str = "KITTY_INFER_TOOL_FORMAT";
/// Reasoning-effort override (`low` | `medium`; unset/other = engine default).
/// Read by `to_request` (K2 effort probe: `think_fast`/`think_faster` tags).
pub const ENV_EFFORT: &str = "KITTY_INFER_REASONING_EFFORT";

/// Validated tool-call format for the template context. Returns the
/// lowercase `json`/`xml`/`xml_typed` value, else `None` (template default
/// wins). Pure and unit-testable.
fn tool_format(raw: Option<&str>) -> Option<String> {
    match raw.map(str::trim) {
        Some("json" | "xml" | "xml_typed") => raw.map(|v| v.trim().to_string()),
        _ => None,
    }
}

/// Sampler overrides carried from `RequestOptions` into the builder.
/// All-`None` = engine defaults (plain `chat` path).
#[derive(Debug, Clone, Copy, Default)]
struct SamplerOpts {
    max_len: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
}

/// Env overrides for sampler instrumentation (A/B recipe runs):
/// `KITTY_INFER_TEMP`, `KITTY_INFER_TOP_P`, `KITTY_INFER_TOP_K`. Invalid or
/// absent values leave the field untouched. Pure and unit-testable (env
/// read stays at the call site).
pub const ENV_TEMP: &str = "KITTY_INFER_TEMP";
pub const ENV_TOP_P: &str = "KITTY_INFER_TOP_P";
pub const ENV_TOP_K: &str = "KITTY_INFER_TOP_K";

fn sampler_env_override(
    samp: SamplerOpts,
    temp: Option<&str>,
    top_p: Option<&str>,
    top_k: Option<&str>,
) -> SamplerOpts {
    let mut out = samp;
    if let Some(t) = temp
        .map(str::trim)
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|t| t.is_finite() && (0.0..=2.0).contains(t))
    {
        out.temperature = Some(t);
    }
    if let Some(p) = top_p
        .map(str::trim)
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|p| p.is_finite() && (0.0..=1.0).contains(p))
    {
        out.top_p = Some(p);
    }
    if let Some(k) = top_k
        .map(str::trim)
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|k| *k > 0)
    {
        out.top_k = Some(k.min(1000));
    }
    out
}

/// Opt-in decode-time grammar mask (W3/T3.1 spike): `KITTY_INFER_GRAMMAR=1`
/// threads `GrammarConfig::to_json_schema` into the engine sampler via the
/// existing `RequestBuilder::set_constraint` (no engine change — the
/// `Constraint::JsonSchema → llg_factory → Llguidance` mask path already
/// exists: `build_sequence_recognizer` + `pipeline/llg.rs`). Unset/other =
/// post-hoc `GrammarClient` only (previous behavior, default).
pub const ENV_GRAMMAR: &str = "KITTY_INFER_GRAMMAR";

/// Pure and unit-testable; anything else (unset/invalid) keeps post-hoc only.
#[cfg(any(feature = "kitty-infer", test))]
fn grammar_enabled(raw: Option<&str>) -> bool {
    matches!(
        raw.map(|v| v.trim()),
        Some("1") | Some("true") | Some("yes")
    )
}

/// Build the llguidance JSON-Schema mask for this request. `None` = no
/// decode-time constraint (mask off, empty tool set, or no tools passed).
/// Pure and unit-testable.
#[cfg(any(feature = "kitty-infer", test))]
fn grammar_schema_for(
    tools: Option<&Vec<crate::tools::ToolDefinition>>,
    enabled: bool,
) -> Option<serde_json::Value> {
    if !enabled {
        return None;
    }
    let tools = tools?;
    if tools.is_empty() {
        return None;
    }
    Some(crate::grammar::GrammarConfig::from_tools(tools.clone()).to_json_schema())
}

/// Map harness options onto sampler overrides. Pure and unit-testable:
/// out-of-range values fall back to engine defaults rather than erroring
/// mid-turn.
fn sampler_opts_from(options: Option<crate::client::RequestOptions>) -> SamplerOpts {
    let mut out = SamplerOpts::default();
    if let Some(o) = options {
        // Clamp to something sane: harness-derived caps can exceed the
        // resident window when they were computed against another backend.
        out.max_len = o
            .num_predict
            .filter(|n| *n > 0)
            .map(|n| (n as usize).min(8192));
        out.temperature = o
            .temperature
            .map(|t| t as f64)
            .filter(|t| t.is_finite() && (0.0..=2.0).contains(t));
        out.top_p = o
            .top_p
            .map(|p| p as f64)
            .filter(|p| p.is_finite() && (0.0..=1.0).contains(p));
        out.top_k = o.top_k.filter(|k| *k > 0).map(|k| (k as usize).min(1000));
    }
    out
}

/// Fallback id for an id-less tool result: positional `tool-{idx}` among
/// the tool messages of the request (idx 0 keeps the historical `"tool-0"`).
/// Pure and unit-testable.
fn tool_fallback_id(id: Option<&str>, idx: usize) -> String {
    id.map(str::to_string)
        .unwrap_or_else(|| format!("tool-{idx}"))
}

fn thinking_enabled(raw: Option<&str>) -> bool {
    !matches!(
        raw.map(|v| v.trim()),
        Some("1") | Some("true") | Some("yes")
    )
}

/// Resolved configuration for one embedded resident.
#[derive(Debug, Clone)]
pub struct KittyInferConfig {
    pub model_path: PathBuf,
    pub tokenizer_sidecar_dir: Option<PathBuf>,
    pub ctx_len: usize,
}

impl KittyInferConfig {
    pub fn from_env() -> Result<Self> {
        let raw = std::env::var(ENV_MODEL).map_err(|_| {
            RuntimeError::InvalidArgument(format!(
                "embedded backend selected ({ENV_BACKEND}={BACKEND_KITTY_INFER}) but \
                 {ENV_MODEL} is not set — set it to a local .gguf file \
                 (e.g. {ENV_MODEL}=~/.cache/kitty/models/Qwen3-4B-Q4_K_M.gguf)"
            ))
        })?;
        let ctx_len = match std::env::var(ENV_CTX) {
            Err(_) => DEFAULT_CTX,
            Ok(raw) => raw
                .trim()
                .parse::<usize>()
                .ok()
                .filter(|v| *v > 0)
                .ok_or_else(|| {
                    RuntimeError::InvalidArgument(format!(
                        "{ENV_CTX}={raw:?} is not a positive integer — unset it or \
                         set a value like {ENV_CTX}=4096"
                    ))
                })?,
        };
        Self::from_path(
            PathBuf::from(raw),
            std::env::var(ENV_TOKENIZER_DIR).ok().map(PathBuf::from),
            ctx_len,
        )
    }

    pub fn from_path(
        model_path: PathBuf,
        tokenizer_sidecar_dir: Option<PathBuf>,
        ctx_len: usize,
    ) -> Result<Self> {
        if model_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            != Some("gguf".to_string())
        {
            return Err(RuntimeError::InvalidArgument(format!(
                "embedded backend needs a local .gguf file, got `{}` — \
                 hint: KITTY_INFER_MODEL must point at a GGUF (try the humo-verified \
                 Qwen3-4B-Q4_K_M.gguf or MiniCPM5-2B-Q4_K_M.gguf)",
                model_path.display()
            )));
        }
        if !model_path.is_file() {
            return Err(RuntimeError::ModelNotFound(format!(
                "GGUF not found at {} — hint: check KITTY_INFER_MODEL; humo-verified \
                 models live under ~/.cache/kitty/models/",
                model_path.display()
            )));
        }
        Ok(Self {
            model_path,
            tokenizer_sidecar_dir,
            ctx_len: ctx_len.max(512),
        })
    }
}

// ---------------------------------------------------------------------------
// Pure mapping helpers (feature-independent, unit-tested without a model)
// ---------------------------------------------------------------------------

/// Map a kitty role string to a mistralrs role. Tool results never reach
/// this classifier: `to_request` intercepts `role == "tool"` first and
/// threads [`Message::tool_call_id`] through `add_tool_message` (fallback
/// `"tool-0"` documented on the constructor).
#[cfg(any(feature = "kitty-infer", test))]
fn map_role(role: &str) -> (&'static str, bool) {
    match role {
        "system" => ("system", false),
        "assistant" => ("assistant", false),
        _ => ("user", false),
    }
}

// ---------------------------------------------------------------------------
// Loaded client (feature-gated; mirrors LlamaCppClient structure)
// ---------------------------------------------------------------------------

/// Process-wide loaded models, keyed by [`ModelSpec`](mistralrs::serving::ModelSpec)
/// id. The single-flight lock lives in `global_registry`; this map only
/// makes a registry-shared load visible to every client (a per-client slot
/// would diverge: client B would miss a model client A just built).
/// Pruned to the winner on every successful load, so the map honors the
/// single-resident contract instead of pinning evicted models in VRAM.
/// Plain std mutex, never held across await.
#[cfg(feature = "kitty-infer")]
static LOADED: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<String, Arc<Model>>>,
> = std::sync::OnceLock::new();

#[cfg(feature = "kitty-infer")]
fn loaded_map() -> &'static std::sync::Mutex<std::collections::HashMap<String, Arc<Model>>> {
    LOADED.get_or_init(Default::default)
}

/// Embedded kitty-infer backend: single-resident GGUF behind [`ChatClient`].
///
/// Concurrency contract (D1): serialization lives in `global_registry`,
/// which holds its guard across evict → load → store and reuses the
/// resident for same-spec waiters — two concurrent `ensure_loaded` (same
/// or different clients, same spec) compile the model exactly once, and
/// both read it from [`loaded_map`]. Cross-spec races serialize with
/// last-writer-wins residency, same as the registry metadata.
#[cfg(feature = "kitty-infer")]
#[derive(Clone)]
pub struct KittyInferClient {
    config: KittyInferConfig,
}

#[cfg(feature = "kitty-infer")]
impl KittyInferClient {
    pub fn new(config: KittyInferConfig) -> Self {
        Self { config }
    }

    pub fn from_env() -> Result<Self> {
        Ok(Self::new(KittyInferConfig::from_env()?))
    }

    pub fn config(&self) -> &KittyInferConfig {
        &self.config
    }

    fn spec_id(&self) -> Option<String> {
        ki::ModelSpec::auto(&format!("local:{}", self.config.model_path.display()))
            .ok()
            .map(|spec| spec.id)
    }

    pub fn is_ready(&self) -> bool {
        // Non-blocking probe only; never loads here.
        self.spec_id()
            .and_then(|id| loaded_map().lock().ok().map(|m| m.contains_key(&id)))
            .unwrap_or(false)
    }

    fn friendly(err: impl std::fmt::Display) -> RuntimeError {
        let classified = ki::KittyError::classify(&err.to_string());
        let body = classified.body();
        RuntimeError::KittyInfer(format!("{} — hint: {}", body.error, body.hint))
    }

    /// Load (or reuse) the single resident. Evict-before-load + OOM retry
    /// come from `global_registry`; the `gpu_fraction` attempt scales the
    /// context (the KV lever this client controls), and `vram_precheck`
    /// fails fast with byte counts before touching the engine.
    async fn ensure_loaded(&self) -> Result<Arc<Model>> {
        let cfg = self.config.clone();
        let file_name = cfg
            .model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        let parent = cfg
            .model_path
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string());
        let extra = ki::extra_f32_prefixes(std::env::var(ENV_F32_PREFIXES).ok().as_deref());
        let dtype = ki::resolve_dtype(&file_name, &extra);
        let tok_json =
            ki::resolve_tokenizer_json(&cfg.model_path, cfg.tokenizer_sidecar_dir.as_deref())
                .map_err(Self::friendly)?;
        let file_len = ki::file_bytes(&cfg.model_path).map_err(Self::friendly)?;
        // Spill validation is relative to the model's own layer total
        // (read from the GGUF header, no load): n > total always bails.
        // Unknown total allows through (documented: legacy had no per-model
        // view either) with the byte pre-check below staying fail-closed.
        // Internal knob documented ≤12 (measured 8GB envelope).
        let gpu_layers = parse_gpu_layers(std::env::var(ENV_GPU_LAYERS).ok().as_deref());
        let total_layers = gpu_layers
            .is_some()
            .then(|| gguf_block_count(&cfg.model_path))
            .flatten();
        if let Some(n) = gpu_layers {
            validate_gpu_layers(n, total_layers)?;
        }
        let spec = ki::ModelSpec::auto(&format!("local:{}", cfg.model_path.display()))
            .map_err(Self::friendly)?;
        let spec_id = spec.id.clone();

        // Fast path: another client (or an earlier turn) already built it.
        if let Some(model) = loaded_map()
            .lock()
            .ok()
            .and_then(|m| m.get(&spec_id).cloned())
        {
            return Ok(model);
        }

        let resident = ki::global_registry()
            .load_with_retry(spec, |fraction| {
                let file_name = file_name.clone();
                let parent = parent.clone();
                let tok_json = tok_json.clone();
                let spec_id = spec_id.clone();
                let gpu_layers = gpu_layers;
                let total_layers = total_layers;
                async move {
                    // Attempt ctx scales with the retry fraction (1.0 → ~0.66):
                    // KV cache is the OOM lever this client owns.
                    let ctx = ((cfg.ctx_len as f32 * fraction) as usize).max(512);
                    let free = ki::health::snapshot()
                        .devices
                        .first()
                        .map(|d| d.total_bytes.saturating_sub(d.used_bytes));
                    // Layered estimate: with spill only n/total of the file
                    // sits on GPU. Warn above headroom (not fail — the
                    // scaled pre-check below stays fail-closed with bytes).
                    if let (Some(n), Some(t)) = (gpu_layers, total_layers) {
                        let est = spill_estimate_bytes(file_len, Some(n), Some(t), fraction, ctx);
                        if let Some(free_bytes) = free {
                            if est as f64 > free_bytes as f64 * SPILL_WARN_HEADROOM {
                                tracing::warn!(
                                    estimate_bytes = est,
                                    free_bytes,
                                    gpu_layers = n,
                                    total_layers = t,
                                    "spill estimate within headroom of free VRAM; \
                                     prefill transients may still OOM — failure will be loud"
                                );
                            }
                        }
                    }
                    // Scale the precheck input so a spill passes on its real
                    // GPU share instead of the full file (spill-20 passed the
                    // legacy check and OOMed late — the HOLD this fixes).
                    let eff_file = match (gpu_layers, total_layers) {
                        (Some(n), Some(t)) if t > 0 => {
                            ((file_len as f64 * n.min(t) as f64 / t as f64) as u64).max(1)
                        }
                        _ => file_len,
                    };
                    ki::vram_precheck(eff_file, free, fraction, ctx)
                        .map_err(|e| anyhow::anyhow!("{e:#}"))?;
                    let mut builder = GgufModelBuilder::new(parent, vec![file_name])
                        .with_logging()
                        .with_max_model_len(ctx)
                        .with_dtype(dtype.clone())
                        // Paged KV capacity must follow the attempt ctx: the
                        // builder default pins ContextSize(4096) = 128 blocks,
                        // which starves agent prompts (measured K2 panic).
                        .with_paged_attn(
                            PagedAttentionMetaBuilder::default()
                                .with_gpu_memory(mistralrs::MemoryGpuConfig::ContextSize(ctx))
                                .build()?,
                        );
                    // Pre-validated above (relative clamp); the spill itself is
                    // honored verbatim here so load matches what passed checks.
                    if let Some(n) = gpu_layers {
                        builder = builder.with_device_mapping(mistralrs::DeviceMapSetting::Map(
                            mistralrs::DeviceMapMetadata::from_num_device_layers(vec![
                                mistralrs::DeviceLayerMapMetadata {
                                    ordinal: 0,
                                    layers: n,
                                },
                            ]),
                        ));
                    }
                    if let Some(tj) = tok_json {
                        builder = builder.with_tokenizer_json(tj.to_string_lossy().into_owned());
                    }
                    let model = builder
                        .build()
                        .await
                        .map_err(|e| anyhow::anyhow!("{e:#}"))?;
                    loaded_map()
                        .lock()
                        .map_err(|_| anyhow::anyhow!("loaded-model map is poisoned"))?
                        .insert(spec_id, Arc::new(model));
                    Ok(())
                }
            })
            .await
            .map_err(Self::friendly)?;
        tracing::info!(
            model = %resident.spec.id,
            gpu_fraction = resident.gpu_fraction,
            ctx = cfg.ctx_len,
            gpu_layers = ?gpu_layers,
            thinking = thinking_enabled(std::env::var(ENV_NO_THINK).ok().as_deref()),
            "kitty-infer resident ready"
        );
        // Prune evicted models so the map honors single-residence.
        let mut map = loaded_map()
            .lock()
            .map_err(|_| RuntimeError::KittyInfer("loaded-model map is poisoned".to_string()))?;
        map.retain(|id, _| id == &resident.spec.id);
        map.get(&resident.spec.id).cloned().ok_or_else(|| {
            RuntimeError::KittyInfer(
                "load reported success but no model is resident — retry once; \
                 if it repeats, capture RUST_LOG=info output"
                    .to_string(),
            )
        })
    }

    fn to_request(
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        grammar_schema: Option<serde_json::Value>,
    ) -> RequestBuilder {
        let mut req = RequestBuilder::new().enable_thinking(thinking_enabled(
            std::env::var(ENV_NO_THINK).ok().as_deref(),
        ));
        if let Some(fmt) = tool_format(std::env::var(ENV_TOOL_FORMAT).ok().as_deref()) {
            req = req.with_tool_call_format(fmt);
        }
        match std::env::var(ENV_EFFORT).ok().as_deref() {
            Some("low") => req = req.with_reasoning_effort(ReasoningEffort::Low),
            Some("medium") => req = req.with_reasoning_effort(ReasoningEffort::Medium),
            _ => {}
        }
        let mut tool_idx = 0usize;
        for m in messages {
            let text = m.content.clone().unwrap_or_default();
            if m.role == "tool" {
                // Threaded tool_call_id (harness-filled); id-less results
                // get a positional `tool-{idx}` fallback (idx 0 keeps the
                // legacy `"tool-0"`). First id-less result is debug-level
                // (legacy single-tool path); idx ≥ 1 is genuinely ambiguous
                // and stays warn-level — bounded by tool count, never a loop.
                let id = tool_fallback_id(m.tool_call_id.as_deref(), tool_idx);
                if m.tool_call_id.is_none() {
                    if tool_idx == 0 {
                        tracing::debug!(
                            "tool result without tool_call_id; positional fallback \"tool-0\""
                        );
                    } else {
                        tracing::warn!(
                            tool_idx,
                            "id-less tool result past the first cannot pair reliably; \
                             positional fallback used (multi-tool turns need ids)"
                        );
                    }
                }
                tool_idx += 1;
                req = req.add_tool_message(text, id);
                continue;
            }
            let (kind, _) = map_role(&m.role);
            match kind {
                "system" => {
                    req = req.add_message(TextMessageRole::System, text);
                }
                "assistant" if m.tool_calls.as_ref().is_some_and(|c| !c.is_empty()) => {
                    let calls = m
                        .tool_calls
                        .unwrap_or_default()
                        .into_iter()
                        .map(|c| mistralrs::ToolCallResponse {
                            index: 0,
                            id: c.id.unwrap_or_else(|| "call-0".to_string()),
                            tp: mistralrs::ToolCallType::Function,
                            function: mistralrs::CalledFunction {
                                name: c.function.name,
                                arguments: c.function.arguments.to_string(),
                            },
                        })
                        .collect();
                    req = req.add_message_with_tool_call(TextMessageRole::Assistant, text, calls);
                }
                "assistant" => {
                    req = req.add_message(TextMessageRole::Assistant, text);
                }
                _ => {
                    req = req.add_message(TextMessageRole::User, text);
                }
            }
        }
        if let Some(tools) = tools {
            if !tools.is_empty() {
                req = req.set_tools(
                    tools
                        .into_iter()
                        .map(|t| mistralrs::Tool {
                            tp: mistralrs::ToolType::Function,
                            function: mistralrs::Function {
                                name: t.function.name,
                                description: Some(t.function.description),
                                parameters: Some(t.function.parameters),
                                strict: None,
                            },
                        })
                        .collect(),
                );
            }
        }
        // W3/T3.1: decode-time llguidance mask. Pre-existing engine path
        // (`Constraint::JsonSchema` → `build_sequence_recognizer` →
        // `pipeline/llg.rs` factory → `SequenceRecognizer::Llguidance`
        // biasing logits in `sample_sequence`); a bad schema fails the
        // request loudly, never silently unconstrained.
        if let Some(schema) = grammar_schema {
            req = req.set_constraint(Constraint::JsonSchema(schema));
        }
        req
    }

    async fn chat_inner(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        samp: SamplerOpts,
    ) -> Result<ChatResponse> {
        let loaded = self.ensure_loaded().await?;
        // W3/T3.0 plumbing: GrammarConfig (from this request's tools) →
        // JSON-Schema mask → pipeline llg. Opt-in via KITTY_INFER_GRAMMAR.
        let grammar_schema = grammar_schema_for(
            tools.as_ref(),
            grammar_enabled(std::env::var(ENV_GRAMMAR).ok().as_deref()),
        );
        let mut req = Self::to_request(messages, tools, grammar_schema);
        if let Some(n) = samp.max_len {
            req = req.set_sampler_max_len(n);
        }
        if let Some(t) = samp.temperature {
            req = req.set_sampler_temperature(t);
        }
        if let Some(p) = samp.top_p {
            req = req.set_sampler_topp(p);
        }
        if let Some(k) = samp.top_k {
            req = req.set_sampler_topk(k);
        }
        let resp = loaded
            .send_chat_request(req)
            .await
            .map_err(Self::friendly)?;
        let choice = resp.choices.first().ok_or_else(|| {
                RuntimeError::KittyInfer(
                    "engine returned zero choices — retry once; hint: check RUST_LOG=info for pipeline errors".to_string(),
                )
            })?;
        let tool_calls = choice.message.tool_calls.as_ref().map(|calls| {
            calls
                .iter()
                .map(|c| crate::client::ToolCall {
                    id: Some(c.id.clone()),
                    function: crate::client::ToolCallFunction {
                        name: c.function.name.clone(),
                        arguments: serde_json::from_str(&c.function.arguments)
                            .unwrap_or(serde_json::Value::String(c.function.arguments.clone())),
                    },
                })
                .collect()
        });
        Ok(ChatResponse {
            model: model.to_string(),
            message: Message {
                role: "assistant".to_string(),
                content: choice.message.content.clone(),
                thinking: None,
                tool_calls,
                tool_call_id: None,
            },
            done: true,
            eval_count: Some(resp.usage.completion_tokens as i64),
            eval_duration: None,
            prompt_eval_count: Some(resp.usage.prompt_tokens as i64),
        })
    }
}

#[cfg(feature = "kitty-infer")]
impl std::fmt::Debug for KittyInferClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KittyInferClient")
            .field("model_path", &self.config.model_path)
            .field("ready", &self.is_ready())
            .finish()
    }
}

#[cfg(feature = "kitty-infer")]
impl ChatClient for KittyInferClient {
    fn chat<'a>(
        &'a self,
        model: &'a str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>> {
        Box::pin(async move {
            self.chat_inner(model, messages, tools, SamplerOpts::default())
                .await
        })
    }

    /// Honor `RequestOptions` (token cap + sampler). The trait default
    /// IGNORES options, which left embedded generation unbounded (up to
    /// `max_model_len`) and turned agent turns into 25-min timeouts.
    /// `num_predict` clamps to 8192 (harness caps computed against another
    /// backend can exceed the resident window); temperature/top_p/top_k
    /// pass through when finite and in range, else engine defaults hold.
    fn chat_with_options<'a>(
        &'a self,
        model: &'a str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        options: Option<crate::client::RequestOptions>,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>> {
        let samp = sampler_env_override(
            sampler_opts_from(options),
            std::env::var(ENV_TEMP).ok().as_deref(),
            std::env::var(ENV_TOP_P).ok().as_deref(),
            std::env::var(ENV_TOP_K).ok().as_deref(),
        );
        Box::pin(async move { self.chat_inner(model, messages, tools, samp).await })
    }

    fn count_tokens<'a>(
        &'a self,
        _model: &'a str,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<usize>> + Send + 'a>> {
        Box::pin(async move {
            // Heuristic on purpose: tokenizing must not trigger a 5GB load.
            Ok(text.len().div_ceil(4))
        })
    }

    fn health_check<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            // File-level probe, never loads the model.
            if !self.config.model_path.is_file() {
                return Err(RuntimeError::ModelNotFound(format!(
                    "GGUF not found at {} — hint: check {ENV_MODEL}",
                    self.config.model_path.display()
                )));
            }
            check_gguf_arch(&self.config.model_path)?;
            let file_name = self
                .config
                .model_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            if ki::needs_external_tokenizer(file_name) {
                ki::resolve_tokenizer_json(
                    &self.config.model_path,
                    self.config.tokenizer_sidecar_dir.as_deref(),
                )
                .map_err(Self::friendly)?;
            }
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// No-feature stub: honest compile-time-off error (never a silent fallback)
// ---------------------------------------------------------------------------

/// Selected backend is unavailable in this build. Always compiled so the CLI
/// can name the missing feature without cfg-conditional call sites.
#[cfg(not(feature = "kitty-infer"))]
pub fn unavailable_error() -> RuntimeError {
    RuntimeError::InvalidArgument(
        "KITTY_BACKEND=kitty-infer needs the `kitty-infer` cargo feature — \
         rebuild with --features kitty-infer (mono M3: engine/mistralrs interno)"
            .to_string(),
    )
}

#[cfg(test)]
mod mapping_tests {
    use super::map_role;

    #[test]
    fn gpu_layers_parse() {
        use super::parse_gpu_layers;
        assert_eq!(parse_gpu_layers(None), None);
        assert_eq!(parse_gpu_layers(Some("12")), Some(12));
        assert_eq!(parse_gpu_layers(Some(" 12 ")), Some(12));
        assert_eq!(parse_gpu_layers(Some("0")), None);
        assert_eq!(parse_gpu_layers(Some("nope")), None);
    }

    #[test]
    fn spill_validation_is_relative() {
        use super::validate_gpu_layers;
        assert!(validate_gpu_layers(24, Some(28)).is_ok());
        assert!(validate_gpu_layers(28, Some(28)).is_ok());
        assert!(validate_gpu_layers(37, Some(36)).is_err());
        // Unknown total: allowed (documented), byte pre-check stays fail-closed.
        assert!(validate_gpu_layers(99, None).is_ok());
    }

    #[test]
    fn spill_estimate_scales_and_fails_fast() {
        use super::spill_estimate_bytes;
        // f32 fraction widens exactly like the implementation does.
        let frac = 0.9f32.clamp(0.0, 1.0) as f64;
        let kv = 7372u64 * 256 * 1024;
        // K2 spill-20 shape: 10GB file, 20/36 layers.
        let est = spill_estimate_bytes(10_000_000_000, Some(20), Some(36), 0.9, 7372);
        assert_eq!(est, (10_000_000_000f64 * (20.0 / 36.0) * frac) as u64 + kv);
        assert!(est < 10_000_000_000);
        // No spill info: legacy full-file shape.
        let full = spill_estimate_bytes(10_000_000_000, None, None, 0.9, 7372);
        assert_eq!(full, (10_000_000_000f64 * frac) as u64 + kv);
        assert!(est < full);
    }

    #[test]
    fn gguf_block_count_reads_header() {
        use super::gguf_block_count;
        fn u32le(v: u32, out: &mut Vec<u8>) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        fn u64le(v: u64, out: &mut Vec<u8>) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        fn kv_str(out: &mut Vec<u8>, key: &str, val: &str) {
            u64le(key.len() as u64, out);
            out.extend_from_slice(key.as_bytes());
            u32le(8, out);
            u64le(val.len() as u64, out);
            out.extend_from_slice(val.as_bytes());
        }
        fn kv_u32(out: &mut Vec<u8>, key: &str, val: u32) {
            u64le(key.len() as u64, out);
            out.extend_from_slice(key.as_bytes());
            u32le(4, out);
            out.extend_from_slice(&val.to_le_bytes());
        }
        let mut blob = vec![];
        blob.extend_from_slice(b"GGUF");
        u32le(3, &mut blob);
        u64le(0, &mut blob); // tensor_count
        u64le(2, &mut blob); // kv_count
        kv_str(&mut blob, "general.architecture", "k2-horizon");
        kv_u32(&mut blob, "k2-horizon.block_count", 36);
        let dir = std::env::temp_dir();
        let path = dir.join("kitty-infer-test-header.gguf");
        std::fs::write(&path, &blob).unwrap();
        assert_eq!(gguf_block_count(&path), Some(36));
        // Truncated / wrong magic / wrong arch: None, never zero/panic.
        std::fs::write(&path, &blob[..10]).unwrap();
        assert_eq!(gguf_block_count(&path), None);
        std::fs::write(&path, b"NOPE").unwrap();
        assert_eq!(gguf_block_count(&path), None);
        let mut blob2 = vec![];
        blob2.extend_from_slice(b"GGUF");
        u32le(3, &mut blob2);
        u64le(0, &mut blob2);
        u64le(2, &mut blob2);
        kv_str(&mut blob2, "general.architecture", "llama");
        kv_u32(&mut blob2, "k2-horizon.block_count", 36);
        std::fs::write(&path, &blob2).unwrap();
        assert_eq!(gguf_block_count(&path), None);
        // Order-independent: block_count before the architecture key.
        let mut blob3 = vec![];
        blob3.extend_from_slice(b"GGUF");
        u32le(3, &mut blob3);
        u64le(0, &mut blob3);
        u64le(2, &mut blob3);
        kv_u32(&mut blob3, "qwen3.block_count", 36);
        kv_str(&mut blob3, "general.architecture", "qwen3");
        std::fs::write(&path, &blob3).unwrap();
        assert_eq!(gguf_block_count(&path), Some(36));
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn gguf_arch_early_check_accepts_rejects_and_fails_open() {
        use super::{check_gguf_arch, gguf_architecture, is_supported_gguf_arch};
        fn blob_with_arch(arch: &str) -> Vec<u8> {
            fn u32le(v: u32, out: &mut Vec<u8>) {
                out.extend_from_slice(&v.to_le_bytes());
            }
            fn u64le(v: u64, out: &mut Vec<u8>) {
                out.extend_from_slice(&v.to_le_bytes());
            }
            fn kv_str(out: &mut Vec<u8>, key: &str, val: &str) {
                u64le(key.len() as u64, out);
                out.extend_from_slice(key.as_bytes());
                u32le(8, out);
                u64le(val.len() as u64, out);
                out.extend_from_slice(val.as_bytes());
            }
            let mut blob = vec![];
            blob.extend_from_slice(b"GGUF");
            u32le(3, &mut blob);
            u64le(0, &mut blob);
            u64le(1, &mut blob);
            kv_str(&mut blob, "general.architecture", arch);
            blob
        }
        let dir = std::env::temp_dir();
        let path = dir.join("kitty-infer-test-arch-check.gguf");
        assert!(is_supported_gguf_arch("qwen3"));
        assert!(is_supported_gguf_arch("k2-horizon"));
        assert!(!is_supported_gguf_arch("falcon"));
        std::fs::write(&path, blob_with_arch("qwen3")).unwrap();
        assert_eq!(gguf_architecture(&path), Some("qwen3".to_string()));
        assert!(check_gguf_arch(&path).is_ok());
        std::fs::write(&path, blob_with_arch("falcon")).unwrap();
        assert_eq!(gguf_architecture(&path), Some("falcon".to_string()));
        assert!(check_gguf_arch(&path).is_err());
        std::fs::write(&path, b"NOPE").unwrap();
        assert_eq!(gguf_architecture(&path), None);
        assert!(check_gguf_arch(&path).is_ok());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn tool_fallback_ids_are_positional() {
        use super::tool_fallback_id;
        assert_eq!(tool_fallback_id(Some("call_9"), 0), "call_9");
        assert_eq!(tool_fallback_id(Some("call_9"), 3), "call_9");
        assert_eq!(tool_fallback_id(None, 0), "tool-0");
        assert_eq!(tool_fallback_id(None, 1), "tool-1");
        assert_eq!(tool_fallback_id(None, 7), "tool-7");
    }

    #[test]
    fn sampler_opts_mapping() {
        use super::sampler_opts_from;
        use crate::client::RequestOptions;
        let d = sampler_opts_from(None);
        assert_eq!(
            (d.max_len, d.temperature, d.top_p, d.top_k),
            (None, None, None, None)
        );
        let o = sampler_opts_from(Some(RequestOptions {
            temperature: Some(1.0),
            top_p: Some(0.95),
            top_k: Some(40),
            num_predict: Some(512),
            ..Default::default()
        }));
        assert_eq!(o.max_len, Some(512));
        assert_eq!(o.temperature, Some(1.0));
        assert!((o.top_p.unwrap() - 0.95).abs() < 1e-6); // f32→f64 widening
        assert_eq!(o.top_k, Some(40));
        // Out-of-range falls back to defaults; huge caps clamp.
        let o = sampler_opts_from(Some(RequestOptions {
            temperature: Some(99.0),
            top_p: Some(-1.0),
            top_k: Some(0),
            num_predict: Some(100_000),
            ..Default::default()
        }));
        assert_eq!((o.temperature, o.top_p, o.top_k), (None, None, None));
        assert_eq!(o.max_len, Some(8192));
    }

    #[test]
    fn sampler_env_override_wins() {
        use super::{sampler_env_override, SamplerOpts};
        let base = SamplerOpts {
            max_len: Some(512),
            temperature: Some(0.3),
            top_p: None,
            top_k: None,
        };
        let o = sampler_env_override(base, Some("1.0"), Some("0.95"), None);
        assert_eq!(o.max_len, Some(512));
        assert_eq!(o.temperature, Some(1.0));
        assert!((o.top_p.unwrap() - 0.95).abs() < 1e-9);
        assert_eq!(o.top_k, None);
        // Invalid values leave fields untouched.
        let o = sampler_env_override(base, Some("nope"), Some("7"), Some("0"));
        assert_eq!((o.temperature, o.top_p, o.top_k), (Some(0.3), None, None));
    }

    #[test]
    fn tool_format_validated() {
        use super::tool_format;
        assert_eq!(tool_format(None), None);
        assert_eq!(tool_format(Some("json")), Some("json".to_string()));
        assert_eq!(tool_format(Some(" xml ")), Some("xml".to_string()));
        assert_eq!(
            tool_format(Some("xml_typed")),
            Some("xml_typed".to_string())
        );
        assert_eq!(tool_format(Some("JSON")), None);
        assert_eq!(tool_format(Some("yaml")), None);
        assert_eq!(tool_format(Some("")), None);
    }

    #[test]
    fn thinking_toggle() {
        use super::thinking_enabled;
        assert!(thinking_enabled(None));
        assert!(thinking_enabled(Some("0")));
        assert!(thinking_enabled(Some("nope")));
        assert!(!thinking_enabled(Some("1")));
        assert!(!thinking_enabled(Some(" true ")));
        assert!(!thinking_enabled(Some("yes")));
    }

    #[test]
    fn grammar_mask_is_opt_in_and_needs_tools() {
        use super::{grammar_enabled, grammar_schema_for};
        assert!(grammar_enabled(Some("1")));
        assert!(grammar_enabled(Some(" true ")));
        assert!(grammar_enabled(Some("yes")));
        assert!(!grammar_enabled(None));
        assert!(!grammar_enabled(Some("0")));
        assert!(!grammar_enabled(Some("nope")));
        assert!(!grammar_enabled(Some("")));

        // Mask off → None even with tools; on but no/empty tools → None.
        let tools = vec![crate::tools::ToolDefinition::new(
            "Read",
            "Read a file",
            serde_json::json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
        )];
        assert!(grammar_schema_for(Some(&tools), false).is_none());
        assert!(grammar_schema_for(None, true).is_none());
        assert!(grammar_schema_for(Some(&vec![]), true).is_none());
        // On + tools → mask carrying the tool-name enum (plumbing anchor:
        // this Value is what `to_request` hands to `set_constraint`).
        let schema = grammar_schema_for(Some(&tools), true).expect("mask must build");
        let names: Vec<&str> = schema["properties"]["name"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["Read"]);
    }

    #[test]
    fn roles_map_with_tool_id_threaded() {
        // Non-tool roles map plainly; `tool` is intercepted before the
        // classifier (see `to_request`), so it falls through to `user` here
        // and must never be relied upon for id correlation.
        assert_eq!(map_role("system"), ("system", false));
        assert_eq!(map_role("assistant"), ("assistant", false));
        assert_eq!(map_role("user"), ("user", false));
        assert_eq!(map_role("tool"), ("user", false));
        assert_eq!(map_role("weird"), ("user", false));
        // The id itself travels on the message, with a documented fallback.
        use crate::client::Message;
        assert_eq!(
            Message::tool_result_with_id("out", Some("call_7".into())).tool_call_id,
            Some("call_7".to_string())
        );
        assert_eq!(Message::tool_result("out").tool_call_id, None);
    }

    #[test]
    fn backend_switch_is_exact() {
        use super::backend_selected;
        assert!(backend_selected(Some("kitty-infer")));
        assert!(backend_selected(Some("  kitty-infer  "))); // surrounding whitespace trimmed
        assert!(!backend_selected(Some("KITTY-INFER"))); // case-sensitive
        assert!(!backend_selected(Some("ollama")));
        assert!(!backend_selected(None));
    }

    /// T3.2 A/B k=3: post-hoc (`KITTY_INFER_GRAMMAR` unset) vs decode-time
    /// mask (`=1`). Needs a local GGUF (`KITTY_INFER_MODEL`, e.g.
    /// Qwen3-4B-Q4_K_M.gguf — no sidecar needed). Run single-threaded:
    /// `cargo test -p kitty-runtime --features kitty-infer bench_grammar_ab
    ///  -- --ignored --test-threads=1`. Skips honestly when no model is set.
    #[cfg(feature = "kitty-infer")]
    #[tokio::test]
    #[ignore = "needs local GGUF via KITTY_INFER_MODEL"]
    async fn bench_grammar_ab_k3() {
        use crate::client::{ChatClient, Message, RequestOptions};

        let Ok(_) = std::env::var(super::ENV_MODEL) else {
            eprintln!("SKIP bench_grammar_ab_k3: KITTY_INFER_MODEL not set");
            return;
        };
        let tools = vec![crate::tools::ToolDefinition::new(
            "Read",
            "Read a file",
            serde_json::json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
        )];
        let grammar = crate::grammar::GrammarConfig::from_tools(tools.clone());
        // str::repeat is on &str: build prompt without format! noise.
        let prompt =
            "Call the Read tool with path \"/tmp/x\". Reply with ONLY the tool call, no prose. "
                .repeat(1);
        let options = Some(RequestOptions {
            num_predict: Some(64),
            temperature: Some(0.0),
            ..Default::default()
        });
        let mut report = String::new();
        for (arm, grammar_env) in [("posthoc", "0"), ("mask", "1")] {
            // SAFETY: this test runs alone (--test-threads=1); no other
            // test reads KITTY_INFER_GRAMMAR concurrently.
            unsafe { std::env::set_var(super::ENV_GRAMMAR, grammar_env) };
            let client = super::KittyInferClient::from_env().expect("config from env");
            if let Err(e) = client.health_check().await {
                eprintln!("SKIP bench_grammar_ab_k3: health_check failed: {e}");
                unsafe { std::env::remove_var(super::ENV_GRAMMAR) };
                return;
            }
            let t0 = std::time::Instant::now();
            let mut valid = 0;
            let mut total = 0;
            for _ in 0..3 {
                let resp = client
                    .chat_with_options(
                        "bench",
                        vec![Message::user(prompt.clone())],
                        Some(tools.clone()),
                        options.clone(),
                    )
                    .await
                    .expect("generation must succeed");
                if let Some(calls) = resp.message.tool_calls {
                    for c in &calls {
                        total += 1;
                        let json = serde_json::json!({
                            "name": c.function.name,
                            "arguments": c.function.arguments
                        })
                        .to_string();
                        valid += grammar.validate(&json) as usize;
                    }
                }
            }
            let dt = t0.elapsed();
            report.push_str(&format!("arm={arm} valid={valid}/{total} wall={dt:?}\n"));
        }
        unsafe { std::env::remove_var(super::ENV_GRAMMAR) };
        eprintln!("BENCH-AB-k3:\n{report}");
        assert!(report.contains("arm=posthoc") && report.contains("arm=mask"));
    }

    #[test]
    fn config_rejects_non_gguf_and_missing() {
        use super::KittyInferConfig;
        assert!(KittyInferConfig::from_path("/tmp/x.safetensors".into(), None, 4096).is_err());
        assert!(KittyInferConfig::from_path(
            "/tmp/does-not-exist-kitty-infer.gguf".into(),
            None,
            4096
        )
        .is_err());
    }

    #[test]
    fn k2_horizon_path_plumbing() {
        // K2 resolves through the generic GGUF path (no per-arch switch):
        // extension + existence gate, sidecar dir + ctx preserved for the
        // loader (tokenizer/dtype policy lives in kitty-infer infer_policy).
        use super::KittyInferConfig;
        let dir = std::env::temp_dir();
        let fake = dir.join("kitty-infer-test-K2-Horizon-4B-BF16.gguf");
        std::fs::write(&fake, b"GGUF").unwrap();
        let cfg =
            KittyInferConfig::from_path(fake.clone(), Some(dir.join("tokenizers")), 2048).unwrap();
        assert_eq!(cfg.model_path, fake);
        assert_eq!(cfg.tokenizer_sidecar_dir, Some(dir.join("tokenizers")));
        assert_eq!(cfg.ctx_len, 2048);
        // Ctx floor holds for K2's huge native window alike.
        let cfg = KittyInferConfig::from_path(fake.clone(), None, 0).unwrap();
        assert_eq!(cfg.ctx_len, 512);
        std::fs::remove_file(&fake).unwrap();
    }
}

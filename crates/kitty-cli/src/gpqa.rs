//! GPQA Diamond offline-first scaffolding (F1 + G1 remediation).
//!
//! - Metadatos versionados: dataset id, revisión, hash SHA-256, licencia/atribución.
//! - Caché materializada fuera del repo (XDG/HOME), nunca dentro de `benches/` ni del workspace.
//! - Validación offline-first con pre-registro explícito: `bench gpqa` exige
//!   `--pre-registration <PATH>` que fija suite, n=198, dataset id, 40-hex revision,
//!   64-hex sha256 no placeholder, modelo, límites, grading, compatibilidad,
//!   precondiciones y artefactos. Si falta/inválido ⇒ `blocked` sin invocar el modelo.
//! - Preflight verifica que el pre-registro coincide exactamente con el sidecar y
//!   que el SHA-256 del fichero coincide con el SHA esperado del pre-registro.
//!   Rechaza `main`, ceros, formato inválido y cualquier discrepancia.
//! - No descarga, no registra prompts, no usa `HF_TOKEN`.
//! - Loader solo desde materializado validado; 4 alternativas barajadas determinista.
//! - Grading por letra A-D normalizada; resultado etiquetado como aproximación local no oficial.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

// ── Metadatos versionados (fuente única, placeholders bloqueantes) ────

/// Identificador canónico del dataset en Hugging Face.
pub const GPQA_DATASET_ID: &str = "Idavidrein/gpqa";
/// Config / split de GPQA Diamond (198 preguntas).
pub const GPQA_CONFIG: &str = "gpqa_diamond";
/// Revisión fijada (commit 40-hex). Placeholder `main` es intencionalmente
/// bloqueante: G1 exige 40 hex inmutable; `main` siempre se rechaza hasta
/// que el usuario aporte un pre-registro con commit real.
#[allow(dead_code)]
pub const GPQA_REVISION: &str = "main";
/// SHA-256 esperado del artefacto materializado (hex, 64 chars, lower).
/// Placeholder de ceros es bloqueante: G1 exige 64 hex no placeholder.
#[allow(dead_code)]
pub const GPQA_EXPECTED_SHA256: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";
/// Licencia y condición anti-fuga (resumen, no sustituye el texto oficial).
#[allow(dead_code)]
pub const GPQA_LICENSE: &str = "Gated dataset — requires HF acceptance; see https://huggingface.co/datasets/Idavidrein/gpqa (Apache-2.0 derived, check upstream)";
/// Atribución requerida.
#[allow(dead_code)]
pub const GPQA_ATTRIBUTION: &str =
    "Rein et al., GPQA: A Graduate-Level Google-Proof Q&A Benchmark (2023)";
/// Versión del wrapper local (no del dataset upstream).
#[allow(dead_code)]
pub const GPQA_WRAPPER_VERSION: &str = "0.1.0-gpqa-local-v1";
/// Número de preguntas en GPQA Diamond.
pub const GPQA_NUM_QUESTIONS: usize = 198;
/// Método de grading declarado en el resultado.
pub const GPQA_GRADING_METHOD: &str = "letter_A-D_permuted";
/// Nivel de compatibilidad — etiqueta honesta.
pub const GPQA_COMPATIBILITY: &str = "local_approximation_not_official";
/// Descripción de la permutación determinista (question_id + revision validada).
#[allow(dead_code)]
pub const GPQA_PERMUTATION: &str = "deterministic_shuffle_seed_fnv1a_question_id_and_revision";
/// Plantilla declarada en el resultado.
#[allow(dead_code)]
pub const GPQA_TEMPLATE: &str =
    "gpqa_template_v1: Question + 4 shuffled options labeled A-D, answer single letter A-D";

// ── Estructura de metadatos (sidecar junto a la caché) ───────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpqaMetadata {
    /// Dataset id canónico (p.ej. "Idavidrein/gpqa").
    pub dataset_id: String,
    /// Revisión fijada (commit 40-hex).
    pub revision: String,
    /// Hash SHA-256 hex del artefacto materializado (64 hex lower).
    pub sha256: String,
    /// Licencia resumida.
    pub license: String,
    /// Atribución.
    pub attribution: String,
    /// Versión del wrapper local.
    pub version: String,
    /// Método de grading.
    pub grading_method: String,
    /// Nivel de compatibilidad.
    pub compatibility: String,
    /// Plantilla usada.
    pub template: String,
    /// Permutación usada.
    pub permutation: String,
}

impl GpqaMetadata {
    /// Metadatos esperados (placeholders bloqueantes).
    #[allow(dead_code)]
    pub fn expected() -> Self {
        Self {
            dataset_id: GPQA_DATASET_ID.to_string(),
            revision: GPQA_REVISION.to_string(),
            sha256: GPQA_EXPECTED_SHA256.to_string(),
            license: GPQA_LICENSE.to_string(),
            attribution: GPQA_ATTRIBUTION.to_string(),
            version: GPQA_WRAPPER_VERSION.to_string(),
            grading_method: GPQA_GRADING_METHOD.to_string(),
            compatibility: GPQA_COMPATIBILITY.to_string(),
            template: GPQA_TEMPLATE.to_string(),
            permutation: GPQA_PERMUTATION.to_string(),
        }
    }
}

// ── Pre-registro explícito (creado antes de medir) ────────────────────

/// Límites/configuración fijados en el pre-registro.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpqaLimits {
    pub limit: usize,
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    #[serde(default)]
    pub max_iterations: Option<usize>,
    #[serde(default)]
    pub thinking_budget: Option<usize>,
    #[serde(default)]
    pub context_window: Option<usize>,
}

/// Manifiesto de pre-registro para GPQA Diamond.
/// Sin prompts, fija todo lo requerido por G1 antes de medir.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpqaPreRegistration {
    /// Suite, debe ser `gpqa` o `gpqa_diamond`.
    pub suite: String,
    /// Número de preguntas, debe ser 198.
    pub n: usize,
    pub dataset_id: String,
    /// Revisión 40 hex inmutable.
    pub revision: String,
    /// SHA-256 64 hex no placeholder.
    pub sha256: String,
    /// Modelo usado para la medición.
    pub model: String,
    /// Límites / configuración.
    pub limits: GpqaLimits,
    pub grading_method: String,
    pub compatibility: String,
    /// Precondiciones (objeto JSON, al menos un campo).
    pub preconditions: serde_json::Value,
    /// Artefactos esperados (al menos uno).
    pub expected_artifacts: Vec<String>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub attribution: Option<String>,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub permutation: Option<String>,
}

// ── Errores ──────────────────────────────────────────────────────────

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum GpqaError {
    /// Caché no encontrada.
    CacheMissing { path: PathBuf, help: String },
    /// Manifest sidecar no encontrado.
    ManifestMissing { path: PathBuf, help: String },
    /// Pre-registro no encontrado o no proporcionado.
    PreRegistrationMissing { path: Option<PathBuf>, help: String },
    /// Revisión no coincide o formato inválido.
    RevisionMismatch { expected: String, found: String },
    InvalidRevisionFormat(String),
    /// Hash no coincide o formato inválido.
    HashMismatch { expected: String, got: String, path: PathBuf },
    InvalidHashFormat(String),
    PlaceholderHash(String),
    PlaceholderRevision(String),
    /// JSON de metadata mal formado.
    InvalidMetadata(String),
    /// Pre-registro inválido.
    InvalidPreRegistration(String),
    /// Caché dentro del workspace.
    CacheInsideWorkspace { cache: PathBuf, workspace: PathBuf },
    /// Conteo de preguntas no es 198.
    CountMismatch { expected: usize, got: usize },
    /// Otro error de IO / parsing.
    Io(String),
    /// Bloqueo genérico pre-registro.
    Blocked(String),
}

impl std::fmt::Display for GpqaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CacheMissing { path, help } => {
                write!(f, "GPQA Diamond cache not found at {}\n{}", path.display(), help)
            }
            Self::ManifestMissing { path, help } => {
                write!(f, "GPQA manifest not found at {}\n{}", path.display(), help)
            }
            Self::PreRegistrationMissing { path, help } => {
                if let Some(p) = path {
                    write!(f, "GPQA pre-registration manifest not found at {}\n{}", p.display(), help)
                } else {
                    write!(f, "GPQA pre-registration required but not provided.\n{}", help)
                }
            }
            Self::RevisionMismatch { expected, found } => {
                write!(f, "GPQA revision mismatch: expected {expected}, found {found}. Re-materialize the pinned revision.")
            }
            Self::InvalidRevisionFormat(s) => write!(f, "Invalid GPQA revision format (expected 40 hex chars, got '{s}'): revision must be full 40-char commit hash, not 'main'"),
            Self::HashMismatch { expected, got, path } => {
                write!(f, "GPQA integrity check failed for {}: expected sha256 {expected}, got {got}. File may be corrupt or not the pinned revision.", path.display())
            }
            Self::InvalidHashFormat(s) => write!(f, "Invalid SHA-256 format (expected 64 hex chars): {s}"),
            Self::PlaceholderHash(s) => write!(f, "Placeholder SHA-256 rejected (all zeros or invalid): {s}. Provide real pinned sha256 in pre-registration"),
            Self::PlaceholderRevision(s) => write!(f, "Placeholder revision rejected: '{s}'. Provide full 40-hex commit hash in pre-registration"),
            Self::InvalidMetadata(s) => write!(f, "Invalid GPQA metadata JSON: {s}"),
            Self::InvalidPreRegistration(s) => write!(f, "Invalid GPQA pre-registration manifest: {s}"),
            Self::CacheInsideWorkspace { cache, workspace } => {
                write!(f, "GPQA cache path {} is inside workspace {} — cache must be outside repo (e.g. $HOME/.cache/kitty/gpqa/). Use KITTY_GPQA_CACHE outside workspace.", cache.display(), workspace.display())
            }
            Self::CountMismatch { expected, got } => {
                write!(f, "GPQA Diamond must have exactly {expected} questions, found {got}")
            }
            Self::Io(s) => write!(f, "{s}"),
            Self::Blocked(s) => write!(f, "GPQA blocked: {s}"),
        }
    }
}

impl std::error::Error for GpqaError {}

// ── Helpers de validación ──────────────────────────────────────────

fn is_all_zeros(s: &str) -> bool {
    s.chars().all(|c| c == '0')
}

fn is_hex(s: &str) -> bool {
    s.chars().all(|c| c.is_ascii_hexdigit())
}

/// Valida revisión 40 hex, rechaza placeholders como `main`.
pub fn validate_revision_format(rev: &str) -> Result<(), GpqaError> {
    let t = rev.trim();
    if t == "main" || t == "master" || t == "HEAD" || t.is_empty() {
        return Err(GpqaError::PlaceholderRevision(t.to_string()));
    }
    if t.len() != 40 {
        return Err(GpqaError::InvalidRevisionFormat(t.to_string()));
    }
    if !is_hex(t) {
        return Err(GpqaError::InvalidRevisionFormat(t.to_string()));
    }
    if is_all_zeros(t) {
        return Err(GpqaError::PlaceholderRevision(t.to_string()));
    }
    Ok(())
}

/// Valida hash 64 hex, rechaza placeholder all-zero.
pub fn validate_hash_format(hash: &str) -> Result<(), GpqaError> {
    let h = hash.trim();
    if h.len() != 64 {
        return Err(GpqaError::InvalidHashFormat(hash.to_string()));
    }
    if !is_hex(h) {
        return Err(GpqaError::InvalidHashFormat(hash.to_string()));
    }
    if is_all_zeros(h) {
        return Err(GpqaError::PlaceholderHash(hash.to_string()));
    }
    Ok(())
}

// ── Paths fuera del repo ───────────────────────────────────────────

/// Directorio base del cache fuera del repo.
/// Orden: `KITTY_GPQA_CACHE` (fichero directo) → `XDG_CACHE_HOME` → `HOME/.cache`.
pub fn gpqa_cache_path() -> PathBuf {
    if let Ok(p) = std::env::var("KITTY_GPQA_CACHE") {
        let trimmed = p.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        let trimmed = xdg.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed).join("kitty/gpqa/diamond.jsonl");
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache/kitty/gpqa/diamond.jsonl")
}

/// Manifest sidecar al cache (mismo dir, distinto nombre).
pub fn gpqa_manifest_path() -> PathBuf {
    let cache = gpqa_cache_path();
    cache.with_file_name("gpqa_manifest.json")
}

/// Resuelve el workspace root para validación de caché fuera del repo.
/// Usa `resolve` heurístico: intenta `KITTY_WORKSPACE_ROOT` env, luego
/// `CARGO_MANIFEST_DIR` parent (kitty-cli -> workspace), luego current_dir.
fn resolve_workspace_root() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("KITTY_WORKSPACE_ROOT") {
        let trimmed = p.trim();
        if !trimmed.is_empty() {
            return Some(PathBuf::from(trimmed));
        }
    }
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let p = PathBuf::from(manifest_dir);
        // crates/kitty-cli -> workspace root = parent of crates
        if let Some(parent) = p.parent() {
            if parent.join("Cargo.toml").exists() || parent.join(".git").exists() {
                return Some(parent.to_path_buf());
            }
            if let Some(gp) = parent.parent() {
                if gp.join("Cargo.toml").exists() || gp.join(".git").exists() {
                    return Some(gp.to_path_buf());
                }
            }
        }
        return Some(p);
    }
    std::env::current_dir().ok()
}

/// Comprueba si `cache` está dentro de `workspace` (lexical, sin canonicalize).
/// Determinista y testeable: normaliza ambas rutas y comprueba prefijo.
pub fn is_cache_inside_workspace(cache: &Path, workspace: &Path) -> bool {
    let cache_norm = normalize_path(cache);
    let ws_norm = normalize_path(workspace);
    cache_norm.starts_with(&ws_norm)
}

fn normalize_path(p: &Path) -> PathBuf {
    // Elimina `.`, `..` léxicamente sin tocar FS (determinista).
    let mut out = PathBuf::new();
    for comp in p.components() {
        use std::path::Component;
        match comp {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Asegura que la caché está fuera del workspace y no es relativa; si está dentro o es relativa retorna error.
/// Determinista y testeable. Rechaza `KITTY_GPQA_CACHE` relativo para impedir bypass de workspace.
pub fn ensure_cache_outside_workspace(cache: &Path) -> Result<(), GpqaError> {
    // Rechaza rutas relativas: un `KITTY_GPQA_CACHE=crates/cache.json` podría resolverse dentro del workspace
    // según el cwd y eludir la comprobación de prefijo; por eso se exige absoluta.
    if !cache.is_absolute() {
        let ws = resolve_workspace_root().unwrap_or_else(|| PathBuf::from("<workspace>"));
        return Err(GpqaError::CacheInsideWorkspace {
            cache: cache.to_path_buf(),
            workspace: ws,
        });
    }
    if let Some(ws) = resolve_workspace_root() {
        if is_cache_inside_workspace(cache, &ws) {
            return Err(GpqaError::CacheInsideWorkspace {
                cache: cache.to_path_buf(),
                workspace: ws,
            });
        }
    }
    Ok(())
}

/// Mensaje de ayuda para preparar el dataset (sin prompts).
fn prepare_help(cache: &Path, manifest: &Path) -> String {
    format!(
        "Preflight bloqueado (offline-first, gated).\n\
         GPQA Diamond es un dataset gated que requiere aceptar términos en https://huggingface.co/datasets/{dataset} .\n\
         Este harness NUNCA descarga automáticamente ni usa HF_TOKEN; tampoco registra prompts.\n\
         Para materializar:\n\
           1) Acepta términos en HF y obtén acceso.\n\
           2) Descarga el artefacto de la revisión 40-hex pinneada (config `{config}`) con la herramienta oficial de HF.\n\
           3) Coloca el fichero en `{cache}` (fuera del repo, no se commitea).\n\
           4) Coloca el sidecar manifest en `{manifest}` con dataset_id, 40-hex revision, 64-hex sha256, license, attribution.\n\
           5) Crea el pre-registro con `kitty bench --suite gpqa --pre-registration <PATH>` que fije suite=n=198,dataset,revision,sha256,model,límites,grading,compatibilidad,precondiciones, artefactos.\n\
           6) El hash debe ser SHA-256 del fichero; calcúlalo con `sha256sum {cache}` y ponlo en ambos manifests. La ejecución validará pre-registro==sidecar==fichero sin red.\n\
         Resultado GPQA se etiqueta siempre como \"aproximación local no oficial\" (no leaderboard). Caché permanece fuera del repo."
        , dataset = GPQA_DATASET_ID, config = GPQA_CONFIG, cache = cache.display(), manifest = manifest.display(),
    )
}

pub fn pre_registration_help_text() -> String {
    pre_registration_help()
}

fn pre_registration_help() -> String {
    format!(
        "Pre-registro requerido para GPQA: bench gpqa exige --pre-registration <PATH>\n\
         El fichero JSON debe fijar sin prompts al menos:\n\
           suite=\"gpqa_diamond\", n=198, dataset_id=\"{}\", revision=\"<40-hex>\", sha256=\"<64-hex no zeros>\",\n\
           model=\"<model>\", limits={{limit, timeout_secs, max_iterations, ...}}, grading_method=\"{}\",\n\
           compatibility=\"{}\", preconditions={{...}}, expected_artifacts=[...]\n\
         Revisión debe ser commit completo 40 hex (no 'main'), sha256 64 hex no placeholder all-zero.\n\
         Si falta/inválido ⇒ bloqueado y nunca se invoca el modelo.",
        GPQA_DATASET_ID, GPQA_GRADING_METHOD, GPQA_COMPATIBILITY
    )
}

// ── Parsing de metadata ────────────────────────────────────────────

/// Parsea JSON de metadata (sidecar) a `GpqaMetadata`.
pub fn parse_metadata(json_str: &str) -> Result<GpqaMetadata, GpqaError> {
    serde_json::from_str::<GpqaMetadata>(json_str)
        .map_err(|e| GpqaError::InvalidMetadata(e.to_string()))
}

/// Parsea JSON de pre-registro a `GpqaPreRegistration`.
pub fn parse_pre_registration(json_str: &str) -> Result<GpqaPreRegistration, GpqaError> {
    serde_json::from_str::<GpqaPreRegistration>(json_str)
        .map_err(|e| GpqaError::InvalidPreRegistration(e.to_string()))
}

/// Valida un pre-registro (sin IO, solo campos).
pub fn validate_pre_registration(pre: &GpqaPreRegistration) -> Result<(), GpqaError> {
    let suite_ok = pre.suite == "gpqa" || pre.suite == "gpqa_diamond" || pre.suite == "gpqa_diamond_local";
    if !suite_ok {
        return Err(GpqaError::InvalidPreRegistration(format!(
            "suite must be gpqa or gpqa_diamond, got '{}'",
            pre.suite
        )));
    }
    if pre.n != GPQA_NUM_QUESTIONS {
        return Err(GpqaError::InvalidPreRegistration(format!(
            "n must be {}, got {}",
            GPQA_NUM_QUESTIONS, pre.n
        )));
    }
    if pre.dataset_id != GPQA_DATASET_ID {
        return Err(GpqaError::InvalidPreRegistration(format!(
            "dataset_id must be {}, got {}",
            GPQA_DATASET_ID, pre.dataset_id
        )));
    }
    validate_revision_format(&pre.revision)?;
    validate_hash_format(&pre.sha256)?;
    if pre.model.trim().is_empty() {
        return Err(GpqaError::InvalidPreRegistration("model must not be empty".to_string()));
    }
    if pre.limits.limit != GPQA_NUM_QUESTIONS {
        return Err(GpqaError::InvalidPreRegistration(format!(
            "limits.limit must be {}, got {}",
            GPQA_NUM_QUESTIONS, pre.limits.limit
        )));
    }
    if pre.grading_method != GPQA_GRADING_METHOD {
        return Err(GpqaError::InvalidPreRegistration(format!(
            "grading_method must be {}, got {}",
            GPQA_GRADING_METHOD, pre.grading_method
        )));
    }
    if pre.compatibility != GPQA_COMPATIBILITY {
        return Err(GpqaError::InvalidPreRegistration(format!(
            "compatibility must be {}, got {}",
            GPQA_COMPATIBILITY, pre.compatibility
        )));
    }
    if !pre.preconditions.is_object() || pre.preconditions.as_object().map(|o| o.is_empty()).unwrap_or(true) {
        return Err(GpqaError::InvalidPreRegistration("preconditions must be non-empty object".to_string()));
    }
    if pre.expected_artifacts.is_empty() {
        return Err(GpqaError::InvalidPreRegistration("expected_artifacts must be non-empty".to_string()));
    }
    Ok(())
}

// ── SHA-256 puro (sin deps externas) ───────────────────────────────

/// Implementación SHA-256 mínima (FIPS 180-4), sin dependencias externas.
/// Retorna 32 bytes.
fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    const H0: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    #[inline]
    fn rotr(x: u32, n: u32) -> u32 {
        (x >> n) | (x << (32 - n))
    }

    // Padding
    let mut padded = data.to_vec();
    let bit_len = (data.len() as u64) * 8;
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0x00);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    let mut h = H0;

    for chunk in padded.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[i * 4], chunk[i * 4 + 1], chunk[i * 4 + 2], chunk[i * 4 + 3]]);
        }
        for i in 16..64 {
            let s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            let s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];

        for i in 0..64 {
            let s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for (i, v) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

/// Hex lower de SHA-256 de bytes.
pub fn sha256_hex(data: &[u8]) -> String {
    let bytes = sha256_bytes(data);
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Calcula SHA-256 hex de un fichero.
/// Lee el fichero completo en memoria (GPQA diamond <10 MB); no es streaming incremental.
pub fn sha256_file(path: &Path) -> Result<String, GpqaError> {
    let data = std::fs::read(path).map_err(|e| GpqaError::Io(format!("failed to read {}: {e}", path.display())))?;
    Ok(sha256_hex(&data))
}

// ── Validación de integridad ───────────────────────────────────────

/// Verifica que la caché materializada coincide con `expected`.
/// Comprueba existencia, formato de hash y que el hash computado coincide.
#[allow(dead_code)]
pub fn verify_integrity(expected: &GpqaMetadata, cache_path: &Path) -> Result<(), GpqaError> {
    if !cache_path.exists() {
        let manifest = cache_path.with_file_name("gpqa_manifest.json");
        return Err(GpqaError::CacheMissing {
            path: cache_path.to_path_buf(),
            help: prepare_help(cache_path, &manifest),
        });
    }
    validate_hash_format(&expected.sha256)?;
    validate_revision_format(&expected.revision)?;
    let got = sha256_file(cache_path)?;
    if !got.eq_ignore_ascii_case(expected.sha256.trim()) {
        return Err(GpqaError::HashMismatch {
            expected: expected.sha256.clone(),
            got,
            path: cache_path.to_path_buf(),
        });
    }
    Ok(())
}

/// Verifica integridad usando el hash declarado en el manifest sidecar.
pub fn verify_integrity_with_manifest(manifest: &GpqaMetadata, cache_path: &Path) -> Result<(), GpqaError> {
    validate_hash_format(&manifest.sha256)?;
    validate_revision_format(&manifest.revision)?;
    let got = sha256_file(cache_path)?;
    if !got.eq_ignore_ascii_case(manifest.sha256.trim()) {
        return Err(GpqaError::HashMismatch {
            expected: manifest.sha256.clone(),
            got,
            path: cache_path.to_path_buf(),
        });
    }
    if manifest.dataset_id != GPQA_DATASET_ID {
        return Err(GpqaError::InvalidMetadata(format!(
            "dataset_id mismatch: expected {}, found {}",
            GPQA_DATASET_ID, manifest.dataset_id
        )));
    }
    Ok(())
}

// ── Preflight offline-first con pre-registro explícito ─────────────

/// Preflight que exige pre-registro explícito y valida sidecar + fichero.
/// Nunca invoca el modelo si falta/inválido ⇒ `blocked`.
pub fn preflight_gpqa_with_prereg(pre_reg_path: &Path) -> Result<(GpqaPreRegistration, GpqaMetadata), GpqaError> {
    if !pre_reg_path.exists() {
        return Err(GpqaError::PreRegistrationMissing {
            path: Some(pre_reg_path.to_path_buf()),
            help: pre_registration_help(),
        });
    }
    let pre_str = std::fs::read_to_string(pre_reg_path)
        .map_err(|e| GpqaError::Io(format!("failed to read pre-registration {}: {e}", pre_reg_path.display())))?;
    let pre = parse_pre_registration(&pre_str)?;
    validate_pre_registration(&pre)?;

    let cache = gpqa_cache_path();
    let manifest_path = gpqa_manifest_path();

    ensure_cache_outside_workspace(&cache)?;

    if !manifest_path.exists() {
        return Err(GpqaError::ManifestMissing {
            path: manifest_path.clone(),
            help: prepare_help(&cache, &manifest_path),
        });
    }
    if !cache.exists() {
        return Err(GpqaError::CacheMissing {
            path: cache.clone(),
            help: prepare_help(&cache, &manifest_path),
        });
    }

    let manifest_str = std::fs::read_to_string(&manifest_path)
        .map_err(|e| GpqaError::Io(format!("failed to read manifest {}: {e}", manifest_path.display())))?;
    let manifest = parse_metadata(&manifest_str)?;

    // Validar formato de ambos
    validate_revision_format(&manifest.revision)?;
    validate_hash_format(&manifest.sha256)?;

    // Coincidencia exacta pre-registro == sidecar para los 3 campos críticos
    if manifest.dataset_id != pre.dataset_id {
        return Err(GpqaError::InvalidPreRegistration(format!(
            "pre-registration dataset_id {} != sidecar {}",
            pre.dataset_id, manifest.dataset_id
        )));
    }
    if manifest.revision.to_ascii_lowercase() != pre.revision.to_ascii_lowercase() {
        return Err(GpqaError::RevisionMismatch {
            expected: pre.revision.clone(),
            found: manifest.revision.clone(),
        });
    }
    if manifest.sha256.to_ascii_lowercase() != pre.sha256.to_ascii_lowercase() {
        return Err(GpqaError::HashMismatch {
            expected: pre.sha256.clone(),
            got: manifest.sha256.clone(),
            path: manifest_path.clone(),
        });
    }

    // SHA del fichero debe coincidir con el del pre-registro (autoridad).
    let got = sha256_file(&cache)?;
    if !got.eq_ignore_ascii_case(pre.sha256.trim()) {
        return Err(GpqaError::HashMismatch {
            expected: pre.sha256.clone(),
            got,
            path: cache.clone(),
        });
    }
    // También con sidecar (redundante pero explícito).
    verify_integrity_with_manifest(&manifest, &cache)?;

    // Validar licencia/atribución no vacías.
    if manifest.license.trim().is_empty() {
        return Err(GpqaError::InvalidMetadata("license must not be empty".to_string()));
    }
    if manifest.attribution.trim().is_empty() {
        return Err(GpqaError::InvalidMetadata("attribution must not be empty".to_string()));
    }

    Ok((pre, manifest))
}

/// Preflight legacy sin pre-registro: siempre bloqueado, explica requisito.
/// Conservado para compatibilidad pero siempre retorna `PreRegistrationMissing`.
#[allow(dead_code)]
pub fn preflight_gpqa() -> Result<GpqaMetadata, GpqaError> {
    Err(GpqaError::PreRegistrationMissing {
        path: None,
        help: pre_registration_help(),
    })
}

// ── Loader (solo materializado validado) ───────────────────────────

/// Representación mínima de una pregunta GPQA Diamond sin exponer prompts en logs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpqaQuestion {
    /// ID estable de la pregunta.
    pub id: String,
    /// Enunciado (no loguear).
    pub question: String,
    /// Respuesta correcta textual (no loguear en claro fuera del grading).
    pub correct_answer: String,
    /// Tres distractores.
    pub distractors: Vec<String>,
    #[serde(default)]
    pub explanation: Option<String>,
}

/// Carga preguntas desde el fichero materializado validado.
/// Retorna error si no hay exactamente 198 preguntas o si algún registro es malformado.
/// Nunca hace panic por datos malformados.
pub fn load_questions(cache_path: &Path) -> Result<Vec<GpqaQuestion>, GpqaError> {
    let data = std::fs::read_to_string(cache_path)
        .map_err(|e| GpqaError::Io(format!("failed to read GPQA cache {}: {e}", cache_path.display())))?;
    let trimmed_all = data.trim();
    let mut out: Vec<GpqaQuestion> = Vec::new();

    // Soporta JSON array completo o JSONL (una por línea).
    if trimmed_all.starts_with('[') {
        let arr: Vec<GpqaQuestion> = serde_json::from_str(trimmed_all)
            .map_err(|e| GpqaError::InvalidMetadata(format!("failed to parse GPQA JSON array: {e}")))?;
        for q in &arr {
            if q.distractors.len() != 3 {
                return Err(GpqaError::InvalidMetadata(format!(
                    "question {} must have exactly 3 distractors, found {}",
                    q.id,
                    q.distractors.len()
                )));
            }
        }
        out = arr;
    } else {
        for (idx, line) in data.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let q: GpqaQuestion = serde_json::from_str(trimmed)
                .map_err(|e| GpqaError::InvalidMetadata(format!("failed to parse GPQA line {}: {e}", idx + 1)))?;
            if q.distractors.len() != 3 {
                return Err(GpqaError::InvalidMetadata(format!(
                    "question {} must have exactly 3 distractors, found {}",
                    q.id,
                    q.distractors.len()
                )));
            }
            out.push(q);
        }
    }

    if out.len() != GPQA_NUM_QUESTIONS {
        return Err(GpqaError::CountMismatch {
            expected: GPQA_NUM_QUESTIONS,
            got: out.len(),
        });
    }
    Ok(out)
}

// ── Permutación determinista de 4 opciones ─────────────────────────

fn fnv1a64(question_id: &str, revision: &str) -> u64 {
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;
    let mut hash = FNV_OFFSET;
    for b in question_id.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    // Mezcla la revisión real validada del pre-registro (no el placeholder)
    for b in revision.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Baraja determinísticamente 4 opciones y retorna (permutadas, índice_correcto).
/// La semilla mezcla `question_id` y la `revision` real ya validada (no placeholder).
/// Retorna error en lugar de panic si distractores no son 3.
pub fn permute_four(
    question_id: &str,
    revision: &str,
    correct: String,
    distractors: Vec<String>,
) -> Result<(Vec<String>, usize), GpqaError> {
    if distractors.len() != 3 {
        return Err(GpqaError::InvalidPreRegistration(format!(
            "permute_four requires exactly 3 distractors, got {}",
            distractors.len()
        )));
    }
    let mut items: Vec<(String, bool)> = Vec::with_capacity(4);
    items.push((correct, true));
    for d in distractors {
        items.push((d, false));
    }

    let mut seed = fnv1a64(question_id, revision);
    let mut next_u64 = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        seed
    };

    for i in (1..items.len()).rev() {
        let j = (next_u64() as usize) % (i + 1);
        items.swap(i, j);
    }

    let mut permuted = Vec::with_capacity(4);
    let mut correct_idx = 0;
    for (idx, (opt, is_correct)) in items.into_iter().enumerate() {
        if is_correct {
            correct_idx = idx;
        }
        permuted.push(opt);
    }
    Ok((permuted, correct_idx))
}

/// Convierte índice 0..3 a letra A-D. Retorna error en lugar de panic.
pub fn idx_to_letter(idx: usize) -> Result<char, GpqaError> {
    match idx {
        0 => Ok('A'),
        1 => Ok('B'),
        2 => Ok('C'),
        3 => Ok('D'),
        _ => Err(GpqaError::InvalidPreRegistration(format!("invalid idx for GPQA options: {idx}"))),
    }
}

// ── Normalización y grading A-D ───────────────────────────────────

pub fn normalize_answer(raw: &str) -> Option<char> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let upper = trimmed.to_ascii_uppercase();
    let stripped: String = upper.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    if stripped.len() == 1 {
        let c = stripped.chars().next().unwrap();
        if ('A'..='D').contains(&c) {
            return Some(c);
        }
    }
    let chars: Vec<char> = upper.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if !('A'..='D').contains(&c) {
            continue;
        }
        let prev_ok = i == 0 || !chars[i - 1].is_ascii_alphabetic();
        let next_ok = i + 1 == chars.len() || !chars[i + 1].is_ascii_alphabetic();
        if prev_ok && next_ok {
            return Some(c);
        }
    }
    for &c in &chars {
        if ('A'..='D').contains(&c) {
            return Some(c);
        }
    }
    None
}

pub fn grade_answer(normalized: Option<char>, correct_letter: char) -> bool {
    match normalized {
        Some(l) => l == correct_letter,
        None => false,
    }
}

// ── Formato de prompt ──────────────────────────────────────────────

/// Construye el prompt para una pregunta con opciones ya permutadas.
/// Retorna error si no hay exactamente 4 opciones, nunca panic.
pub fn format_prompt(question: &str, options: &[String]) -> Result<String, GpqaError> {
    if options.len() != 4 {
        return Err(GpqaError::InvalidPreRegistration(format!(
            "format_prompt requires exactly 4 options, got {}",
            options.len()
        )));
    }
    let mut out = String::new();
    out.push_str(question.trim());
    out.push_str("\n\nOptions:\n");
    for (idx, opt) in options.iter().enumerate() {
        let letter = idx_to_letter(idx)?;
        out.push_str(&format!("{} ) {}\n", letter, opt.trim()));
    }
    out.push_str("\nRespond with a single letter A, B, C, or D.");
    Ok(out)
}

// ── Resultado con proveniencia ─────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpqaResult {
    pub suite: String,
    pub version: String,
    pub dataset_id: String,
    pub revision: String,
    pub sha256: String,
    pub license: String,
    pub attribution: String,
    pub grading_method: String,
    pub compatibility: String,
    pub permutation: String,
    pub template: String,
    pub label: String,
    pub preconditions: serde_json::Value,
    pub total: usize,
    pub correct: usize,
    pub accuracy: f64,
    pub note: String,
    /// Procedencia íntegra del pre-registro (G1.5).
    pub pre_registration_path: String,
    pub pre_registration_sha256: String,
    pub limits: GpqaLimits,
    pub expected_artifacts: Vec<String>,
}

impl GpqaResult {
    #[cfg(test)]
    pub fn new(manifest: &GpqaMetadata, total: usize, correct: usize) -> Self {
        // Constructor legacy para tests: pre-registro vacío pero etiqueta preservada.
        let preconditions = serde_json::json!({
            "cache_path": gpqa_cache_path().display().to_string(),
            "validated": true,
            "offline": true,
            "requires": "HF acceptance + manual materialization + pre-registration (gated)"
        });
        Self::new_with_prereg(
            manifest,
            &GpqaPreRegistration {
                suite: "gpqa_diamond".to_string(),
                n: total,
                dataset_id: manifest.dataset_id.clone(),
                revision: manifest.revision.clone(),
                sha256: manifest.sha256.clone(),
                model: String::new(),
                limits: GpqaLimits {
                    limit: total,
                    timeout_secs: None,
                    max_iterations: None,
                    thinking_budget: None,
                    context_window: None,
                },
                grading_method: manifest.grading_method.clone(),
                compatibility: manifest.compatibility.clone(),
                preconditions: preconditions.clone(),
                expected_artifacts: vec![],
                version: Some(manifest.version.clone()),
                license: Some(manifest.license.clone()),
                attribution: Some(manifest.attribution.clone()),
                template: Some(manifest.template.clone()),
                permutation: Some(manifest.permutation.clone()),
            },
            Path::new(""),
            "",
            total,
            correct,
        )
    }

    pub fn new_with_prereg(
        manifest: &GpqaMetadata,
        pre: &GpqaPreRegistration,
        pre_path: &Path,
        pre_sha256: &str,
        total: usize,
        correct: usize,
    ) -> Self {
        let accuracy = if total > 0 { correct as f64 / total as f64 } else { 0.0 };
        Self {
            suite: "gpqa_diamond_local".to_string(),
            version: manifest.version.clone(),
            dataset_id: manifest.dataset_id.clone(),
            revision: manifest.revision.clone(),
            sha256: manifest.sha256.clone(),
            license: manifest.license.clone(),
            attribution: manifest.attribution.clone(),
            grading_method: manifest.grading_method.clone(),
            compatibility: manifest.compatibility.clone(),
            permutation: manifest.permutation.clone(),
            template: manifest.template.clone(),
            label: "aproximación local no oficial — no leaderboard oficial".to_string(),
            preconditions: pre.preconditions.clone(),
            total,
            correct,
            accuracy,
            note: "Resultado local aproximado; no comparable a leaderboard oficial de GPQA (requiere harness oficial). No se registran prompts.".to_string(),
            pre_registration_path: pre_path.display().to_string(),
            pre_registration_sha256: pre_sha256.to_string(),
            limits: pre.limits.clone(),
            expected_artifacts: pre.expected_artifacts.clone(),
        }
    }
}

// ── Tests unitarios (F1.4 + G1) ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn valid_pre_json(rev: &str, hash: &str) -> String {
        serde_json::json!({
            "suite": "gpqa_diamond",
            "n": 198,
            "dataset_id": GPQA_DATASET_ID,
            "revision": rev,
            "sha256": hash,
            "model": "ornith:9b",
            "limits": {"limit": 198, "timeout_secs": 30},
            "grading_method": GPQA_GRADING_METHOD,
            "compatibility": GPQA_COMPATIBILITY,
            "preconditions": {"hf_acceptance": true},
            "expected_artifacts": ["gpqa_results.json"]
        }).to_string()
    }

    #[test]
    fn parse_metadata_valid() {
        let json = serde_json::json!({
            "dataset_id": GPQA_DATASET_ID,
            "revision": "a".repeat(64)[..40].to_string(),
            "sha256": "a".repeat(64),
            "license": GPQA_LICENSE,
            "attribution": GPQA_ATTRIBUTION,
            "version": GPQA_WRAPPER_VERSION,
            "grading_method": GPQA_GRADING_METHOD,
            "compatibility": GPQA_COMPATIBILITY,
            "template": GPQA_TEMPLATE,
            "permutation": GPQA_PERMUTATION
        })
        .to_string();
        let meta = parse_metadata(&json).unwrap();
        assert_eq!(meta.dataset_id, GPQA_DATASET_ID);
        assert_eq!(meta.sha256.len(), 64);
    }

    #[test]
    fn parse_metadata_invalid_json_fails() {
        let err = parse_metadata("{ not json }").unwrap_err();
        assert!(matches!(err, GpqaError::InvalidMetadata(_)));
    }

    #[test]
    fn parse_metadata_missing_field_fails() {
        let err = parse_metadata(r#"{"dataset_id":"x"}"#).unwrap_err();
        assert!(matches!(err, GpqaError::InvalidMetadata(_)));
    }

    #[test]
    fn validate_revision_rejects_placeholders() {
        assert!(validate_revision_format("main").is_err());
        assert!(validate_revision_format("").is_err());
        assert!(validate_revision_format("abc").is_err());
        assert!(validate_revision_format(&"0".repeat(40)).is_err());
        assert!(validate_revision_format(&"g".repeat(40)).is_err());
        assert!(validate_revision_format(&"a".repeat(40)).is_ok());
        assert!(validate_revision_format(&"A".repeat(40)).is_ok());
    }

    #[test]
    fn validate_hash_rejects_placeholder_and_format() {
        assert!(validate_hash_format(&"a".repeat(64)).is_ok());
        assert!(validate_hash_format("abc").is_err());
        assert!(validate_hash_format(&"0".repeat(64)).is_err());
        assert!(validate_hash_format("").is_err());
        assert!(validate_hash_format(&"g".repeat(64)).is_err());
    }

    #[test]
    fn validate_pre_registration_ok_and_rejects() {
        let rev = "a".repeat(40);
        let hash = "b".repeat(64);
        let ok = parse_pre_registration(&valid_pre_json(&rev, &hash)).unwrap();
        assert!(validate_pre_registration(&ok).is_ok());

        // n !=198
        let mut bad = ok.clone();
        bad.n = 10;
        assert!(validate_pre_registration(&bad).is_err());

        // revision placeholder
        let mut bad = ok.clone();
        bad.revision = "main".into();
        assert!(validate_pre_registration(&bad).is_err());

        // hash placeholder zeros
        let mut bad = ok.clone();
        bad.sha256 = "0".repeat(64);
        assert!(validate_pre_registration(&bad).is_err());

        // suite invalid
        let mut bad = ok.clone();
        bad.suite = "polyglot".into();
        assert!(validate_pre_registration(&bad).is_err());

        // empty preconditions
        let mut bad = ok.clone();
        bad.preconditions = serde_json::json!({});
        assert!(validate_pre_registration(&bad).is_err());

        // empty artifacts
        let mut bad = ok;
        bad.expected_artifacts = vec![];
        assert!(validate_pre_registration(&bad).is_err());
    }

    #[test]
    fn verify_integrity_success_and_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("cache.bin");
        std::fs::write(&file, b"hello gpqa").unwrap();
        let hash = sha256_hex(b"hello gpqa");
        let rev = "a".repeat(40);

        let meta_ok = GpqaMetadata {
            dataset_id: GPQA_DATASET_ID.to_string(),
            revision: rev.clone(),
            sha256: hash.clone(),
            license: GPQA_LICENSE.to_string(),
            attribution: GPQA_ATTRIBUTION.to_string(),
            version: GPQA_WRAPPER_VERSION.to_string(),
            grading_method: GPQA_GRADING_METHOD.to_string(),
            compatibility: GPQA_COMPATIBILITY.to_string(),
            template: GPQA_TEMPLATE.to_string(),
            permutation: GPQA_PERMUTATION.to_string(),
        };
        assert!(verify_integrity(&meta_ok, &file).is_ok());

        let meta_bad = GpqaMetadata { sha256: "b".repeat(64), ..meta_ok.clone() };
        let err = verify_integrity(&meta_bad, &file).unwrap_err();
        assert!(matches!(err, GpqaError::HashMismatch { .. }));
    }

    #[test]
    fn verify_integrity_rejects_placeholder_rev_and_hash() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("cache.bin");
        std::fs::write(&file, b"x").unwrap();
        let mut meta = GpqaMetadata {
            dataset_id: GPQA_DATASET_ID.to_string(),
            revision: "main".into(),
            sha256: "a".repeat(64),
            license: GPQA_LICENSE.to_string(),
            attribution: GPQA_ATTRIBUTION.to_string(),
            version: GPQA_WRAPPER_VERSION.to_string(),
            grading_method: GPQA_GRADING_METHOD.to_string(),
            compatibility: GPQA_COMPATIBILITY.to_string(),
            template: GPQA_TEMPLATE.to_string(),
            permutation: GPQA_PERMUTATION.to_string(),
        };
        assert!(matches!(verify_integrity(&meta, &file), Err(GpqaError::PlaceholderRevision(_)) | Err(GpqaError::InvalidRevisionFormat(_))));
        meta.revision = "a".repeat(40);
        meta.sha256 = "0".repeat(64);
        assert!(matches!(verify_integrity(&meta, &file), Err(GpqaError::PlaceholderHash(_))));
    }

    #[test]
    fn sha256_known_vectors() {
        assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        // hello gpqa deterministic
        let h1 = sha256_hex(b"hello gpqa");
        let h2 = sha256_hex(b"hello gpqa");
        assert_eq!(h1, h2);
    }

    #[test]
    fn is_cache_inside_workspace_deterministic() {
        let ws = PathBuf::from("/home/user/Proyectos/Kitty-Code");
        assert!(is_cache_inside_workspace(&PathBuf::from("/home/user/Proyectos/Kitty-Code/.cache/kitty/gpqa/diamond.jsonl"), &ws));
        assert!(is_cache_inside_workspace(&PathBuf::from("/home/user/Proyectos/Kitty-Code/crates/kitty-cli/../bench.json"), &ws));
        assert!(!is_cache_inside_workspace(&PathBuf::from("/home/user/.cache/kitty/gpqa/diamond.jsonl"), &ws));
        assert!(!is_cache_inside_workspace(&PathBuf::from("/tmp/gpqa.jsonl"), &ws));
        // Normalization of `..` and `.`
        assert!(!is_cache_inside_workspace(&PathBuf::from("/home/user/other/../.cache/kitty"), &PathBuf::from("/home/user/Proyectos/Kitty-Code")));
    }

    #[test]
    fn load_questions_rejects_not_198_and_bad_distractors_no_panic() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("gpqa.jsonl");
        // 1 question only -> count mismatch
        let q = serde_json::json!({"id":"q1","question":"q?","correct_answer":"a","distractors":["b","c","d"]}).to_string();
        std::fs::write(&file, q).unwrap();
        let err = load_questions(&file).unwrap_err();
        assert!(matches!(err, GpqaError::CountMismatch { .. }), "got {err:?}");

        // Bad distractors -> InvalidMetadata, not panic
        // Build file with 198 copies of bad distractors to pass count check but fail on distractor len
        let many: String = (0..198).map(|i| serde_json::json!({"id":format!("q{i}"),"question":"q?","correct_answer":"a","distractors":["b","c"]}).to_string()).collect::<Vec<_>>().join("\n");
        std::fs::write(&file, many).unwrap();
        let err = load_questions(&file).unwrap_err();
        assert!(matches!(err, GpqaError::InvalidMetadata(_)), "got {err:?}");

        // Exactly 198 valid -> ok
        let many_ok: String = (0..198).map(|i| serde_json::json!({"id":format!("q{i}"),"question":"q?","correct_answer":"a","distractors":["b","c","d"]}).to_string()).collect::<Vec<_>>().join("\n");
        std::fs::write(&file, many_ok).unwrap();
        assert!(load_questions(&file).is_ok());
    }

    #[test]
    fn permute_four_deterministic_and_no_panic_on_bad_len() {
        let rev = "a".repeat(40);
        let res = permute_four("q123", &rev, "correct".into(), vec!["a".into(), "b".into(), "c".into()]).unwrap();
        let (opts1, idx1) = res;
        let (opts2, idx2) = permute_four("q123", &rev, "correct".into(), vec!["a".into(), "b".into(), "c".into()]).unwrap();
        assert_eq!(opts1, opts2);
        assert_eq!(idx1, idx2);
        assert_eq!(opts1.iter().filter(|o| *o == "correct").count(), 1);
        assert_eq!(opts1[idx1], "correct");

        // Different revision must change permutation (seed includes revision)
        let rev2 = "b".repeat(40);
        let (opts_r2, _) = permute_four("q123", &rev2, "correct".into(), vec!["a".into(), "b".into(), "c".into()]).unwrap();
        assert_ne!(opts1, opts_r2, "revision must affect permutation");

        // Wrong len -> error, not panic
        let err = permute_four("q1", &rev, "c".into(), vec!["a".into()]).unwrap_err();
        assert!(matches!(err, GpqaError::InvalidPreRegistration(_)));
    }

    #[test]
    fn idx_to_letter_and_format_prompt_no_panic() {
        assert_eq!(idx_to_letter(0).unwrap(), 'A');
        assert_eq!(idx_to_letter(3).unwrap(), 'D');
        assert!(idx_to_letter(4).is_err());
        assert!(format_prompt("q?", &["a".into(), "b".into(), "c".into()]).is_err());
        assert!(format_prompt("q?", &["a".into(), "b".into(), "c".into(), "d".into()]).is_ok());
    }

    #[test]
    fn normalize_and_grade() {
        assert_eq!(normalize_answer("A"), Some('A'));
        assert_eq!(normalize_answer("Answer: C"), Some('C'));
        assert_eq!(normalize_answer(""), None);
        assert!(grade_answer(Some('A'), 'A'));
        assert!(!grade_answer(Some('B'), 'A'));
    }

    #[test]
    fn gpqa_result_labels_as_approximation() {
        let manifest = GpqaMetadata {
            dataset_id: GPQA_DATASET_ID.to_string(),
            revision: "a".repeat(40),
            sha256: "b".repeat(64),
            license: GPQA_LICENSE.to_string(),
            attribution: GPQA_ATTRIBUTION.to_string(),
            version: GPQA_WRAPPER_VERSION.to_string(),
            grading_method: GPQA_GRADING_METHOD.to_string(),
            compatibility: GPQA_COMPATIBILITY.to_string(),
            template: GPQA_TEMPLATE.to_string(),
            permutation: GPQA_PERMUTATION.to_string(),
        };
        let res = GpqaResult::new(&manifest, 198, 50);
        assert!(res.label.contains("aproximación local no oficial"));
        assert_eq!(res.total, 198);
    }
}

//! Embedded-inference policy for `kitty run` (F2).
//!
//! Two open decisions, resolved here so every embedder shares them:
//! 1. dtype default F16 (`Auto`) + per-model F32 override list. Initial
//!    entry `qwen2.5-coder*`: isolated F16-CUDA NaN on sm_75 (humo F1);
//!    same file is healthy on CPU and in F32-CUDA.
//! 2. MiniCPM5 tokenizer sidecar-or-honest-error: the `minicpm5`
//!    pre-tokenizer has no standalone-conversion profile, so the embedded
//!    GGUF tokenizer cannot be used alone. Either an explicit
//!    `tokenizer.json` is found (sidecar dir or next to the model) or load
//!    fails with the exact recovery step — never a silent wrong tokenizer.

use std::path::{Path, PathBuf};

use crate::ModelDType;

/// Lowercase filename prefixes forced to F32 on CUDA.
pub const F32_OVERRIDE_PREFIXES: &[&str] = &["qwen2.5-coder"];

/// Extra env-driven F32 prefixes (`KITTY_INFER_F32_PREFIXES`, comma-separated).
/// Pure helper so embedders (and tests) share the parsing.
pub fn extra_f32_prefixes(env_raw: Option<&str>) -> Vec<String> {
    env_raw
        .unwrap_or("")
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Resolve compute dtype for a GGUF file name. Default is `Auto` (the F16
/// path); F32 when the name matches a built-in or env-provided prefix.
pub fn resolve_dtype(file_name: &str, extra_prefixes: &[String]) -> ModelDType {
    let lower = file_name.to_lowercase();
    if F32_OVERRIDE_PREFIXES
        .iter()
        .any(|p| lower.starts_with(p))
        || extra_prefixes.iter().any(|p| lower.starts_with(p))
    {
        ModelDType::F32
    } else {
        ModelDType::Auto
    }
}

/// Families whose GGUF pre-tokenizer has no standalone-conversion profile
/// and therefore need an external `tokenizer.json`:
/// `(filename prefix, sidecar subdir, HF source repo)`.
const EXTERNAL_TOKENIZER_FAMILIES: &[(&str, &str, &str)] = &[
    ("minicpm5", "MiniCPM5-2B", "openbmb/MiniCPM5-2B"),
    ("k2-horizon", "K2-Horizon-3.7B", "IFM/K2-Horizon-3.7B"),
];

fn external_tokenizer_family(file_name: &str) -> Option<(&'static str, &'static str, &'static str)> {
    let lower = file_name.to_lowercase();
    EXTERNAL_TOKENIZER_FAMILIES
        .iter()
        .find(|(prefix, _, _)| lower.starts_with(prefix))
        .copied()
}

/// Whether this GGUF needs an external `tokenizer.json` (MiniCPM5, K2-Horizon).
pub fn needs_external_tokenizer(file_name: &str) -> bool {
    external_tokenizer_family(file_name).is_some()
}

/// Reference sidecar layout: `<sidecar_dir>/<family-subdir>/tokenizer.json`
/// (e.g. `~/.cache/kitty/tokenizers/MiniCPM5-2B/tokenizer.json`).
/// Takes the family subdir explicitly — the old MiniCPM5-only default hid
/// which family was being resolved (D9).
pub fn sidecar_candidate(sidecar_dir: &Path, family_subdir: &str) -> PathBuf {
    sidecar_dir.join(family_subdir).join("tokenizer.json")
}

/// Resolve the explicit `tokenizer.json` for models that need one.
/// Returns `None` when the embedded tokenizer suffices. Errors honestly
/// (with the recovery step) when no sidecar is available.
pub fn resolve_tokenizer_json(
    model_path: &Path,
    sidecar_dir: Option<&Path>,
) -> anyhow::Result<Option<PathBuf>> {
    let file_name = model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    let Some((_, subdir, source)) = external_tokenizer_family(file_name) else {
        return Ok(None);
    };
    if let Some(dir) = sidecar_dir {
        let cand = sidecar_candidate(dir, subdir);
        if cand.is_file() {
            return Ok(Some(cand));
        }
    }
    if let Some(parent) = model_path.parent() {
        let sibling = parent.join("tokenizer.json");
        if sibling.is_file() {
            return Ok(Some(sibling));
        }
    }
    let pre = file_name
        .split(['-', '_'])
        .next()
        .unwrap_or("custom")
        .to_lowercase();
    anyhow::bail!(
        "GGUF `{}` needs its original tokenizer.json (the `{pre}` \
         pre-tokenizer has no standalone-conversion profile). Fetch it with \
         `hf download {source} tokenizer.json --local-dir <sidecar>/{subdir}` \
         and set the sidecar dir (KITTY_INFER_TOKENIZER_DIR), or place tokenizer.json next to the GGUF",
        model_path.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_default_and_override() {
        assert!(matches!(
            resolve_dtype("Qwen3-4B-Q4_K_M.gguf", &[]),
            ModelDType::Auto
        ));
        assert!(matches!(
            resolve_dtype("qwen2.5-coder-7b-q4_k_m.gguf", &[]),
            ModelDType::F32
        ));
        assert!(matches!(
            resolve_dtype("QWEN2.5-CODER-7B-Q4_K_M.GGUF", &[]),
            ModelDType::F32
        ));
        let extra = extra_f32_prefixes(Some("my-model, other"));
        assert!(matches!(
            resolve_dtype("my-model-q8.gguf", &extra),
            ModelDType::F32
        ));
        assert!(matches!(
            resolve_dtype("other-q8.gguf", &extra_f32_prefixes(None)),
            ModelDType::Auto
        ));
    }

    #[test]
    fn minicpm5_sidecar_or_honest_error() {
        assert!(needs_external_tokenizer("MiniCPM5-2B-Q4_K_M.gguf"));
        assert!(!needs_external_tokenizer("Qwen3-4B-Q4_K_M.gguf"));
        // No sidecar anywhere -> honest error naming the recovery.
        let err = resolve_tokenizer_json(Path::new("/models/MiniCPM5-2B-Q4_K_M.gguf"), None)
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("tokenizer.json"), "must name recovery: {msg}");
        assert!(msg.contains("openbmb/MiniCPM5-2B"), "must name source: {msg}");
        // Plain GGUF -> None, no error.
        assert!(
            resolve_tokenizer_json(Path::new("/models/Qwen3-4B-Q4_K_M.gguf"), None)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn sidecar_layout_reference() {
        assert_eq!(
            sidecar_candidate(Path::new("/s"), "MiniCPM5-2B"),
            PathBuf::from("/s/MiniCPM5-2B/tokenizer.json")
        );
        assert_eq!(
            sidecar_candidate(Path::new("/s"), "K2-Horizon-3.7B"),
            PathBuf::from("/s/K2-Horizon-3.7B/tokenizer.json")
        );
    }

    #[test]
    fn k2_horizon_sidecar_or_honest_error() {
        assert!(needs_external_tokenizer("K2-Horizon-4B-BF16.gguf"));
        let err = resolve_tokenizer_json(Path::new("/models/K2-Horizon-4B-BF16.gguf"), None)
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("tokenizer.json"), "must name recovery: {msg}");
        assert!(msg.contains("IFM/K2-Horizon-3.7B"), "must name source: {msg}");
    }
}

//! Argument building for GGUF loads (no vLLM here).
//!
//! - `quant_for_format`: GGUF files are already quantized (keep); raw
//!   safetensors/pth hints an in-situ quant instead of failing later.
//! - `effective_ctx`: `min(ctx_len, max_model_len)`.
//! - `inject_template`: explicit `--chat-template` path wins; otherwise keep
//!   the embedded/discovered template untouched.

use std::path::{Path, PathBuf};

/// Quantization hint per weight format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantHint {
    /// GGUF: already quantized, load as-is.
    Keep,
    /// Raw weights: suggest in-situ quant (label only, caller applies).
    Suggest(String),
}

/// Quant policy by file format/extension.
pub fn quant_for_format(path: &Path) -> QuantHint {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "gguf" => QuantHint::Keep,
        "safetensors" | "bin" | "pth" => QuantHint::Suggest("Q4_K_M".to_string()),
        _ => QuantHint::Suggest("Q4_K_M".to_string()),
    }
}

/// Effective context: never exceed the model's max.
pub fn effective_ctx(ctx_len: usize, max_model_len: Option<usize>) -> usize {
    match max_model_len {
        Some(cap) => ctx_len.min(cap),
        None => ctx_len,
    }
}

/// Resolved GGUF launch arguments.
#[derive(Debug, Clone)]
pub struct GgufArgs {
    pub model: PathBuf,
    pub ctx_len: usize,
    pub max_model_len: Option<usize>,
    pub n_gpu_layers: Option<usize>,
    pub chat_template_path: Option<PathBuf>,
}

impl GgufArgs {
    pub fn new(model: PathBuf, ctx_len: usize) -> Self {
        Self {
            model,
            ctx_len,
            max_model_len: None,
            n_gpu_layers: None,
            chat_template_path: None,
        }
    }

    pub fn effective_ctx_len(&self) -> usize {
        effective_ctx(self.ctx_len, self.max_model_len)
    }

    pub fn quant_hint(&self) -> QuantHint {
        quant_for_format(&self.model)
    }
}

/// Template injection: an explicit path wins; otherwise `None` (keep the
/// model's embedded template). Never synthesizes a fake template.
pub fn inject_template(args: &mut GgufArgs, explicit: Option<PathBuf>) {
    if explicit.is_some() {
        args.chat_template_path = explicit;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_and_ctx() {
        assert_eq!(quant_for_format(Path::new("m.gguf")), QuantHint::Keep);
        assert!(matches!(
            quant_for_format(Path::new("m.safetensors")),
            QuantHint::Suggest(_)
        ));
        assert_eq!(effective_ctx(8192, Some(4096)), 4096);
        assert_eq!(effective_ctx(2048, Some(4096)), 2048);
    }

    #[test]
    fn template_wins() {
        let mut a = GgufArgs::new(PathBuf::from("m.gguf"), 4096);
        inject_template(&mut a, None);
        assert!(a.chat_template_path.is_none());
        inject_template(&mut a, Some(PathBuf::from("/t.jinja")));
        assert_eq!(a.chat_template_path, Some(PathBuf::from("/t.jinja")));
    }
}

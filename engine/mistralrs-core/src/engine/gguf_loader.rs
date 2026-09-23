//! GGUF load planning for `kitty run` on consumer GPUs (gate F1).
//!
//! - `resolve_n_gpu_layers`: clamp requested layers to `[0, n_layers]`.
//!   `requested > n_layers` (or `> ctx` pressure) resolves down, never up.
//! - `vram_precheck`: estimate model + KV bytes vs free VRAM *before* load.
//! - `backend_mismatch_warn`: CPU-with-GPU-layers and CUDA-with-zero-layer hints.
//! - `LogBuffer`: bounded ring keeping the real failure cause.

use std::collections::VecDeque;
use std::path::Path;

use candle_core::Device;

/// Planned GGUF load.
#[derive(Debug, Clone)]
pub struct GgufLoadPlan {
    pub file: String,
    pub n_gpu_layers: usize,
    pub ctx_len: usize,
    pub gpu_fraction: f32,
}

/// Clamp requested GPU layers into `[0, n_layers]`.
/// `None` means "all layers on GPU" (consumer default).
pub fn resolve_n_gpu_layers(requested: Option<usize>, n_layers: usize) -> usize {
    match requested {
        None => n_layers,
        Some(n) => n.min(n_layers),
    }
}

/// Resolve context length against the model's max. Never grows past the cap.
pub fn resolve_ctx_len(requested: usize, max_model_len: Option<usize>) -> usize {
    match max_model_len {
        Some(cap) => requested.min(cap),
        None => requested,
    }
}

/// Heuristic VRAM estimate, kept honest and documented:
/// model_vram = file_bytes * gpu_fraction; kv_vram ~= ctx_len * 256 KiB
/// per token of resident KV (dense decoder average for 1-35B at Q4).
/// Fails *before* load when estimate exceeds free bytes.
pub fn vram_precheck(
    file_bytes: u64,
    free_bytes: Option<u64>,
    gpu_fraction: f32,
    ctx_len: usize,
) -> anyhow::Result<()> {
    let Some(free) = free_bytes else {
        // No GPU snapshot (CPU box): nothing to pre-check.
        return Ok(());
    };
    let model_vram = (file_bytes as f64 * gpu_fraction.clamp(0.0, 1.0) as f64) as u64;
    let kv_vram = ctx_len as u64 * 256 * 1024;
    let total = model_vram.saturating_add(kv_vram);
    if total > free {
        anyhow::bail!(
            "VRAM pre-check failed: need ~{total} bytes (model ~{model_vram} + KV ~{kv_vram} for ctx {ctx_len}) but only {free} bytes free; use a smaller quant, fewer --gpu-layers, or --ctx 4096"
        );
    }
    Ok(())
}

/// Warn when the device and layer placement disagree. Returns the warning.
pub fn backend_mismatch_warn(device: &Device, n_gpu_layers: usize) -> Option<String> {
    match device {
        Device::Cpu if n_gpu_layers > 0 => Some(format!(
            "backend mismatch: CPU device with n_gpu_layers={n_gpu_layers}; all layers will run on CPU — pass --gpu-layers 0 to silence or use a CUDA build"
        )),
        Device::Cuda(_) if n_gpu_layers == 0 => Some(
            "backend mismatch: CUDA device with n_gpu_layers=0; model will run fully on CPU — omit --gpu-layers to use all GPU layers".to_string(),
        ),
        _ => None,
    }
}

/// File size helper with the real OS error preserved.
pub fn file_bytes(path: &Path) -> anyhow::Result<u64> {
    Ok(std::fs::metadata(path)
        .map_err(|e| anyhow::anyhow!("cannot stat GGUF `{}`: {e}", path.display()))?
        .len())
}

/// Bounded log buffer keeping the real cause (no truncation of the last error).
pub struct LogBuffer {
    ring: VecDeque<String>,
    cap: usize,
}

impl LogBuffer {
    pub fn new(cap: usize) -> Self {
        Self { ring: VecDeque::with_capacity(cap.max(1)), cap: cap.max(1) }
    }

    pub fn push(&mut self, line: impl Into<String>) {
        if self.ring.len() >= self.cap {
            self.ring.pop_front();
        }
        self.ring.push_back(line.into());
    }

    /// Last line = the real cause to surface.
    pub fn cause(&self) -> Option<&str> {
        self.ring.back().map(|s| s.as_str())
    }

    pub fn tail(&self, n: usize) -> Vec<&str> {
        let n = n.min(self.ring.len());
        self.ring.iter().skip(self.ring.len() - n).map(|s| s.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_layers() {
        assert_eq!(resolve_n_gpu_layers(None, 32), 32);
        assert_eq!(resolve_n_gpu_layers(Some(99), 32), 32);
        assert_eq!(resolve_n_gpu_layers(Some(10), 32), 10);
        assert_eq!(resolve_ctx_len(8192, Some(4096)), 4096);
    }

    #[test]
    fn precheck_fails_with_cause() {
        let err = vram_precheck(8 * 1024 * 1024 * 1024, Some(4 * 1024 * 1024 * 1024), 0.9, 8192)
            .unwrap_err();
        assert!(format!("{err:#}").contains("VRAM pre-check failed"));
    }
}

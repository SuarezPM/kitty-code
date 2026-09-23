//! Unified GPU snapshot, cached 1s.
//!
//! Tries `nvidia-smi`, then `rocm-smi`. No polling loops, no fragile
//! `ldconfig`/`vulkaninfo` probes: if neither tool answers, the snapshot
//! reports `cpu` with an empty device list and the caller surfaces a hint.
//! Commands are argv-based (no shell) and bounded.

use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceStatus {
    pub name: String,
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub util_percent: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSnapshot {
    pub backend: String,
    pub devices: Vec<GpuDeviceStatus>,
    /// Seconds since UNIX epoch, for cache transparency.
    pub fetched_at_secs: u64,
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn run_nvidia_smi() -> Option<Vec<GpuDeviceStatus>> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut devices = Vec::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 4 {
            continue;
        }
        let mib = |s: &str| s.split_whitespace().next().unwrap_or("0").parse::<u64>().unwrap_or(0) * 1024 * 1024;
        devices.push(GpuDeviceStatus {
            name: parts[0].to_string(),
            total_bytes: mib(parts[1]),
            used_bytes: mib(parts[2]),
            util_percent: parts[3].split_whitespace().next().and_then(|s| s.parse().ok()),
        });
    }
    if devices.is_empty() { None } else { Some(devices) }
}

fn run_rocm_smi() -> Option<Vec<GpuDeviceStatus>> {
    // `rocm-smi --showmeminfo vram --csv` is stable across ROCm 5/6; parse leniently.
    let out = std::process::Command::new("rocm-smi")
        .args(["--showmeminfo", "vram", "--csv"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut devices = Vec::new();
    for line in text.lines().skip(1) {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 3 {
            continue;
        }
        let total_mb = parts.get(1).and_then(|s| s.trim().parse::<u64>().ok()).unwrap_or(0);
        let used_mb = parts.get(2).and_then(|s| s.trim().parse::<u64>().ok()).unwrap_or(0);
        devices.push(GpuDeviceStatus {
            name: parts[0].trim().to_string(),
            total_bytes: total_mb * 1024 * 1024,
            used_bytes: used_mb * 1024 * 1024,
            util_percent: None,
        });
    }
    if devices.is_empty() { None } else { Some(devices) }
}

fn fresh_snapshot() -> GpuSnapshot {
    if let Some(devices) = run_nvidia_smi() {
        return GpuSnapshot { backend: "cuda".to_string(), devices, fetched_at_secs: now_secs() };
    }
    if let Some(devices) = run_rocm_smi() {
        return GpuSnapshot { backend: "rocm".to_string(), devices, fetched_at_secs: now_secs() };
    }
    GpuSnapshot { backend: "cpu".to_string(), devices: Vec::new(), fetched_at_secs: now_secs() }
}

static CACHE: OnceLock<Mutex<(Option<GpuSnapshot>, Option<Instant>)>> = OnceLock::new();

fn cache() -> &'static Mutex<(Option<GpuSnapshot>, Option<Instant>)> {
    CACHE.get_or_init(|| Mutex::new((None, None)))
}

/// Cached snapshot (1s TTL). Cheap to call per request.
pub fn snapshot() -> GpuSnapshot {
    let lock = cache();
    let guard = lock.lock().unwrap();
    if let (Some(snap), Some(at)) = (guard.0.clone(), guard.1) {
        if at.elapsed() < Duration::from_secs(1) {
            return snap;
        }
    }
    drop(guard);
    let fresh = fresh_snapshot();
    *lock.lock().unwrap() = (Some(fresh.clone()), Some(Instant::now()));
    fresh
}

/// Free bytes on the first device, if any GPU is present.
pub fn first_device_free_bytes() -> Option<u64> {
    let snap = snapshot();
    snap.devices.first().map(|d| d.total_bytes.saturating_sub(d.used_bytes))
}

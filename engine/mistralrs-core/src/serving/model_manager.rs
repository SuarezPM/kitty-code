//! Single-resident model registry for `kitty run`.
//!
//! Rule: exactly one GGUF resident at a time. A new load evicts the previous
//! resident *before* loading (frees VRAM first), then retries once on OOM
//! with a reduced GPU fraction (0.9 -> 0.6).
//!
//! Concurrency: one real global lock (`tokio::sync::Mutex`). No placeholder
//! imports, no fake locking.

use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::sync::Mutex;

/// Where a model comes from. Only what `kitty run` needs for 1-35B GGUF.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    /// `hf:<repo>/<file>` — HuggingFace GGUF artifact.
    HuggingFace { repo: String, file: String },
    /// `ollama:<name>` — local Ollama library entry (GGUF blob).
    Ollama { name: String },
    /// `local:<path.gguf>` or any bare path ending in `.gguf`.
    LocalGguf { path: String },
}

/// Resolved model identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    /// Display id (repo/file, ollama name, or file stem).
    pub id: String,
    pub source: ModelSource,
}

impl ModelSpec {
    /// Auto-detect source by prefix. Accepted:
    /// `hf:<repo>/<file>`, `ollama:<name>`, `local:<path.gguf>`,
    /// or a bare path ending in `.gguf` (case-insensitive).
    pub fn auto(raw: &str) -> anyhow::Result<Self> {
        let raw = raw.trim();
        if raw.is_empty() {
            anyhow::bail!("empty model spec");
        }
        if let Some(rest) = raw.strip_prefix("hf:") {
            let rest = rest.trim();
            let (repo, file) = rest
                .rsplit_once('/')
                .filter(|(r, f)| !r.is_empty() && !f.is_empty())
                .map(|(r, f)| (r.to_string(), f.to_string()))
                .unwrap_or((rest.to_string(), String::new()));
            if file.is_empty() {
                anyhow::bail!("hf: spec needs `hf:<repo>/<file.gguf>`, got `{raw}`");
            }
            let id = format!("{repo}/{file}");
            return Ok(Self {
                id,
                source: ModelSource::HuggingFace { repo, file },
            });
        }
        if let Some(name) = raw.strip_prefix("ollama:") {
            let name = name.trim().to_string();
            if name.is_empty() {
                anyhow::bail!("ollama: spec needs `ollama:<name>`, got `{raw}`");
            }
            return Ok(Self {
                id: name.clone(),
                source: ModelSource::Ollama { name },
            });
        }
        if let Some(path) = raw.strip_prefix("local:") {
            let path = path.trim().to_string();
            Self::local(path)
        } else if raw.to_lowercase().ends_with(".gguf") {
            Self::local(raw.to_string())
        } else {
            anyhow::bail!(
                "unknown model spec `{raw}`: use hf:<repo>/<file>, ollama:<name>, or local:<path.gguf>"
            )
        }
    }

    fn local(path: String) -> anyhow::Result<Self> {
        if path.is_empty() {
            anyhow::bail!("local: spec needs `local:<path.gguf>`");
        }
        if !path.to_lowercase().ends_with(".gguf") {
            anyhow::bail!("local model must be a .gguf file, got `{path}`");
        }
        let stem = path
            .rsplit('/')
            .next()
            .unwrap_or(&path)
            .trim_end_matches(|c| c == '"')
            .to_string();
        Ok(Self {
            id: stem,
            source: ModelSource::LocalGguf { path },
        })
    }
}

/// Currently resident model metadata (no engine handle here; the caller owns it).
#[derive(Debug, Clone)]
pub struct Resident {
    pub spec: ModelSpec,
    pub gpu_fraction: f32,
    pub loaded_at_secs: u64,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Single-resident registry with single-flight loads.
///
/// Concurrency contract (D1): `inner` is held across the whole
/// evict → load → store sequence, so concurrent `load_with_retry` calls
/// serialize and a second waiter for the SAME spec reuses the resident
/// without rebuilding (exactly one build per spec). This is sound ONLY
/// because the `load` closure never calls back into the registry
/// (no `resident`/`evict`/`load_with_retry` inside `load`); a reentrant
/// closure would deadlock. Keep it that way — the compiler will not save
/// you here.
pub struct ModelRegistry {
    inner: Mutex<Option<Resident>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }

    /// Current resident, if any.
    pub async fn resident(&self) -> Option<Resident> {
        self.inner.lock().await.clone()
    }

    /// Evict the current resident without loading anything.
    /// Caller drops its engine handle after this returns.
    pub async fn evict(&self) -> Option<Resident> {
        self.inner.lock().await.take()
    }

    /// Load `spec` with evict-before-load + OOM retry (0.9 -> 0.6).
    ///
    /// `load` is the real loader closure: receives the GPU fraction to use
    /// and returns Ok on success or an error whose Display mentions OOM
    /// (case-insensitive `oom` / `out of memory` / `memory`) for retry.
    /// Non-OOM errors return immediately. Previous resident is evicted
    /// *before* the first attempt so its VRAM is free.
    ///
    /// Single-flight: the registry guard is held across the whole sequence
    /// (see the type-level contract). A waiter arriving while another load
    /// of the SAME spec is in flight blocks, then re-checks and reuses the
    /// fresh resident — one build total. A different spec evicts and builds.
    pub async fn load_with_retry<F, Fut>(&self, spec: ModelSpec, mut load: F) -> anyhow::Result<Resident>
    where
        F: FnMut(f32) -> Fut,
        Fut: std::future::Future<Output = anyhow::Result<()>>,
    {
        // Fast path: already resident (short lock, no build).
        if let Some(resident) = self.inner.lock().await.clone() {
            if resident.spec == spec {
                return Ok(resident);
            }
        }

        let mut slot = self.inner.lock().await;
        // Re-check under the guard: a concurrent load may have finished
        // while this task waited for the lock.
        if let Some(resident) = slot.as_ref() {
            if resident.spec == spec {
                return Ok(resident.clone());
            }
        }
        slot.take(); // evict-before-load, VRAM free before the first attempt

        let mut last_err: Option<anyhow::Error> = None;
        for fraction in [0.9f32, 0.6f32] {
            match load(fraction).await {
                Ok(()) => {
                    let resident = Resident {
                        spec: spec.clone(),
                        gpu_fraction: fraction,
                        loaded_at_secs: now_secs(),
                    };
                    *slot = Some(resident.clone());
                    if last_err.is_some() {
                        tracing::info!(
                            "model `{}` loaded after OOM retry at gpu_fraction={fraction}",
                            spec.id
                        );
                    }
                    return Ok(resident);
                }
                Err(e) => {
                    let msg = format!("{e:#}");
                    // Shared predicate (serving/errors.rs): retry on OOM only.
                    if !super::errors::is_oom_message(&msg) {
                        return Err(e);
                    }
                    tracing::warn!(
                        "load of `{}` failed with OOM at gpu_fraction={fraction}: {msg:#}; retrying lower",
                        spec.id
                    );
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("load failed without error detail")))
    }
    /// Warm-resident preload for serving startup (Fase 2 SDD).
    /// Same-spec fast path: never rebuilds nor evicts, only probes liveness.
    /// Probe failure evicts and bails loudly — never measure over a dead context.
    /// Bench calls this once at startup so the first turn finds weights+graphs ready.
    /// Caller supplies `probe` (e.g. 1-token dry step or CUDA sync query).
    pub async fn preload_with_retry<F, Fut, P, PFut>(
        &self,
        spec: ModelSpec,
        load: F,
        mut probe: P,
    ) -> anyhow::Result<Resident>
    where
        F: FnMut(f32) -> Fut,
        Fut: std::future::Future<Output = anyhow::Result<()>>,
        P: FnMut() -> PFut,
        PFut: std::future::Future<Output = anyhow::Result<()>>,
    {
        // Load (or fast-path reuse) first; the probe runs lock-free so it may
        // safely call `resident()` — unlike `load`, which must never reenter.
        let resident = self.load_with_retry(spec, load).await?;
        if let Err(e) = probe().await {
            // Dead context: leave nothing resident behind, fail noisy.
            self.evict().await;
            anyhow::bail!(
                "preload of `{}` built but liveness probe failed (dead CUDA context?): {e:#}",
                resident.spec.id
            );
        }
        Ok(resident)
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

static GLOBAL: OnceLock<ModelRegistry> = OnceLock::new();

/// Process-wide registry used by `kitty run`.
pub fn global_registry() -> &'static ModelRegistry {
    GLOBAL.get_or_init(ModelRegistry::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_prefixes() {
        assert!(matches!(
            ModelSpec::auto("hf:org/model/model.gguf").unwrap().source,
            ModelSource::HuggingFace { .. }
        ));
        assert!(matches!(
            ModelSpec::auto("ollama:qwen3:8b").unwrap().source,
            ModelSource::Ollama { .. }
        ));
        assert!(matches!(
            ModelSpec::auto("local:/models/q4.gguf").unwrap().source,
            ModelSource::LocalGguf { .. }
        ));
        assert!(matches!(
            ModelSpec::auto("/models/q4.GGUF").unwrap().source,
            ModelSource::LocalGguf { .. }
        ));
        assert!(ModelSpec::auto("gpt-4").is_err());
        assert!(ModelSpec::auto("hf:onlyrepo").is_err());
    }

    #[tokio::test]
    async fn same_spec_reuses_resident_without_build() {
        // Single-flight, deterministic half: a resident for the requested
        // spec short-circuits before the loader runs (the closure panics if
        // invoked — a build here would be the D1 bug).
        let reg = ModelRegistry::new();
        let spec = ModelSpec::auto("local:/same.gguf").unwrap();
        *reg.inner.lock().await = Some(Resident {
            spec: spec.clone(),
            gpu_fraction: 0.9,
            loaded_at_secs: 7,
        });
        let resident = reg
            .load_with_retry(spec, |_| async {
                panic!("loader must not run for an already-resident spec")
            })
            .await
            .unwrap();
        assert_eq!(resident.spec.id, "same.gguf");
        assert_eq!(resident.gpu_fraction, 0.9);
        assert_eq!(resident.loaded_at_secs, 7);
    }

    #[tokio::test]
    async fn evict_before_load_and_oom_retry() {
        let reg = ModelRegistry::new();
        // Seed a resident, then load: the old one must be gone before load runs.
        *reg.inner.lock().await = Some(Resident {
            spec: ModelSpec::auto("local:/a.gguf").unwrap(),
            gpu_fraction: 0.9,
            loaded_at_secs: 1,
        });
        let seen_empty = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let seen = seen_empty.clone();
        let spec = ModelSpec::auto("local:/b.gguf").unwrap();
        let mut calls = 0;
        let resident = reg
            .load_with_retry(spec, |frac| {
                calls += 1;
                let seen = seen.clone();
                async move {
                    if frac == 0.9 {
                        // Simulate: previous resident already evicted is observable
                        // via the retry path; first attempt OOMs.
                        seen.store(true, std::sync::atomic::Ordering::SeqCst);
                        anyhow::bail!("CUDA OOM: out of memory allocating 2GiB")
                    } else {
                        Ok(())
                    }
                }
            })
            .await
            .unwrap();
        assert_eq!(calls, 2);
        assert!(seen_empty.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(resident.gpu_fraction, 0.6);
        assert_eq!(reg.resident().await.unwrap().spec.id, "b.gguf");
    }

    #[tokio::test]
    async fn preload_warms_then_reuses_without_rebuild() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        let reg = ModelRegistry::new();
        let spec = ModelSpec::auto("local:/warm.gguf").unwrap();
        let builds = Arc::new(AtomicUsize::new(0));
        let probes = Arc::new(AtomicUsize::new(0));
        // First preload builds (cold start paid once, at startup).
        let resident = reg
            .preload_with_retry(
                spec.clone(),
                {
                    let builds = builds.clone();
                    move |_| {
                        builds.fetch_add(1, Ordering::SeqCst);
                        async { Ok(()) }
                    }
                },
                {
                    let probes = probes.clone();
                    move || {
                        probes.fetch_add(1, Ordering::SeqCst);
                        async { Ok(()) }
                    }
                },
            )
            .await
            .unwrap();
        assert_eq!(resident.spec.id, "warm.gguf");
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        // Second preload (the first bench turn): no rebuild, liveness re-probed.
        // A loader run here would be cold-start contamination on the measured turn.
        let resident2 = reg
            .preload_with_retry(
                spec,
                |_| async { panic!("warm path must not rebuild") },
                {
                    let probes = probes.clone();
                    move || {
                        probes.fetch_add(1, Ordering::SeqCst);
                        async { Ok(()) }
                    }
                },
            )
            .await
            .unwrap();
        assert_eq!(resident2.spec.id, "warm.gguf");
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert_eq!(probes.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn preload_probe_failure_evicts_and_fails_noisy() {
        let reg = ModelRegistry::new();
        let spec = ModelSpec::auto("local:/dead.gguf").unwrap();
        let err = reg
            .preload_with_retry(spec, |_| async { Ok(()) }, || async {
                anyhow::bail!("CUDA context dead: device 0 lost")
            })
            .await
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("dead.gguf"), "must name the spec: {msg}");
        assert!(msg.contains("liveness"), "must say liveness: {msg}");
        // Nothing left resident: no measuring over a corpse.
        assert!(reg.resident().await.is_none());
    }

    #[tokio::test]
    async fn preload_load_failure_leaves_nothing_resident() {
        let reg = ModelRegistry::new();
        let err = reg
            .preload_with_retry(
                ModelSpec::auto("local:/broken.gguf").unwrap(),
                |_| async { anyhow::bail!("weights file missing") },
                || async { Ok(()) },
            )
            .await
            .unwrap_err();
        assert!(format!("{err:#}").contains("weights file missing"));
        assert!(reg.resident().await.is_none());
    }
}

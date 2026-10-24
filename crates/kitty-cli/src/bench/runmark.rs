//! runmark — fingerprint BLAKE3 + sidecar + exit codes (Capa 0, Lane B).
//!
//! Fingerprint canónico de celda bench sobre claves ordenadas:
//! `(model,digest,backend,profile,ctx,max_iter,hops,suite,dataset_sha,
//! timeout_cell,seed,git_head,dirty,memory,systemone,toolset)`.
//! El JSON canónico usa `BTreeMap` (claves ordenadas byte a byte); el
//! fingerprint es `blake3(canonical_bytes)` en hex.
//!
//! Sidecar `<out>.runmark.json`: escritura tmp+fsync+rename (atómica en el
//! mismo dir). Rerun de la misma celda se rechaza ANTES del LLM
//! (`check_rerun_rejected`): si el sidecar ya existe con el mismo
//! fingerprint, el bench no corre.
//!
//! Exit codes: 0 pass / 1 gate-fail / 2 inconcluso / 3 integridad.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Exit codes unificados (B2, Capa 0): 0 pass / 1 gate-fail /
/// 2 inconcluso (1 rep o noise, nunca publicable) / 3 integridad.
/// 2 = inconcluso: 1 rep sin k>=3, evidencia insuficiente, o noise
/// (|delta|<2 sin significación). Sin doble sentido: noise ES inconcluso,
/// no "pass con ruido". Comparte mapping con `common::bench_exit_code` y
/// `bench/mod.rs:317` (todos usan estas consts).
pub const EXIT_PASS: i32 = 0;
/// Gate no superado (o regresión vs baseline).
pub const EXIT_GATE_FAIL: i32 = 1;
/// Inconcluso: 1 rep sin k>=3, evidencia insuficiente o noise (|delta|<2).
/// Nunca publicable (sin SIGNAL). Mapeo B2: el report genérico
/// (`bench/mod.rs`) mapea `Signal::NoSignal` con razón `k<3` (single-rep,
/// reps=1) a este exit aunque el gate pase.
pub const EXIT_INCONCLUSIVE: i32 = 2;
/// Integridad: rerun misma celda, leak tripwire (KITTY_RUN_LEAKED),
/// sidecar corrupto.
pub const EXIT_INTEGRITY: i32 = 3;

/// Fingerprint de celda bench (todos los ejes que cambian el resultado).
#[derive(Debug, Clone)]
pub struct Runmark {
    pub model: String,
    pub digest: String,
    pub backend: String,
    pub profile: String,
    pub ctx: u32,
    pub max_iter: usize,
    pub hops: usize,
    pub suite: String,
    pub dataset_sha: String,
    pub timeout_cell: u64,
    pub seed: u64,
    pub git_head: String,
    pub dirty: bool,
    pub memory: bool,
    pub systemone: String,
    pub toolset: String,
}

impl Runmark {
    /// JSON canónico con claves ordenadas (BTreeMap ⇒ orden alfabético).
    pub fn canonical_json(&self) -> serde_json::Value {
        let mut m = BTreeMap::new();
        m.insert("backend".to_string(), serde_json::json!(self.backend));
        m.insert("ctx".to_string(), serde_json::json!(self.ctx));
        m.insert("dataset_sha".to_string(), serde_json::json!(self.dataset_sha));
        m.insert("digest".to_string(), serde_json::json!(self.digest));
        m.insert("dirty".to_string(), serde_json::json!(self.dirty));
        m.insert("git_head".to_string(), serde_json::json!(self.git_head));
        m.insert("hops".to_string(), serde_json::json!(self.hops));
        m.insert("max_iter".to_string(), serde_json::json!(self.max_iter));
        m.insert("memory".to_string(), serde_json::json!(self.memory));
        m.insert("model".to_string(), serde_json::json!(self.model));
        m.insert("profile".to_string(), serde_json::json!(self.profile));
        m.insert("seed".to_string(), serde_json::json!(self.seed));
        m.insert("suite".to_string(), serde_json::json!(self.suite));
        m.insert("systemone".to_string(), serde_json::json!(self.systemone));
        m.insert(
            "timeout_cell".to_string(),
            serde_json::json!(self.timeout_cell),
        );
        m.insert("toolset".to_string(), serde_json::json!(self.toolset));
        serde_json::Value::Object(m.into_iter().collect())
    }

    /// Bytes canónicos (JSON compacto, claves ordenadas).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(&self.canonical_json()).unwrap_or_default()
    }

    /// Fingerprint BLAKE3 en hex sobre los bytes canónicos.
    pub fn fingerprint(&self) -> String {
        blake3::hash(&self.canonical_bytes()).to_hex().to_string()
    }
}

/// `<out>.runmark.json` (p. ej. `bench.json` → `bench.json.runmark.json`).
pub fn sidecar_path(out: &Path) -> PathBuf {
    let mut s = out.as_os_str().to_owned();
    s.push(".runmark.json");
    PathBuf::from(s)
}

/// Lee el fingerprint guardado en un sidecar existente (`None` si no hay o
/// si no parsea — corrupto se trata como integridad en el llamador).
pub fn load_existing_fingerprint(sidecar: &Path) -> Option<String> {
    let text = std::fs::read_to_string(sidecar).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    v.get("fingerprint").and_then(|f| f.as_str()).map(|s| s.to_string())
}

/// Rechaza rerun de la misma celda ANTES del LLM (F1):
/// si `<out>.runmark.json` ya existe con el MISMO fingerprint, error de
/// integridad (exit 3). Distinta celda (distinto fingerprint) ⇒ Ok (el
/// llamador sobrescribe el sidecar al cerrar el run).
pub fn check_rerun_rejected(out: &Path, fingerprint: &str) -> Result<()> {
    let sidecar = sidecar_path(out);
    if let Some(existing) = load_existing_fingerprint(&sidecar) {
        if existing == fingerprint {
            anyhow::bail!(
                "runmark: misma celda ya registrada en {} (fingerprint {fingerprint}); \
                 rerun rechazado antes del LLM — borre el sidecar o cambie --out",
                sidecar.display()
            );
        }
    }
    Ok(())
}

/// Escribe el sidecar tmp+fsync+rename (atómico en el mismo dir).
pub fn write_sidecar_atomic(out: &Path, runmark: &Runmark) -> Result<()> {
    let sidecar = sidecar_path(out);
    let parent = sidecar.parent().filter(|p| !p.as_os_str().is_empty());
    if let Some(dir) = parent {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("runmark: crear dir {}", dir.display()))?;
    }
    let fingerprint = runmark.fingerprint();
    let payload = serde_json::json!({
        "fingerprint": fingerprint,
        "runmark": runmark.canonical_json(),
    });
    let bytes = serde_json::to_vec_pretty(&payload)?;
    // Tmp en el mismo dir para que rename sea atómico.
    let tmp_name = format!(
        ".{}.tmp.{}",
        sidecar
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "runmark".to_string()),
        std::process::id()
    );
    let tmp_path = sidecar
        .parent()
        .map(|p| p.join(&tmp_name))
        .unwrap_or_else(|| PathBuf::from(&tmp_name));
    let file = std::fs::File::create(&tmp_path)
        .with_context(|| format!("runmark: crear tmp {}", tmp_path.display()))?;
    use std::io::Write as _;
    (&file)
        .write_all(&bytes)
        .with_context(|| format!("runmark: escribir tmp {}", tmp_path.display()))?;
    file.sync_all()
        .with_context(|| format!("runmark: fsync tmp {}", tmp_path.display()))?;
    drop(file);
    std::fs::rename(&tmp_path, &sidecar)
        .with_context(|| format!("runmark: rename {} → {}", tmp_path.display(), sidecar.display()))?;
    // fsync del dir (mejor esfuerzo; no falla el bench).
    if let Some(dir) = parent {
        if let Ok(d) = std::fs::File::open(dir) {
            let _ = d.sync_all();
        }
    }
    Ok(())
}

/// BLAKE3 hex de un string (dataset_sha, digests best-effort).
pub fn sha_of_str(s: &str) -> String {
    blake3::hash(s.as_bytes()).to_hex().to_string()
}

#[cfg(test)]
mod runmark_tests {
    use super::*;

    fn sample() -> Runmark {
        Runmark {
            model: "ornith:9b".into(),
            digest: "sha256-abc".into(),
            backend: "ollama".into(),
            profile: "medium".into(),
            ctx: 16384,
            max_iter: 20,
            hops: 0,
            suite: "demo-12".into(),
            dataset_sha: "deadbeef".into(),
            timeout_cell: 60,
            seed: 42,
            git_head: "abc123".into(),
            dirty: false,
            memory: false,
            systemone: "off".into(),
            toolset: "Read,Edit,Write,Bash,Glob,Grep".into(),
        }
    }

    #[test]
    fn canonical_keys_sorted_and_stable() {
        let r = sample();
        let v = r.canonical_json();
        let obj = v.as_object().unwrap();
        let keys: Vec<&String> = obj.keys().collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted, "claves canónicas ordenadas");
        assert_eq!(
            keys.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            vec![
                "backend", "ctx", "dataset_sha", "digest", "dirty", "git_head", "hops",
                "max_iter", "memory", "model", "profile", "seed", "suite", "systemone",
                "timeout_cell", "toolset"
            ]
        );
        // Estable: dos serializaciones idénticas.
        assert_eq!(r.canonical_bytes(), r.canonical_bytes());
        assert_eq!(r.fingerprint(), r.fingerprint());
        assert_eq!(r.fingerprint().len(), 64);
    }

    #[test]
    fn fingerprint_changes_when_any_axis_changes() {
        let base = sample().fingerprint();
        let mut r = sample();
        r.hops = 1;
        assert_ne!(r.fingerprint(), base);
        let mut r = sample();
        r.timeout_cell = 61;
        assert_ne!(r.fingerprint(), base);
        let mut r = sample();
        r.memory = true;
        assert_ne!(r.fingerprint(), base);
    }

    #[test]
    fn sidecar_tmp_fsync_rename_and_rerun_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("bench.json");
        std::fs::write(&out, "{}").unwrap();
        let r = sample();
        write_sidecar_atomic(&out, &r).unwrap();
        let sidecar = sidecar_path(&out);
        assert!(sidecar.exists(), "sidecar creado");
        let fp = r.fingerprint();
        assert_eq!(load_existing_fingerprint(&sidecar).as_deref(), Some(fp.as_str()));
        // Misma celda ⇒ rechazo antes del LLM.
        assert!(check_rerun_rejected(&out, &fp).is_err());
        // Distinta celda ⇒ Ok.
        let mut other = sample();
        other.hops = 2;
        assert!(check_rerun_rejected(&out, &other.fingerprint()).is_ok());
        // Sin sidecar ⇒ Ok.
        let fresh = dir.path().join("fresh.json");
        assert!(check_rerun_rejected(&fresh, &fp).is_ok());
    }

    #[test]
    fn exit_codes_are_distinct_contract() {
        assert_eq!(EXIT_PASS, 0);
        assert_eq!(EXIT_GATE_FAIL, 1);
        assert_eq!(EXIT_INCONCLUSIVE, 2);
        assert_eq!(EXIT_INTEGRITY, 3);
    }
}

//! runmark — fingerprint BLAKE3 + sidecar + exit codes (Capa 0, Lane B).
//!
//! Fingerprint canónico de celda bench sobre claves ordenadas:
//! `(model,digest,backend,profile,ctx,max_iter,hops,suite,dataset_sha,
//! timeout_cell,seed,rep,git_head,dirty,memory,systemone,toolset)`.
//! El JSON canónico usa `BTreeMap` (claves ordenadas byte a byte); el
//! fingerprint es `blake3(canonical_bytes)` en hex.
//!
//! `rep` es el eje de repetición (hueco para k=2 futuro: hoy siempre 0,
//! ppc sigue k=1 — NO ejecuta k=2; cambiar `rep` cambia el fingerprint).
//!
//! Sidecar `<out>.runmark.json`: escritura tmp+fsync+rename (atómica en el
//! mismo dir, solo caché legible). La decisión rerun sí/no vive en el ledger
//! SQLite (`bench/ledger.rs`, tabla `runs`, `UNIQUE(fingerprint,arm,rep)`):
//! `check_rerun_rejected` delega al ledger (una sola fuente de verdad).
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
/// `bench/mod.rs` (todos usan estas consts).
///
/// Mapping único (todos los runners usan estas consts, sin literales):
/// - `ppc::decide_exit`: `Signal`→0, `NoSignal(k<3)`→2, `NoSignal`+tasks<3→2,
///   resto (ledger sucio / CI solapa con tareas suficientes)→1; integridad→3.
///   k=1 ⇒ siempre NoSignal; el exit depende de la razón (k<3/n<3 ⇒ 2,
///   ledger sucio/solapa ⇒ 1).
/// - `demo12`: 1 rep (k=1) siempre NoSignal → 2 (incluso con ledger sucio;
///   demo12 no usa `decide_exit`); integridad→3.
/// - `bench/mod.rs` genérico: `NoSignal(k<3)` (single-rep, reps=1)→2 aunque
///   el gate pase; si no, `bench_exit_code` (0/1/2); integridad→3.
/// - `tb2::run_tb2_official`: gate 50% → 0 pass / 1 gate-fail (sin `Signal`:
///   es wrapper Harbor, no serie k>=3; el 2 no aplica a su gate).
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

/// F2a E4 / F1b: features del backend declaradas por celda (`kv_fp8`,
/// `response_cache`, `kv_retention`, `paged_mem_config`). Misma forma que
/// `inferperf::InferperfFeatures` (Lane D) para que la comparación de líneas
/// sea auditable: booleanos para flags, strings para políticas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendFeatures {
    /// KV-cache cuantizado a FP8 en el servidor (Oilamma `OLLAMA_KV_CACHE_TYPE`).
    pub kv_fp8: bool,
    /// Response cache idempotente del harness (E3, opt-in).
    pub response_cache: bool,
    /// Retención KV entre runs (E4/F2c: `off` hasta pasar el test de aislamiento).
    pub kv_retention: String,
    /// Config de memoria paginada del backend embebido (kitty-infer) o `n/a`.
    pub paged_mem_config: String,
}

impl BackendFeatures {
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
    /// `response_cache` se resuelve por separado (opt-in env); aquí solo la
    /// forma por backend.
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

    /// Con el flag de response cache ya resuelto (env/opt-in).
    pub fn for_bench(backend: &str, response_cache: bool) -> Self {
        let mut f = Self::for_backend(backend);
        f.response_cache = response_cache;
        f
    }

    /// JSON canónico (claves ordenadas por BTreeMap del llamador).
    pub fn canonical_json(&self) -> serde_json::Value {
        serde_json::json!({
            "kv_fp8": self.kv_fp8,
            "kv_retention": self.kv_retention,
            "paged_mem_config": self.paged_mem_config,
            "response_cache": self.response_cache,
        })
    }
}

/// Fingerprint de celda bench (todos los ejes que cambian el resultado,
/// incluido `rep`: cambiar de rep cambia el fingerprint).
/// F2a E4: suma `backend_features` + `available_context_tokens` +
/// `terminal_reason` + `invalid_tool_calls` al canónico.
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
    /// Eje de repetición (hueco k=2 futuro; hoy siempre 0, ppc k=1).
    /// Single-rep ⇒ k<3 ⇒ `decide_signal` siempre NoSignal + exit 2.
    pub rep: u32,
    pub git_head: String,
    pub dirty: bool,
    pub memory: bool,
    pub systemone: String,
    pub toolset: String,
    /// F2a E4: features del backend (cambian el resultado).
    pub backend_features: BackendFeatures,
    /// F2a E4 / F1a: tokens de contexto disponibles para la celda.
    pub available_context_tokens: u32,
    /// F1b: razón terminal de la celda (`n/a` pre-run; el manifiesto
    /// post-run vuelca el agregado por ejercicio en `details`).
    pub terminal_reason: String,
    /// F2a E2: llamadas a tools inválidas acumuladas en la celda.
    pub invalid_tool_calls: u64,
}

impl Runmark {
    /// JSON canónico con claves ordenadas (BTreeMap ⇒ orden alfabético).
    pub fn canonical_json(&self) -> serde_json::Value {
        let mut m = BTreeMap::new();
        m.insert(
            "available_context_tokens".to_string(),
            serde_json::json!(self.available_context_tokens),
        );
        m.insert("backend".to_string(), serde_json::json!(self.backend));
        m.insert(
            "backend_features".to_string(),
            self.backend_features.canonical_json(),
        );
        m.insert("ctx".to_string(), serde_json::json!(self.ctx));
        m.insert("dataset_sha".to_string(), serde_json::json!(self.dataset_sha));
        m.insert("digest".to_string(), serde_json::json!(self.digest));
        m.insert("dirty".to_string(), serde_json::json!(self.dirty));
        m.insert("git_head".to_string(), serde_json::json!(self.git_head));
        m.insert("hops".to_string(), serde_json::json!(self.hops));
        m.insert(
            "invalid_tool_calls".to_string(),
            serde_json::json!(self.invalid_tool_calls),
        );
        m.insert("max_iter".to_string(), serde_json::json!(self.max_iter));
        m.insert("memory".to_string(), serde_json::json!(self.memory));
        m.insert("model".to_string(), serde_json::json!(self.model));
        m.insert("profile".to_string(), serde_json::json!(self.profile));
        m.insert("rep".to_string(), serde_json::json!(self.rep));
        m.insert("seed".to_string(), serde_json::json!(self.seed));
        m.insert("suite".to_string(), serde_json::json!(self.suite));
        m.insert("systemone".to_string(), serde_json::json!(self.systemone));
        m.insert(
            "terminal_reason".to_string(),
            serde_json::json!(self.terminal_reason),
        );
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
/// Hoy solo caché legible + verificación de escritura (la decisión rerun
/// vive en el ledger); ejercitado en tests.
#[allow(dead_code)]
pub fn load_existing_fingerprint(sidecar: &Path) -> Option<String> {
    let text = std::fs::read_to_string(sidecar).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    v.get("fingerprint").and_then(|f| f.as_str()).map(|s| s.to_string())
}

/// Rechaza rerun de la misma celda ANTES del LLM (F1).
///
/// DELEGA AL LEDGER (una sola fuente de verdad): abre
/// `ledger::ledger_db_path(out)` y rechaza si ya existe una fila con el
/// mismo fingerprint. El sidecar (`<out>.runmark.json`) es solo caché
/// legible — NO decide. La reserva atómica real la hace `Ledger::begin`
/// (UNIQUE `fingerprint,arm,rep`) en los runners antes del primer LLM.
pub fn check_rerun_rejected(out: &Path, fingerprint: &str) -> Result<()> {
    let db = super::ledger::ledger_db_path(out);
    // Sin ledger aún ⇒ sin filas ⇒ Ok (primera corrida crea el ledger en
    // `Ledger::begin`). Abrir aquí crea el fichero vacío: barato y local.
    let ledger = match super::ledger::Ledger::open(&db) {
        Ok(l) => l,
        Err(e) => {
            // Fail-closed solo si el path existe pero no abre (corrupto);
            // si el dir no es escribible el begin del runner fallará igual.
            anyhow::bail!(
                "runmark: no se pudo abrir el ledger {}: {e:#}",
                db.display()
            )
        }
    };
    let dup = ledger
        .exists_fingerprint(fingerprint)
        .with_context(|| format!("runmark: consultar ledger {}", db.display()))?;
    if dup {
        anyhow::bail!(
            "runmark: misma celda ya registrada en {} (fingerprint {fingerprint}); \
             rerun rechazado antes del LLM — borre el ledger o cambie --out",
            db.display()
        );
    }
    Ok(())
}

/// Rechazo exacto por celda (`fingerprint,arm,rep`): lo que `Ledger::begin`
/// impone vía UNIQUE. El wrapper `check_rerun_rejected` (solo fingerprint)
/// es más estricto a propósito (cualquier arm/rep con igual fingerprint
/// rechaza: el fingerprint ya liga `rep`). Solo k=2 futuro + tests usan la
/// variante exacta hoy (los runners usan el wrapper pre-`begin`).
#[allow(dead_code)] // k=2 futuro (exact-triple); ejercitado en tests
pub fn check_rerun_rejected_cell(
    out: &Path,
    fingerprint: &str,
    arm: &str,
    rep: i64,
) -> Result<()> {
    let db = super::ledger::ledger_db_path(out);
    let ledger = super::ledger::Ledger::open(&db)
        .with_context(|| format!("runmark: abrir ledger {}", db.display()))?;
    let dup = ledger
        .exists(fingerprint, arm, rep)
        .with_context(|| format!("runmark: consultar ledger {}", db.display()))?;
    if dup {
        anyhow::bail!(
            "runmark: misma celda ya registrada en {} (fingerprint {fingerprint} arm {arm} rep {rep}); \
             rerun rechazado antes del LLM — borre el ledger o cambie --out",
            db.display()
        );
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
            rep: 0,
            git_head: "abc123".into(),
            dirty: false,
            memory: false,
            systemone: "off".into(),
            toolset: "Read,Edit,Write,Bash,Glob,Grep".into(),
            backend_features: BackendFeatures::for_bench("ollama", false),
            available_context_tokens: 16384,
            terminal_reason: "n/a".into(),
            invalid_tool_calls: 0,
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
                "available_context_tokens", "backend", "backend_features", "ctx",
                "dataset_sha", "digest", "dirty", "git_head", "hops", "invalid_tool_calls",
                "max_iter", "memory", "model", "profile", "rep", "seed", "suite", "systemone",
                "terminal_reason", "timeout_cell", "toolset"
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
    fn fingerprint_changes_with_backend_features() {
        // F2a E4: cada eje nuevo cambia el hash (evidencia Oracle).
        let base = sample().fingerprint();
        let mut r = sample();
        r.backend_features.response_cache = true;
        assert_ne!(r.fingerprint(), base, "response_cache debe cambiar el hash");
        let mut r = sample();
        r.backend_features.kv_fp8 = true;
        assert_ne!(r.fingerprint(), base, "kv_fp8 debe cambiar el hash");
        let mut r = sample();
        r.backend_features.kv_retention = "session".to_string();
        assert_ne!(r.fingerprint(), base, "kv_retention debe cambiar el hash");
        let mut r = sample();
        r.backend_features.paged_mem_config = "paged".to_string();
        assert_ne!(
            r.fingerprint(),
            base,
            "paged_mem_config debe cambiar el hash"
        );
        let mut r = sample();
        r.available_context_tokens = 8192;
        assert_ne!(
            r.fingerprint(),
            base,
            "available_context_tokens debe cambiar el hash"
        );
        let mut r = sample();
        r.terminal_reason = "text".to_string();
        assert_ne!(
            r.fingerprint(),
            base,
            "terminal_reason debe cambiar el hash"
        );
        let mut r = sample();
        r.invalid_tool_calls = 3;
        assert_ne!(
            r.fingerprint(),
            base,
            "invalid_tool_calls debe cambiar el hash"
        );
        // Forma canónica de backend_features (misma que inferperf Lane D).
        let v = sample().canonical_json();
        let bf = v.get("backend_features").expect("backend_features canónico");
        assert_eq!(bf.get("kv_fp8"), Some(&serde_json::json!(false)));
        assert_eq!(bf.get("response_cache"), Some(&serde_json::json!(false)));
        assert_eq!(bf.get("kv_retention"), Some(&serde_json::json!("off")));
        assert!(bf.get("paged_mem_config").is_some());
    }

    #[test]
    fn rep_en_fingerprint_cambia_hash() {
        // Eje rep (hueco k=2 futuro): rep 0 vs 1 ⇒ distinto fingerprint,
        // misma celda por lo demás. ppc sigue k=1 (rep siempre 0).
        let base = sample().fingerprint();
        let mut r = sample();
        r.rep = 1;
        assert_ne!(r.fingerprint(), base, "rep debe cambiar el hash");
        assert_eq!(r.fingerprint().len(), 64);
        // Canónico contiene rep.
        assert_eq!(
            sample().canonical_json().get("rep"),
            Some(&serde_json::json!(0))
        );
        assert_eq!(r.canonical_json().get("rep"), Some(&serde_json::json!(1)));
    }

    #[test]
    fn sidecar_tmp_fsync_rename_and_rerun_rejected() {
        // Hermético: el ledger lee KITTY_LEDGER_DB; sin lock otro test
        // (ledger_db_path_env) fijaría ese env en paralelo y este test
        // abriría el ledger equivocado ⇒ flaky. Mismo lock compartido.
        let _lock = super::super::ledger::test_ledger_env_lock();
        let _guard = super::super::ledger::LedgerEnvGuard::remove();
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("bench.json");
        std::fs::write(&out, "{}").unwrap();
        let r = sample();
        write_sidecar_atomic(&out, &r).unwrap();
        let sidecar = sidecar_path(&out);
        assert!(sidecar.exists(), "sidecar creado (caché legible)");
        let fp = r.fingerprint();
        assert_eq!(load_existing_fingerprint(&sidecar).as_deref(), Some(fp.as_str()));
        // El sidecar SOLO ya no decide: la fuente única es el ledger.
        // Sin fila en el ledger ⇒ Ok aunque el sidecar exista.
        assert!(check_rerun_rejected(&out, &fp).is_ok());
        // Tras reservar la celda en el ledger ⇒ rechazo antes del LLM.
        {
            use super::super::ledger::{Ledger, LedgerStamp};
            let db = super::super::ledger::ledger_db_path(&out);
            let ledger = Ledger::open(&db).unwrap();
            ledger
                .begin(&LedgerStamp {
                    fingerprint: fp.clone(),
                    suite: "demo-12".to_string(),
                    arm: "default".to_string(),
                    rep: 0,
                    git_head: "abc123".to_string(),
                    dirty: false,
                    started_at: "2026-10-24T00:00:00Z".to_string(),
                })
                .unwrap();
        }
        assert!(check_rerun_rejected(&out, &fp).is_err());
        // Distinta celda ⇒ Ok.
        let mut other = sample();
        other.hops = 2;
        assert!(check_rerun_rejected(&out, &other.fingerprint()).is_ok());
        // Celda exacta (fingerprint,arm,rep) ⇒ rechazo; distinto rep ⇒ Ok.
        assert!(check_rerun_rejected_cell(&out, &fp, "default", 0).is_err());
        assert!(check_rerun_rejected_cell(&out, &fp, "default", 1).is_ok());
        // Sin ledger ⇒ Ok.
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

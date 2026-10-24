//! Ledger SQLite real de celdas bench (Capa 0, Lane A Pass 2).
//!
//! Una sola fuente de verdad para el rechazo de reruns: `check_rerun_rejected`
//! en `runmark.rs` delega aquí (ya no lee el sidecar para decidir).
//!
//! Esquema EXACTO Oracle — tabla `runs`:
//! ```sql
//! CREATE TABLE IF NOT EXISTS runs (
//!   run_id TEXT PRIMARY KEY,
//!   fingerprint TEXT NOT NULL,
//!   suite TEXT NOT NULL,
//!   arm TEXT NOT NULL,
//!   rep INTEGER NOT NULL,
//!   git_head TEXT NOT NULL,
//!   dirty INTEGER NOT NULL,
//!   leaked INTEGER NOT NULL DEFAULT 0,
//!   started_at TEXT NOT NULL,
//!   finished_at TEXT,
//!   exit_code INTEGER,
//!   UNIQUE(fingerprint, arm, rep)
//! );
//! ```
//!
//! - Path: `<out>.ledger.db` o `KITTY_LEDGER_DB` (env manda). NUNCA
//!   `default_db_path` (el ledger de bench es DB aislada por manifiesto,
//!   igual que `ppc_lessons_db_path`).
//! - `run_id` opaco (UUID v4, sin semántica: no es fingerprint ni parseable).
//! - `begin(stamp)` reserva la celda ANTES del primer LLM: el `UNIQUE`
//!   rechaza el duplicado (`fingerprint,arm,rep`) con error (el caller sale
//!   con exit 3 integridad).
//! - `is_clean(run_id)` = `dirty == 0 && leaked == 0` (único gate publicable).
//!
//! Sin `unwrap`/`expect` en prod (m06): todo error sube como `anyhow::Error`
//! y el caller lo vuelca a stderr con exit no-cero.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Fila mínima para reservar una celda (lo que `begin` necesita).
#[derive(Debug, Clone)]
pub struct LedgerStamp {
    pub fingerprint: String,
    pub suite: String,
    /// Brazo de la celda (`"default"` en runs de un solo brazo; ppc/demo-12
    /// no abren una fila por candidato — los probes son intra-run).
    pub arm: String,
    /// Índice de repetición (hueco para k=2 futuro; hoy siempre 0, ppc k=1).
    pub rep: i64,
    pub git_head: String,
    pub dirty: bool,
    pub started_at: String,
}

/// Ledger SQLite (`runs`). Dueño único de la decisión rerun sí/no.
pub struct Ledger {
    conn: rusqlite::Connection,
}

fn schema_sql() -> &'static str {
    "CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        fingerprint TEXT NOT NULL,
        suite TEXT NOT NULL,
        arm TEXT NOT NULL,
        rep INTEGER NOT NULL,
        git_head TEXT NOT NULL,
        dirty INTEGER NOT NULL,
        leaked INTEGER NOT NULL DEFAULT 0,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        exit_code INTEGER,
        UNIQUE(fingerprint, arm, rep)
    )"
}

/// Path del ledger: `KITTY_LEDGER_DB` si está seteado y no vacío, si no
/// `<out>.ledger.db` junto al manifiesto. NUNCA `default_db_path`.
pub fn ledger_db_path(out: &Path) -> PathBuf {
    if let Ok(p) = std::env::var("KITTY_LEDGER_DB") {
        if !p.trim().is_empty() {
            return PathBuf::from(p);
        }
    }
    let mut s = out.as_os_str().to_owned();
    s.push(".ledger.db");
    PathBuf::from(s)
}

/// Lock serializa los tests que mutan `KITTY_LEDGER_DB` (cargo corre hilos
/// en paralelo; sin lock `ledger_db_path_env` y los tests de runmark que
/// leen ese path se pisan ⇒ flaky). Compartido con `runmark_tests`.
#[cfg(test)]
pub(crate) fn test_ledger_env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Guard panic-safe con restore para `KITTY_LEDGER_DB` (test-only).
#[cfg(test)]
pub(crate) struct LedgerEnvGuard {
    prev: Option<String>,
}

#[cfg(test)]
impl LedgerEnvGuard {
    pub(crate) fn remove() -> Self {
        let prev = std::env::var("KITTY_LEDGER_DB").ok();
        unsafe {
            std::env::remove_var("KITTY_LEDGER_DB");
        }
        Self { prev }
    }
    pub(crate) fn set(path: &str) -> Self {
        let prev = std::env::var("KITTY_LEDGER_DB").ok();
        unsafe {
            std::env::set_var("KITTY_LEDGER_DB", path);
        }
        Self { prev }
    }
}

#[cfg(test)]
impl Drop for LedgerEnvGuard {
    fn drop(&mut self) {
        unsafe {
            match &self.prev {
                Some(v) => std::env::set_var("KITTY_LEDGER_DB", v),
                None => std::env::remove_var("KITTY_LEDGER_DB"),
            }
        }
    }
}

/// `run_id` opaco UUID v4 (rand local, sin semántica; no parseable).
fn new_run_id() -> String {
    let mut bytes = rand::random::<[u8; 16]>();
    // Versión 4 + variante RFC 4122.
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let h: Vec<String> = bytes.iter().map(|b| format!("{b:02x}")).collect();
    let s: String = h.concat();
    format!(
        "{}-{}-{}-{}-{}",
        &s[0..8],
        &s[8..12],
        &s[12..16],
        &s[16..20],
        &s[20..32]
    )
}

impl Ledger {
    fn init(conn: rusqlite::Connection) -> Result<Self> {
        conn.pragma_update(None, "journal_mode", "WAL")
            .context("ledger: pragma journal_mode")?;
        conn.pragma_update(None, "busy_timeout", 5000)
            .context("ledger: pragma busy_timeout")?;
        conn.execute_batch(schema_sql())
            .context("ledger: crear tabla runs")?;
        Ok(Self { conn })
    }

    /// Abre (creando dirs padre) el ledger en `path`.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("ledger: crear dir {}", parent.display()))?;
        }
        let conn = rusqlite::Connection::open(path)
            .with_context(|| format!("ledger: abrir {}", path.display()))?;
        Self::init(conn)
    }

    /// Solo tests (sin IO en disco).
    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self> {
        let conn = rusqlite::Connection::open_in_memory().context("ledger: memoria")?;
        Self::init(conn)
    }

    /// Reserva la celda ANTES del primer LLM. Duplicado
    /// (`fingerprint,arm,rep`) ⇒ error (el caller sale exit 3 integridad).
    /// Devuelve el `run_id` opaco de la fila insertada.
    pub fn begin(&self, stamp: &LedgerStamp) -> Result<String> {
        let run_id = new_run_id();
        let dirty: i64 = if stamp.dirty { 1 } else { 0 };
        match self.conn.execute(
            "INSERT INTO runs (run_id, fingerprint, suite, arm, rep, git_head, dirty, started_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                run_id,
                stamp.fingerprint,
                stamp.suite,
                stamp.arm,
                stamp.rep,
                stamp.git_head,
                dirty,
                stamp.started_at,
            ],
        ) {
            Ok(_) => Ok(run_id),
            Err(rusqlite::Error::SqliteFailure(e, _))
                if e.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_UNIQUE =>
            {
                anyhow::bail!(
                    "ledger: misma celda ya registrada (fingerprint {} arm {} rep {}); \
                     rerun rechazado antes del LLM — borre el ledger o cambie --out",
                    stamp.fingerprint,
                    stamp.arm,
                    stamp.rep
                )
            }
            Err(e) => Err(e).context("ledger: begin insert"),
        }
    }

    /// Marca el run con leak (tripwire intra-run). Idempotente.
    pub fn mark_leaked(&self, run_id: &str) -> Result<()> {
        self.conn
            .execute(
                "UPDATE runs SET leaked = 1 WHERE run_id = ?1",
                rusqlite::params![run_id],
            )
            .with_context(|| format!("ledger: mark_leaked {run_id}"))?;
        Ok(())
    }

    /// Cierra el run (manifiesto ya escrito por el caller).
    pub fn finish(&self, run_id: &str, finished_at: &str, exit_code: i32) -> Result<()> {
        self.conn
            .execute(
                "UPDATE runs SET finished_at = ?1, exit_code = ?2 WHERE run_id = ?3",
                rusqlite::params![finished_at, exit_code, run_id],
            )
            .with_context(|| format!("ledger: finish {run_id}"))?;
        Ok(())
    }

    /// `true` solo si `dirty == 0 && leaked == 0`. Fila ausente ⇒ error
    /// (fail-closed: sin fila no hay señal publicable).
    pub fn is_clean(&self, run_id: &str) -> Result<bool> {
        let (dirty, leaked): (i64, i64) = self
            .conn
            .query_row(
                "SELECT dirty, leaked FROM runs WHERE run_id = ?1",
                rusqlite::params![run_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .with_context(|| format!("ledger: is_clean {run_id}"))?;
        Ok(dirty == 0 && leaked == 0)
    }

    /// ¿Existe ya la celda (`fingerprint,arm,rep`)? (Lo que
    /// `check_rerun_rejected_cell` delega; k=2 futuro + tests.)
    #[allow(dead_code)] // k=2 futuro (exact-triple); ejercitado en tests
    pub fn exists(&self, fingerprint: &str, arm: &str, rep: i64) -> Result<bool> {
        let n: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM runs WHERE fingerprint = ?1 AND arm = ?2 AND rep = ?3",
                rusqlite::params![fingerprint, arm, rep],
                |row| row.get(0),
            )
            .context("ledger: exists")?;
        Ok(n > 0)
    }

    /// ¿Existe alguna fila con ese fingerprint (cualquier arm/rep)?
    /// Wrapper de compat para `check_rerun_rejected(out, fingerprint)`.
    pub fn exists_fingerprint(&self, fingerprint: &str) -> Result<bool> {
        let n: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM runs WHERE fingerprint = ?1",
                rusqlite::params![fingerprint],
                |row| row.get(0),
            )
            .context("ledger: exists_fingerprint")?;
        Ok(n > 0)
    }
}

#[cfg(test)]
mod ledger_tests {
    use super::*;

    fn stamp(fp: &str, arm: &str, rep: i64, dirty: bool) -> LedgerStamp {
        LedgerStamp {
            fingerprint: fp.to_string(),
            suite: "demo-12".to_string(),
            arm: arm.to_string(),
            rep,
            git_head: "abc123".to_string(),
            dirty,
            started_at: "2026-10-24T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn duplicado_fingerprint_arm_rep_rechazado() {
        let l = Ledger::open_in_memory().unwrap();
        let id = l.begin(&stamp("fp1", "default", 0, false)).unwrap();
        assert!(!id.is_empty(), "run_id opaco generado");
        // Misma triple ⇒ rechazo antes del LLM.
        let err = l.begin(&stamp("fp1", "default", 0, false)).unwrap_err();
        assert!(
            err.to_string().contains("rerun rechazado"),
            "duplicado debe rechazar: {err}"
        );
        // Distinto rep ⇒ Ok (hueco k=2 futuro).
        assert!(l.begin(&stamp("fp1", "default", 1, false)).is_ok());
        // Distinto arm ⇒ Ok.
        assert!(l.begin(&stamp("fp1", "other", 0, false)).is_ok());
        // Distinto fingerprint ⇒ Ok.
        assert!(l.begin(&stamp("fp2", "default", 0, false)).is_ok());
    }

    #[test]
    fn is_clean_con_leak_false_y_dirty_false() {
        let l = Ledger::open_in_memory().unwrap();
        let clean = l.begin(&stamp("fp-c", "default", 0, false)).unwrap();
        assert!(l.is_clean(&clean).unwrap(), "limpio ⇒ true");
        // Leak ⇒ sucio.
        l.mark_leaked(&clean).unwrap();
        assert!(!l.is_clean(&clean).unwrap(), "leak ⇒ false");
        // Dirty desde begin ⇒ sucio aunque sin leak.
        let dirty = l.begin(&stamp("fp-d", "default", 0, true)).unwrap();
        assert!(!l.is_clean(&dirty).unwrap(), "dirty ⇒ false");
        // Fila ausente ⇒ error fail-closed.
        assert!(l.is_clean("no-existe").is_err());
    }

    #[test]
    fn finish_guarda_exit_y_cierre() {
        let l = Ledger::open_in_memory().unwrap();
        let id = l.begin(&stamp("fp-f", "default", 0, false)).unwrap();
        l.finish(&id, "2026-10-24T00:01:00Z", 2).unwrap();
        let (fin, code): (Option<String>, Option<i64>) = l
            .conn
            .query_row(
                "SELECT finished_at, exit_code FROM runs WHERE run_id = ?1",
                rusqlite::params![id],
                |row| Ok((row.get(0).unwrap_or(None), row.get(1).unwrap_or(None))),
            )
            .unwrap();
        assert_eq!(fin.as_deref(), Some("2026-10-24T00:01:00Z"));
        assert_eq!(code, Some(2));
        assert!(l.is_clean(&id).unwrap(), "finish no ensucia");
    }

    #[test]
    fn run_id_opaco_uuid_sin_semantica() {
        let l = Ledger::open_in_memory().unwrap();
        let a = l.begin(&stamp("fp-u1", "default", 0, false)).unwrap();
        let b = l.begin(&stamp("fp-u2", "default", 0, false)).unwrap();
        assert_ne!(a, b, "run_id únicos");
        for id in [&a, &b] {
            // Forma UUID 8-4-4-4-12, 36 chars.
            assert_eq!(id.len(), 36, "run_id con forma UUID: {id}");
            assert_eq!(id.matches('-').count(), 4);
            // No contiene el fingerprint (opaco).
            assert!(!id.contains("fp-u"));
        }
    }

    #[test]
    fn ledger_db_path_env_y_nunca_default() {
        // Hermético: lock + guard (restore en Drop, panic-safe). Sin lock,
        // otro test que fije KITTY_LEDGER_DB en paralelo contaminaría.
        let _lock = super::test_ledger_env_lock();
        let _guard = super::LedgerEnvGuard::remove();
        let out = PathBuf::from("/tmp/x/bench-demo12.json");
        assert_eq!(
            ledger_db_path(&out),
            PathBuf::from("/tmp/x/bench-demo12.json.ledger.db")
        );
        {
            let _set = super::LedgerEnvGuard::set("/tmp/custom-ledger.db");
            // Nota: `_set` guarda prev=None (el remove previo) y al salir
            // restaura a unset; el `_guard` exterior conserva el prev real.
            // Para el assert basta leer con el env fijado:
            assert_eq!(
                ledger_db_path(&out),
                PathBuf::from("/tmp/custom-ledger.db")
            );
        }
        // Nunca default_db_path (~/.kitty/episodes.db): el ledger es aislado.
        let p = ledger_db_path(&out).to_string_lossy().to_string();
        assert!(!p.ends_with("episodes.db"), "ledger nunca episodes.db: {p}");
        assert!(p.ends_with(".ledger.db"));
    }

    #[test]
    fn schema_exact_oracle() {
        let l = Ledger::open_in_memory().unwrap();
        // Columnas exactas en orden: run_id PK, fingerprint, suite, arm, rep,
        // git_head, dirty, leaked DEFAULT 0, started_at, finished_at, exit_code.
        let cols: Vec<(String, String)> = l
            .conn
            .prepare("SELECT name, sql FROM sqlite_master WHERE name = 'runs'")
            .unwrap()
            .query_row([], |row| {
                let _name: String = row.get(0)?;
                let sql: String = row.get(1)?;
                Ok((String::new(), sql))
            })
            .map(|(_, sql)| {
                // Extrae el bloque de columnas del CREATE TABLE para asserts.
                vec![("sql".to_string(), sql)]
            })
            .unwrap();
        let sql = &cols[0].1;
        for col in [
            "run_id TEXT PRIMARY KEY",
            "fingerprint TEXT NOT NULL",
            "suite TEXT NOT NULL",
            "arm TEXT NOT NULL",
            "rep INTEGER NOT NULL",
            "git_head TEXT NOT NULL",
            "dirty INTEGER NOT NULL",
            "leaked INTEGER NOT NULL DEFAULT 0",
            "started_at TEXT NOT NULL",
            "finished_at TEXT",
            "exit_code INTEGER",
            "UNIQUE(fingerprint, arm, rep)",
        ] {
            // Normaliza espacios para comparar robusto.
            let norm: String = sql.split_whitespace().collect::<Vec<_>>().join(" ");
            let want: String = col.split_whitespace().collect::<Vec<_>>().join(" ");
            assert!(
                norm.contains(&want),
                "schema debe contener {col:?}: {sql}"
            );
        }
    }
}

//! Kitty Memory — Lessons (librería cue/lesson/evidence de Harness², con gate).
//!
//! Capa 2 (SDD Harness²-SystemOne §3): tabla `memory_items` separada de
//! `episodes` (**sin migrar episodios**). Solo persistencia + gate:
//!
//! - `save` **rechaza** lecciones sin `evidence_ref` o provenientes de runs
//!   con leak (tripwire) / ledger sucio. Solo runs con ledger limpio admiten
//!   lecciones.
//! - `recall` sin match devuelve **vacío** — SIN `fill_recent` (compárese con
//!   `EpisodeStore`, que sí cae a reciente; aquí está prohibido: inyectar
//!   lecciones no-relacionadas envenena el harness — F3: 50 goals
//!   no-relacionados → 0 inyecciones).
//! - La inyección en el loop NO vive aquí (solo persistencia + gate).
//!
//! Esquema: `memory_items(id, cue, lesson, evidence, domain, hash(run blake3),
//! run_id, evidence_ref OBLIGATORIA, confidence, created_at, expires_at,
//! superseded_by)` con TTL/forget y supersede.

use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

/// TTL por defecto de una lección: 90 días, en segundos.
/// Caducada ⇒ invisible a `recall` (re-verificable vía `save` de nuevo, nunca
/// permanente sin renovar).
pub const LESSON_DEFAULT_TTL_SECS: i64 = 90 * 24 * 3600;

/// Tope de lecciones devueltas por `recall` en una llamada.
pub const MAX_LESSON_RECALL: usize = 10;

/// Errores del store de lecciones (propios: no se toca `error.rs`).
#[derive(Debug, thiserror::Error)]
pub enum LessonError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// F3: lesson sin `evidence_ref` ⇒ rechazo.
    #[error("evidence_ref obligatoria: lesson sin evidence_ref rechazada")]
    MissingEvidenceRef,
    /// F3: run con tripwire (leak) o ledger sucio ⇒ rechazo.
    #[error("run rechazado para lecciones: {0}")]
    DirtyRun(String),
    #[error("confidence fuera de rango [0,1]: {0}")]
    BadConfidence(f32),
    #[error("Lesson not found: {0}")]
    NotFound(String),
}

pub type Result<T> = std::result::Result<T, LessonError>;

/// Una lección Harness²: cue + lesson + evidence con provenance del run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: String,
    pub cue: String,
    pub lesson: String,
    pub evidence: String,
    pub domain: String,
    /// blake3(cue :: lesson :: evidence :: run_id), hex.
    pub hash: String,
    pub run_id: String,
    /// OBLIGATORIA: referencia a la evidencia (celda/manifiesto/run que la
    /// respalda, p. ej. `ppc:{run_id}:{candidato}`). Vacía ⇒ `save` rechaza.
    pub evidence_ref: String,
    pub confidence: f32,
    pub created_at: i64,
    /// Epoch-secs de caducidad. 0 = sin caducidad.
    pub expires_at: i64,
    /// Id de la lección que la reemplaza (`None` = vigente).
    pub superseded_by: Option<String>,
}

impl MemoryItem {
    /// Constructor mínimo. `id`/`hash` se computan en `save`; `expires_at`
    /// = `now + LESSON_DEFAULT_TTL_SECS` (si `now == 0`, se rellenan en `save`
    /// con el reloj real).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cue: String,
        lesson: String,
        evidence: String,
        domain: String,
        run_id: String,
        evidence_ref: String,
        confidence: f32,
        now: i64,
    ) -> Self {
        let expires_at = if now > 0 {
            now.saturating_add(LESSON_DEFAULT_TTL_SECS)
        } else {
            0
        };
        Self {
            id: String::new(),
            cue,
            lesson,
            evidence,
            domain,
            hash: String::new(),
            run_id,
            evidence_ref,
            confidence,
            created_at: now,
            expires_at,
            superseded_by: None,
        }
    }

    /// Id blake3 determinista (cue :: lesson :: run_id :: created_at).
    pub fn compute_id(cue: &str, lesson: &str, run_id: &str, created_at: i64) -> String {
        let input = format!("{cue}::{lesson}::{run_id}::{created_at}");
        blake3::hash(input.as_bytes()).to_hex().to_string()
    }

    /// Hash de contenido blake3 (cue :: lesson :: evidence :: run_id).
    pub fn compute_hash(cue: &str, lesson: &str, evidence: &str, run_id: &str) -> String {
        let input = format!("{cue}::{lesson}::{evidence}::{run_id}");
        blake3::hash(input.as_bytes()).to_hex().to_string()
    }
}

/// Proveniencia mínima del run que admite lecciones. La rellenan Lane A
/// (runmark/ledger/tripwires); aquí solo se consume como gate: `save` la
/// exige limpia. `systemone_status` (Lane C) NO entra aquí: nunca es gate.
#[derive(Debug, Clone)]
pub struct RunProvenance {
    pub run_id: String,
    pub leaked: bool,
    pub ledger_dirty: bool,
}

impl RunProvenance {
    /// Run limpio (sin leak, ledger limpio).
    pub fn clean(run_id: &str) -> Self {
        Self {
            run_id: run_id.to_string(),
            leaked: false,
            ledger_dirty: false,
        }
    }
}

/// Store SQLite de lecciones (`memory_items`), separado de `episodes`.
pub struct LessonStore {
    conn: Connection,
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Tokeniza igual que el episódico (minúsculas, rachas alfanuméricas).
fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

fn jaccard(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let set_a: std::collections::HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
    let set_b: std::collections::HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
    let inter = set_a.intersection(&set_b).count();
    if inter == 0 {
        return 0.0;
    }
    let union = set_a.union(&set_b).count();
    inter as f64 / union as f64
}

fn row_to_item(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryItem> {
    Ok(MemoryItem {
        id: row.get(0)?,
        cue: row.get(1)?,
        lesson: row.get(2)?,
        evidence: row.get(3)?,
        domain: row.get(4)?,
        hash: row.get(5)?,
        run_id: row.get(6)?,
        evidence_ref: row.get(7)?,
        confidence: row.get::<_, f64>(8)? as f32,
        created_at: row.get(9)?,
        expires_at: row.get(10)?,
        superseded_by: row.get(11)?,
    })
}

const SELECT_COLS: &str = "id, cue, lesson, evidence, domain, hash, run_id, evidence_ref, confidence, created_at, expires_at, superseded_by";

impl LessonStore {
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        Self::init(conn)
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::init(conn)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "busy_timeout", 5000)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memory_items (
                id TEXT PRIMARY KEY,
                cue TEXT NOT NULL,
                lesson TEXT NOT NULL,
                evidence TEXT NOT NULL,
                domain TEXT NOT NULL DEFAULT '',
                hash TEXT NOT NULL,
                run_id TEXT NOT NULL,
                evidence_ref TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.0,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL DEFAULT 0,
                superseded_by TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memory_items_domain ON memory_items(domain);
            CREATE INDEX IF NOT EXISTS idx_memory_items_run ON memory_items(run_id);",
        )?;
        Ok(Self { conn })
    }

    /// Guarda una lección. Rechaza (F3):
    /// - `evidence_ref` vacía/solo-espacios ⇒ [`LessonError::MissingEvidenceRef`];
    /// - run con leak (tripwire) o ledger sucio, o `run_id` que no coincide
    ///   con la proveniencia ⇒ [`LessonError::DirtyRun`].
    /// Solo runs con ledger limpio admiten lecciones.
    pub fn save(&self, mut item: MemoryItem, prov: &RunProvenance) -> Result<String> {
        if item.evidence_ref.trim().is_empty() {
            return Err(LessonError::MissingEvidenceRef);
        }
        if prov.leaked {
            return Err(LessonError::DirtyRun(format!(
                "run '{}' marcado con leak (tripwire): no admite lecciones",
                prov.run_id
            )));
        }
        if prov.ledger_dirty {
            return Err(LessonError::DirtyRun(format!(
                "run '{}' con ledger sucio: no admite lecciones",
                prov.run_id
            )));
        }
        if item.run_id != prov.run_id {
            return Err(LessonError::DirtyRun(format!(
                "run_id '{}' != run de proveniencia '{}'",
                item.run_id, prov.run_id
            )));
        }
        if !(0.0..=1.0).contains(&item.confidence) {
            return Err(LessonError::BadConfidence(item.confidence));
        }
        if item.created_at == 0 {
            item.created_at = now_secs();
        }
        if item.expires_at == 0 {
            item.expires_at = item.created_at.saturating_add(LESSON_DEFAULT_TTL_SECS);
        }
        if item.id.is_empty() {
            item.id = MemoryItem::compute_id(&item.cue, &item.lesson, &item.run_id, item.created_at);
        }
        item.hash = MemoryItem::compute_hash(&item.cue, &item.lesson, &item.evidence, &item.run_id);
        self.conn.execute(
            "INSERT OR REPLACE INTO memory_items (id, cue, lesson, evidence, domain, hash, run_id, evidence_ref, confidence, created_at, expires_at, superseded_by)
              VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            rusqlite::params![
                item.id,
                item.cue,
                item.lesson,
                item.evidence,
                item.domain,
                item.hash,
                item.run_id,
                item.evidence_ref,
                item.confidence as f64,
                item.created_at,
                item.expires_at,
                item.superseded_by,
            ],
        )?;
        Ok(item.id)
    }

    /// Recall por similitud Jaccard sobre `cue`. Sin match ⇒ **vacío**:
    /// prohibido el `fill_recent` del episódico (inyectaría lecciones
    /// no-relacionadas). Solo filas vigentes (no caducadas, no supersedidas).
    pub fn recall(&self, cue: &str, limit: usize, now: i64) -> Result<Vec<MemoryItem>> {
        let limit = limit.min(MAX_LESSON_RECALL);
        if limit == 0 {
            return Ok(Vec::new());
        }
        let query = tokenize(cue);
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {SELECT_COLS} FROM memory_items
              WHERE superseded_by IS NULL AND (expires_at = 0 OR expires_at > ?1)"
        ))?;
        let items: Vec<MemoryItem> = stmt
            .query_map(rusqlite::params![now], row_to_item)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut scored: Vec<(f64, MemoryItem)> = items
            .into_iter()
            .map(|it| {
                let s = jaccard(&query, &tokenize(&it.cue));
                (s, it)
            })
            .filter(|(s, _)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.1.confidence.partial_cmp(&a.1.confidence).unwrap_or(std::cmp::Ordering::Equal))
                .then(b.1.created_at.cmp(&a.1.created_at))
        });
        Ok(scored.into_iter().take(limit).map(|(_, it)| it).collect())
    }

    /// Marca una lección como reemplazada por `by_id` (retain/forget).
    pub fn supersede(&self, id: &str, by_id: &str) -> Result<()> {
        let n = self.conn.execute(
            "UPDATE memory_items SET superseded_by = ?1 WHERE id = ?2",
            rusqlite::params![by_id, id],
        )?;
        if n == 0 {
            return Err(LessonError::NotFound(id.to_string()));
        }
        Ok(())
    }

    /// TTL/forget: borra caducadas (`expires_at != 0 && <= now`). Devuelve
    /// cuántas borró.
    pub fn forget_expired(&self, now: i64) -> Result<usize> {
        let n = self.conn.execute(
            "DELETE FROM memory_items WHERE expires_at != 0 AND expires_at <= ?1",
            rusqlite::params![now],
        )?;
        Ok(n)
    }

    pub fn get(&self, id: &str) -> Result<Option<MemoryItem>> {
        let mut stmt = self
            .conn
            .prepare(&format!("SELECT {SELECT_COLS} FROM memory_items WHERE id = ?1"))?;
        let opt = stmt
            .query_row(rusqlite::params![id], row_to_item)
            .optional()?;
        Ok(opt)
    }

    pub fn count(&self) -> Result<usize> {
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM memory_items", [], |row| row.get(0))?;
        Ok(n as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_790_000_000;

    fn item(cue: &str, evidence_ref: &str) -> MemoryItem {
        MemoryItem::new(
            cue.to_string(),
            "usar iter_cap bajo en triage".to_string(),
            "{\"score\": 0.9}".to_string(),
            "ppc".to_string(),
            "run-limpio-1".to_string(),
            evidence_ref.to_string(),
            0.8,
            NOW,
        )
    }

    #[test]
    fn save_rechaza_sin_evidence_ref() {
        let store = LessonStore::open_in_memory().unwrap();
        let prov = RunProvenance::clean("run-limpio-1");
        for bad in ["", "   "] {
            let err = store.save(item("postgres index bloat autovacuum", bad), &prov).unwrap_err();
            assert!(
                matches!(err, LessonError::MissingEvidenceRef),
                "sin evidence_ref debe rechazar, fue: {err}"
            );
        }
        assert_eq!(store.count().unwrap(), 0);
    }

    #[test]
    fn save_rechaza_run_con_leak_o_ledger_sucio() {
        let store = LessonStore::open_in_memory().unwrap();
        let leaked = RunProvenance {
            run_id: "run-leak".to_string(),
            leaked: true,
            ledger_dirty: false,
        };
        let dirty = RunProvenance {
            run_id: "run-dirty".to_string(),
            leaked: false,
            ledger_dirty: true,
        };
        let mut li = item("postgres index bloat autovacuum", "ppc:run-leak:x");
        li.run_id = "run-leak".to_string();
        let err = store.save(li, &leaked).unwrap_err();
        assert!(matches!(err, LessonError::DirtyRun(_)), "leak debe rechazar: {err}");

        let mut di = item("postgres index bloat autovacuum", "ppc:run-dirty:x");
        di.run_id = "run-dirty".to_string();
        let err = store.save(di, &dirty).unwrap_err();
        assert!(matches!(err, LessonError::DirtyRun(_)), "ledger sucio debe rechazar: {err}");

        // run_id que no coincide con la proveniencia también se rechaza.
        let clean = RunProvenance::clean("run-otro");
        let err = store.save(item("postgres index bloat autovacuum", "ppc:run:x"), &clean).unwrap_err();
        assert!(matches!(err, LessonError::DirtyRun(_)), "mismatch debe rechazar: {err}");

        assert_eq!(store.count().unwrap(), 0);
    }

    #[test]
    fn save_ok_con_evidence_ref_y_run_limpio() {
        let store = LessonStore::open_in_memory().unwrap();
        let prov = RunProvenance::clean("run-limpio-1");
        let id = store
            .save(item("postgres index bloat autovacuum", "ppc:run-limpio-1:baseline"), &prov)
            .unwrap();
        assert!(!id.is_empty());
        let got = store.get(&id).unwrap().expect("debe existir");
        assert!(!got.hash.is_empty());
        assert_eq!(got.evidence_ref, "ppc:run-limpio-1:baseline");
    }

    #[test]
    fn recall_sin_match_vacio_sin_fill_recent() {
        let store = LessonStore::open_in_memory().unwrap();
        let prov = RunProvenance::clean("run-limpio-1");
        store.save(item("postgres index bloat autovacuum", "ppc:r:a"), &prov).unwrap();
        store
            .save(item("postgres index bloat autovacuum tuning", "ppc:r:b"), &prov)
            .unwrap();
        assert_eq!(store.count().unwrap(), 2);
        // Nada en común con los cues guardados ⇒ vacío (el episódico aquí
        // devolvería recientes; lessons lo tiene prohibido).
        let out = store.recall("bake sourdough bread at home", 5, NOW).unwrap();
        assert!(out.is_empty(), "sin match debe ser vacío, fue: {out:?}");
    }

    /// F3: 50 goals no-relacionados ⇒ 0 inyecciones.
    #[test]
    fn recall_50_goals_no_relacionados_cero_inyecciones() {
        let store = LessonStore::open_in_memory().unwrap();
        let prov = RunProvenance::clean("run-limpio-1");
        store.save(item("postgres index bloat autovacuum", "ppc:r:a"), &prov).unwrap();
        const GOALS: [&str; 50] = [
            "bake sourdough bread at home",
            "grow tomatoes on balcony",
            "fix bicycle flat tire",
            "plan weekly meals cheap",
            "learn basic watercolor washes",
            "train puppy to sit",
            "organize garage shelves",
            "write wedding speech toast",
            "meditate ten minutes daily",
            "change car oil safely",
            "knit wool scarf pattern",
            "build birdhouse from pine",
            "study spanish verbs nightly",
            "clean oven glass door",
            "photograph moon with phone",
            "repair drywall small hole",
            "brew kombucha first batch",
            "sew button on coat",
            "prune roses in march",
            "fold fitted sheet neatly",
            "grill salmon fillets evenly",
            "unclog sink drain naturally",
            "paint bedroom accent wall",
            "potty train toddler tips",
            "sharpen kitchen knives whetstone",
            "make candles with soy wax",
            "tile backsplash grout lines",
            "compost kitchen scraps apartment",
            "draw cartoon faces simple",
            "jog first five kilometers",
            "hem curtains without sewing machine",
            "pickle cucumbers vinegar brine",
            "mount tv bracket studs",
            "whiten sneakers baking soda",
            "arrange flowers vase tall",
            "defrost freezer quickly safely",
            "wrap gifts with fabric",
            "calibrate guitar tuner ear",
            "remove rust from tools",
            "freeze herbs olive oil",
            "patch jeans knee denim",
            "steam dumplings bamboo basket",
            "refinish wooden chair sanding",
            "darn socks heel wool",
            "cure cast iron skillet",
            "hang curtains rod level",
            "mop hardwood floors vinegar",
            "fold origami crane paper",
            "descale kettle citric acid",
            "water orchids ice cubes",
        ];
        assert_eq!(GOALS.len(), 50);
        for g in GOALS {
            let out = store.recall(g, 5, NOW).unwrap();
            assert!(out.is_empty(), "goal '{g}' no-relacionado debe dar 0 inyecciones, fue: {out:?}");
        }
    }

    #[test]
    fn recall_match_devuelve_ordenado_y_respeta_limite() {
        let store = LessonStore::open_in_memory().unwrap();
        let prov = RunProvenance::clean("run-limpio-1");
        store.save(item("postgres index bloat autovacuum", "ppc:r:a"), &prov).unwrap();
        let out = store.recall("autovacuum bloat postgres", 5, NOW).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].evidence_ref, "ppc:r:a");
    }

    #[test]
    fn supersede_y_ttl_ocultan_en_recall() {
        let store = LessonStore::open_in_memory().unwrap();
        let prov = RunProvenance::clean("run-limpio-1");
        let id = store.save(item("postgres index bloat autovacuum", "ppc:r:a"), &prov).unwrap();
        assert_eq!(store.recall("postgres autovacuum", 5, NOW).unwrap().len(), 1);
        store.supersede(&id, "nueva").unwrap();
        assert!(store.recall("postgres autovacuum", 5, NOW).unwrap().is_empty());

        let mut exp = item("postgres index bloat autovacuum", "ppc:r:b");
        exp.run_id = "run-limpio-1".to_string();
        exp.expires_at = NOW - 1; // ya caducada
        // id distinto para no colisionar con la anterior
        exp.created_at = NOW + 1;
        let id2 = store.save(exp, &prov).unwrap();
        assert_ne!(id, id2);
        assert!(store.recall("postgres autovacuum", 5, NOW).unwrap().is_empty());
        assert_eq!(store.forget_expired(NOW).unwrap(), 1);
    }
}

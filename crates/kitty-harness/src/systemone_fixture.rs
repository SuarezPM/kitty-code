//! Fixture versionado para FUTURA calibración System-One (Lane C).
//!
//! Solo esquema + loader + ejemplos sintéticos. NO hay calibración real:
//! - Los 8 ejemplos van marcados `synthetic: true` y no derivan
//!   `(confidence, correct)` (los manifiestos no los derivan; Oracle).
//! - NUNCA se mide ECE sobre el train: la calibración futura medirá ECE
//!   en holdout por tarea/dominio (ver [`holdout_split`]).
//! - Puro `std` + `serde_json` (ya dependencia del crate); el loader de
//!   fichero es la única I/O y vive aquí, fuera de `decide`.
//!
//! Esquema JSON v1:
//! ```json
//! {
//!   "version": 1,
//!   "samples": [
//!     {"goal": "...", "outcome": true, "domain": "...",
//!      "task": "...", "synthetic": true}
//!   ]
//! }
//! ```

use serde::{Deserialize, Serialize};

/// Versión del esquema soportada por [`load_str`].
pub const FIXTURE_SCHEMA_VERSION: u32 = 1;

/// Una muestra de calibración futura: meta + desenlace + partición.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationSample {
    /// Meta / texto de la salida a puntuar.
    pub goal: String,
    /// Desenlace observado (`true` = done).
    pub outcome: bool,
    /// Dominio para holdout (p. ej. `"closer-smoke"`).
    pub domain: String,
    /// Tarea para holdout (p. ej. `"fixed-task"`).
    pub task: String,
    /// `true` ⇒ sintético, no evidencia real.
    #[serde(default)]
    pub synthetic: bool,
}

/// Fixture validado: versión + muestras no vacías.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationFixture {
    pub version: u32,
    pub samples: Vec<CalibrationSample>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RawFixture {
    version: u32,
    samples: Vec<CalibrationSample>,
}

/// Error del loader (rechaza corrupto sin pánico).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FixtureError {
    /// JSON inválido.
    Parse(String),
    /// `version` distinta de [`FIXTURE_SCHEMA_VERSION`].
    UnsupportedVersion { got: u32, want: u32 },
    /// Sin muestras.
    Empty,
    /// Muestra con `goal`, `domain` o `task` vacíos (índice incluido).
    MissingField { index: usize },
}

impl std::fmt::Display for FixtureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "fixture JSON inválido: {e}"),
            Self::UnsupportedVersion { got, want } => {
                write!(f, "fixture version={got} no soportada (want {want})")
            }
            Self::Empty => write!(f, "fixture sin muestras"),
            Self::MissingField { index } => {
                write!(f, "muestra {index} con goal/domain/task vacíos")
            }
        }
    }
}

impl std::error::Error for FixtureError {}

/// Carga y valida un fixture desde JSON en memoria. Puro, sin I/O.
pub fn load_str(json: &str) -> Result<CalibrationFixture, FixtureError> {
    let raw: RawFixture =
        serde_json::from_str(json).map_err(|e| FixtureError::Parse(e.to_string()))?;
    if raw.version != FIXTURE_SCHEMA_VERSION {
        return Err(FixtureError::UnsupportedVersion {
            got: raw.version,
            want: FIXTURE_SCHEMA_VERSION,
        });
    }
    if raw.samples.is_empty() {
        return Err(FixtureError::Empty);
    }
    for (i, s) in raw.samples.iter().enumerate() {
        if s.goal.trim().is_empty() || s.domain.trim().is_empty() || s.task.trim().is_empty() {
            return Err(FixtureError::MissingField { index: i });
        }
    }
    Ok(CalibrationFixture {
        version: raw.version,
        samples: raw.samples,
    })
}

/// Carga y valida un fixture desde fichero. Única I/O de este módulo.
pub fn load_file(path: &std::path::Path) -> Result<CalibrationFixture, FixtureError> {
    let text =
        std::fs::read_to_string(path).map_err(|e| FixtureError::Parse(e.to_string()))?;
    load_str(&text)
}

/// Fixture sintético embebido (8 ejemplos, todos `synthetic: true`).
/// NO es evidencia de calibración: solo fija el esquema para el futuro.
#[must_use]
pub fn synthetic_fixture() -> CalibrationFixture {
    // `load_str` sobre el JSON canónico: si el esquema cambia, aquí falla.
    load_str(SYNTHETIC_JSON).expect("SYNTHETIC_JSON debe ser un fixture v1 válido")
}

/// JSON canónico del fixture sintético (v1, 8 muestras).
pub const SYNTHETIC_JSON: &str = r#"{
  "version": 1,
  "samples": [
    {"goal": "task complete all tests pass solution saved", "outcome": true, "domain": "closer-smoke", "task": "fixed-task", "synthetic": true},
    {"goal": "done implementation finished verified ok", "outcome": true, "domain": "closer-smoke", "task": "fixed-task", "synthetic": true},
    {"goal": "finished successfully all checks passed", "outcome": true, "domain": "tool-output", "task": "fixed-task", "synthetic": true},
    {"goal": "completed output written tests green", "outcome": true, "domain": "tool-output", "task": "fixed-task", "synthetic": true},
    {"goal": "error failed traceback exception occurred", "outcome": false, "domain": "closer-smoke", "task": "fixed-task", "synthetic": true},
    {"goal": "timeout killed after retries exhausted", "outcome": false, "domain": "tool-output", "task": "fixed-task", "synthetic": true},
    {"goal": "partial progress blocked waiting input", "outcome": false, "domain": "tool-output", "task": "holdout-task", "synthetic": true},
    {"goal": "failed assertion mismatch needs fix", "outcome": false, "domain": "closer-smoke", "task": "holdout-task", "synthetic": true}
  ]
}"#;

/// Partición holdout por dominio: las muestras cuyo `domain` es
/// `holdout_domain` van al holdout; el resto al train.
///
/// La futura calibración medirá ECE SOLO en el holdout, nunca sobre el train.
#[must_use]
pub fn holdout_split(
    fixture: &CalibrationFixture,
    holdout_domain: &str,
) -> (Vec<CalibrationSample>, Vec<CalibrationSample>) {
    let mut train = Vec::new();
    let mut held = Vec::new();
    for s in &fixture.samples {
        if s.domain == holdout_domain {
            held.push(s.clone());
        } else {
            train.push(s.clone());
        }
    }
    (train, held)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sintetico_carga_y_todo_marcado() {
        let f = synthetic_fixture();
        assert_eq!(f.version, FIXTURE_SCHEMA_VERSION);
        assert!((5..=10).contains(&f.samples.len()));
        assert!(
            f.samples.iter().all(|s| s.synthetic),
            "todo ejemplo sintético va marcado synthetic:true"
        );
        assert!(f.samples.iter().all(|s| !s.goal.trim().is_empty()));
    }

    #[test]
    fn loader_rechaza_corrupto() {
        // Versión no soportada.
        let bad = r#"{"version": 999, "samples": [{"goal": "g", "outcome": true, "domain": "d", "task": "t", "synthetic": true}]}"#;
        assert!(matches!(
            load_str(bad),
            Err(FixtureError::UnsupportedVersion { got: 999, .. })
        ));
        // Vacío.
        assert!(matches!(
            load_str(r#"{"version": 1, "samples": []}"#),
            Err(FixtureError::Empty)
        ));
        // Campo vacío.
        let missing = r#"{"version": 1, "samples": [{"goal": "", "outcome": true, "domain": "d", "task": "t", "synthetic": true}]}"#;
        assert!(matches!(
            load_str(missing),
            Err(FixtureError::MissingField { .. })
        ));
        // JSON roto.
        assert!(matches!(load_str("{no json"), Err(FixtureError::Parse(_))));
    }

    #[test]
    fn holdout_por_dominio_particiona() {
        let f = synthetic_fixture();
        let (train, held) = holdout_split(&f, "tool-output");
        assert!(!train.is_empty() && !held.is_empty());
        assert_eq!(train.len() + held.len(), f.samples.len());
        assert!(held.iter().all(|s| s.domain == "tool-output"));
        assert!(train.iter().all(|s| s.domain != "tool-output"));
    }

    #[test]
    fn loader_acepta_fichero_valido() {
        let dir = std::env::temp_dir();
        let path = dir.join("kitty_systemone_fixture_test.json");
        std::fs::write(&path, SYNTHETIC_JSON).expect("write temp fixture");
        let f = load_file(&path).expect("fixture válido desde fichero");
        assert_eq!(f.samples.len(), 8);
        let _ = std::fs::remove_file(&path);
    }
}

//! W2 TUI: persistencia de sesiones para `resume`.
//!
//! Cada sesión es un JSON en `~/.kitty/sessions/<id>.json` con el modelo, la
//! ventana de contexto y el transcript (pares user/assistant). Al abrir la TUI
//! se puede listar (`list_sessions`) y cargar (`load_session`) la más reciente
//! u otra, y la TUI repuebla el historial de mensajes para que el agent tenga
//! contexto al primer turno.
//!
//! Decisiones:
//!   - ID = timestamp UNIX en segundos + sufijo corto, ordenable y legible.
//!   - Guardado tras cada turno completado (idempotente, barato).
//!   - Sin migraciones: todos los campos llevan `#[serde(default)]`.

use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Una entrada del transcript.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Entry {
    pub role: String,
    pub content: String,
}

/// Sesión persistida.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub model: String,
    pub context_window: usize,
    pub created_at: u64,
    #[serde(default)]
    pub entries: Vec<Entry>,
}

impl Session {
    pub fn new(model: &str, context_window: usize) -> Self {
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            id: format!("{created_at}-{}", &uuidish()[..6]),
            model: model.to_string(),
            context_window,
            created_at,
            entries: Vec::new(),
        }
    }

    pub fn push_user(&mut self, content: impl Into<String>) {
        self.entries.push(Entry {
            role: "user".into(),
            content: content.into(),
        });
    }

    pub fn push_assistant(&mut self, content: impl Into<String>) {
        self.entries.push(Entry {
            role: "assistant".into(),
            content: content.into(),
        });
    }

    /// Última línea del transcript, para el preview de la lista.
    pub fn preview(&self) -> String {
        self.entries
            .last()
            .map(|e| {
                let s = e.content.replace('\n', " ");
                if s.chars().count() > 60 {
                    let cut: String = s.chars().take(60).collect();
                    format!("{cut}…")
                } else {
                    s
                }
            })
            .unwrap_or_else(|| "(vacía)".into())
    }
}

/// Directorio de sesiones (`~/.kitty/sessions`).
pub fn sessions_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".kitty").join("sessions")
}

/// Guarda (o sobreescribe) una sesión. Crea el directorio si no existe.
pub fn save_session(session: &Session) -> std::io::Result<PathBuf> {
    let dir = sessions_dir();
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.json", session.id));
    let json = serde_json::to_string_pretty(session)?;
    fs::write(&path, json)?;
    Ok(path)
}

/// Lista las sesiones ordenadas por `created_at` descendente.
/// Un archivo corrupto se ignora (no rompe el arranque de la TUI).
pub fn list_sessions() -> Vec<Session> {
    let dir = sessions_dir();
    let mut out = Vec::new();
    let Ok(rd) = fs::read_dir(&dir) else {
        return out;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        if let Ok(json) = fs::read_to_string(&path) {
            if let Ok(s) = serde_json::from_str::<Session>(&json) {
                out.push(s);
            }
        }
    }
    out.sort_by_key(|s| std::cmp::Reverse(s.created_at));
    out
}

/// Id pseudoaleatorio corto (sin dep de `uuid`): sufijo de timestamp+pid.
fn uuidish() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    format!("{:x}{:x}", nanos, std::process::id())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Aislamos HOME en un tempdir para no tocar el ~/.kitty real.
    fn with_tmp_home<F: FnOnce(&str)>(f: F) {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().to_str().unwrap().to_string();
        // Env var global: serializamos con un lock por proceso de test.
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _g = LOCK.lock().unwrap();
        let prev = std::env::var("HOME").ok();
        std::env::set_var("HOME", &home);
        // Aíslo el directorio de sesiones para que un test no vea archivos
        // de otro (el HOME temporal es nuevo, pero el lock se comparte).
        let _ = fs::remove_dir_all(sessions_dir());
        f(&home);
        match prev {
            Some(p) => std::env::set_var("HOME", p),
            None => std::env::remove_var("HOME"),
        }
    }

    #[test]
    fn save_load_roundtrip() {
        with_tmp_home(|_| {
            let mut s = Session::new("qwen3:8b", 16384);
            s.push_user("hola");
            s.push_assistant("¿qué tal?");
            let path = save_session(&s).unwrap();
            assert!(path.exists());

            let loaded: Session = serde_json::from_str(
                &fs::read_to_string(sessions_dir().join(format!("{}.json", s.id))).unwrap(),
            )
            .unwrap();
            assert_eq!(loaded.model, "qwen3:8b");
            assert_eq!(loaded.entries.len(), 2);
            assert_eq!(loaded.entries[0].role, "user");
            assert_eq!(loaded.entries[1].content, "¿qué tal?");
        });
    }

    #[test]
    fn list_orders_by_created_desc() {
        with_tmp_home(|_| {
            let mut a = Session::new("m", 100);
            a.created_at = 100;
            let mut b = Session::new("m", 100);
            b.created_at = 200;
            save_session(&a).unwrap();
            save_session(&b).unwrap();
            let list = list_sessions();
            assert_eq!(list.len(), 2);
            assert_eq!(list[0].created_at, 200);
            assert_eq!(list[1].created_at, 100);
        });
    }

    #[test]
    fn corrupt_file_is_skipped_not_fatal() {
        with_tmp_home(|_| {
            let dir = sessions_dir();
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("bad.json"), "{ no es json").unwrap();
            let s = Session::new("m", 100);
            save_session(&s).unwrap();
            let list = list_sessions();
            assert_eq!(list.len(), 1, "el corrupto se ignora, el bueno cuenta");
        });
    }

    #[test]
    fn preview_truncates_long_content() {
        let mut s = Session::new("m", 100);
        s.push_user("x".repeat(200));
        let p = s.preview();
        assert!(p.chars().count() <= 62, "preview acotado: {p}");
        assert!(p.ends_with('…'));
    }

    #[test]
    fn empty_session_preview_is_placeholder() {
        let s = Session::new("m", 100);
        assert_eq!(s.preview(), "(vacía)");
    }
}

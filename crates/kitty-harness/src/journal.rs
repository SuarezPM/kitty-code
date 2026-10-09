//! Edit journal writer — harness side of W3 `kitty undo` (F3, Gate A).
//!
//! Format mirrors `kitty-cli/src/journal.rs` field-for-field
//! (`path, before, after, recorded_at` in that order) so `kitty undo`
//! reverts agent-loop edits exactly like MCP-server edits. That file is
//! owned by the cli lane — this module duplicates the schema deliberately
//! (harness must not depend on the binary crate) and documents the mirror.
//!
//! Enablement: explicit [`ToolRegistry::with_journal`] path, or the
//! `KITTY_JOURNAL` env var (same name the cli honors). There is NO implicit
//! `~/.kitty/journal.jsonl` default here: the default lives in the cli lane
//! (`default_journal_path`), which passes the resolved path in. Recording is
//! best-effort — a journal I/O error warns but never fails the tool call.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Env override for the journal path (same name the cli honors).
pub const ENV_JOURNAL: &str = "KITTY_JOURNAL";

/// One recorded file mutation. Field names AND order mirror
/// `kitty-cli/src/journal.rs::EditEntry` — keep them in sync.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JournalEntry {
    /// Absolute (or as-given) file path.
    pub path: String,
    /// File bytes before the edit (`None` = file did not exist).
    pub before: Option<String>,
    /// File bytes after the edit.
    pub after: String,
    /// Unix seconds when recorded.
    pub recorded_at: u64,
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Resolve the journal path: explicit `KITTY_JOURNAL` env (non-empty) or
/// `None` = journaling disabled. No implicit default (cli lane owns it).
#[must_use]
pub fn resolve_journal_path() -> Option<PathBuf> {
    std::env::var(ENV_JOURNAL)
        .ok()
        .filter(|p| !p.trim().is_empty())
        .map(PathBuf::from)
}

/// Append one entry (creates parent dirs). Sync by design: one small
/// append per successful edit, same as the cli recorder.
pub fn append_entry(journal: &Path, entry: &JournalEntry) -> std::io::Result<()> {
    if let Some(parent) = journal.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut line = serde_json::to_string(entry)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    line.push('\n');
    use std::io::Write as _;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(journal)?;
    file.write_all(line.as_bytes())
}

/// Build an entry with the current timestamp.
#[must_use]
pub fn entry_for(path: &str, before: Option<String>, after: String) -> JournalEntry {
    JournalEntry {
        path: path.to_string(),
        before,
        after,
        recorded_at: now_secs(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_field_order_matches_cli() {
        // `kitty undo` parses positionally-agnostic JSON, but keep the wire
        // order identical to the cli EditEntry for byte-level compatibility.
        let e = entry_for("a.txt", Some("v1".into()), "v2".into());
        let json = serde_json::to_string(&e).unwrap();
        let (p, b, a, r) = (
            json.find("\"path\"").unwrap(),
            json.find("\"before\"").unwrap(),
            json.find("\"after\"").unwrap(),
            json.find("\"recorded_at\"").unwrap(),
        );
        assert!(p < b && b < a && a < r, "field order drifted: {json}");
    }

    #[test]
    fn append_roundtrips_entry() {
        let dir = tempfile::tempdir().unwrap();
        let journal = dir.path().join("j.jsonl");
        let e = entry_for("f.txt", None, "hello".into());
        append_entry(&journal, &e).unwrap();
        let text = std::fs::read_to_string(&journal).unwrap();
        let back: JournalEntry = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn resolve_honors_explicit_env_only() {
        let _g = crate::model_profile::test_env_lock();
        let dir = tempfile::tempdir().unwrap();
        let journal = dir.path().join("j.jsonl");
        let _e = crate::model_profile::EnvRestoreGuard::set(
            ENV_JOURNAL,
            journal.to_string_lossy().as_ref(),
        );
        assert_eq!(resolve_journal_path(), Some(journal));
    }

    #[test]
    fn resolve_ignores_empty_env() {
        let _g = crate::model_profile::test_env_lock();
        let _e = crate::model_profile::EnvRestoreGuard::set(ENV_JOURNAL, "   ");
        assert_eq!(resolve_journal_path(), None);
    }
}

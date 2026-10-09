//! Edit journal + `kitty undo` (W3).
//!
//! The journal is an append-only JSONL file (default
//! `~/.kitty/journal.jsonl`, override with `--journal` or
//! `KITTY_JOURNAL`). Each line records one file mutation with its
//! pre-image, so [`undo_last`] can restore the previous bytes.
//!
//! Current wiring (lane boundary: harness is read-only for us):
//! - `McpServer::call_tool` records `Edit`/`Write` successes via
//!   [`record_file_edit`] (snapshot before → execute → snapshot after).
//! - Agent-loop edits (`kitty run`) are recorded by the harness hook itself
//!   (`ToolRegistry::execute` snapshots before/after on success). This lane
//!   arms that hook via [`ensure_run_journal_env`]: the harness resolves its
//!   path as *explicit builder first, then the same `KITTY_JOURNAL` env*, so
//!   setting the env to the resolved default is the whole cli-side wiring.
//! - `kitty undo` reverts whatever the journal holds, regardless of which
//!   path recorded it.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Env override for the journal path (tests set it per-case instead).
pub const ENV_JOURNAL: &str = "KITTY_JOURNAL";

/// One recorded file mutation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EditEntry {
    /// Absolute (or as-given) file path.
    pub path: String,
    /// File bytes before the edit (`None` = file did not exist).
    pub before: Option<String>,
    /// File bytes after the edit.
    pub after: String,
    /// Unix seconds when recorded.
    pub recorded_at: u64,
}

/// Default journal path: `~/.kitty/journal.jsonl`.
#[must_use]
pub fn default_journal_path() -> PathBuf {
    if let Ok(p) = std::env::var(ENV_JOURNAL) {
        if !p.trim().is_empty() {
            return PathBuf::from(p);
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".kitty").join("journal.jsonl")
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Read all entries (missing file = empty journal, never an error).
pub fn read_entries(journal: &Path) -> Result<Vec<EditEntry>> {
    let bytes = match std::fs::read(journal) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e).with_context(|| format!("reading {}", journal.display())),
    };
    let text = String::from_utf8_lossy(&bytes);
    let mut entries = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let entry: EditEntry = serde_json::from_str(line)
            .with_context(|| format!("{}: línea {} corrupta", journal.display(), i + 1))?;
        entries.push(entry);
    }
    Ok(entries)
}

/// Append one entry (creates parent dirs).
pub fn append_entry(journal: &Path, entry: &EditEntry) -> Result<()> {
    if let Some(parent) = journal.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    let mut line = serde_json::to_string(entry)?;
    line.push('\n');
    use std::io::Write as _;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(journal)
        .with_context(|| format!("abriendo {}", journal.display()))?;
    file.write_all(line.as_bytes())?;
    Ok(())
}

/// Record a mutation with explicit before/after snapshots.
pub fn record_file_edit(
    journal: &Path,
    path: &Path,
    before: Option<String>,
    after: String,
) -> Result<()> {
    append_entry(
        journal,
        &EditEntry {
            path: path.to_string_lossy().into_owned(),
            before,
            after,
            recorded_at: now_secs(),
        },
    )
}

/// Snapshot helper: current file text or `None` when absent/unreadable.
#[must_use]
pub fn snapshot(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

/// Serialize tests that mutate `KITTY_JOURNAL` or drive successful
/// `Edit`/`Write` through `ToolRegistry::execute` (test-only).
///
/// The harness hook resolves its journal path per `execute` from the
/// process-global env, so a test swapping the env while another test
/// executes an edit would cross-record into the wrong journal. Hold this
/// across the whole env-mutating / edit-executing section.
#[cfg(test)]
pub(crate) fn test_journal_env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Arm the agent-loop journal hook for `kitty run` (C3, Gate B).
///
/// The hook itself lives in `kitty-harness` (`ToolRegistry::execute` records
/// successful `Edit`/`Write` with before/after snapshots). The registry
/// inside `Agent` is not reachable from this crate, but the harness resolves
/// its path as *explicit builder first, then this same `KITTY_JOURNAL` env* —
/// so ensuring the env holds the resolved default IS the cli-side wiring
/// (no harness change, no format change: still append-only JSONL, one
/// [`EditEntry`] per line).
///
/// An explicit non-empty `KITTY_JOURNAL` is left untouched; unset/empty
/// falls back to [`default_journal_path`]. Returns the path the run will
/// journal to. Performs no I/O itself (entries/dirs are created lazily on
/// the first recorded edit).
pub fn ensure_run_journal_env() -> PathBuf {
    if let Ok(p) = std::env::var(ENV_JOURNAL) {
        if !p.trim().is_empty() {
            return PathBuf::from(p);
        }
    }
    let default = default_journal_path();
    std::env::set_var(ENV_JOURNAL, &default);
    default
}

/// Undo the most recent entry: restores `before` (`None` = delete the file)
/// and truncates that line from the journal. Returns the reverted path, or
/// `None` when the journal is empty.
pub fn undo_last(journal: &Path) -> Result<Option<PathBuf>> {
    let mut entries = read_entries(journal)?;
    let Some(last) = entries.pop() else {
        return Ok(None);
    };
    let target = PathBuf::from(&last.path);
    match last.before {
        Some(before) => {
            if let Some(parent) = target.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            std::fs::write(&target, before)
                .with_context(|| format!("restaurando {}", target.display()))?;
        }
        None => {
            if target.exists() {
                std::fs::remove_file(&target)
                    .with_context(|| format!("eliminando {}", target.display()))?;
            }
        }
    }
    // Rewrite the journal without the reverted line (atomic-ish: tmp+rename).
    let tmp = journal.with_extension("tmp");
    {
        use std::io::Write as _;
        let mut file = std::fs::File::create(&tmp)?;
        for e in &entries {
            let mut line = serde_json::to_string(e)?;
            line.push('\n');
            file.write_all(line.as_bytes())?;
        }
    }
    std::fs::rename(&tmp, journal)?;
    Ok(Some(target))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_journal() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let journal = dir.path().join("journal.jsonl");
        (dir, journal)
    }

    #[test]
    fn missing_journal_reads_empty() {
        let (_dir, journal) = tmp_journal();
        assert!(read_entries(&journal).unwrap().is_empty());
    }

    #[test]
    fn record_then_undo_restores_bytes() {
        let (_dir, journal) = tmp_journal();
        let file_dir = tempfile::tempdir().unwrap();
        let file = file_dir.path().join("a.txt");
        std::fs::write(&file, "v1").unwrap();

        record_file_edit(&journal, &file, snapshot(&file), "v2".to_string()).unwrap();
        std::fs::write(&file, "v2").unwrap();

        let undone = undo_last(&journal).unwrap().expect("entry");
        assert_eq!(undone, file);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "v1");
        assert!(read_entries(&journal).unwrap().is_empty());
    }

    #[test]
    fn undo_created_file_deletes_it() {
        let (_dir, journal) = tmp_journal();
        let file_dir = tempfile::tempdir().unwrap();
        let file = file_dir.path().join("new.txt");

        record_file_edit(&journal, &file, None, "hello".to_string()).unwrap();
        std::fs::write(&file, "hello").unwrap();

        undo_last(&journal).unwrap().expect("entry");
        assert!(!file.exists());
    }

    #[test]
    fn undo_empty_journal_is_none() {
        let (_dir, journal) = tmp_journal();
        assert!(undo_last(&journal).unwrap().is_none());
    }

    #[test]
    fn undo_pops_only_last_entry() {
        let (_dir, journal) = tmp_journal();
        let dir = tempfile::tempdir().unwrap();
        let fa = dir.path().join("a.txt");
        let fb = dir.path().join("b.txt");
        std::fs::write(&fa, "a1").unwrap();
        std::fs::write(&fb, "b1").unwrap();

        record_file_edit(&journal, &fa, Some("a1".into()), "a2".into()).unwrap();
        std::fs::write(&fa, "a2").unwrap();
        record_file_edit(&journal, &fb, Some("b1".into()), "b2".into()).unwrap();
        std::fs::write(&fb, "b2").unwrap();

        undo_last(&journal).unwrap().expect("entry");
        assert_eq!(std::fs::read_to_string(&fb).unwrap(), "b1");
        // La primera entrada sigue intacta.
        assert_eq!(std::fs::read_to_string(&fa).unwrap(), "a2");
        assert_eq!(read_entries(&journal).unwrap().len(), 1);
    }

    /// C3: `ensure_run_journal_env` preserves an explicit `KITTY_JOURNAL`
    /// and falls back to the default (setting the env) when unset/empty.
    /// Single test (not split) so the env mutation can't race with itself;
    /// no other test in this binary reads `KITTY_JOURNAL`.
    #[test]
    fn ensure_run_journal_env_defaults_and_preserves() {
        let _lock = test_journal_env_lock();
        let prev = std::env::var(ENV_JOURNAL).ok();
        struct Restore(Option<String>);
        impl Drop for Restore {
            fn drop(&mut self) {
                match &self.0 {
                    Some(v) => std::env::set_var(ENV_JOURNAL, v),
                    None => std::env::remove_var(ENV_JOURNAL),
                }
            }
        }
        let _restore = Restore(prev);

        // Explicit non-empty value: untouched, returned verbatim.
        let (_dir, explicit) = tmp_journal();
        std::env::set_var(ENV_JOURNAL, &explicit);
        assert_eq!(ensure_run_journal_env(), explicit);
        assert_eq!(
            std::env::var(ENV_JOURNAL).unwrap(),
            explicit.to_str().unwrap()
        );
        assert!(
            !explicit.exists(),
            "ensure must not touch the journal file itself"
        );

        // Unset: falls back to the default AND publishes it for the harness hook.
        std::env::remove_var(ENV_JOURNAL);
        let resolved = ensure_run_journal_env();
        assert_eq!(resolved, default_journal_path());
        assert_eq!(PathBuf::from(std::env::var(ENV_JOURNAL).unwrap()), resolved);

        // Empty: same as unset (mirrors `default_journal_path`).
        std::env::set_var(ENV_JOURNAL, "   ");
        let resolved = ensure_run_journal_env();
        assert_eq!(resolved, default_journal_path());
    }
}

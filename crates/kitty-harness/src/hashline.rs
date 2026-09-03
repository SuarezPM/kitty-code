//! Hashline editor — reads files tagged per-line, edits by hash.
//!
//! Format: `LINE:HASH|content`
//! - Each line gets a BLAKE3 hash of its content.
//! - Edits target a specific hash, not text content.
//! - If the file changed since last read, the edit is rejected.

use blake3::Hasher;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum HashlineError {
    #[error("File not loaded: {0}")]
    NotLoaded(String),

    #[error("File changed since last read (expected hash {expected}, got new content)")]
    FileChanged { expected: String },

    #[error("No line matched hash: {0}")]
    HashNotFound(String),

    #[error("Ambiguous hash prefix '{0}' matches multiple lines")]
    AmbiguousPrefix(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// A single line with its hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineHash {
    pub line_num: usize,
    pub hash: String,
    pub content: String,
}

impl LineHash {
    /// Compute the BLAKE3 hash of a line's content.
    pub fn hash_line(line: &str) -> String {
        let mut hasher = Hasher::new();
        hasher.update(line.as_bytes());
        hex::encode(hasher.finalize().as_bytes())
    }

    /// Format as `LINE:HASH|content`.
    pub fn to_tagged(&self) -> String {
        format!("{}:{}|{}", self.line_num, self.hash, self.content)
    }
}

/// Snapshot of a file's lines with hashes.
#[derive(Debug, Clone)]
pub struct FileSnapshot {
    pub path: PathBuf,
    pub file_hash: String,
    pub lines: Vec<LineHash>,
}

impl FileSnapshot {
    /// Recompute the overall file hash (hash of all line hashes concatenated).
    pub fn compute_file_hash(lines: &[LineHash]) -> String {
        let mut hasher = Hasher::new();
        for line in lines {
            hasher.update(line.hash.as_bytes());
        }
        hex::encode(hasher.finalize().as_bytes())
    }
}

/// Hashline editor — reads files with line hashes, edits by hash.
pub struct HashlineEditor {
    snapshots: HashMap<PathBuf, FileSnapshot>,
}

impl HashlineEditor {
    pub fn new() -> Self {
        Self {
            snapshots: HashMap::new(),
        }
    }

    /// Read a file and tag each line with a BLAKE3 hash.
    pub fn read_file(&mut self, path: impl AsRef<Path>) -> Result<FileSnapshot, HashlineError> {
        let path = path.as_ref().to_path_buf();
        let content = std::fs::read_to_string(&path)?;
        let lines = Self::tag_lines(&content);
        let file_hash = FileSnapshot::compute_file_hash(&lines);
        let snapshot = FileSnapshot {
            path: path.clone(),
            file_hash,
            lines,
        };
        self.snapshots.insert(path, snapshot.clone());
        Ok(snapshot)
    }

    /// Tag each line of content with line number and BLAKE3 hash.
    fn tag_lines(content: &str) -> Vec<LineHash> {
        content
            .lines()
            .enumerate()
            .map(|(i, line)| LineHash {
                line_num: i + 1,
                hash: LineHash::hash_line(line),
                content: line.to_string(),
            })
            .collect()
    }

    /// Edit a file by replacing the line with the given hash prefix.
    ///
    /// Uses prefix matching: if `old_hash` is 8 chars, it matches any full
    /// hash starting with those 8 chars. If the prefix matches multiple lines,
    /// returns `AmbiguousPrefix`. If fewer than 6 chars, falls through to
    /// exact match.
    ///
    /// Returns the new file content. Does NOT write to disk — caller decides.
    /// If the file changed since last read, returns `FileChanged` error.
    pub fn edit_by_hash(
        &self,
        path: impl AsRef<Path>,
        old_hash: &str,
        new_content: &str,
    ) -> Result<String, HashlineError> {
        let path = path.as_ref();
        let snapshot = self
            .snapshots
            .get(path)
            .ok_or_else(|| HashlineError::NotLoaded(path.display().to_string()))?;

        // Verify file hasn't changed
        let current_hash = Self::compute_current_file_hash(path)?;
        if current_hash != snapshot.file_hash {
            return Err(HashlineError::FileChanged {
                expected: snapshot.file_hash.clone(),
            });
        }

        // Collect matching lines using prefix match (min 6 chars for prefix)
        let use_prefix = old_hash.len() >= 6;
        let matching_indices: Vec<usize> = snapshot
            .lines
            .iter()
            .enumerate()
            .filter(|(_, line)| {
                if use_prefix {
                    line.hash.starts_with(old_hash)
                } else {
                    line.hash == old_hash
                }
            })
            .map(|(i, _)| i)
            .collect();

        if matching_indices.is_empty() {
            return Err(HashlineError::HashNotFound(old_hash.to_string()));
        }

        if matching_indices.len() > 1 {
            return Err(HashlineError::AmbiguousPrefix(old_hash.to_string()));
        }

        // Exactly one match — replace it
        let target_idx = matching_indices[0];
        let result: Vec<String> = snapshot
            .lines
            .iter()
            .enumerate()
            .map(|(i, line)| {
                if i == target_idx {
                    new_content.to_string()
                } else {
                    line.content.clone()
                }
            })
            .collect();

        Ok(result.join("\n"))
    }

    /// Edit by hash prefix AND line number for disambiguation.
    ///
    /// Verifies the hash prefix matches the line at `line_num`, then replaces.
    /// Returns `HashNotFound` if the prefix doesn't match the specified line.
    pub fn edit_by_hash_and_line(
        &self,
        path: impl AsRef<Path>,
        old_hash: &str,
        line_num: usize,
        new_content: &str,
    ) -> Result<String, HashlineError> {
        let path = path.as_ref();
        let snapshot = self
            .snapshots
            .get(path)
            .ok_or_else(|| HashlineError::NotLoaded(path.display().to_string()))?;

        // Verify file hasn't changed
        let current_hash = Self::compute_current_file_hash(path)?;
        if current_hash != snapshot.file_hash {
            return Err(HashlineError::FileChanged {
                expected: snapshot.file_hash.clone(),
            });
        }

        // Find the line at the given line number
        let target = snapshot
            .lines
            .iter()
            .find(|l| l.line_num == line_num)
            .ok_or_else(|| HashlineError::HashNotFound(old_hash.to_string()))?;

        // Verify the hash prefix matches
        let use_prefix = old_hash.len() >= 6;
        let hash_matches = if use_prefix {
            target.hash.starts_with(old_hash)
        } else {
            target.hash == old_hash
        };

        if !hash_matches {
            return Err(HashlineError::HashNotFound(old_hash.to_string()));
        }

        // Replace that specific line
        let result: Vec<String> = snapshot
            .lines
            .iter()
            .map(|line| {
                if line.line_num == line_num {
                    new_content.to_string()
                } else {
                    line.content.clone()
                }
            })
            .collect();

        Ok(result.join("\n"))
    }

    /// Compute the current file hash on disk (without loading into snapshot).
    fn compute_current_file_hash(path: &Path) -> Result<String, HashlineError> {
        let content = std::fs::read_to_string(path)?;
        let lines = Self::tag_lines(&content);
        Ok(FileSnapshot::compute_file_hash(&lines))
    }

    /// Get a snapshot if loaded.
    pub fn get_snapshot(&self, path: impl AsRef<Path>) -> Option<&FileSnapshot> {
        self.snapshots.get(path.as_ref())
    }

    /// Format a snapshot as tagged lines (LINE:HASH|content format).
    pub fn format_tagged(snapshot: &FileSnapshot) -> String {
        snapshot
            .lines
            .iter()
            .map(|l| l.to_tagged())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Default for HashlineEditor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_file(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn hash_line_deterministic() {
        let h1 = LineHash::hash_line("hello");
        let h2 = LineHash::hash_line("hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_line_different_for_different_content() {
        let h1 = LineHash::hash_line("hello");
        let h2 = LineHash::hash_line("world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn read_file_tags_lines() {
        let f = temp_file("line1\nline2\nline3");
        let mut editor = HashlineEditor::new();
        let snapshot = editor.read_file(f.path()).unwrap();
        assert_eq!(snapshot.lines.len(), 3);
        assert_eq!(snapshot.lines[0].content, "line1");
        assert_eq!(snapshot.lines[1].content, "line2");
        assert_eq!(snapshot.lines[2].content, "line3");
        assert_eq!(snapshot.lines[0].line_num, 1);
    }

    #[test]
    fn edit_by_hash_works() {
        let f = temp_file("aaa\nbbb\nccc");
        let mut editor = HashlineEditor::new();
        let snapshot = editor.read_file(f.path()).unwrap();
        let hash_b = snapshot.lines[1].hash.clone();
        let result = editor.edit_by_hash(f.path(), &hash_b, "BBB").unwrap();
        assert_eq!(result, "aaa\nBBB\nccc");
    }

    #[test]
    fn edit_by_hash_rejects_changed_file() {
        let f = temp_file("aaa\nbbb\nccc");
        let mut editor = HashlineEditor::new();
        let _snapshot = editor.read_file(f.path()).unwrap();

        // Modify the file externally
        let mut f2 = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(f.path())
            .unwrap();
        write!(f2, "aaa\nXXX\nccc").unwrap();
        drop(f2);

        let err = editor.edit_by_hash(f.path(), "dummy", "new").unwrap_err();
        assert!(matches!(err, HashlineError::FileChanged { .. }));
    }

    #[test]
    fn edit_by_hash_rejects_unknown_hash() {
        let f = temp_file("aaa\nbbb\nccc");
        let mut editor = HashlineEditor::new();
        let _snapshot = editor.read_file(f.path()).unwrap();
        let err = editor
            .edit_by_hash(f.path(), "deadbeef", "new")
            .unwrap_err();
        assert!(matches!(err, HashlineError::HashNotFound(_)));
    }

    #[test]
    fn format_tagged_works() {
        let lines = vec![
            LineHash {
                line_num: 1,
                hash: "abc123".to_string(),
                content: "hello".to_string(),
            },
            LineHash {
                line_num: 2,
                hash: "def456".to_string(),
                content: "world".to_string(),
            },
        ];
        let snapshot = FileSnapshot {
            path: PathBuf::from("test.rs"),
            file_hash: "root".to_string(),
            lines,
        };
        let tagged = HashlineEditor::format_tagged(&snapshot);
        assert_eq!(tagged, "1:abc123|hello\n2:def456|world");
    }

    #[test]
    fn edit_by_hash_prefix_matches_full_hash() {
        let f = temp_file("aaa\nbbb\nccc");
        let mut editor = HashlineEditor::new();
        let snapshot = editor.read_file(f.path()).unwrap();
        let full_hash = &snapshot.lines[1].hash;
        let prefix = &full_hash[..8.min(full_hash.len())];
        let result = editor.edit_by_hash(f.path(), prefix, "BBB").unwrap();
        assert_eq!(result, "aaa\nBBB\nccc");
    }

    #[test]
    fn edit_by_hash_full_64_char_hash_still_works() {
        let f = temp_file("aaa\nbbb\nccc");
        let mut editor = HashlineEditor::new();
        let snapshot = editor.read_file(f.path()).unwrap();
        let full_hash = snapshot.lines[1].hash.clone();
        assert_eq!(full_hash.len(), 64); // BLAKE3 produces 64 hex chars
        let result = editor.edit_by_hash(f.path(), &full_hash, "BBB").unwrap();
        assert_eq!(result, "aaa\nBBB\nccc");
    }

    #[test]
    fn edit_by_hash_ambiguous_prefix_returns_error() {
        // Create file with two identical lines
        let f = temp_file("same\nsame\ndifferent");
        let mut editor = HashlineEditor::new();
        let snapshot = editor.read_file(f.path()).unwrap();
        let hash_line0 = &snapshot.lines[0].hash;
        let hash_line1 = &snapshot.lines[1].hash;
        // Identical lines → identical hashes
        assert_eq!(hash_line0, hash_line1);
        let prefix = &hash_line0[..8.min(hash_line0.len())];
        let err = editor
            .edit_by_hash(f.path(), prefix, "REPLACED")
            .unwrap_err();
        assert!(matches!(err, HashlineError::AmbiguousPrefix(_)));
    }

    #[test]
    fn edit_by_hash_and_line_disambiguates() {
        // Create file with two identical lines
        let f = temp_file("same\nsame\ndifferent");
        let mut editor = HashlineEditor::new();
        let snapshot = editor.read_file(f.path()).unwrap();
        let hash_line0 = &snapshot.lines[0].hash;
        let prefix = &hash_line0[..8.min(hash_line0.len())];
        let result = editor
            .edit_by_hash_and_line(f.path(), prefix, 1, "FIRST")
            .unwrap();
        assert_eq!(result, "FIRST\nsame\ndifferent");
    }

    #[test]
    fn edit_by_hash_short_prefix_falls_to_exact_match() {
        let f = temp_file("aaa\nbbb\nccc");
        let mut editor = HashlineEditor::new();
        let snapshot = editor.read_file(f.path()).unwrap();
        // 5-char prefix should NOT match (too short for prefix mode)
        let short_prefix = &snapshot.lines[1].hash[..5];
        let err = editor
            .edit_by_hash(f.path(), short_prefix, "BBB")
            .unwrap_err();
        // Falls through to exact match, which won't find a 5-char hash
        assert!(matches!(err, HashlineError::HashNotFound(_)));
    }

    #[test]
    fn edit_by_hash_and_line_rejects_mismatched_hash() {
        let f = temp_file("aaa\nbbb\nccc");
        let mut editor = HashlineEditor::new();
        let snapshot = editor.read_file(f.path()).unwrap();
        let hash_a = &snapshot.lines[0].hash;
        let prefix_a = &hash_a[..8.min(hash_a.len())];
        // Try to edit line 2 using line 0's hash prefix → should fail
        let err = editor
            .edit_by_hash_and_line(f.path(), prefix_a, 2, "BBB")
            .unwrap_err();
        assert!(matches!(err, HashlineError::HashNotFound(_)));
    }

    /// Mutation-testing regression (found by cargo-mutants): the
    /// short-hash path (`old_hash.len() < 6`, exact comparison) must always
    /// reject, because a sub-6-char hash can never equal a 64-char line
    /// hash. Flipping `==` to `!=` there would silently accept any short
    /// hash — this test pins the rejection.
    #[test]
    fn edit_by_hash_and_line_rejects_short_hash() {
        let f = temp_file("aaa\nbbb\nccc");
        let mut editor = HashlineEditor::new();
        editor.read_file(f.path()).unwrap();
        for short in ["a", "abc", "abcde"] {
            let err = editor
                .edit_by_hash_and_line(f.path(), short, 1, "XXX")
                .unwrap_err();
            assert!(matches!(err, HashlineError::HashNotFound(_)));
        }
    }
}

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

    /// Edit a file by replacing the line with the given hash.
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

        // Find and replace the line
        let mut found = false;
        let result: Vec<String> = snapshot
            .lines
            .iter()
            .map(|line| {
                if line.hash == old_hash {
                    found = true;
                    new_content.to_string()
                } else {
                    line.content.clone()
                }
            })
            .collect();

        if !found {
            return Err(HashlineError::HashNotFound(old_hash.to_string()));
        }

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

        let err = editor
            .edit_by_hash(f.path(), "dummy", "new")
            .unwrap_err();
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
}

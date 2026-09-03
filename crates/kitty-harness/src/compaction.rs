//! Structured compaction summaries with Goal/Constraints/Progress format.
//!
//! When context compaction triggers, the agent builds a [`CompactionSummary`]
//! from recent messages and tool results, then serializes it as structured
//! markdown for the system prompt. This gives small models a clearer
//! representation of session state than free-form prose summaries.

use serde::{Deserialize, Serialize};

/// A structured compaction summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionSummary {
    /// High-level goal of the current task.
    pub goal: String,
    /// Constraints and user preferences discovered during the session.
    pub constraints: Vec<String>,
    /// Tasks that have been completed.
    pub progress_done: Vec<String>,
    /// Tasks currently in progress.
    pub progress_in_progress: Vec<String>,
    /// Tasks that are blocked.
    pub progress_blocked: Vec<String>,
    /// Key architectural/design decisions made.
    pub key_decisions: Vec<String>,
    /// Concrete next steps to continue work.
    pub next_steps: Vec<String>,
    /// Critical context that must survive compaction.
    pub critical_context: Vec<String>,
    /// Files that were read during this segment.
    pub read_files: Vec<String>,
    /// Files that were modified during this segment.
    pub modified_files: Vec<String>,
}

impl CompactionSummary {
    /// Create an empty summary.
    pub fn empty() -> Self {
        Self {
            goal: String::new(),
            constraints: Vec::new(),
            progress_done: Vec::new(),
            progress_in_progress: Vec::new(),
            progress_blocked: Vec::new(),
            key_decisions: Vec::new(),
            next_steps: Vec::new(),
            critical_context: Vec::new(),
            read_files: Vec::new(),
            modified_files: Vec::new(),
        }
    }

    /// Format as structured markdown for the system prompt.
    pub fn format_markdown(&self) -> String {
        let mut out = String::new();

        out.push_str("## Goal\n");
        if self.goal.is_empty() {
            out.push_str("(none)\n");
        } else {
            out.push_str(&self.goal);
            out.push('\n');
        }

        out.push_str("\n## Constraints & Preferences\n");
        if self.constraints.is_empty() {
            out.push_str("(none)\n");
        } else {
            for c in &self.constraints {
                out.push_str(&format!("- {c}\n"));
            }
        }

        out.push_str("\n## Progress\n");
        if self.progress_done.is_empty()
            && self.progress_in_progress.is_empty()
            && self.progress_blocked.is_empty()
        {
            out.push_str("(none)\n");
        }
        if !self.progress_done.is_empty() {
            out.push_str("Done:\n");
            for d in &self.progress_done {
                out.push_str(&format!("- \u{2705} {d}\n"));
            }
        }
        if !self.progress_in_progress.is_empty() {
            out.push_str("In Progress:\n");
            for p in &self.progress_in_progress {
                out.push_str(&format!("- \u{1f504} {p}\n"));
            }
        }
        if !self.progress_blocked.is_empty() {
            out.push_str("Blocked:\n");
            for b in &self.progress_blocked {
                out.push_str(&format!("- \u{1f6ab} {b}\n"));
            }
        }

        out.push_str("\n## Key Decisions\n");
        if self.key_decisions.is_empty() {
            out.push_str("(none)\n");
        } else {
            for d in &self.key_decisions {
                out.push_str(&format!("- {d}\n"));
            }
        }

        out.push_str("\n## Next Steps\n");
        if self.next_steps.is_empty() {
            out.push_str("(none)\n");
        } else {
            for s in &self.next_steps {
                out.push_str(&format!("- {s}\n"));
            }
        }

        out.push_str("\n## Critical Context\n");
        if self.critical_context.is_empty() {
            out.push_str("(none)\n");
        } else {
            for c in &self.critical_context {
                out.push_str(&format!("- {c}\n"));
            }
        }

        if !self.read_files.is_empty() {
            out.push_str("\n<read-files>\n");
            for f in &self.read_files {
                out.push_str(&format!("  {f}\n"));
            }
            out.push_str("</read-files>\n");
        }

        if !self.modified_files.is_empty() {
            out.push_str("\n<modified-files>\n");
            for f in &self.modified_files {
                out.push_str(&format!("  {f}\n"));
            }
            out.push_str("</modified-files>\n");
        }

        out
    }

    /// Total number of items across all progress categories.
    pub fn progress_count(&self) -> usize {
        self.progress_done.len() + self.progress_in_progress.len() + self.progress_blocked.len()
    }
}

/// Extract file operations from a list of tool results.
///
/// Returns `(read_files, modified_files)` using simple heuristics on
/// tool output text. Intentionally lightweight — a more robust version
/// would parse structured tool call arguments, but this covers the
/// compaction-summary use case without requiring the full tool history.
pub fn extract_file_operations(results: &[String]) -> (Vec<String>, Vec<String>) {
    let mut read_files = Vec::new();
    let mut modified_files = Vec::new();
    let mut seen_read = std::collections::HashSet::new();
    let mut seen_write = std::collections::HashSet::new();

    for result in results {
        // Simple heuristic: look for file paths in tool results
        if result.contains("Wrote ") || result.contains("wrote ") {
            if let Some(path) = extract_path(result) {
                if seen_write.insert(path.clone()) {
                    modified_files.push(path);
                }
            }
        }
        if result.contains("Read ") || result.contains("read ") || result.contains("Read file") {
            if let Some(path) = extract_path(result) {
                if seen_read.insert(path.clone()) {
                    read_files.push(path);
                }
            }
        }
    }

    (read_files, modified_files)
}

fn extract_path(text: &str) -> Option<String> {
    // Look for common path patterns — file extensions common in coding projects
    const EXTENSIONS: &[&str] = &[
        ".rs", ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".java", ".c", ".cpp", ".h", ".rb",
        ".sh", ".toml", ".yaml", ".yml", ".json", ".md", ".txt", ".html", ".css",
    ];
    for word in text.split_whitespace() {
        // Strip leading/trailing quote characters
        let cleaned = word
            .trim_start_matches('`')
            .trim_start_matches('"')
            .trim_start_matches('\'')
            .trim_end_matches('`')
            .trim_end_matches('"')
            .trim_end_matches('\'');
        if cleaned.contains('/') && EXTENSIONS.iter().any(|ext| cleaned.ends_with(ext)) {
            return Some(cleaned.to_string());
        }
        if word.contains('/') && EXTENSIONS.iter().any(|ext| word.ends_with(ext)) {
            return Some(word.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_summary() {
        let s = CompactionSummary::empty();
        let md = s.format_markdown();
        assert!(md.contains("## Goal"));
        assert!(md.contains("## Progress"));
        assert!(md.contains("## Constraints"));
        assert!(md.contains("## Key Decisions"));
        assert!(md.contains("## Next Steps"));
        assert!(md.contains("## Critical Context"));
    }

    #[test]
    fn summary_with_content() {
        let mut s = CompactionSummary::empty();
        s.goal = "Implement feature X".to_string();
        s.progress_done.push("Task 1 complete".to_string());
        s.next_steps.push("Start task 2".to_string());

        let md = s.format_markdown();
        assert!(md.contains("Implement feature X"));
        assert!(md.contains("\u{2705} Task 1 complete"));
        assert!(md.contains("Start task 2"));
    }

    #[test]
    fn summary_in_progress_and_blocked() {
        let mut s = CompactionSummary::empty();
        s.progress_in_progress.push("Wiring module".to_string());
        s.progress_blocked.push("Waiting for CI".to_string());

        let md = s.format_markdown();
        assert!(md.contains("\u{1f504} Wiring module"));
        assert!(md.contains("\u{1f6ab} Waiting for CI"));
    }

    #[test]
    fn summary_file_lists() {
        let mut s = CompactionSummary::empty();
        s.read_files.push("src/main.rs".to_string());
        s.modified_files.push("src/lib.rs".to_string());

        let md = s.format_markdown();
        assert!(md.contains("<read-files>"));
        assert!(md.contains("src/main.rs"));
        assert!(md.contains("<modified-files>"));
        assert!(md.contains("src/lib.rs"));
    }

    #[test]
    fn progress_count() {
        let mut s = CompactionSummary::empty();
        assert_eq!(s.progress_count(), 0);
        s.progress_done.push("a".into());
        s.progress_in_progress.push("b".into());
        s.progress_blocked.push("c".into());
        assert_eq!(s.progress_count(), 3);
    }

    #[test]
    fn extract_file_operations_writes() {
        let results = vec![
            "Wrote src/main.rs".to_string(),
            "Read src/lib.rs".to_string(),
        ];
        let (read, modified) = extract_file_operations(&results);
        assert!(modified.contains(&"src/main.rs".to_string()));
        assert!(read.contains(&"src/lib.rs".to_string()));
    }

    #[test]
    fn extract_file_operations_dedup() {
        let results = vec![
            "Wrote src/main.rs".to_string(),
            "wrote src/main.rs again".to_string(),
        ];
        let (_, modified) = extract_file_operations(&results);
        assert_eq!(modified.len(), 1);
    }

    #[test]
    fn extract_path_various_extensions() {
        assert!(extract_path("wrote src/main.rs").is_some());
        assert!(extract_path("Read lib/utils.py").is_some());
        assert!(extract_path("wrote build/config.toml").is_some());
        // No slash → not a path
        assert!(extract_path("open README").is_none());
        // No recognized extension
        assert!(extract_path("wrote some/dir/file.xyz").is_none());
    }
}

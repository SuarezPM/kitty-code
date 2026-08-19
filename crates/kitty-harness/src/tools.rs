//! Tool implementations for the agent loop.
//!
//! Each tool type has a dedicated handler. Safety guards:
//! - Write-guard: refuses Write on existing files (force Edit).
//! - Read-before-edit guard: refuses Edit if file not read this session.
//! - Bash: sanitize command, block dangerous patterns via Guard.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::guard::{Guard, GuardVerdict};
use crate::hashline::HashlineEditor;
use crate::parser::ParsedToolCall;

#[derive(Error, Debug)]
pub enum ToolError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Write blocked on existing file: {path} — use Edit instead")]
    WriteBlocked { path: String },

    #[error("Edit blocked: file not read this session: {0}")]
    EditBlocked(String),

    #[error("Bash command blocked: {0}")]
    BashBlocked(String),

    #[error("Tool not found: {0}")]
    NotFound(String),

    #[error("Invalid arguments for {tool}: {reason}")]
    InvalidArgs { tool: String, reason: String },

    #[error("Hashline error: {0}")]
    Hashline(#[from] crate::hashline::HashlineError),

    #[error("Glob error: {0}")]
    Glob(String),

    #[error("Grep error: {0}")]
    Grep(String),
}

/// Result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub success: bool,
    pub output: String,
    /// Whether this tool call should be retried (e.g., file changed).
    pub retry: bool,
}

/// The tool registry manages tool execution and session state.
pub struct ToolRegistry {
    /// Files read this session (for read-before-edit guard).
    read_files: HashSet<PathBuf>,
    /// Files written this session (for write guard).
    written_files: HashSet<PathBuf>,
    /// The hashline editor for file operations.
    pub hashline: HashlineEditor,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            read_files: HashSet::new(),
            written_files: HashSet::new(),
            hashline: HashlineEditor::new(),
        }
    }

    /// Execute a parsed tool call.
    pub fn execute(&mut self, tc: &ParsedToolCall) -> Result<ToolResult, ToolError> {
        match tc.name.as_str() {
            "Read" => self.execute_read(&tc.arguments),
            "Edit" => self.execute_edit(&tc.arguments),
            "Write" => self.execute_write(&tc.arguments),
            "Bash" => self.execute_bash(&tc.arguments),
            "Glob" => self.execute_glob(&tc.arguments),
            "Grep" => self.execute_grep(&tc.arguments),
            other => Err(ToolError::NotFound(other.to_string())),
        }
    }

    /// Execute a Read tool call.
    fn execute_read(&mut self, args: &serde_json::Value) -> Result<ToolResult, ToolError> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs {
                tool: "Read".to_string(),
                reason: "missing 'path'".to_string(),
            })?;

        let path_buf = PathBuf::from(path);

        // Read with hashline tagging
        let snapshot = self.hashline.read_file(&path_buf)?;
        self.read_files.insert(path_buf);

        // Format output with line numbers and hashes
        let tagged: Vec<String> = snapshot
            .lines
            .iter()
            .map(|l| {
                format!(
                    "{}:{}|{}",
                    l.line_num,
                    &l.hash[..8.min(l.hash.len())],
                    l.content
                )
            })
            .collect();

        Ok(ToolResult {
            success: true,
            output: tagged.join("\n"),
            retry: false,
        })
    }

    /// Execute an Edit tool call (hashline-based).
    fn execute_edit(&mut self, args: &serde_json::Value) -> Result<ToolResult, ToolError> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs {
                tool: "Edit".to_string(),
                reason: "missing 'path'".to_string(),
            })?;

        let old_hash = args
            .get("hash")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs {
                tool: "Edit".to_string(),
                reason: "missing 'hash' (use line hash from Read output)".to_string(),
            })?;

        let new_content = args
            .get("new")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs {
                tool: "Edit".to_string(),
                reason: "missing 'new' content".to_string(),
            })?;

        let path_buf = PathBuf::from(path);

        // Read-before-edit guard
        if !self.read_files.contains(&path_buf) {
            return Err(ToolError::EditBlocked(path.to_string()));
        }

        match self.hashline.edit_by_hash(&path_buf, old_hash, new_content) {
            Ok(new_file_content) => {
                std::fs::write(&path_buf, &new_file_content)?;
                // Re-read to get updated hashes
                self.hashline.read_file(&path_buf)?;
                Ok(ToolResult {
                    success: true,
                    output: format!("Edit applied to {path}"),
                    retry: false,
                })
            }
            Err(crate::hashline::HashlineError::FileChanged { .. }) => Ok(ToolResult {
                success: false,
                output: format!("File {path} changed since last read. Re-read and retry."),
                retry: true,
            }),
            Err(crate::hashline::HashlineError::HashNotFound(h)) => {
                // Fallback: try search-replace using old content
                self.fallback_search_replace(path, &h, new_content)
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Fallback: if hash not found, try search-replace on the file content.
    fn fallback_search_replace(
        &mut self,
        path: &str,
        _hash: &str,
        new_content: &str,
    ) -> Result<ToolResult, ToolError> {
        // If the hash wasn't found, the line might have changed.
        // Re-read and try to find the content by text.
        let path_buf = PathBuf::from(path);
        let snapshot = self.hashline.read_file(&path_buf)?;

        // Try to find a line that partially matches
        let matching_line = snapshot.lines.iter().find(|l| {
            // Check if the new content's context matches
            l.content.contains(new_content) || new_content.contains(&l.content)
        });

        if let Some(line) = matching_line {
            let result = self.hashline.edit_by_hash(path, &line.hash, new_content)?;
            std::fs::write(&path_buf, &result)?;
            self.hashline.read_file(&path_buf)?;
            Ok(ToolResult {
                success: true,
                output: format!("Edit applied (fallback) to {path}"),
                retry: false,
            })
        } else {
            Ok(ToolResult {
                success: false,
                output: format!(
                    "Cannot find target line in {path}. \
                     File may have changed. Please re-read and retry."
                ),
                retry: true,
            })
        }
    }

    /// Execute a Write tool call.
    fn execute_write(&mut self, args: &serde_json::Value) -> Result<ToolResult, ToolError> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs {
                tool: "Write".to_string(),
                reason: "missing 'path'".to_string(),
            })?;

        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs {
                tool: "Write".to_string(),
                reason: "missing 'content'".to_string(),
            })?;

        let path_buf = PathBuf::from(path);

        // Write guard: refuse Write on existing files
        if path_buf.exists() && !self.written_files.contains(&path_buf) {
            return Err(ToolError::WriteBlocked {
                path: path.to_string(),
            });
        }

        // Create parent directories if needed
        if let Some(parent) = path_buf.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(&path_buf, content)?;

        // Update read_files so the file can be edited afterwards
        self.read_files.insert(path_buf.clone());
        self.written_files.insert(path_buf);

        Ok(ToolResult {
            success: true,
            output: format!("Wrote {path}"),
            retry: false,
        })
    }

    /// Execute a Bash tool call.
    fn execute_bash(&self, args: &serde_json::Value) -> Result<ToolResult, ToolError> {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs {
                tool: "Bash".to_string(),
                reason: "missing 'command'".to_string(),
            })?;

        // Guard: check the full chain
        let verdicts = Guard::check_chain(command);
        for verdict in &verdicts {
            match verdict {
                GuardVerdict::Block(reason) => {
                    return Err(ToolError::BashBlocked(reason.clone()));
                }
                GuardVerdict::Warn(reason) => {
                    // For now, treat warnings as blocks too (no interactive confirmation)
                    return Err(ToolError::BashBlocked(format!(
                        "{reason} (interactive confirmation not available)"
                    )));
                }
                GuardVerdict::Allow => {}
            }
        }

        // Execute the command
        let output = std::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let mut result = String::new();
        if !stdout.is_empty() {
            result.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push_str("\n--- stderr ---\n");
            }
            result.push_str(&stderr);
        }

        // Truncate very long output
        let max_output = 10_000;
        if result.len() > max_output {
            result.truncate(max_output);
            result.push_str("\n... (output truncated)");
        }

        Ok(ToolResult {
            success: output.status.success(),
            output: result,
            retry: false,
        })
    }

    /// Execute a Glob tool call.
    fn execute_glob(&self, args: &serde_json::Value) -> Result<ToolResult, ToolError> {
        let pattern = args
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs {
                tool: "Glob".to_string(),
                reason: "missing 'pattern'".to_string(),
            })?;

        let entries: Vec<String> = glob::glob(pattern)
            .map_err(|e| ToolError::Glob(e.to_string()))?
            .filter_map(|entry| entry.ok())
            .map(|path| path.display().to_string())
            .collect();

        if entries.is_empty() {
            Ok(ToolResult {
                success: true,
                output: "No files matched".to_string(),
                retry: false,
            })
        } else {
            Ok(ToolResult {
                success: true,
                output: entries.join("\n"),
                retry: false,
            })
        }
    }

    /// Execute a Grep tool call.
    fn execute_grep(&self, args: &serde_json::Value) -> Result<ToolResult, ToolError> {
        let pattern = args
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs {
                tool: "Grep".to_string(),
                reason: "missing 'pattern'".to_string(),
            })?;

        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or(".");

        let include = args.get("include").and_then(|v| v.as_str());

        let re = regex::Regex::new(pattern)
            .map_err(|e| ToolError::Grep(format!("Invalid regex: {e}")))?;

        let mut results = Vec::new();
        let walker = walkdir::WalkDir::new(path)
            .into_iter()
            .filter_entry(|e| !e.file_name().to_string_lossy().starts_with('.'));

        for entry in walker.filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }

            let entry_path = entry.path();
            if let Some(incl) = include {
                if let Some(ext) = entry_path.extension() {
                    if ext != incl.trim_start_matches('.') {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            if let Ok(content) = std::fs::read_to_string(entry_path) {
                for (line_num, line) in content.lines().enumerate() {
                    if re.is_match(line) {
                        results.push(format!(
                            "{}:{}:{}",
                            entry_path.display(),
                            line_num + 1,
                            line.trim()
                        ));
                    }
                }
            }
        }

        // Truncate if too many results
        let max_results = 200;
        if results.len() > max_results {
            let total = results.len();
            results.truncate(max_results);
            results.push(format!("... and {total} total matches (truncated)"));
        }

        Ok(ToolResult {
            success: true,
            output: if results.is_empty() {
                "No matches found".to_string()
            } else {
                results.join("\n")
            },
            retry: false,
        })
    }

    /// Check if a file has been read this session.
    pub fn is_file_read(&self, path: &Path) -> bool {
        self.read_files.contains(path)
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the standard tool definitions for the agent.
pub fn standard_tool_definitions() -> Vec<kitty_runtime::ToolDefinition> {
    vec![
        kitty_runtime::ToolDefinition::new(
            "Read",
            "Read a file with line hashes for safe editing. Returns LINE:HASH|content format.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read"
                    }
                },
                "required": ["path"]
            }),
        ),
        kitty_runtime::ToolDefinition::new(
            "Edit",
            "Edit a file by line hash (from Read output). Refuses if file changed since read.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to edit"
                    },
                    "hash": {
                        "type": "string",
                        "description": "Line hash from Read output"
                    },
                    "new": {
                        "type": "string",
                        "description": "New content for the line"
                    }
                },
                "required": ["path", "hash", "new"]
            }),
        ),
        kitty_runtime::ToolDefinition::new(
            "Write",
            "Write a new file. Blocked on existing files — use Edit instead.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "File content"
                    }
                },
                "required": ["path", "content"]
            }),
        ),
        kitty_runtime::ToolDefinition::new(
            "Bash",
            "Execute a shell command. Dangerous commands are blocked.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute"
                    }
                },
                "required": ["command"]
            }),
        ),
        kitty_runtime::ToolDefinition::new(
            "Glob",
            "Find files matching a glob pattern.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., **/*.rs)"
                    }
                },
                "required": ["pattern"]
            }),
        ),
        kitty_runtime::ToolDefinition::new(
            "Grep",
            "Search file contents with regex.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in"
                    },
                    "include": {
                        "type": "string",
                        "description": "File extension filter (e.g., rs)"
                    }
                },
                "required": ["pattern"]
            }),
        ),
    ]
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
    fn read_file_records_session() {
        let f = temp_file("line1\nline2");
        let mut registry = ToolRegistry::new();
        let args = serde_json::json!({"path": f.path().to_str().unwrap()});
        let result = registry.execute_read(&args).unwrap();
        assert!(result.success);
        assert!(registry.is_file_read(f.path()));
    }

    #[test]
    fn write_guard_blocks_existing_file() {
        let f = temp_file("existing");
        let mut registry = ToolRegistry::new();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "content": "new content"
        });
        let err = registry.execute_write(&args).unwrap_err();
        assert!(matches!(err, ToolError::WriteBlocked { .. }));
    }

    #[test]
    fn write_allows_new_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new_file.txt");
        let mut registry = ToolRegistry::new();
        let args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "content": "hello"
        });
        let result = registry.execute_write(&args).unwrap();
        assert!(result.success);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello");
    }

    #[test]
    fn edit_blocked_without_read() {
        let f = temp_file("line1\nline2");
        let mut registry = ToolRegistry::new();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "hash": "dummy",
            "new": "replaced"
        });
        let err = registry.execute_edit(&args).unwrap_err();
        assert!(matches!(err, ToolError::EditBlocked(_)));
    }

    #[test]
    fn bash_blocks_dangerous_commands() {
        let registry = ToolRegistry::new();
        let args = serde_json::json!({"command": "rm -rf /"});
        let err = registry.execute_bash(&args).unwrap_err();
        assert!(matches!(err, ToolError::BashBlocked(_)));
    }

    #[test]
    fn bash_allows_safe_commands() {
        let registry = ToolRegistry::new();
        let args = serde_json::json!({"command": "echo hello"});
        let result = registry.execute_bash(&args).unwrap();
        assert!(result.success);
        assert!(result.output.contains("hello"));
    }

    #[test]
    fn glob_finds_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.path().join("b.txt"), "hello").unwrap();

        let registry = ToolRegistry::new();
        let pattern = format!("{}/*.rs", dir.path().display());
        let args = serde_json::json!({"pattern": pattern});
        let result = registry.execute_glob(&args).unwrap();
        assert!(result.success);
        assert!(result.output.contains("a.rs"));
        assert!(!result.output.contains("b.txt"));
    }

    #[test]
    fn tool_not_found() {
        let mut registry = ToolRegistry::new();
        let tc = ParsedToolCall {
            name: "Nonexistent".to_string(),
            arguments: serde_json::json!({}),
        };
        let err = registry.execute(&tc).unwrap_err();
        assert!(matches!(err, ToolError::NotFound(_)));
    }
}

//! Tool implementations for the agent loop.
//!
//! Each tool type has a dedicated handler. Safety guards:
//! - Write-guard: refuses Write on existing files (force Edit).
//! - Read-before-edit guard: refuses Edit if file not read this session.
//! - Bash: sanitize command, block dangerous patterns via Guard.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use thiserror::Error;

use crate::guard::{Guard, GuardVerdict};
use crate::hashline::HashlineEditor;
use crate::parser::ParsedToolCall;
use crate::sandbox::{Sandbox, SandboxConfig, SandboxError};

#[derive(Error, Debug)]
pub enum ToolError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Write blocked on existing file: {path} — use Edit instead")]
    WriteBlocked { path: String },

    #[error("File not loaded: {0} — must Read before Edit")]
    EditBlocked(String),

    #[error("Bash command blocked: {0}")]
    BashBlocked(String),

    #[error("Sandbox error: {0}")]
    Sandbox(String),

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
    /// Bash command timeout in seconds (F1.3). Default: 60.
    bash_timeout_secs: u64,
    /// Max lines for Read tool output (F3.1). Default: 300.
    max_read_lines: usize,
    /// Kernel containment armed on every Bash child before spawn
    /// (`None` only when explicitly disabled — see [`Self::without_sandbox`]).
    bash_sandbox: Option<Sandbox>,
}

/// Default Bash-sandbox allowlist root: the process working directory
/// (the workspace, for kitty-cli usage).
fn default_workspace_root() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::with_timeout(60)
    }

    /// Create a registry with a custom bash timeout (seconds).
    pub fn with_timeout(bash_timeout_secs: u64) -> Self {
        Self {
            read_files: HashSet::new(),
            written_files: HashSet::new(),
            hashline: HashlineEditor::new(),
            bash_timeout_secs,
            max_read_lines: 300,
            bash_sandbox: Some(Sandbox::new(SandboxConfig::new(default_workspace_root()))),
        }
    }

    /// Create a registry with a custom Read truncation limit (F3.1).
    pub fn with_read_limit(max_read_lines: usize) -> Self {
        Self {
            read_files: HashSet::new(),
            written_files: HashSet::new(),
            hashline: HashlineEditor::new(),
            bash_timeout_secs: 60,
            max_read_lines,
            bash_sandbox: Some(Sandbox::new(SandboxConfig::new(default_workspace_root()))),
        }
    }

    /// Rebuild the Bash sandbox with `root` as the FS allowlist workspace.
    pub fn with_workspace(mut self, root: impl Into<PathBuf>) -> Self {
        self.bash_sandbox = Some(Sandbox::new(SandboxConfig::new(root)));
        self
    }

    /// Inject a fully-configured [`Sandbox`] (custom policy / tests).
    pub fn with_sandbox(mut self, sb: Sandbox) -> Self {
        self.bash_sandbox = Some(sb);
        self
    }

    /// Explicitly disable kernel containment for Bash children. Every
    /// activation still logs; prefer the loud per-spawn `KITTY_SANDBOX=off`
    /// opt-out when the disable decision is environmental.
    pub fn without_sandbox(mut self) -> Self {
        self.bash_sandbox = None;
        self
    }

    /// Execute a parsed tool call.
    pub async fn execute(&mut self, tc: &ParsedToolCall) -> Result<ToolResult, ToolError> {
        match tc.name.as_str() {
            "Read" => self.execute_read(&tc.arguments),
            "Edit" => self.execute_edit(&tc.arguments),
            "Write" => self.execute_write(&tc.arguments),
            "Bash" => self.execute_bash(&tc.arguments).await,
            "Glob" => self.execute_glob(&tc.arguments),
            "Grep" => self.execute_grep(&tc.arguments),
            other => Err(ToolError::NotFound(other.to_string())),
        }
    }

    /// Execute a Read tool call.
    fn execute_read(&mut self, args: &serde_json::Value) -> Result<ToolResult, ToolError> {
        let path =
            args.get("path")
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

        let total_lines = tagged.len();
        let capped = &tagged[..total_lines.min(self.max_read_lines)];

        let mut output = capped.join("\n");
        if total_lines > self.max_read_lines {
            let omitted = total_lines - self.max_read_lines;
            output.push_str(&format!(
                "\n... ({omitted} lines omitted — use Grep to search specific content)"
            ));
        }

        Ok(ToolResult {
            success: true,
            output,
            retry: false,
        })
    }

    /// Execute an Edit tool call (hashline-based).
    fn execute_edit(&mut self, args: &serde_json::Value) -> Result<ToolResult, ToolError> {
        let path =
            args.get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidArgs {
                    tool: "Edit".to_string(),
                    reason: "missing 'path'".to_string(),
                })?;

        let old_hash =
            args.get("hash")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidArgs {
                    tool: "Edit".to_string(),
                    reason: "missing 'hash' (use line hash from Read output)".to_string(),
                })?;

        let new_content =
            args.get("new")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidArgs {
                    tool: "Edit".to_string(),
                    reason: "missing 'new' content".to_string(),
                })?;

        let line_num = args
            .get("line")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);

        let path_buf = PathBuf::from(path);

        // Read-before-edit guard — scoped to session via snapshot
        if self.hashline.get_snapshot(&path_buf).is_none() {
            return Err(ToolError::EditBlocked(path.to_string()));
        }

        // If line number provided, use disambiguation path
        if let Some(line) = line_num {
            match self
                .hashline
                .edit_by_hash_and_line(&path_buf, old_hash, line, new_content)
            {
                Ok(new_file_content) => {
                    std::fs::write(&path_buf, &new_file_content)?;
                    self.hashline.read_file(&path_buf)?;
                    Ok(ToolResult {
                        success: true,
                        output: format!("Edit applied to {path} (line {line})"),
                        retry: false,
                    })
                }
                Err(crate::hashline::HashlineError::FileChanged { .. }) => Ok(ToolResult {
                    success: false,
                    output: format!("File {path} changed since last read. Re-read and retry."),
                    retry: true,
                }),
                Err(e) => Err(e.into()),
            }
        } else {
            // Normal path — prefix match
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
                Err(crate::hashline::HashlineError::AmbiguousPrefix(h)) => Ok(ToolResult {
                    success: false,
                    output: format!(
                        "Hash prefix '{h}' matches multiple lines in {path}. \
                         Re-run Edit with the 'line' parameter set to the line number to disambiguate."
                    ),
                    retry: false,
                }),
                Err(e) => Err(e.into()),
            }
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
        let path =
            args.get("path")
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

        // Write guard: refuse Write on existing files (force Edit)
        // Layer 1: spec-required check — existing file not in written_files → block
        if path_buf.exists() && !self.written_files.contains(&path_buf) {
            return Err(ToolError::WriteBlocked {
                path: path.to_string(),
            });
        }
        // Layer 2: strict invariant — even files we created via Write in this
        // session must be edited via Edit, not overwritten via Write again.
        // This enforces "Primera vez en tmpdir fresco permite, segunda bloquea".
        if path_buf.exists() {
            return Err(ToolError::WriteBlocked {
                path: path.to_string(),
            });
        }

        // Create parent directories if needed
        if let Some(parent) = path_buf.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(&path_buf, content)?;

        // Update session state so the file can be edited afterwards without extra Read
        self.read_files.insert(path_buf.clone());
        self.written_files.insert(path_buf.clone());
        // Populate hashline snapshot for read-before-edit invariant
        let _ = self.hashline.read_file(&path_buf);

        Ok(ToolResult {
            success: true,
            output: format!("Wrote {path}"),
            retry: false,
        })
    }

    /// Execute a Bash tool call.
    ///
    /// The command runs under `sh -c` with a hard timeout (F1.3): a command
    /// that outlives `bash_timeout_secs` is killed and reported as a failed
    /// tool result (`success: false`), never as a fatal error propagating out
    /// of the agent loop.
    ///
    /// Containment is layered: `Guard::check_chain` (policy) runs first, then
    /// `Sandbox::apply` arms Landlock/seccomp/namespaces/RLIMIT_AS on the
    /// child just before exec. Sandbox setup failures are fail-closed: the
    /// command is NOT executed and a [`ToolError::Sandbox`] with an actionable
    /// message is returned. Setting `KITTY_SANDBOX=off` in the environment
    /// disables containment explicitly — loudly logged on every activation,
    /// never silent.
    async fn execute_bash(&self, args: &serde_json::Value) -> Result<ToolResult, ToolError> {
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

        // Kernel containment AFTER guard policy (ordering contract documented
        // at `Sandbox::apply`). `KITTY_SANDBOX=off` is honored ONLY when set;
        // every activation logs a warning so an unsandboxed run is never
        // silent.
        let sandbox: Option<&Sandbox> = match (
            &self.bash_sandbox,
            std::env::var("KITTY_SANDBOX").as_deref(),
        ) {
            (Some(_), Ok("off")) => {
                tracing::warn!(
                    "KITTY_SANDBOX=off: Bash command runs WITHOUT kernel \
                         containment (explicit opt-out)"
                );
                None
            }
            (sb_opt, _) => sb_opt.as_ref(),
        };

        // Execute the command with a timeout. `kill_on_drop(true)` guarantees
        // the child is killed if the future is cancelled by the timeout.
        let timeout = Duration::from_secs(self.bash_timeout_secs);
        let run = async {
            let mut cmd = tokio::process::Command::new("sh");
            cmd.arg("-c")
                .arg(command)
                .kill_on_drop(true)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            if let Some(sb) = sandbox {
                sb.apply(&mut cmd).map_err(|e| match e {
                    SandboxError::Unsupported(_) | SandboxError::Setup(_) => {
                        ToolError::Sandbox(format!(
                            "{e}; command NOT executed (fail-closed). Fix the host \
                             (needs Linux >= 5.13 with unprivileged user namespaces) \
                             or set KITTY_SANDBOX=off to disable containment explicitly"
                        ))
                    }
                    other => ToolError::Sandbox(other.to_string()),
                })?;
            }
            // When containment is armed, a spawn failure usually means the
            // kernel refused the namespace/seccomp chain at exec time even
            // though the cheap preflight passed (e.g. Docker's default
            // seccomp profile blocks unshare(CLONE_NEWUSER) while the sysctl
            // files still look permissive). Surface that as an actionable
            // sandbox error instead of a bare errno.
            let child = cmd.spawn().map_err(|e| {
                if sandbox.is_some() {
                    ToolError::Sandbox(format!(
                        "spawn failed while arming containment: {e}. This host \
                         or container likely blocks user namespaces; set \
                         KITTY_SANDBOX=off to disable containment explicitly"
                    ))
                } else {
                    ToolError::Io(e)
                }
            })?;
            child.wait_with_output().await.map_err(ToolError::Io)
        };

        let output = match tokio::time::timeout(timeout, run).await {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => return Err(e),
            Err(_elapsed) => {
                // Timed out: the child was killed by kill_on_drop when the
                // future was dropped. Report as a failed tool result.
                return Ok(ToolResult {
                    success: false,
                    output: format!(
                        "command timed out after {}s (killed)",
                        self.bash_timeout_secs
                    ),
                    retry: false,
                });
            }
        };

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

        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        let include = args.get("include").and_then(|v| v.as_str());

        let re = regex::Regex::new(pattern)
            .map_err(|e| ToolError::Grep(format!("Invalid regex: {e}")))?;

        let mut results = Vec::new();
        let walker = ignore::WalkBuilder::new(path)
            .hidden(false)
            .git_ignore(true)
            .build();

        for entry in walker.filter_map(|e| e.ok()) {
            if !entry.file_type().is_some_and(|ft| ft.is_file()) {
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
                        "description": "Line hash from Read output (8-char prefix or full hash)"
                    },
                    "new": {
                        "type": "string",
                        "description": "New content for the line"
                    },
                    "line": {
                        "type": "integer",
                        "description": "Line number to disambiguate when hash prefix matches multiple lines"
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

    #[tokio::test]
    async fn bash_blocks_dangerous_commands() {
        let registry = ToolRegistry::new();
        let args = serde_json::json!({"command": "rm -rf /"});
        let err = registry.execute_bash(&args).await.unwrap_err();
        assert!(matches!(err, ToolError::BashBlocked(_)));
    }

    #[tokio::test]
    async fn bash_allows_safe_commands() {
        let registry = ToolRegistry::new();
        let args = serde_json::json!({"command": "echo hello"});
        let result = registry.execute_bash(&args).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("hello"));
    }

    #[tokio::test]
    async fn bash_command_substitution_blocked() {
        // F1.1 regression: `echo $(...)` must be blocked, not silently allowed.
        let registry = ToolRegistry::new();
        let args = serde_json::json!({"command": "echo $(rm -rf /)"});
        let err = registry.execute_bash(&args).await.unwrap_err();
        assert!(matches!(err, ToolError::BashBlocked(_)));
    }

    #[tokio::test]
    async fn bash_timeout_returns_failed_result() {
        // F1.3 regression: a command that hangs must be killed and reported as
        // a failed tool result — never hang forever, never abort the session.
        let registry = ToolRegistry::with_timeout(1); // 1 second
        let args = serde_json::json!({"command": "sleep 10"});
        let result = registry.execute_bash(&args).await.unwrap();
        assert!(!result.success);
        assert!(
            result.output.contains("timed out"),
            "got: {}",
            result.output
        );
        assert!(!result.retry);
    }

    /// Gate-3 wiring regression: the sandbox must not defeat the Bash
    /// timeout. The double-fork middle process used to hold std's exec-error
    /// channel open until the workload finished, so `spawn()` blocked the
    /// runtime thread and `tokio::time::timeout` never fired (observed with
    /// `sleep 10` under a 1s budget: success after 10s instead of a timed-out
    /// result at ~1s).
    #[tokio::test]
    async fn bash_timeout_still_fires_under_sandbox() {
        if !crate::sandbox::containment_available() {
            println!(
                "SKIP: kernel containment unavailable (Landlock/userns) — \
                 cannot verify sandbox wiring honestly"
            );
            return;
        }
        let started = std::time::Instant::now();
        let registry = ToolRegistry::with_timeout(1);
        let args = serde_json::json!({"command": "sleep 10"});
        let result = registry.execute_bash(&args).await.unwrap();
        let elapsed = started.elapsed();
        println!("sandboxed timeout test elapsed: {elapsed:?}");
        assert!(
            !result.success,
            "sleep must be killed, got: {}",
            result.output
        );
        assert!(
            result.output.contains("timed out"),
            "got: {}",
            result.output
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "timeout must fire near the 1s budget, took {elapsed:?} — \
             spawn is blocking on the sandbox exec-error channel"
        );
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

    #[tokio::test]
    async fn tool_not_found() {
        let mut registry = ToolRegistry::new();
        let tc = ParsedToolCall {
            name: "Nonexistent".to_string(),
            arguments: serde_json::json!({}),
        };
        let err = registry.execute(&tc).await.unwrap_err();
        assert!(matches!(err, ToolError::NotFound(_)));
    }

    #[test]
    fn read_truncates_large_file() {
        // F3.1: Read output is capped at max_read_lines with a truncation message.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.txt");
        let lines: Vec<String> = (1..=500).map(|i| format!("line {i}")).collect();
        std::fs::write(&path, lines.join("\n")).unwrap();

        let mut registry = ToolRegistry::with_read_limit(300);
        let args = serde_json::json!({"path": path.to_str().unwrap()});
        let result = registry.execute_read(&args).unwrap();
        assert!(result.success);
        assert!(
            result.output.contains("200 lines omitted"),
            "got: {}",
            result.output
        );
        assert!(
            result
                .output
                .contains("use Grep to search specific content"),
            "got: {}",
            result.output
        );
        // First line present
        assert!(result.output.contains("1:"));
        // Line 300 present (last included)
        assert!(result.output.contains("300:"));
        // Line 301 absent (truncated)
        assert!(!result.output.contains("301:"));
    }

    #[test]
    fn grep_respects_gitignore() {
        // F3.2: Grep uses ignore crate — files in .gitignore'd dirs are excluded.
        let dir = tempfile::tempdir().unwrap();
        // Init a git repo so ignore crate picks up .gitignore
        std::process::Command::new("git")
            .args(["init", dir.path().to_str().unwrap()])
            .output()
            .unwrap();
        // Create .gitignore excluding target/
        std::fs::write(dir.path().join(".gitignore"), "target/\n").unwrap();
        // Create a file that SHOULD match (not gitignored)
        std::fs::write(dir.path().join("src.rs"), "needle here").unwrap();
        // Create a file in target/ that should NOT match
        let target_dir = dir.path().join("target");
        std::fs::create_dir_all(&target_dir).unwrap();
        std::fs::write(target_dir.join("build.rs"), "needle here").unwrap();

        let registry = ToolRegistry::new();
        let args = serde_json::json!({
            "pattern": "needle",
            "path": dir.path().to_str().unwrap()
        });
        let result = registry.execute_grep(&args).unwrap();
        assert!(result.success);
        assert!(
            result.output.contains("src.rs"),
            "should find src.rs, got: {}",
            result.output
        );
        assert!(
            !result.output.contains("target/build.rs"),
            "should NOT find target/build.rs, got: {}",
            result.output
        );
    }

    #[test]
    fn edit_blocked_message_mentions_must_read() {
        let f = temp_file("line1\nline2");
        let mut registry = ToolRegistry::new();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "hash": "deadbeef",
            "new": "replaced"
        });
        let err = registry.execute_edit(&args).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("File not loaded"), "got: {msg}");
        assert!(msg.contains("must Read before Edit"), "got: {msg}");
    }

    #[test]
    fn edit_after_read_succeeds() {
        let f = temp_file("hello\nworld");
        let mut registry = ToolRegistry::new();
        // First read
        let read_args = serde_json::json!({"path": f.path().to_str().unwrap()});
        let read_res = registry.execute_read(&read_args).unwrap();
        assert!(read_res.success);
        // Extract hash from first line
        let first_line = read_res.output.lines().next().unwrap();
        // Format is "1:HASH|content"
        let hash = first_line
            .split(':')
            .nth(1)
            .unwrap()
            .split('|')
            .next()
            .unwrap();
        let edit_args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "hash": hash,
            "new": "HELLO"
        });
        let edit_res = registry.execute_edit(&edit_args).unwrap();
        assert!(edit_res.success);
    }

    #[test]
    fn write_blocks_existing_file_message_mentions_edit() {
        let f = temp_file("existing");
        let mut registry = ToolRegistry::new();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "content": "new"
        });
        let err = registry.execute_write(&args).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("use Edit instead"), "got: {msg}");
    }

    #[test]
    fn write_then_edit_without_extra_read_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new.txt");
        let mut registry = ToolRegistry::new();
        let write_args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "content": "line1\nline2"
        });
        registry.execute_write(&write_args).unwrap();
        // Now edit without explicit Read — should succeed because Write populated snapshot
        let read_args = serde_json::json!({"path": path.to_str().unwrap()});
        let read_res = registry.execute_read(&read_args).unwrap();
        let hash = read_res
            .output
            .lines()
            .next()
            .unwrap()
            .split(':')
            .nth(1)
            .unwrap()
            .split('|')
            .next()
            .unwrap();
        let edit_args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "hash": hash,
            "new": "LINE1"
        });
        let edit_res = registry.execute_edit(&edit_args).unwrap();
        assert!(edit_res.success);
    }

    #[tokio::test]
    async fn bash_blocks_redirect_write() {
        let registry = ToolRegistry::new();
        let args = serde_json::json!({"command": "cat > /tmp/evil.txt"});
        let err = registry.execute_bash(&args).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("do not redirect") || msg.contains("Write"),
            "got: {msg}"
        );
    }

    #[tokio::test]
    async fn bash_allows_redirect_to_dev_null() {
        let registry = ToolRegistry::new();
        let args = serde_json::json!({"command": "echo hi > /dev/null"});
        let result = registry.execute_bash(&args).await.unwrap();
        assert!(
            result.success || !result.output.contains("blocked"),
            "should allow /dev/null, got: {:?}",
            result
        );
        // If it blocks, it's a failure — but /dev/null should be allowed
        // Verify no BashBlocked error was thrown
    }

    #[tokio::test]
    async fn bash_blocks_tee_write() {
        let registry = ToolRegistry::new();
        let args = serde_json::json!({"command": "echo x | tee file.log"});
        let err = registry.execute_bash(&args).await.unwrap_err();
        assert!(matches!(err, ToolError::BashBlocked(_)));
    }

    #[tokio::test]
    async fn bash_allows_tee_without_file() {
        let registry = ToolRegistry::new();
        let args = serde_json::json!({"command": "echo hi | tee"});
        let result = registry.execute_bash(&args).await.unwrap();
        assert!(result.success);
    }

    /// Gate-3 finding 1: execute_bash must arm kernel containment. A
    /// Guard-passing command (`cp`, no redirect/substitution) that writes
    /// OUTSIDE the sandbox fs_allow must FAIL under execute_bash, while the
    /// SAME command succeeds without the sandbox (negative control proving
    /// the denial comes from Landlock, not DAC).
    #[tokio::test]
    async fn bash_write_outside_workspace_denied_by_sandbox_with_negative_control() {
        if !crate::sandbox::containment_available() {
            println!(
                "SKIP: kernel containment unavailable (Landlock/userns) — \
                 cannot verify sandbox wiring honestly"
            );
            return;
        }
        if std::env::var("KITTY_SANDBOX").as_deref() == Ok("off") {
            println!("SKIP: KITTY_SANDBOX=off set — containment intentionally disabled");
            return;
        }

        // Source inside the allow-root; victim OUTSIDE any allow-root
        // (fs_allow below deliberately excludes both tempdirs' parents).
        let allow_root = tempfile::tempdir().unwrap();
        let source = allow_root.path().join("src.txt");
        std::fs::write(&source, "payload").unwrap();
        let victim_dir = tempfile::tempdir().unwrap();
        let victim = victim_dir.path().join("victim.txt");
        std::fs::write(&victim, b"original").unwrap();

        let sb = Sandbox::new(crate::sandbox::SandboxConfig {
            fs_allow: vec![allow_root.path().to_path_buf()],
            net_deny: true,
            mem_limit_mb: 512,
            cpu_timeout_secs: 30,
        });
        let registry = ToolRegistry::new().with_sandbox(sb);

        let command = format!("cp {} {}", source.display(), victim.display());
        let args = serde_json::json!({ "command": command });

        // Sandboxed: cp opens the destination O_WRONLY|O_TRUNC -> needs
        // WriteFile (+Truncate) rights outside fs_allow -> Landlock denies.
        let result = registry.execute_bash(&args).await.unwrap();
        println!(
            "sandboxed cp result: success={} output={}",
            result.success, result.output
        );
        assert!(
            !result.success,
            "CONTAINMENT BROKEN: Guard-passing write outside fs_allow \
             succeeded under execute_bash"
        );
        assert_eq!(
            std::fs::read(&victim).unwrap(),
            b"original",
            "victim file must be untouched by the denied write"
        );

        // Negative control: identical command WITHOUT sandbox succeeds.
        let plain = ToolRegistry::new().without_sandbox();
        let ctl = plain.execute_bash(&args).await.unwrap();
        println!(
            "negative-control cp result: success={} output={}",
            ctl.success, ctl.output
        );
        assert!(
            ctl.success,
            "negative control FAILED — plain cp was already denied \
             (output: {}); denial cannot be attributed to the sandbox",
            ctl.output
        );
        assert_eq!(std::fs::read(&victim).unwrap(), b"payload");
    }
}

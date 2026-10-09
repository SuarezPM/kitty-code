//! Tool implementations for the agent loop.
//!
//! Each tool type has a dedicated handler. Safety guards:
//! - Write-guard: refuses Write on existing files (force Edit).
//! - Read-before-edit guard: refuses Edit if file not read this session.
//! - Bash: sanitize command, block dangerous patterns via Guard.
//!
//! Edit supports two protocols:
//! - Hash-snapshot: `{path, hash, new[, line]}` — line-hash based (default).
//! - Str-replace (aider-compatible): `{path, old_string, new_string[, replace_all]}`.
//!
//! CRLF note: line-ending normalization in str_replace mode is FILE-GLOBAL —
//! matching is done LF-normalized and, if the file on disk uses CRLF, ALL
//! line endings are restored uniformly on write. A mixed LF/CRLF file comes
//! out uniform after an edit (all-CRLF or all-LF), never mixed.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

use crate::hashline::HashlineEditor;
use crate::parser::ParsedToolCall;
use crate::sandbox::{Sandbox, SandboxConfig};

#[cfg(test)]
use std::io::Write;

/// Shared test helper (W2): single definition used by `tools/*` submodule tests.
#[cfg(test)]
pub(crate) fn temp_file(content: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(content.as_bytes()).unwrap();
    f
}

// W2 split by symbol: normalization lives in `tools/normalize.rs`.
// Re-exported here so `crate::tools::normalize_tool_call` (used by
// `agent/response.rs`) and intra-crate callers keep working unchanged.
mod normalize;
pub(crate) use normalize::{attempt_raw_repair, normalize_tool_call, raw_only_arguments};
// W2 splits (por símbolo): io = Read/Edit/Write, bash = Bash+containment,
// search = Glob/Grep, lint = gates + cargo-check + diff/reshow feedback.
// `tools.rs` queda como fachada fina: tipos + dispatcher + re-exports.
mod bash;
mod io;
mod lint;
mod search;
pub(crate) use lint::{
    append_lint_feedback, edit_check_enabled, first_changed_line, reshow_region,
    run_extension_check, truncate_lines, unified_diff,
};

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
    pub(crate) read_files: HashSet<PathBuf>,
    /// Files written this session (for write guard).
    pub(crate) written_files: HashSet<PathBuf>,
    /// Successful file-modifying executions this session (Edit/Write, plus
    /// Bash commands whose Guard analysis detected write targets). Consumed
    /// by the verification loop to decide whether test failures are the
    /// agent's doing.
    pub(crate) file_modifications: usize,
    /// The hashline editor for file operations (external code goes
    /// through ToolRegistry methods so the read-before-edit invariant holds).
    pub(crate) hashline: HashlineEditor,
    /// Bash command timeout in seconds (F1.3). Default: 60.
    pub(crate) bash_timeout_secs: u64,
    /// Max characters returned by Bash output (F4.1). Default: 5_000.
    pub(crate) bash_output_cap: usize,
    /// Max lines for Read tool output (F3.1). Default: 300.
    pub(crate) max_read_lines: usize,
    /// Kernel containment armed on every Bash child before spawn
    /// (`None` only when explicitly disabled — see [`Self::without_sandbox`]).
    pub(crate) bash_sandbox: Option<Sandbox>,
    /// W0 (SDD estado-arte): accumulated `Guard::check_chain` wall (ms) for
    /// the current turn. Atomic so `run_bash_command_in` (`&self`) can
    /// record without changing its public signature; reset per turn by the
    /// agent loop via [`Self::reset_turn_timing`]. Truncated to ms — a
    /// single check is ~µs, so this is often 0; still measured, never a
    /// placeholder.
    pub(crate) turn_guard_ms: AtomicU64,
    /// W0: accumulated Bash child-execution wall (ms) for the current turn
    /// (the `SandboxOutcome::duration_ms` quantity, taken at this spawn
    /// site). Same reset/read discipline as [`Self::turn_guard_ms`].
    pub(crate) turn_sandbox_ms: AtomicU64,
    /// F3 (Gate A): edit-journal path for `kitty undo`. `None` = journaling
    /// disabled (default — fully compatible: no file touched, no behavior
    /// change). Set via [`Self::with_journal`]; the cli lane passes its
    /// resolved `default_journal_path()` here on the `run` path.
    pub(crate) journal: Option<PathBuf>,
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
            file_modifications: 0,
            hashline: HashlineEditor::new(),
            bash_timeout_secs,
            bash_output_cap: 5_000,
            max_read_lines: 300,
            bash_sandbox: Some(Sandbox::new(SandboxConfig::new(default_workspace_root()))),
            turn_guard_ms: AtomicU64::new(0),
            turn_sandbox_ms: AtomicU64::new(0),
            journal: None,
        }
    }

    /// F3: enable the edit journal (W3 `kitty undo`) at `path`. Only
    /// successful `Edit`/`Write` executions are recorded (before/after
    /// snapshots); recording is best-effort and never fails the tool call.
    pub fn with_journal(mut self, path: PathBuf) -> Self {
        self.journal = Some(path);
        self
    }

    /// Resolve the active journal path: explicit builder first, then the
    /// `KITTY_JOURNAL` env (same name the cli honors). `None` = disabled.
    fn journal_path(&self) -> Option<PathBuf> {
        self.journal
            .clone()
            .or_else(crate::journal::resolve_journal_path)
    }

    /// W0: zero the per-turn timing accumulators. Called by the agent loop
    /// before `process_response` so the turn's `TurnEvent` only carries its
    /// own guard/sandbox time.
    pub(crate) fn reset_turn_timing(&self) {
        self.turn_guard_ms.store(0, Ordering::Relaxed);
        self.turn_sandbox_ms.store(0, Ordering::Relaxed);
    }

    /// W0: read (and reset) the current turn's accumulated
    /// `(guard_ms, sandbox_ms)`. The swap-reset keeps verification-loop
    /// Bash runs (which happen outside any turn) from leaking into the
    /// next turn's event.
    pub(crate) fn take_turn_timing(&self) -> (u64, u64) {
        (
            self.turn_guard_ms.swap(0, Ordering::Relaxed),
            self.turn_sandbox_ms.swap(0, Ordering::Relaxed),
        )
    }

    /// Override the default Read window cap (chainable, f8 T2.3/P4 Rider B).
    /// The agent wires [`ModelProfile::read_cap`] here so the active profile
    /// actually governs the default window; tests use it directly.
    pub fn with_read_cap(mut self, max_read_lines: usize) -> Self {
        self.max_read_lines = max_read_lines;
        self
    }

    /// Override the default Bash output cap (F4.1).
    /// The agent wires [`ModelProfile::bash_output_cap`] here so the active
    /// profile governs the truncation threshold.
    pub fn with_bash_output_cap(mut self, bash_output_cap: usize) -> Self {
        self.bash_output_cap = bash_output_cap;
        self
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

    /// Normalize + repair prelude shared by [`Self::execute`] and the W5
    /// speculative prefetch (same code path ⇒ same verdicts, no drift).
    /// `Ok(call)` = dispatchable; `Err(result)` = early failed result
    /// (unrepairable args) to return as-is.
    pub(crate) fn prepare_call(tc: &ParsedToolCall) -> Result<ParsedToolCall, ToolResult> {
        // Alias normalization: weaker models emit near-miss tool names and
        // argument keys; both fold onto the canonical forms at this single
        // choke point shared by the agent loop and the MCP server. Unknown
        // names pass through untouched so existing diagnostics still apply.
        let call = normalize_tool_call(tc);

        // G3-5: a salvaged `_raw`-only arguments object means the model's
        // JSON body could not be parsed. Attempt JSON repair pipeline before
        // giving up: parse the raw body, then apply schema-aware coercion.
        if let Some(raw) = raw_only_arguments(&call.arguments) {
            if let Some(repaired) = attempt_raw_repair(raw) {
                // Repair succeeded — rebuild the call with repaired arguments
                // and re-normalize (aliases may have been in the raw body).
                Ok(normalize_tool_call(&ParsedToolCall {
                    name: call.name,
                    arguments: repaired,
                    id: call.id.clone(),
                }))
            } else {
                Err(ToolResult {
                    success: false,
                    output: format!(
                        "Tool call arguments were not valid JSON, so the call \
                         could not be applied.\nRecovered body (truncated):\n{}\n\n\
                         Re-send the call as proper JSON: \
                         {{\"name\": \"<Tool>\", \"arguments\": {{...}}}}.",
                        truncate_lines(raw, 6)
                    ),
                    retry: true,
                })
            }
        } else {
            Ok(call)
        }
    }

    /// Execute a parsed tool call.
    pub async fn execute(&mut self, tc: &ParsedToolCall) -> Result<ToolResult, ToolError> {
        let call = match Self::prepare_call(tc) {
            Ok(call) => call,
            Err(early) => return Ok(early),
        };

        // F3 (Gate A): journal pre-image for Edit/Write so agent-loop edits
        // become revertible via `kitty undo` (same schema as the cli
        // journal). Snapshot BEFORE dispatch; recorded only when the
        // execution below reports success. Bash write-target commands are
        // intentionally NOT journaled (multi-target restores are the cli
        // lane's decision, mirroring `McpServer::call_tool`).
        let journal_path = self.journal_path();
        let journal_before: Option<Option<String>> = match (&journal_path, call.name.as_str()) {
            (Some(_), "Edit" | "Write") => {
                let p = call
                    .arguments
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                Some(tokio::fs::read_to_string(p).await.ok())
            }
            _ => None,
        };

        let result = match call.name.as_str() {
            "Read" => self.execute_read(&call.arguments).await?,
            "Edit" => self.execute_edit(&call.arguments).await?,
            "Write" => self.execute_write(&call.arguments).await?,
            "Bash" => self.execute_bash(&call.arguments).await?,
            "Glob" => self.execute_glob(&call.arguments).await?,
            "Grep" => self.execute_grep(&call.arguments).await?,
            other => return Err(ToolError::NotFound(other.to_string())),
        };

        // F3: post-image on success. Best-effort: a journal I/O error warns
        // but never fails the (already successful) tool call.
        if let (Some(journal), Some(before)) = (&journal_path, &journal_before) {
            if result.success {
                if let Some(path) = call.arguments.get("path").and_then(|v| v.as_str()) {
                    let after = tokio::fs::read_to_string(path).await.unwrap_or_default();
                    let entry = crate::journal::entry_for(path, before.clone(), after);
                    if let Err(e) = crate::journal::append_entry(journal, &entry) {
                        tracing::warn!(
                            journal = %journal.display(),
                            error = %e,
                            "edit journal append failed (tool result kept)"
                        );
                    }
                }
            }
        }
        if result.success {
            match call.name.as_str() {
                "Edit" | "Write" => self.file_modifications += 1,
                "Bash" => {
                    // A successful shell command counts as a file modification
                    // only when Guard's write-target analysis saw one (redirects,
                    // tee, dd, sed -i …). Read-only commands stay neutral.
                    if let Some(cmd) = call.arguments.get("command").and_then(|v| v.as_str()) {
                        if !crate::guard::detect_write_targets(cmd).is_empty() {
                            self.file_modifications += 1;
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(result)
    }

    /// Number of successful file-modifying executions this session
    /// (Edit/Write plus write-target Bash commands).
    pub fn file_modifications(&self) -> usize {
        self.file_modifications
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
            "To modify an existing file: Read it, then use Edit with old_string/new_string \
             (exact text copied from the file). Never use Write on existing files.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to edit"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact text to replace, copied from the file (str_replace mode). Include enough surrounding lines to be unique."
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text for old_string"
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence of old_string instead of failing on multiple matches (default false)"
                    },
                    "hash": {
                        "type": "string",
                        "description": "Alternative line-hash mode: line hash from Read output"
                    },
                    "new": {
                        "type": "string",
                        "description": "Line-hash mode: new content for the line"
                    },
                    "line": {
                        "type": "integer",
                        "description": "Line-hash mode: line number to disambiguate when hash prefix matches multiple lines"
                    }
                },
                "required": ["path"]
            }),
        ),
        kitty_runtime::ToolDefinition::new(
            "Write",
            "Create a NEW file. Never use Write on existing files — to modify an existing file: Read it, then use Edit with old_string/new_string.",
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

    /// Gate-3 wiring regression: the sandbox must not defeat the Bash
    /// timeout. The double-fork middle process used to hold std's exec-error
    /// channel open until the workload finished, so `spawn()` blocked the
    /// runtime thread and `tokio::time::timeout` never fired (observed with
    /// `sleep 10` under a 1s budget: success after 10s instead of a timed-out
    /// result at ~1s).

    #[tokio::test]
    async fn tool_not_found() {
        let mut registry = ToolRegistry::new();
        let tc = ParsedToolCall {
            name: "Nonexistent".to_string(),
            arguments: serde_json::json!({}),
            id: None,
        };
        let err = registry.execute(&tc).await.unwrap_err();
        assert!(matches!(err, ToolError::NotFound(_)));
    }

    fn read_journal(path: &std::path::Path) -> Vec<crate::journal::JournalEntry> {
        let text = std::fs::read_to_string(path).unwrap();
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    /// F3: `Write` of a new file records `{before: None, after}` via `execute`.
    #[tokio::test]
    async fn f3_journal_records_write_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let journal = dir.path().join("journal.jsonl");
        let target = dir.path().join("new.txt");
        let mut registry = ToolRegistry::new().with_journal(journal.clone());
        let tc = ParsedToolCall {
            name: "Write".to_string(),
            arguments: serde_json::json!({
                "path": target.to_str().unwrap(),
                "content": "hello",
            }),
            id: None,
        };
        let result = registry.execute(&tc).await.unwrap();
        assert!(result.success);
        let entries = read_journal(&journal);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].path, target.to_str().unwrap());
        assert_eq!(entries[0].before, None);
        assert_eq!(entries[0].after, "hello");
    }

    /// F3: `Read` + `Edit` records `{before, after}` with the pre-image.
    #[tokio::test]
    async fn f3_journal_records_edit_with_before() {
        let dir = tempfile::tempdir().unwrap();
        let journal = dir.path().join("journal.jsonl");
        let target = dir.path().join("e.txt");
        std::fs::write(&target, "line1\nline2").unwrap();
        let mut registry = ToolRegistry::new().with_journal(journal.clone());
        let path = target.to_str().unwrap();
        registry
            .execute(&ParsedToolCall {
                name: "Read".to_string(),
                arguments: serde_json::json!({"path": path}),
                id: None,
            })
            .await
            .unwrap();
        let edit = registry
            .execute(&ParsedToolCall {
                name: "Edit".to_string(),
                arguments: serde_json::json!({
                    "path": path,
                    "old_string": "line1",
                    "new_string": "LINE1",
                }),
                id: None,
            })
            .await
            .unwrap();
        assert!(edit.success, "edit failed: {}", edit.output);
        let entries = read_journal(&journal);
        assert_eq!(entries.len(), 1, "only the Edit is journaled, not the Read");
        assert_eq!(entries[0].before.as_deref(), Some("line1\nline2"));
        assert!(entries[0].after.contains("LINE1"));
        assert_eq!(std::fs::read_to_string(&target).unwrap(), entries[0].after);
    }

    /// F3: failed executions (EditBlocked) record nothing.
    #[tokio::test]
    async fn f3_journal_skips_failed_edit() {
        let dir = tempfile::tempdir().unwrap();
        let journal = dir.path().join("journal.jsonl");
        let target = dir.path().join("e.txt");
        std::fs::write(&target, "data").unwrap();
        let mut registry = ToolRegistry::new().with_journal(journal.clone());
        // No prior Read → EditBlocked error, no mutation, no entry.
        let err = registry
            .execute(&ParsedToolCall {
                name: "Edit".to_string(),
                arguments: serde_json::json!({
                    "path": target.to_str().unwrap(),
                    "old_string": "data",
                    "new_string": "DATA",
                }),
                id: None,
            })
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::EditBlocked(_)));
        assert!(
            !journal.exists(),
            "failed edits must not create journal entries"
        );
    }

    /// F3: journaling is disabled by default (no path, no env) — compatible.
    #[tokio::test]
    async fn f3_journal_disabled_by_default() {
        let _g = crate::model_profile::test_env_lock();
        let _e = crate::model_profile::EnvRestoreGuard::remove(crate::journal::ENV_JOURNAL);
        let registry = ToolRegistry::new();
        assert_eq!(registry.journal_path(), None);
    }
}

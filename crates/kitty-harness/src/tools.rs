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
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use thiserror::Error;

use crate::guard::{Guard, GuardVerdict};
use crate::hashline::HashlineEditor;
use crate::parser::ParsedToolCall;
use crate::redact::redact_secrets;
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
    /// Successful file-modifying executions this session (Edit/Write, plus
    /// Bash commands whose Guard analysis detected write targets). Consumed
    /// by the verification loop to decide whether test failures are the
    /// agent's doing.
    file_modifications: usize,
    /// The hashline editor for file operations.
    pub hashline: HashlineEditor,
    /// Bash command timeout in seconds (F1.3). Default: 60.
    bash_timeout_secs: u64,
    /// Max characters returned by Bash output (F4.1). Default: 5_000.
    bash_output_cap: usize,
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
            file_modifications: 0,
            hashline: HashlineEditor::new(),
            bash_timeout_secs,
            bash_output_cap: 5_000,
            max_read_lines: 300,
            bash_sandbox: Some(Sandbox::new(SandboxConfig::new(default_workspace_root()))),
        }
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

    /// Execute a parsed tool call.
    pub async fn execute(&mut self, tc: &ParsedToolCall) -> Result<ToolResult, ToolError> {
        // Alias normalization: weaker models emit near-miss tool names and
        // argument keys; both fold onto the canonical forms at this single
        // choke point shared by the agent loop and the MCP server. Unknown
        // names pass through untouched so existing diagnostics still apply.
        let call = normalize_tool_call(tc);

        // G3-5: a salvaged `_raw`-only arguments object means the model's
        // JSON body could not be parsed. Attempt JSON repair pipeline before
        // giving up: parse the raw body, then apply schema-aware coercion.
        let call = if let Some(raw) = raw_only_arguments(&call.arguments) {
            if let Some(repaired) = attempt_raw_repair(raw) {
                // Repair succeeded — rebuild the call with repaired arguments
                // and re-normalize (aliases may have been in the raw body).
                normalize_tool_call(&ParsedToolCall {
                    name: call.name,
                    arguments: repaired,
                })
            } else {
                return Ok(ToolResult {
                    success: false,
                    output: format!(
                        "Tool call arguments were not valid JSON, so the call \
                         could not be applied.\nRecovered body (truncated):\n{}\n\n\
                         Re-send the call as proper JSON: \
                         {{\"name\": \"<Tool>\", \"arguments\": {{...}}}}.",
                        truncate_lines(raw, 6)
                    ),
                    retry: true,
                });
            }
        } else {
            call
        };

        let result = match call.name.as_str() {
            "Read" => self.execute_read(&call.arguments)?,
            "Edit" => self.execute_edit(&call.arguments).await?,
            "Write" => self.execute_write(&call.arguments).await?,
            "Bash" => self.execute_bash(&call.arguments).await?,
            "Glob" => self.execute_glob(&call.arguments)?,
            "Grep" => self.execute_grep(&call.arguments)?,
            other => return Err(ToolError::NotFound(other.to_string())),
        };
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

    /// Execute a Read tool call.
    ///
    /// Output is an HONEST window (f8 T2.3): the FIRST line is a header
    /// `"{path}: lines {start}-{end} of {total}"` naming the exact returned
    /// range, followed by hashline-tagged `LINE:hash|content` lines.
    ///
    /// Optional args (weak-model tolerant — invalid values are CLAMPED,
    /// never strict errors):
    /// - `offset`: 1-based first line, default 1 (`<1` → 1; non-numeric
    ///   strings → default).
    /// - `limit`: max lines returned, default = registry read cap
    ///   ([`Self::with_read_cap`], profile-driven) (`<1` → cap).
    ///
    /// The hashline snapshot is always the FULL file, so edits outside the
    /// displayed window succeed normally — the window is display-only.
    fn execute_read(&mut self, args: &serde_json::Value) -> Result<ToolResult, ToolError> {
        let path =
            args.get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidArgs {
                    tool: "Read".to_string(),
                    reason: "missing 'path'".to_string(),
                })?;

        let path_buf = PathBuf::from(path);

        // Read with hashline tagging (full-file snapshot regardless of window)
        let snapshot = self.hashline.read_file(&path_buf)?;
        self.read_files.insert(path_buf);

        // Lenient numeric coercion: real numbers and string numerals both
        // work; anything else yields None → caller default.
        let lenient_usize = |v: Option<&serde_json::Value>| -> Option<usize> {
            let v = v?;
            v.as_u64()
                .or_else(|| v.as_str().and_then(|s| s.trim().parse::<u64>().ok()))
                .map(|n| n as usize)
        };

        let total_lines = snapshot.lines.len();
        let cap = self.max_read_lines;

        let offset = lenient_usize(args.get("offset")).unwrap_or(1).max(1);
        let limit_requested = lenient_usize(args.get("limit")).unwrap_or(cap);
        let limit = if limit_requested == 0 {
            cap
        } else {
            limit_requested.min(cap)
        };

        // Clamp the requested start into the file (offset beyond EOF reads
        // from EOF backwards by one line rather than erroring).
        let idx0 = offset.saturating_sub(1).min(total_lines.saturating_sub(1));
        let idx_end = (idx0 + limit).min(total_lines);
        let first_line = snapshot.lines.get(idx0).map_or(0, |l| l.line_num);
        let last_line = snapshot
            .lines
            .get(idx_end.saturating_sub(1))
            .map_or(0, |l| l.line_num);

        // Format output with line numbers and hashes
        let tagged: Vec<String> = snapshot.lines[idx0..idx_end]
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

        let mut output = format!("{path}: lines {first_line}-{last_line} of {total_lines}\n");
        output.push_str(&tagged.join("\n"));

        let omitted_after = total_lines - idx_end;
        if omitted_after > 0 {
            // Content follows the window — suggest the exact next window.
            output.push_str(&format!(
                "\n... ({omitted_after} lines after this window — retry Read with \
                 \"offset\": {}, \"limit\": {limit}, or use Grep to search specific content)",
                idx_end + 1
            ));
        } else if idx0 > 0 {
            // Tail-window read: nothing after, so a next-offset would be
            // bogus — state the mid-file start instead (G2 rider).
            output.push_str(&format!(
                "\n... (window starts at line {first_line} of {total_lines} — prior content exists)"
            ));
        }

        Ok(ToolResult {
            success: true,
            output,
            retry: false,
        })
    }

    /// Execute an Edit tool call.
    ///
    /// Two protocols:
    /// - Str-replace (aider-compatible): args contain `old_string` (+ `new_string`,
    ///   optional `replace_all`). Matches exact text in the current file content.
    /// - Hash-snapshot: args contain `hash` (+ `new`, optional `line`).
    async fn execute_edit(&mut self, args: &serde_json::Value) -> Result<ToolResult, ToolError> {
        let path =
            args.get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidArgs {
                    tool: "Edit".to_string(),
                    reason: "missing 'path'".to_string(),
                })?;

        let path_buf = PathBuf::from(path);

        // Aider-compatible str_replace mode takes precedence when old_string is present.
        if let Some(old_string) = args.get("old_string").and_then(|v| v.as_str()) {
            // E-G1b: empty old_string would match everywhere — fail fast
            // before any occurrence counting or file IO.
            if old_string.is_empty() {
                return Ok(ToolResult {
                    success: false,
                    output: "old_string must not be empty".to_string(),
                    retry: false,
                });
            }
            let mut result = self
                .execute_edit_str_replace(path, &path_buf, old_string, args)
                .await?;
            // E-G1a: dual-mode args — when both old_string and hash are
            // provided, str_replace wins and the hash is silently ignored;
            // surface that explicitly so the model can correct its args.
            if args.get("hash").and_then(|v| v.as_str()).is_some() {
                result.output = format!(
                    "note: hash ignored; str_replace mode active\n{}",
                    result.output
                );
            }
            return Ok(result);
        }

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
                    let before = std::fs::read_to_string(&path_buf)?;
                    std::fs::write(&path_buf, &new_file_content)?;
                    self.hashline.read_file(&path_buf)?;
                    let mut output = format!("Edit applied to {path} (line {line})");
                    output.push_str(
                        &self
                            .build_edit_feedback(path, &path_buf, &before, &new_file_content)
                            .await,
                    );
                    Ok(ToolResult {
                        success: true,
                        output,
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
                    let before = std::fs::read_to_string(&path_buf)?;
                    std::fs::write(&path_buf, &new_file_content)?;
                    // Re-read to get updated hashes
                    self.hashline.read_file(&path_buf)?;
                    let mut output = format!("Edit applied to {path}");
                    output.push_str(
                        &self
                            .build_edit_feedback(path, &path_buf, &before, &new_file_content)
                            .await,
                    );
                    Ok(ToolResult {
                        success: true,
                        output,
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

    /// Str-replace Edit mode (aider-compatible).
    ///
    /// Requires a prior Read this session (same read-before-edit invariant as
    /// the hash protocol). Matching is two-stage:
    ///
    /// 1. EXACT: occurrences of `old_string` in the current file content —
    ///    0 matches → fall through to the tolerance ladder; >1 match without
    ///    `replace_all` → error asking for more context.
    /// 2. FUZZY LADDER ([`crate::edit_replacers::run_ladder`]): graduated
    ///    strategies (line-trim → whitespace-normalized → trimmed-boundary →
    ///    escape-normalized → indent-flexible → context-aware → block-anchor)
    ///    tolerate whitespace/indentation drift from weaker models. Every
    ///    candidate passes the disproportionate-match guard, which rejects
    ///    spans wildly larger than `old_string` so a loose match can never
    ///    swallow half a file. A fuzzy success appends a one-line
    ///    `[edit matched via <strategy>]` note to the output so the model
    ///    learns its drift was tolerated; multiple viable spans without
    ///    `replace_all` still fail with the usual "provide more context".
    ///
    /// CRLF handling (kilocode-style): `\r\n` is normalized to `\n` for
    /// matching, and restored on write when the original file used CRLF.
    async fn execute_edit_str_replace(
        &mut self,
        path: &str,
        path_buf: &Path,
        old_string: &str,
        args: &serde_json::Value,
    ) -> Result<ToolResult, ToolError> {
        let new_string = args
            .get("new_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs {
                tool: "Edit".to_string(),
                reason: "missing 'new_string' (required alongside 'old_string')".to_string(),
            })?;

        let replace_all = args
            .get("replace_all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Read-before-edit guard — same invariant as the hash protocol
        if self.hashline.get_snapshot(path_buf).is_none() {
            return Err(ToolError::EditBlocked(path.to_string()));
        }

        let before_raw = std::fs::read_to_string(path_buf)?;
        let had_crlf = before_raw.contains("\r\n");
        let normalize = |s: &str| s.replace("\r\n", "\n");
        let before = normalize(&before_raw);
        let old_n = normalize(old_string);
        let new_n = normalize(new_string);

        let count = before.matches(&old_n).count();
        if count == 0 {
            // Exact match failed: run the tolerance ladder (guarded fuzzy
            // strategies) before giving up, so whitespace/indentation drift
            // from weaker models still lands atomically.
            let hit = match crate::edit_replacers::run_ladder(&before, &old_n, replace_all) {
                crate::edit_replacers::LadderOutcome::Hit(hit) => hit,
                crate::edit_replacers::LadderOutcome::Ambiguous => {
                    return Ok(ToolResult {
                        success: false,
                        output: format!(
                            "old_string matches multiple locations in {path} \
                             (after whitespace tolerance). Provide more surrounding \
                             context to make it unique, or set replace_all=true."
                        ),
                        retry: false,
                    });
                }
                crate::edit_replacers::LadderOutcome::NotFound => {
                    return Ok(ToolResult {
                        success: false,
                        output: format!(
                            "old_string not found in {path}. \
                             Read the file first, then pass the exact text from it \
                             (whitespace and indentation matter)."
                        ),
                        retry: false,
                    });
                }
            };
            let after_n = if hit.all_occurrences {
                before.replace(&hit.span, &new_n)
            } else {
                before.replacen(&hit.span, &new_n, 1)
            };
            let mut output = format!("Edit applied to {path}");
            output.push_str(&format!("\n\n[edit matched via {}]", hit.label));
            output.push_str(
                &self
                    .finish_str_replace(path_buf, &before, after_n, had_crlf)
                    .await?,
            );
            return Ok(ToolResult {
                success: true,
                output,
                retry: false,
            });
        }
        if count > 1 && !replace_all {
            return Ok(ToolResult {
                success: false,
                output: format!(
                    "old_string matches {count} locations in {path}. \
                     Provide more surrounding context to make it unique, \
                     or set replace_all=true."
                ),
                retry: false,
            });
        }

        let after_n = if replace_all {
            before.replace(&old_n, &new_n)
        } else {
            before.replacen(&old_n, &new_n, 1)
        };

        let mut output = format!("Edit applied to {path}");
        output.push_str(
            &self
                .finish_str_replace(path_buf, &before, after_n, had_crlf)
                .await?,
        );
        Ok(ToolResult {
            success: true,
            output,
            retry: false,
        })
    }

    /// Shared str-replace write-back: restore the file's original CRLF line
    /// endings, persist, refresh the hashline snapshot, and build the edit
    /// feedback (diff + re-show). Returns the feedback text to append.
    async fn finish_str_replace(
        &mut self,
        path_buf: &Path,
        before: &str,
        after_n: String,
        had_crlf: bool,
    ) -> Result<String, ToolError> {
        let after = if had_crlf {
            after_n.replace('\n', "\r\n")
        } else {
            after_n
        };
        std::fs::write(path_buf, &after)?;
        self.hashline.read_file(path_buf)?;
        Ok(self
            .build_edit_feedback(&path_buf.to_string_lossy(), path_buf, before, &after)
            .await)
    }

    /// Feedback appended after every successful edit (either protocol):
    /// unified diff snippet (capped), re-show of the edited region (±10
    /// lines), and — when `KITTY_EDIT_CHECK=1` — a cheap syntax check.
    async fn build_edit_feedback(
        &self,
        path: &str,
        path_buf: &Path,
        before: &str,
        after: &str,
    ) -> String {
        let mut out = String::new();

        let diff = unified_diff(before, after, 40);
        if !diff.is_empty() {
            out.push_str("\n\n--- Diff ---\n");
            out.push_str(&diff);
        }

        if let Some(center) = first_changed_line(before, after) {
            if let Some(snapshot) = self.hashline.get_snapshot(path_buf) {
                let start = center.saturating_sub(10).max(1);
                let end = (center + 10).min(snapshot.lines.len());
                if start <= end {
                    out.push_str(&format!("\n\n--- {path} lines {start}-{end} ---\n"));
                    out.push_str(&reshow_region(&snapshot.lines, center, 10));
                }
            }
        }

        if edit_check_enabled() {
            if let Some(check) = run_extension_check(path_buf) {
                out.push_str(&format!("\n\n--- Check ({}) ---\n", check.label));
                if check.passed {
                    out.push_str("OK");
                } else {
                    out.push_str(&format!("FAILED:\n{}", truncate_lines(&check.detail, 10)));
                }
            }
        }

        append_lint_feedback(&mut out, path_buf).await;

        out
    }

    /// Execute a Write tool call.
    async fn execute_write(&mut self, args: &serde_json::Value) -> Result<ToolResult, ToolError> {
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

        let mut output = format!("Wrote {path}");
        append_lint_feedback(&mut output, &path_buf).await;

        Ok(ToolResult {
            success: true,
            output,
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
        self.run_bash_command(command, Duration::from_secs(self.bash_timeout_secs))
            .await
    }

    /// Run a shell command under the same containment stack as the Bash tool
    /// (`Guard::check_chain` policy → `Sandbox::apply` kernel containment →
    /// hard timeout), inheriting the process working directory. Used by the
    /// Bash tool itself.
    pub async fn run_bash_command(
        &self,
        command: &str,
        timeout: Duration,
    ) -> Result<ToolResult, ToolError> {
        self.run_bash_command_in(command, timeout, None).await
    }

    /// Like [`Self::run_bash_command`] but optionally executing `command` in
    /// `dir`. The verification loop passes the workspace so project test
    /// commands (`make test`, `cargo test`, …) resolve the right project.
    pub async fn run_bash_command_in(
        &self,
        command: &str,
        timeout: Duration,
        dir: Option<&Path>,
    ) -> Result<ToolResult, ToolError> {
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

        // Execute the command with a timeout. The child is spawned in its own
        // process group via `setsid()` so we can kill the entire tree on timeout.
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c")
            .arg(command)
            .kill_on_drop(false)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(dir) = dir {
            cmd.current_dir(dir);
        }
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
        // F3.1: Create a new process group so we can kill the entire tree on
        // timeout. The hook runs inside the forked child before exec.
        #[cfg(unix)]
        unsafe {
            cmd.pre_exec(|| {
                let ret = nix::libc::setsid();
                if ret == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
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
        // Save PID before consuming child with wait_with_output.
        let child_pid = child.id();

        let output = match tokio::time::timeout(timeout, child.wait_with_output()).await {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => return Err(ToolError::Io(e)),
            Err(_elapsed) => {
                // F3.1: Explicitly kill the entire process group on timeout.
                // SIGTERM first for graceful shutdown, then SIGKILL after 2s grace.
                #[cfg(unix)]
                if let Some(id) = child_pid {
                    use nix::sys::signal::{kill, Signal};
                    use nix::unistd::Pid;
                    let pgid = Pid::from_raw(-(id as i32)); // negative = process group
                    let _ = kill(pgid, Signal::SIGTERM);
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    let _ = kill(pgid, Signal::SIGKILL);
                }
                return Ok(ToolResult {
                    success: false,
                    output: format!("command timed out after {}s (killed)", timeout.as_secs()),
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

        // Truncate very long output (F4.1: profile-driven cap).
        let max_output = self.bash_output_cap;
        if result.len() > max_output {
            result.truncate(max_output);
            result.push_str("\n... (output truncated)");
        }

        // F3.2: Redact secrets before returning output to the agent.
        let result = redact_secrets(&result);

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

/// Canonical tool names served by [`ToolRegistry`].
const REAL_TOOLS: [&str; 6] = ["Read", "Edit", "Write", "Bash", "Glob", "Grep"];

/// Common hallucinated synonyms mapped onto the real tools. Never consulted
/// before the exact/case-insensitive real-name check, so a synonym can never
/// shadow a canonical tool name.
const TOOL_SYNONYMS: &[(&str, &str)] = &[
    // str_replace-style editors → Edit
    ("str_replace", "Edit"),
    ("str_replace_editor", "Edit"),
    ("str_replace_based_edit_tool", "Edit"),
    ("edit_file", "Edit"),
    // file viewers → Read
    ("view", "Read"),
    ("read_file", "Read"),
    ("cat", "Read"),
    // file creators → Write
    ("write_file", "Write"),
    ("create_file", "Write"),
    // shell aliases → Bash
    ("shell", "Bash"),
    ("terminal", "Bash"),
    ("run_command", "Bash"),
    // directory listings / glob-style names → Glob
    ("find", "Glob"),
    ("ls", "Glob"),
    ("list_files", "Glob"),
    ("list_dir", "Glob"),
    // search synonyms → Grep
    ("search", "Grep"),
    ("search_files", "Grep"),
];

/// Argument-key aliases folded onto the canonical keys, applied ONLY when the
/// canonical key is absent from the arguments object.
const ARG_KEY_ALIASES: &[(&str, &str)] = &[
    ("file_path", "path"),
    ("filepath", "path"),
    ("file", "path"),
    ("cmd", "command"),
    ("query", "pattern"),
    ("old_str", "old_string"),
    ("old_text", "old_string"),
    ("new_str", "new_string"),
    ("new_text", "new_string"),
];

/// Extract the salvage body when arguments carry ONLY an `_raw` string (the
/// parser's unparsable-JSON retention shape). Mixed objects pass through.
fn raw_only_arguments(args: &serde_json::Value) -> Option<&str> {
    let map = args.as_object()?;
    if map.len() == 1 {
        map.get("_raw")?.as_str()
    } else {
        None
    }
}

/// Attempt to repair a `_raw` salvage body into usable JSON arguments.
///
/// Uses [`crate::json_repair::repair_json`] to fix control chars and
/// invalid escapes. Returns `Some(repaired_value)` on success,
/// `None` if the body cannot be salvaged.
fn attempt_raw_repair(raw: &str) -> Option<serde_json::Value> {
    // First try the parser's structural repair (smart quotes, trailing commas)
    let structurally_repaired = crate::parser::repair_json(raw);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&structurally_repaired) {
        if v.is_object() {
            return Some(v);
        }
    }
    // Then try the low-level repair (control chars, invalid escapes)
    let repaired = crate::json_repair::repair_json(raw);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&repaired) {
        if v.is_object() {
            return Some(v);
        }
    }
    None
}

/// Fold one incoming call onto canonical tool/argument naming.
///
/// Idempotent: canonical names pass through unchanged and aliased keys are
/// only moved when the canonical key is absent, so re-normalizing an already
/// normalized call is a no-op.
pub(crate) fn normalize_tool_call(tc: &ParsedToolCall) -> ParsedToolCall {
    ParsedToolCall {
        name: normalize_tool_name(&tc.name),
        arguments: rekey_arg_aliases(tc.arguments.clone()),
    }
}

/// Resolve `name` onto the canonical casing of a real tool (case-insensitive)
/// or a known hallucinated synonym; unknown names pass through unchanged.
fn normalize_tool_name(name: &str) -> String {
    if let Some(real) = REAL_TOOLS.iter().find(|t| t.eq_ignore_ascii_case(name)) {
        return (*real).to_string();
    }
    if let Some((_, real)) = TOOL_SYNONYMS
        .iter()
        .find(|(s, _)| s.eq_ignore_ascii_case(name))
    {
        return (*real).to_string();
    }
    name.to_string()
}

/// Re-key aliased argument names onto canonical ones; an alias is only moved
/// when the canonical key is absent, and consumed aliases are removed.
fn rekey_arg_aliases(args: serde_json::Value) -> serde_json::Value {
    let serde_json::Value::Object(mut map) = args else {
        return args;
    };
    for (alias, canonical) in ARG_KEY_ALIASES {
        if let Some(value) = map.remove(*alias) {
            map.entry((*canonical).to_string()).or_insert(value);
        }
    }
    serde_json::Value::Object(map)
}

/// Whether per-edit syntax checks are enabled (`KITTY_EDIT_CHECK=1`).
/// Off by default: checks cost a subprocess per edit; benchmarks opt in.
fn edit_check_enabled() -> bool {
    std::env::var("KITTY_EDIT_CHECK").as_deref() == Ok("1")
}

/// Whether the post-edit lint gate is enabled (`KITTY_LINT_GATE=1`).
/// Off by default (same cost rationale as [`edit_check_enabled`]).
fn lint_gate_enabled() -> bool {
    std::env::var("KITTY_LINT_GATE").as_deref() == Ok("1")
}

/// Append the lint-gate section to a tool output when `KITTY_LINT_GATE=1`.
///
/// `.rs` files get `cargo check --message-format=json` on the nearest crate;
/// `.py` files get `python3 -m py_compile`. Other extensions are not gated.
async fn append_lint_feedback(out: &mut String, path_buf: &Path) {
    if !lint_gate_enabled() {
        return;
    }
    let lint = match path_buf.extension().and_then(|e| e.to_str()) {
        Some("rs") => cargo_check_lint(path_buf).await,
        Some("py") => run_extension_check(path_buf),
        _ => None,
    };
    if let Some(lint) = lint {
        out.push_str(&format!("\n\n--- Lint ({}) ---\n", lint.label));
        if lint.passed {
            out.push_str("OK");
        } else {
            out.push_str(&format!("FAILED:\n{}", truncate_lines(&lint.detail, 10)));
        }
    }
}

/// Hard ceiling for one `cargo check` lint run before the child is killed.
const LINT_GATE_TIMEOUT_SECS: u64 = 120;
/// Maximum compiler diagnostics rendered into the tool output.
const LINT_GATE_MAX_DIAGS: usize = 10;

/// Nearest ancestor directory of `path` containing a `Cargo.toml`.
fn nearest_crate_root(path: &Path) -> Option<PathBuf> {
    let mut dir = path.parent();
    while let Some(d) = dir {
        if d.join("Cargo.toml").is_file() {
            return Some(d.to_path_buf());
        }
        dir = d.parent();
    }
    None
}

/// One rendered rustc diagnostic from `cargo check --message-format=json`.
#[derive(Debug)]
struct RustcDiag {
    level: String,
    rendered: String,
}

/// Extract error/warning diagnostics from `cargo check --message-format=json`
/// output (one JSON object per line). Non-compiler-message lines (artifact
/// notifications, diagnostics metadata) are ignored.
fn parse_cargo_check_json(stdout: &str) -> Vec<RustcDiag> {
    let mut diags = Vec::new();
    for line in stdout.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if v.get("reason").and_then(|r| r.as_str()) != Some("compiler-message") {
            continue;
        }
        let Some(msg) = v.get("message") else {
            continue;
        };
        let level = msg
            .get("level")
            .and_then(|l| l.as_str())
            .unwrap_or("")
            .to_string();
        if level != "error" && level != "warning" {
            continue;
        }
        // Prefer rustc's pre-rendered text; fall back to message + primary span.
        let rendered = match msg.get("rendered").and_then(|r| r.as_str()) {
            Some(r) if !r.trim().is_empty() => r.trim().to_string(),
            _ => {
                let text = msg
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("")
                    .to_string();
                let span = msg
                    .get("spans")
                    .and_then(|s| s.as_array())
                    .and_then(|a| {
                        a.iter().find(|s| {
                            s.get("is_primary")
                                .and_then(|p| p.as_bool())
                                .unwrap_or(false)
                        })
                    })
                    .or_else(|| {
                        msg.get("spans")
                            .and_then(|s| s.as_array())
                            .and_then(|a| a.first())
                    });
                match span {
                    Some(s) => {
                        let file = s.get("file_name").and_then(|f| f.as_str()).unwrap_or("?");
                        let line = s.get("line_start").and_then(|l| l.as_u64()).unwrap_or(0);
                        format!("{text} ({file}:{line})")
                    }
                    None => text,
                }
            }
        };
        diags.push(RustcDiag { level, rendered });
    }
    diags
}

/// Create a unique scratch file for capturing subprocess output (std-only;
/// `tempfile` is a dev-dependency in this crate). Opened read+write so the
/// parent can read the captured output back after the child exits. Caller
/// removes the file.
fn scratch_file(label: &str) -> std::io::Result<(std::fs::File, PathBuf)> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("kitty-lint-{}-{}-{}", label, std::process::id(), n));
    // File::create is write-only → reads back with EBADF; open rw explicitly.
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)?;
    Ok((file, path))
}

/// Run `cargo check --message-format=json` on the crate that owns `path` and
/// collect its errors/warnings (async, f8 T4.2): the whole blocking section
/// (scratch files + spawn + poll-to-deadline) runs on the blocking pool via
/// [`tokio::task::spawn_blocking`] so the 100 ms poll never parks an async
/// worker thread. Semantics unchanged: 120 s deadline, kill on timeout, both
/// scratch files cleaned on every exit path. Returns `None` when the file
/// lives outside any Cargo crate or cargo cannot be spawned.
async fn cargo_check_lint(path: &Path) -> Option<CheckOutcome> {
    let owned = path.to_path_buf();
    match tokio::task::spawn_blocking(move || cargo_check_lint_blocking(&owned)).await {
        Ok(outcome) => outcome,
        // G4-2: a panic inside the blocking task must not silently swallow
        // lint feedback — log it, then degrade to None like other gate
        // failures.
        Err(join_err) => {
            tracing::warn!(
                error = %join_err,
                file = %path.display(),
                "lint gate blocking task failed (panic or cancelled); skipping lint feedback"
            );
            None
        }
    }
}

/// Blocking core behind [`cargo_check_lint`]; runs on a blocking-pool thread.
fn cargo_check_lint_blocking(path: &Path) -> Option<CheckOutcome> {
    const LABEL: &str = "cargo check";
    let crate_root = nearest_crate_root(path)?;

    let (mut stdout_file, stdout_path) = scratch_file("out").ok()?;
    // Second open must not leak the first file when it fails.
    let (mut stderr_file, stderr_path) = match scratch_file("err") {
        Ok(pair) => pair,
        Err(_) => {
            let _ = std::fs::remove_file(&stdout_path);
            return None;
        }
    };
    // Duplicate the scratch fds up front; a failed dup must clean up both
    // files (spawn never happened, but the Err arm below still runs).
    let spawned = match (stdout_file.try_clone(), stderr_file.try_clone()) {
        (Ok(stdout_dup), Ok(stderr_dup)) => std::process::Command::new("cargo")
            .args(["check", "--message-format=json"])
            .current_dir(&crate_root)
            .stdout(Stdio::from(stdout_dup))
            .stderr(Stdio::from(stderr_dup))
            .spawn(),
        _ => Err(std::io::Error::other("failed to duplicate lint scratch fd")),
    };
    let mut child = match spawned {
        Ok(child) => child,
        Err(_) => {
            let _ = std::fs::remove_file(&stdout_path);
            let _ = std::fs::remove_file(&stderr_path);
            return None;
        }
    };

    let deadline = std::time::Instant::now() + Duration::from_secs(LINT_GATE_TIMEOUT_SECS);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = std::fs::remove_file(&stdout_path);
                    let _ = std::fs::remove_file(&stderr_path);
                    return Some(CheckOutcome {
                        label: LABEL.to_string(),
                        passed: false,
                        detail: format!(
                            "cargo check timed out after {LINT_GATE_TIMEOUT_SECS}s (killed)"
                        ),
                    });
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(_) => {
                let _ = std::fs::remove_file(&stdout_path);
                let _ = std::fs::remove_file(&stderr_path);
                return None;
            }
        }
    };

    use std::io::{Read, Seek};
    // The child inherited dup'd fds sharing our file offset; rewind to
    // actually read what it wrote. Read through a closure so both scratch
    // files are removed even when a rewind/read fails mid-way.
    let captured = (|| {
        stdout_file.rewind().ok()?;
        let mut stdout = String::new();
        stdout_file.read_to_string(&mut stdout).ok()?;
        stderr_file.rewind().ok()?;
        let mut stderr = String::new();
        stderr_file.read_to_string(&mut stderr).ok()?;
        Some((stdout, stderr))
    })();
    let _ = std::fs::remove_file(&stdout_path);
    let _ = std::fs::remove_file(&stderr_path);
    let (stdout, stderr) = captured?;

    let diags = parse_cargo_check_json(&stdout);
    let has_errors = diags.iter().any(|d| d.level == "error");
    let passed = status.success() && !has_errors;

    let mut detail = String::new();
    for (i, d) in diags.iter().enumerate() {
        if i >= LINT_GATE_MAX_DIAGS {
            detail.push_str(&format!("... ({} more diagnostics)\n", diags.len() - i));
            break;
        }
        detail.push_str(&d.rendered);
        detail.push('\n');
    }
    if !passed && detail.is_empty() {
        // No parsed diagnostics but the check failed — surface raw stderr so
        // the model still gets an actionable signal (e.g. manifest errors).
        detail = stderr;
    }

    Some(CheckOutcome {
        label: LABEL.to_string(),
        passed,
        detail,
    })
}

/// Outcome of a cheap post-edit syntax check.
struct CheckOutcome {
    label: String,
    passed: bool,
    detail: String,
}

/// Map a file extension to a cheap syntax checker and run it.
///
/// Returns `None` when no checker maps to the extension or the checker
/// binary cannot be spawned. For `.go` we use `gofmt -e` and treat any
/// stderr output as failure (gofmt's exit status alone is not reliable
/// across versions for parse errors).
fn run_extension_check(path: &Path) -> Option<CheckOutcome> {
    let ext = path.extension()?.to_str()?;
    let display = path.display().to_string();
    let (label, program, args): (&str, &str, Vec<String>) = match ext {
        "py" => (
            "python3 -m py_compile",
            "python3",
            vec!["-m".into(), "py_compile".into(), display],
        ),
        "js" | "ts" => ("node --check", "node", vec!["--check".into(), display]),
        "sh" => ("bash -n", "bash", vec!["-n".into(), display]),
        "go" => ("gofmt -e", "gofmt", vec!["-e".into(), display]),
        _ => return None,
    };

    let output = std::process::Command::new(program)
        .args(&args)
        .output()
        .ok()?;
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let passed = if ext == "go" {
        output.status.success() && stderr.trim().is_empty()
    } else {
        output.status.success()
    };
    Some(CheckOutcome {
        label: label.to_string(),
        passed,
        detail: stderr,
    })
}

/// 1-based line number in `after` of the first line that differs from
/// `before` (center for the re-show block). `None` when identical.
fn first_changed_line(before: &str, after: &str) -> Option<usize> {
    let b: Vec<&str> = before.lines().collect();
    let a: Vec<&str> = after.lines().collect();
    for (i, (x, y)) in b.iter().zip(a.iter()).enumerate() {
        if x != y {
            return Some(i + 1);
        }
    }
    if b.len() != a.len() {
        return Some(b.len().min(a.len()) + 1);
    }
    None
}

/// Format up to ±`radius` lines around `center` (1-based) from an updated
/// snapshot using the same `LINE:HASH|content` format as Read, so the model
/// can chain hash edits without re-reading the whole file.
fn reshow_region(lines: &[crate::hashline::LineHash], center: usize, radius: usize) -> String {
    let start = center.saturating_sub(radius).max(1);
    let end = (center + radius).min(lines.len());
    if start > end {
        return String::new();
    }
    lines[start - 1..end]
        .iter()
        .map(|l| {
            format!(
                "{}:{}|{}",
                l.line_num,
                &l.hash[..8.min(l.hash.len())],
                l.content
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// difflib-style unified diff between two texts (line-based, LCS).
/// Hunks carry up to 3 context lines; total emitted lines are capped at
/// `max_lines` (a truncation marker is appended when exceeded).
fn unified_diff(before: &str, after: &str, max_lines: usize) -> String {
    #[derive(Clone, Copy, PartialEq)]
    enum OpKind {
        Same,
        Del,
        Add,
    }

    let a: Vec<&str> = before.lines().collect();
    let b: Vec<&str> = after.lines().collect();
    if a == b {
        return String::new();
    }

    const CTX: usize = 3;
    // LCS table is O(n*m); cap input size and fall back to a single
    // prefix/suffix-trimmed hunk for oversized files.
    const LCS_CAP: usize = 600;
    if a.len() > LCS_CAP || b.len() > LCS_CAP {
        return oversized_hunk(&a, &b, CTX, max_lines);
    }
    let (n, m) = (a.len(), b.len());
    let w = m + 1;
    let mut dp = vec![0u32; (n + 1) * w];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i * w + j] = if a[i] == b[j] {
                dp[(i + 1) * w + j + 1] + 1
            } else {
                dp[(i + 1) * w + j].max(dp[i * w + j + 1])
            };
        }
    }
    let mut ops: Vec<(OpKind, &str)> = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < n && j < m {
        if a[i] == b[j] {
            ops.push((OpKind::Same, a[i]));
            i += 1;
            j += 1;
        } else if dp[(i + 1) * w + j] >= dp[i * w + j + 1] {
            ops.push((OpKind::Del, a[i]));
            i += 1;
        } else {
            ops.push((OpKind::Add, b[j]));
            j += 1;
        }
    }
    while i < n {
        ops.push((OpKind::Del, a[i]));
        i += 1;
    }
    while j < m {
        ops.push((OpKind::Add, b[j]));
        j += 1;
    }

    // Absolute line numbers per op index.
    let old_no =
        |idx: usize| -> usize { ops[..idx].iter().filter(|(k, _)| *k != OpKind::Add).count() };
    let new_no =
        |idx: usize| -> usize { ops[..idx].iter().filter(|(k, _)| *k != OpKind::Del).count() };

    let mut out = String::new();
    let mut emitted = 0usize;
    let mut k = 0usize;
    while k < ops.len() {
        if ops[k].0 == OpKind::Same {
            k += 1;
            continue;
        }
        // Hunk start: back up context lines.
        let start = k.saturating_sub(CTX);
        // Hunk end: first run of more than CTX consecutive Same ops.
        let mut end = k;
        let mut run = 0usize;
        while end < ops.len() {
            if ops[end].0 == OpKind::Same {
                run += 1;
                if run > CTX {
                    break;
                }
            } else {
                run = 0;
            }
            end += 1;
        }

        let slice = &ops[start..end];
        let old_count = slice.iter().filter(|(kd, _)| *kd != OpKind::Add).count();
        let new_count = slice.iter().filter(|(kd, _)| *kd != OpKind::Del).count();
        let o_start = old_no(start) + 1;
        let n_start = new_no(start) + 1;
        out.push_str(&format!(
            "@@ -{},{} +{},{} @@\n",
            if old_count == 0 { o_start - 1 } else { o_start },
            old_count,
            if new_count == 0 { n_start - 1 } else { n_start },
            new_count
        ));
        emitted += 1;
        if emitted >= max_lines {
            out.push_str("... (diff truncated)\n");
            return out;
        }
        for (kind, line) in slice {
            let prefix = match kind {
                OpKind::Same => ' ',
                OpKind::Del => '-',
                OpKind::Add => '+',
            };
            out.push(prefix);
            out.push_str(line);
            out.push('\n');
            emitted += 1;
            if emitted >= max_lines {
                out.push_str("... (diff truncated)\n");
                return out;
            }
        }
        k = end;
    }
    out
}

/// Single-hunk diff for oversized files (LCS table would be too large):
/// prefix/suffix-trimmed middle with bounded context. Line numbers are
/// computed from the true offsets, unlike the LCS path.
fn oversized_hunk(a: &[&str], b: &[&str], ctx: usize, max_lines: usize) -> String {
    let mut p = 0usize;
    while p < a.len() && p < b.len() && a[p] == b[p] {
        p += 1;
    }
    let mut s = 0usize;
    while s < a.len() - p && s < b.len() - p && a[a.len() - 1 - s] == b[b.len() - 1 - s] {
        s += 1;
    }

    let lead = p.min(ctx);
    let trail = s.min(ctx);
    let dels = &a[p..a.len() - s];
    let adds = &b[p..b.len() - s];

    let old_start = p + 1 - lead; // 1-based first old line shown
    let new_start = p + 1 - lead; // 1-based first new line shown
    let old_count = lead + dels.len() + trail;
    let new_count = lead + adds.len() + trail;

    let mut out = format!("@@ -{old_start},{old_count} +{new_start},{new_count} @@\n");
    let mut emitted = 1usize;
    if emitted >= max_lines {
        out.push_str("... (diff truncated)\n");
        return out;
    }
    for line in &a[p - lead..p] {
        out.push(' ');
        out.push_str(line);
        out.push('\n');
        emitted += 1;
    }
    for line in dels {
        out.push('-');
        out.push_str(line);
        out.push('\n');
        emitted += 1;
        if emitted >= max_lines {
            out.push_str("... (diff truncated)\n");
            return out;
        }
    }
    for line in adds {
        out.push('+');
        out.push_str(line);
        out.push('\n');
        emitted += 1;
        if emitted >= max_lines {
            out.push_str("... (diff truncated)\n");
            return out;
        }
    }
    for line in &a[a.len() - s..a.len() - s + trail] {
        out.push(' ');
        out.push_str(line);
        out.push('\n');
        emitted += 1;
        if emitted >= max_lines {
            out.push_str("... (diff truncated)\n");
            return out;
        }
    }
    out
}

/// Keep at most `max` lines of `text`.
fn truncate_lines(text: &str, max: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= max {
        text.trim_end().to_string()
    } else {
        let mut kept: Vec<String> = lines[..max].iter().map(|s| s.to_string()).collect();
        kept.push(format!("... ({} more lines)", lines.len() - max));
        kept.join("\n")
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

    #[tokio::test]
    async fn write_guard_blocks_existing_file() {
        let f = temp_file("existing");
        let mut registry = ToolRegistry::new();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "content": "new content"
        });
        let err = registry.execute_write(&args).await.unwrap_err();
        assert!(matches!(err, ToolError::WriteBlocked { .. }));
    }

    #[tokio::test]
    async fn write_allows_new_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new_file.txt");
        let mut registry = ToolRegistry::new();
        let args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "content": "hello"
        });
        let result = registry.execute_write(&args).await.unwrap();
        assert!(result.success);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello");
    }

    #[tokio::test]
    async fn edit_blocked_without_read() {
        let f = temp_file("line1\nline2");
        let mut registry = ToolRegistry::new();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "hash": "dummy",
            "new": "replaced"
        });
        let err = registry.execute_edit(&args).await.unwrap_err();
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
        // F3.1 + f8 T2.3: Read output is capped at max_read_lines, opens with
        // an honest window header, and the omission note suggests offset/limit.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.txt");
        let lines: Vec<String> = (1..=500).map(|i| format!("line {i}")).collect();
        std::fs::write(&path, lines.join("\n")).unwrap();

        let mut registry = ToolRegistry::with_timeout(60).with_read_cap(300);
        let args = serde_json::json!({"path": path.to_str().unwrap()});
        let result = registry.execute_read(&args).unwrap();
        assert!(result.success);
        let expected_header = format!("{}: lines 1-300 of 500", path.display());
        assert!(
            result.output.starts_with(&expected_header),
            "header expected first line, got: {}",
            result.output
        );
        assert!(
            result.output.contains("200 lines after this window"),
            "got: {}",
            result.output
        );
        assert!(
            result.output.contains("\"offset\": 301"),
            "omission note must suggest offset/limit retry, got: {}",
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
    fn read_offset_limit_returns_exact_window() {
        // f8 T2.3 AC: offset=50 & limit=20 → exactly numbered lines 50-69,
        // header first, omission note pointing at the next window.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hundred.txt");
        let lines: Vec<String> = (1..=100).map(|i| format!("content {i}")).collect();
        std::fs::write(&path, lines.join("\n")).unwrap();

        let mut registry = ToolRegistry::new();
        let args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "offset": 50,
            "limit": 20
        });
        let result = registry.execute_read(&args).unwrap();
        assert!(result.success);
        let expected_header = format!("{}: lines 50-69 of 100", path.display());
        assert!(
            result.output.starts_with(&expected_header),
            "got: {}",
            result.output
        );
        let body_lines: Vec<&str> = result.output.lines().skip(1).collect();
        assert_eq!(body_lines.len(), 21, "20 tagged lines + 1 note"); // 20 + omission note
        assert!(body_lines[0].starts_with("50:"), "got: {}", body_lines[0]);
        assert!(body_lines[0].contains("content 50"));
        assert!(body_lines[19].starts_with("69:"));
        assert!(body_lines[19].contains("content 69"));
        assert!(result.output.contains("\"offset\": 70"), "next-window hint");
    }

    #[tokio::test]
    async fn read_clamps_malformed_paging_args_instead_of_erroring() {
        // f8 T2.3 AC: weak-model tolerance — offset<1→1, limit<1→cap,
        // non-numeric strings → defaults; NEVER InvalidArgs.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ten.txt");
        let lines: Vec<String> = (1..=10).map(|i| format!("l {i}")).collect();
        std::fs::write(&path, lines.join("\n")).unwrap();

        let mut registry = ToolRegistry::new();

        // String numerals tolerated.
        let result = registry
            .execute_read(&serde_json::json!({
                "path": path.to_str().unwrap(),
                "offset": "3",
                "limit": "2"
            }))
            .unwrap();
        assert!(result
            .output
            .starts_with(&format!("{}: lines 3-4 of 10", path.display())));

        // offset=0 clamps to 1; negative limit clamps to cap.
        let result = registry
            .execute_read(&serde_json::json!({
                "path": path.to_str().unwrap(),
                "offset": 0,
                "limit": -5
            }))
            .unwrap();
        assert!(result
            .output
            .starts_with(&format!("{}: lines 1-10 of 10", path.display())));

        // Non-numeric strings fall back to defaults (whole file).
        let result = registry
            .execute_read(&serde_json::json!({
                "path": path.to_str().unwrap(),
                "offset": "beginning",
                "limit": "lots"
            }))
            .unwrap();
        assert!(result
            .output
            .starts_with(&format!("{}: lines 1-10 of 10", path.display())));

        // Offset beyond EOF clamps to the last line rather than erroring;
        // the tail-window note must NOT suggest a next offset (G2 rider).
        let result = registry
            .execute_read(&serde_json::json!({
                "path": path.to_str().unwrap(),
                "offset": 999
            }))
            .unwrap();
        assert!(result
            .output
            .starts_with(&format!("{}: lines 10-10 of 10", path.display())));
        assert!(
            result.output.contains("window starts at line 10 of 10"),
            "tail-window note expected, got: {}",
            result.output
        );
        assert!(
            !result.output.contains("\"offset\":"),
            "no bogus next-offset on a tail window, got: {}",
            result.output
        );
    }

    #[tokio::test]
    async fn read_cap_builder_governs_default_window() {
        // f8 T2.3 AC: the profile value plumbed via with_read_cap actually
        // governs the default window (tiny=100 / medium=300 / large=500 are
        // asserted on the profile side in model_profile tests).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("five.txt");
        std::fs::write(&path, "a\nb\nc\nd\ne").unwrap();

        let mut registry = ToolRegistry::with_timeout(60).with_read_cap(2);
        let result = registry
            .execute_read(&serde_json::json!({"path": path.to_str().unwrap()}))
            .unwrap();
        assert!(result
            .output
            .starts_with(&format!("{}: lines 1-2 of 5", path.display())));

        // Edits OUTSIDE the displayed window still succeed — the snapshot is
        // full-file and the window is display-only.
        let hash_args = serde_json::json!({"path": path.to_str().unwrap()});
        registry.execute_read(&hash_args).unwrap();
        let snapshot = registry.hashline.get_snapshot(path.as_path()).unwrap();
        let e_hash = snapshot.lines[4].hash.clone();
        let edit = serde_json::json!({
            "path": path.to_str().unwrap(),
            "hash": &e_hash[..8],
            "new": "E"
        });
        let edit_result = registry.execute_edit(&edit).await.unwrap();
        assert!(edit_result.success, "got: {}", edit_result.output);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "a\nb\nc\nd\nE");
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

    #[tokio::test]
    async fn edit_blocked_message_mentions_must_read() {
        let f = temp_file("line1\nline2");
        let mut registry = ToolRegistry::new();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "hash": "deadbeef",
            "new": "replaced"
        });
        let err = registry.execute_edit(&args).await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("File not loaded"), "got: {msg}");
        assert!(msg.contains("must Read before Edit"), "got: {msg}");
    }

    #[tokio::test]
    async fn edit_after_read_succeeds() {
        let f = temp_file("hello\nworld");
        let mut registry = ToolRegistry::new();
        // First read
        let read_args = serde_json::json!({"path": f.path().to_str().unwrap()});
        let read_res = registry.execute_read(&read_args).unwrap();
        assert!(read_res.success);
        // Extract hash from the first TAGGED line (skip the honest-window header)
        let first_line = read_res.output.lines().nth(1).unwrap();
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
        let edit_res = registry.execute_edit(&edit_args).await.unwrap();
        assert!(edit_res.success);
    }

    #[tokio::test]
    async fn write_blocks_existing_file_message_mentions_edit() {
        let f = temp_file("existing");
        let mut registry = ToolRegistry::new();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "content": "new"
        });
        let err = registry.execute_write(&args).await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("use Edit instead"), "got: {msg}");
    }

    #[tokio::test]
    async fn write_then_edit_without_extra_read_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new.txt");
        let mut registry = ToolRegistry::new();
        let write_args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "content": "line1\nline2"
        });
        registry.execute_write(&write_args).await.unwrap();
        // Now edit without explicit Read — should succeed because Write populated snapshot
        let read_args = serde_json::json!({"path": path.to_str().unwrap()});
        let read_res = registry.execute_read(&read_args).unwrap();
        let hash = read_res
            .output
            .lines()
            .nth(1) // skip the honest-window header (f8 T2.3)
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
        let edit_res = registry.execute_edit(&edit_args).await.unwrap();
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

    // ---- Tool alias normalization (dispatch-time) ----

    #[tokio::test]
    async fn alias_synonym_name_dispatches_to_real_tool() {
        // "str_replace" is not a real tool; it must dispatch to Edit.
        let f = temp_file("alpha\nbeta");
        let mut registry = ToolRegistry::new();
        registry
            .execute_read(&serde_json::json!({"path": f.path().to_str().unwrap()}))
            .unwrap();
        let tc = ParsedToolCall {
            name: "str_replace".to_string(),
            arguments: serde_json::json!({
                "path": f.path().to_str().unwrap(),
                "old_string": "beta",
                "new_string": "BETA"
            }),
        };
        let result = registry.execute(&tc).await.unwrap();
        assert!(result.success, "got: {}", result.output);
        assert_eq!(std::fs::read_to_string(f.path()).unwrap(), "alpha\nBETA");
    }

    #[tokio::test]
    async fn alias_arg_rekey_maps_onto_canonical_keys() {
        // file_path → path, cmd → command (only when canonical absent).
        let f = temp_file("content");
        let mut registry = ToolRegistry::new();
        let tc = ParsedToolCall {
            name: "Read".to_string(),
            arguments: serde_json::json!({"file_path": f.path().to_str().unwrap()}),
        };
        let result = registry.execute(&tc).await.unwrap();
        assert!(result.success, "file_path alias must satisfy Read's `path`");

        let normalized = normalize_tool_call(&ParsedToolCall {
            name: "Bash".to_string(),
            arguments: serde_json::json!({"cmd": "echo hi", "extra": 1}),
        });
        assert_eq!(normalized.arguments["command"], "echo hi");
        assert!(normalized.arguments.get("cmd").is_none());
        assert_eq!(normalized.arguments["extra"], 1);
    }

    #[test]
    fn alias_canonical_key_wins_over_alias() {
        // When the canonical key is already present, the alias is dropped
        // and the canonical value is kept.
        let normalized = normalize_tool_call(&ParsedToolCall {
            name: "Read".to_string(),
            arguments: serde_json::json!({"path": "real.txt", "file_path": "fake.txt"}),
        });
        assert_eq!(normalized.arguments["path"], "real.txt");
        assert!(normalized.arguments.get("file_path").is_none());
    }

    #[test]
    fn alias_real_names_pass_through_with_canonical_casing() {
        // Case-insensitive resolution onto canonical casing...
        assert_eq!(normalize_tool_name("read"), "Read");
        assert_eq!(normalize_tool_name("BASH"), "Bash");
        assert_eq!(normalize_tool_name("Glob"), "Glob");
        // ...and a synonym table entry can never shadow a real name.
        assert_eq!(normalize_tool_name("Edit"), "Edit");
    }

    #[tokio::test]
    async fn alias_unknown_names_pass_through_unchanged() {
        let mut registry = ToolRegistry::new();
        let tc = ParsedToolCall {
            name: "TotallyMadeUp".to_string(),
            arguments: serde_json::json!({}),
        };
        let err = registry.execute(&tc).await.unwrap_err();
        match err {
            ToolError::NotFound(name) => {
                assert_eq!(name, "TotallyMadeUp", "unknown names must pass through");
            }
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn alias_normalization_is_idempotent() {
        // G3-1: the agent loop normalizes upstream of the quality gate and
        // execute() normalizes again — the second pass must be a no-op.
        let once = normalize_tool_call(&ParsedToolCall {
            name: "str_replace".to_string(),
            arguments: serde_json::json!({
                "file_path": "a.rs",
                "old_str": "x",
                "new_str": "y"
            }),
        });
        let twice = normalize_tool_call(&once);
        assert_eq!(once, twice);
        assert_eq!(once.name, "Edit");
        assert_eq!(once.arguments["path"], "a.rs");
        assert_eq!(once.arguments["old_string"], "x");
        assert_eq!(once.arguments["new_string"], "y");
    }

    #[tokio::test]
    async fn raw_only_arguments_get_actionable_diagnostic() {
        // G3-5: salvaged `_raw`-only args must explain WHAT was wrong and
        // show the recovered body instead of a generic missing-key error.
        let mut registry = ToolRegistry::new();
        let tc = ParsedToolCall {
            name: "Read".to_string(),
            arguments: serde_json::json!({
                "_raw": "{\"name\": \"Read\", \"path\": no_close"
            }),
        };
        let result = registry.execute(&tc).await.unwrap();
        assert!(!result.success);
        assert!(
            result.output.contains("not valid JSON"),
            "got: {}",
            result.output
        );
        assert!(
            result.output.contains("no_close"),
            "recovered body must be shown, got: {}",
            result.output
        );
        assert!(
            result.output.contains("Re-send the call as proper JSON"),
            "got: {}",
            result.output
        );
    }

    // ---- Edit str_replace mode (aider-compatible) ----

    #[tokio::test]
    async fn str_replace_success_single_match() {
        let f = temp_file("alpha\nbeta\ngamma");
        let mut registry = ToolRegistry::new();
        registry
            .execute_read(&serde_json::json!({"path": f.path().to_str().unwrap()}))
            .unwrap();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "beta",
            "new_string": "BETA"
        });
        let result = registry.execute_edit(&args).await.unwrap();
        assert!(result.success, "got: {}", result.output);
        assert_eq!(
            std::fs::read_to_string(f.path()).unwrap(),
            "alpha\nBETA\ngamma"
        );
    }

    #[tokio::test]
    async fn str_replace_not_found_error_message() {
        let f = temp_file("alpha\nbeta");
        let mut registry = ToolRegistry::new();
        registry
            .execute_read(&serde_json::json!({"path": f.path().to_str().unwrap()}))
            .unwrap();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "MISSING",
            "new_string": "x"
        });
        let result = registry.execute_edit(&args).await.unwrap();
        assert!(!result.success);
        assert!(
            result.output.contains("old_string not found"),
            "got: {}",
            result.output
        );
        assert!(
            result.output.to_lowercase().contains("read"),
            "must hint to Read first, got: {}",
            result.output
        );
    }

    // ---- E-G1 low findings ----

    #[tokio::test]
    async fn edit_dual_mode_args_notes_hash_ignored() {
        // E-G1a: when BOTH old_string and hash are provided, str_replace mode
        // wins and the output must say the hash was ignored.
        let f = temp_file("alpha\nbeta\ngamma");
        let mut registry = ToolRegistry::new();
        registry
            .execute_read(&serde_json::json!({"path": f.path().to_str().unwrap()}))
            .unwrap();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "beta",
            "new_string": "BETA",
            "hash": "deadbeef"
        });
        let result = registry.execute_edit(&args).await.unwrap();
        assert!(result.success, "got: {}", result.output);
        assert!(
            result
                .output
                .starts_with("note: hash ignored; str_replace mode active"),
            "got: {}",
            result.output
        );
        assert_eq!(
            std::fs::read_to_string(f.path()).unwrap(),
            "alpha\nBETA\ngamma"
        );
    }

    #[tokio::test]
    async fn edit_empty_old_string_fails_immediately() {
        // E-G1b: empty old_string must fail before occurrence counting /
        // file IO (an empty needle would match everywhere).
        let f = temp_file("alpha\nbeta");
        let mut registry = ToolRegistry::new();
        registry
            .execute_read(&serde_json::json!({"path": f.path().to_str().unwrap()}))
            .unwrap();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "",
            "new_string": "x"
        });
        let result = registry.execute_edit(&args).await.unwrap();
        assert!(!result.success);
        assert_eq!(result.output, "old_string must not be empty");
        // File untouched
        assert_eq!(std::fs::read_to_string(f.path()).unwrap(), "alpha\nbeta");
    }

    #[tokio::test]
    async fn str_replace_multiple_matches_error_without_replace_all() {
        let f = temp_file("same\nsame\nother");
        let mut registry = ToolRegistry::new();
        registry
            .execute_read(&serde_json::json!({"path": f.path().to_str().unwrap()}))
            .unwrap();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "same",
            "new_string": "different"
        });
        let result = registry.execute_edit(&args).await.unwrap();
        assert!(!result.success);
        assert!(
            result.output.contains("matches 2 locations"),
            "got: {}",
            result.output
        );
        assert!(
            result.output.contains("more surrounding context"),
            "got: {}",
            result.output
        );
        // File untouched
        assert_eq!(
            std::fs::read_to_string(f.path()).unwrap(),
            "same\nsame\nother"
        );
    }

    #[tokio::test]
    async fn str_replace_replace_all_success() {
        let f = temp_file("same\nmid\nsame");
        let mut registry = ToolRegistry::new();
        registry
            .execute_read(&serde_json::json!({"path": f.path().to_str().unwrap()}))
            .unwrap();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "same",
            "new_string": "uniq",
            "replace_all": true
        });
        let result = registry.execute_edit(&args).await.unwrap();
        assert!(result.success, "got: {}", result.output);
        assert_eq!(
            std::fs::read_to_string(f.path()).unwrap(),
            "uniq\nmid\nuniq"
        );
    }

    #[tokio::test]
    async fn str_replace_crlf_preserved() {
        let f = temp_file("alpha\r\nbeta\r\ngamma");
        let mut registry = ToolRegistry::new();
        registry
            .execute_read(&serde_json::json!({"path": f.path().to_str().unwrap()}))
            .unwrap();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "beta",
            "new_string": "BETA"
        });
        let result = registry.execute_edit(&args).await.unwrap();
        assert!(result.success, "got: {}", result.output);
        let on_disk = std::fs::read_to_string(f.path()).unwrap();
        assert_eq!(on_disk, "alpha\r\nBETA\r\ngamma", "CRLF must be preserved");
    }

    #[tokio::test]
    async fn str_replace_fuzzy_line_trim_succeeds_with_strategy_note() {
        // Indentation drift in old_string — exact match fails, line-trim lands.
        let f = temp_file("fn a() {\n    old();\n}\n");
        let mut registry = ToolRegistry::new();
        registry
            .execute_read(&serde_json::json!({"path": f.path().to_str().unwrap()}))
            .unwrap();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "fn a() {\nold();\n}",
            "new_string": "fn a() {\n    new();\n}"
        });
        let result = registry.execute_edit(&args).await.unwrap();
        assert!(result.success, "got: {}", result.output);
        assert!(
            result.output.contains("[edit matched via line-trim]"),
            "strategy note expected, got: {}",
            result.output
        );
        assert_eq!(
            std::fs::read_to_string(f.path()).unwrap(),
            "fn a() {\n    new();\n}\n"
        );
    }

    #[tokio::test]
    async fn str_replace_fuzzy_crlf_roundtrip_preserved() {
        // Drifted old_string against a CRLF file: fuzzy span replacement must
        // still restore the file's original CRLF endings on disk.
        let f = temp_file("alpha\r\nbeta\r\ngamma");
        let mut registry = ToolRegistry::new();
        registry
            .execute_read(&serde_json::json!({"path": f.path().to_str().unwrap()}))
            .unwrap();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "  beta", // leading-space drift, no \r at all
            "new_string": "BETA"
        });
        let result = registry.execute_edit(&args).await.unwrap();
        assert!(result.success, "got: {}", result.output);
        let on_disk = std::fs::read_to_string(f.path()).unwrap();
        assert_eq!(
            on_disk, "alpha\r\nBETA\r\ngamma",
            "CRLF must survive a fuzzy edit"
        );
    }

    #[tokio::test]
    async fn str_replace_fuzzy_ambiguous_without_replace_all() {
        // Tab-drifted old_string misses exactly but fuzzy-matches BOTH
        // space-indented occurrences — ambiguity without replace_all.
        let f = temp_file("  same\nmid\n  same\nother");
        let mut registry = ToolRegistry::new();
        registry
            .execute_read(&serde_json::json!({"path": f.path().to_str().unwrap()}))
            .unwrap();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "\tsame",
            "new_string": "different"
        });
        let result = registry.execute_edit(&args).await.unwrap();
        assert!(!result.success);
        assert!(
            result.output.contains("multiple locations"),
            "got: {}",
            result.output
        );
        assert_eq!(
            std::fs::read_to_string(f.path()).unwrap(),
            "  same\nmid\n  same\nother",
            "ambiguous fuzzy match must not modify the file"
        );
    }

    #[tokio::test]
    async fn str_replace_fuzzy_replace_all_replaces_every_drifted_occurrence() {
        let f = temp_file("  same\nmid\n  same\nother");
        let mut registry = ToolRegistry::new();
        registry
            .execute_read(&serde_json::json!({"path": f.path().to_str().unwrap()}))
            .unwrap();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "\tsame",
            "new_string": "uniq",
            "replace_all": true
        });
        let result = registry.execute_edit(&args).await.unwrap();
        assert!(result.success, "got: {}", result.output);
        assert_eq!(
            std::fs::read_to_string(f.path()).unwrap(),
            "uniq\nmid\nuniq\nother"
        );
    }

    #[tokio::test]
    async fn str_replace_fuzzy_guard_refuses_disproportionate_span() {
        // Giant indentation drift makes every candidate wildly larger than
        // old_string — the guard must reject and the edit must fail cleanly.
        let pad = " ".repeat(1200);
        let f = temp_file(format!("{pad}a();\n{pad}b();\n").as_str());
        let mut registry = ToolRegistry::new();
        registry
            .execute_read(&serde_json::json!({"path": f.path().to_str().unwrap()}))
            .unwrap();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "a();\nb();",
            "new_string": "x();\ny();"
        });
        let result = registry.execute_edit(&args).await.unwrap();
        assert!(!result.success, "guard must refuse the ballooned span");
        assert_eq!(
            std::fs::read_to_string(f.path()).unwrap(),
            format!("{pad}a();\n{pad}b();\n"),
            "guarded-out fuzzy match must not modify the file"
        );
    }

    #[tokio::test]
    async fn str_replace_multiline_old_string() {
        let f = temp_file("fn a() {\n    old();\n}\n");
        let mut registry = ToolRegistry::new();
        registry
            .execute_read(&serde_json::json!({"path": f.path().to_str().unwrap()}))
            .unwrap();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "fn a() {\n    old();\n}",
            "new_string": "fn a() {\n    new();\n}"
        });
        let result = registry.execute_edit(&args).await.unwrap();
        assert!(result.success, "got: {}", result.output);
        assert_eq!(
            std::fs::read_to_string(f.path()).unwrap(),
            "fn a() {\n    new();\n}\n"
        );
    }

    #[tokio::test]
    async fn str_replace_blocked_without_read() {
        let f = temp_file("alpha\nbeta");
        let mut registry = ToolRegistry::new();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "alpha",
            "new_string": "A"
        });
        let err = registry.execute_edit(&args).await.unwrap_err();
        assert!(matches!(err, ToolError::EditBlocked(_)));
    }

    #[tokio::test]
    async fn str_replace_output_has_diff_and_reshow() {
        let f = temp_file("l1\nl2\nl3\nl4\nl5");
        let mut registry = ToolRegistry::new();
        registry
            .execute_read(&serde_json::json!({"path": f.path().to_str().unwrap()}))
            .unwrap();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "old_string": "l3",
            "new_string": "L3"
        });
        let result = registry.execute_edit(&args).await.unwrap();
        assert!(result.success);
        assert!(
            result.output.contains("--- Diff ---"),
            "got: {}",
            result.output
        );
        assert!(result.output.contains("-l3"), "got: {}", result.output);
        assert!(result.output.contains("+L3"), "got: {}", result.output);
        // Re-show block: tagged lines around the change (Read-style format)
        assert!(
            result.output.contains("|l2"),
            "re-show context missing, got: {}",
            result.output
        );
        assert!(
            result.output.contains("|L3"),
            "re-show edited line missing, got: {}",
            result.output
        );
        assert!(
            result.output.contains("|l4"),
            "re-show context missing, got: {}",
            result.output
        );
    }

    #[tokio::test]
    async fn hash_edit_output_includes_diff_and_reshow() {
        // Feedback block applies to BOTH modes; hash protocol must keep working.
        let f = temp_file("hello\nworld");
        let mut registry = ToolRegistry::new();
        let read_res = registry
            .execute_read(&serde_json::json!({"path": f.path().to_str().unwrap()}))
            .unwrap();
        let hash = read_res
            .output
            .lines()
            .nth(1) // skip the honest-window header (f8 T2.3)
            .unwrap()
            .split(':')
            .nth(1)
            .unwrap()
            .split('|')
            .next()
            .unwrap()
            .to_string();
        let args = serde_json::json!({
            "path": f.path().to_str().unwrap(),
            "hash": hash,
            "new": "HELLO"
        });
        let result = registry.execute_edit(&args).await.unwrap();
        assert!(result.success, "hash protocol broken: {}", result.output);
        assert!(
            result.output.contains("--- Diff ---"),
            "got: {}",
            result.output
        );
        assert!(result.output.contains("+HELLO"), "got: {}", result.output);
        assert!(
            result.output.contains("|world"),
            "re-show missing, got: {}",
            result.output
        );
    }

    #[tokio::test]
    async fn edit_check_disabled_by_default() {
        if std::env::var("KITTY_EDIT_CHECK").as_deref() == Ok("1") {
            println!("SKIP: KITTY_EDIT_CHECK=1 set in ambient environment");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("script.py");
        std::fs::write(&path, "x = 1\nbroken(\n").unwrap(); // syntactically invalid
        let mut registry = ToolRegistry::new();
        registry
            .execute_read(&serde_json::json!({"path": path.to_str().unwrap()}))
            .unwrap();
        let args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "old_string": "x = 1",
            "new_string": "x = 2"
        });
        let result = registry.execute_edit(&args).await.unwrap();
        assert!(result.success);
        assert!(
            !result.output.contains("--- Check ("),
            "check must be off by default, got: {}",
            result.output
        );
    }

    #[test]
    fn extension_check_pass_and_fail_python() {
        if std::process::Command::new("python3")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            let dir = tempfile::tempdir().unwrap();
            let ok = dir.path().join("good.py");
            std::fs::write(&ok, "x = 1\n").unwrap();
            let passed = run_extension_check(&ok).unwrap();
            assert!(passed.passed, "valid .py must pass, got: {}", passed.detail);

            let bad = dir.path().join("bad.py");
            std::fs::write(&bad, "def f(:\n").unwrap();
            let failed = run_extension_check(&bad).unwrap();
            assert!(!failed.passed, "invalid .py must fail");
        } else {
            println!("SKIP: python3 unavailable");
        }
    }

    #[test]
    fn extension_check_bash_n() {
        let dir = tempfile::tempdir().unwrap();
        let ok = dir.path().join("good.sh");
        std::fs::write(&ok, "echo hi\n").unwrap();
        assert!(run_extension_check(&ok).unwrap().passed);

        let bad = dir.path().join("bad.sh");
        std::fs::write(&bad, "if then fi\n").unwrap();
        assert!(!run_extension_check(&bad).unwrap().passed);
    }

    #[test]
    fn unified_diff_basic_and_noop() {
        assert_eq!(unified_diff("a\nb\n", "a\nb\n", 40), "");
        let d = unified_diff("a\nb\nc\n", "a\nX\nc\n", 40);
        assert!(d.contains("@@"), "got: {d}");
        assert!(d.contains("-b\n"), "got: {d}");
        assert!(d.contains("+X\n"), "got: {d}");
        assert!(d.contains(" a\n"), "context line, got: {d}");
    }

    #[test]
    fn unified_diff_caps_output() {
        let old: String = (0..100).map(|i| format!("line{i}\n")).collect();
        let new: String = (0..100).map(|i| format!("ln{i}\n")).collect();
        let d = unified_diff(&old, &new, 40);
        let emitted = d.lines().count();
        assert!(emitted <= 41, "cap exceeded: {emitted} lines");
        assert!(d.contains("diff truncated"));
    }

    // -------------------------------------------------------------------
    // Lint gate (KITTY_LINT_GATE=1)
    // -------------------------------------------------------------------

    fn lint_env_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::model_profile::test_env_lock()
    }

    #[test]
    fn lint_gate_disabled_by_default() {
        let _g = lint_env_lock();
        let prev = std::env::var("KITTY_LINT_GATE").ok();
        std::env::remove_var("KITTY_LINT_GATE");
        assert!(!lint_gate_enabled(), "lint gate must default to off");
        if let Some(v) = prev {
            std::env::set_var("KITTY_LINT_GATE", v);
        }
    }

    #[test]
    fn nearest_crate_root_walks_up() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("a/b")).unwrap();
        std::fs::write(dir.path().join("a/Cargo.toml"), "[package]\n").unwrap();
        let file = dir.path().join("a/b/lib.rs");
        assert_eq!(
            nearest_crate_root(&file),
            Some(dir.path().join("a")),
            "nearest ancestor Cargo.toml wins"
        );
        // No crate anywhere → None
        let orphan_dir = tempfile::tempdir().unwrap();
        assert_eq!(nearest_crate_root(&orphan_dir.path().join("x.rs")), None);
    }

    #[tokio::test]
    async fn cargo_check_lint_reports_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"lint_probe\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        let main = dir.path().join("src/main.rs");
        std::fs::write(&main, "fn main() { let x: i32 = \"not a number\"; }\n").unwrap();

        let outcome = cargo_check_lint(&main)
            .await
            .expect("crate exists so cargo check runs");
        assert!(!outcome.passed, "type error must fail the lint gate");
        assert!(outcome.detail.contains("error"), "got: {}", outcome.detail);
    }

    #[tokio::test]
    async fn cargo_check_lint_passes_on_clean_crate() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"lint_probe_ok\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        let main = dir.path().join("src/main.rs");
        std::fs::write(&main, "fn main() { println!(\"ok\"); }\n").unwrap();

        let outcome = cargo_check_lint(&main)
            .await
            .expect("cargo check should run");
        assert!(
            outcome.passed,
            "clean crate must pass, detail: {}",
            outcome.detail
        );
    }

    /// Regression: `cargo_check_lint` must not leave `kitty-lint-*` scratch
    /// files behind in the temp dir — cleanup is guaranteed on every exit path
    /// (spawn failure, timeout, read error, success).
    #[tokio::test]
    async fn cargo_check_lint_leaves_no_scratch_files() {
        fn leftover_count() -> usize {
            std::env::temp_dir()
                .read_dir()
                .map(|entries| {
                    entries
                        .filter_map(|e| e.ok())
                        .filter(|e| e.file_name().to_string_lossy().starts_with("kitty-lint-"))
                        .count()
                })
                .unwrap_or(0)
        }

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"lint_probe_scratch\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        let main = dir.path().join("src/main.rs");
        std::fs::write(&main, "fn main() {}\n").unwrap();

        // Baseline taken BEFORE the run: sibling tests may legitimately hold
        // their own transient scratch files while their cargo check is still
        // in flight; only files that outlive every run indicate a leak.
        let baseline = leftover_count();
        assert!(
            cargo_check_lint(&main).await.is_some(),
            "cargo check should run"
        );

        // Poll briefly so concurrent lint tests can finish and clean theirs.
        for _ in 0..300 {
            if leftover_count() <= baseline {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        assert!(
            leftover_count() <= baseline,
            "scratch files leaked: {} > baseline {baseline}",
            leftover_count()
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // env lock serializes OS-thread tests; runtime is per-test current-thread
    async fn execute_write_appends_lint_section_when_enabled() {
        let _g = lint_env_lock();
        let prev = std::env::var("KITTY_LINT_GATE").ok();
        std::env::set_var("KITTY_LINT_GATE", "1");

        let dir = tempfile::tempdir().unwrap();
        let py = dir.path().join("bad.py");
        let mut registry = ToolRegistry::new();
        let args = serde_json::json!({
            "path": py.to_str().unwrap(),
            "content": "def f(:\n"
        });
        let result = registry.execute_write(&args).await.unwrap();
        assert!(result.success);
        assert!(
            result.output.contains("--- Lint ("),
            "lint section expected in output, got: {}",
            result.output
        );
        assert!(
            result.output.contains("FAILED"),
            "syntax-broken .py must be flagged, got: {}",
            result.output
        );

        match prev {
            Some(v) => std::env::set_var("KITTY_LINT_GATE", v),
            None => std::env::remove_var("KITTY_LINT_GATE"),
        }
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // env lock serializes OS-thread tests; runtime is per-test current-thread
    async fn execute_write_has_no_lint_section_when_disabled() {
        let _g = lint_env_lock();
        let prev = std::env::var("KITTY_LINT_GATE").ok();
        std::env::remove_var("KITTY_LINT_GATE");

        let dir = tempfile::tempdir().unwrap();
        let py = dir.path().join("fine.py");
        let mut registry = ToolRegistry::new();
        let args = serde_json::json!({
            "path": py.to_str().unwrap(),
            "content": "x = 1\n"
        });
        let result = registry.execute_write(&args).await.unwrap();
        assert!(
            !result.output.contains("--- Lint"),
            "gate off → no lint section, got: {}",
            result.output
        );

        match prev {
            Some(v) => std::env::set_var("KITTY_LINT_GATE", v),
            None => std::env::remove_var("KITTY_LINT_GATE"),
        }
    }

    #[test]
    fn parse_cargo_check_json_extracts_errors_and_warnings() {
        let stdout = concat!(
            r#"{"reason":"compiler-artifact","target":{"name":"x"}}"#,
            "\n",
            r#"{"reason":"compiler-message","message":{"level":"error","rendered":"error[E0308]: mismatched types\n --> src/main.rs:1:1\n"}}"#,
            "\n",
            r#"{"reason":"compiler-message","message":{"level":"warning","message":"unused variable","spans":[{"file_name":"src/lib.rs","line_start":7,"is_primary":true}]}}"#,
            "\n",
            r#"{"reason":"compiler-message","message":{"level":"note","message":"ignored"}}"#,
            "\n"
        );
        let diags = parse_cargo_check_json(stdout);
        assert_eq!(diags.len(), 2, "only error+warning kept, got: {diags:?}");
        assert_eq!(diags[0].level, "error");
        assert!(diags[0].rendered.contains("E0308"));
        assert_eq!(diags[1].level, "warning");
        assert!(diags[1].rendered.contains("src/lib.rs:7"));
    }
}

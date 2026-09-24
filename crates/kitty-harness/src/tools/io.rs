//! Read/Edit/Write tool implementations (W2 split of `tools.rs`, por símbolo).
//!
//! Todo-file 1-op I/O via `tokio::fs` (W1). Lint-gate feedback and diff
//! builders live in `super::lint` (re-exported through `tools.rs`).
use super::{
    append_lint_feedback, edit_check_enabled, first_changed_line, reshow_region,
    run_extension_check, truncate_lines, unified_diff, ToolError, ToolRegistry, ToolResult,
};
use std::path::{Path, PathBuf};

impl ToolRegistry {
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
    /// W2: `pub(crate)` so the `normalize` submodule tests can drive Read.
    pub(crate) async fn execute_read(
        &mut self,
        args: &serde_json::Value,
    ) -> Result<ToolResult, ToolError> {
        let path =
            args.get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidArgs {
                    tool: "Read".to_string(),
                    reason: "missing 'path'".to_string(),
                })?;

        let path_buf = PathBuf::from(path);

        // Read with hashline tagging (full-file snapshot regardless of window)
        let snapshot = self.hashline.read_file(&path_buf).await?;
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
    pub(crate) async fn execute_edit(
        &mut self,
        args: &serde_json::Value,
    ) -> Result<ToolResult, ToolError> {
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
                .await
            {
                Ok(new_file_content) => {
                    let before = tokio::fs::read_to_string(&path_buf).await?;
                    tokio::fs::write(&path_buf, &new_file_content).await?;
                    self.hashline.read_file(&path_buf).await?;
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
            match self.hashline.edit_by_hash(&path_buf, old_hash, new_content).await {
                Ok(new_file_content) => {
                    let before = tokio::fs::read_to_string(&path_buf).await?;
                    tokio::fs::write(&path_buf, &new_file_content).await?;
                    // Re-read to get updated hashes
                    self.hashline.read_file(&path_buf).await?;
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
                    self.fallback_search_replace(path, &h, new_content).await
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
    async fn fallback_search_replace(
        &mut self,
        path: &str,
        _hash: &str,
        new_content: &str,
    ) -> Result<ToolResult, ToolError> {
        // If the hash wasn't found, the line might have changed.
        // Re-read and try to find the content by text.
        let path_buf = PathBuf::from(path);
        let snapshot = self.hashline.read_file(&path_buf).await?;

        // Try to find a line that partially matches
        let matching_line = snapshot.lines.iter().find(|l| {
            // Check if the new content's context matches
            l.content.contains(new_content) || new_content.contains(&l.content)
        });

        if let Some(line) = matching_line {
            let result = self
                .hashline
                .edit_by_hash(path, &line.hash, new_content)
                .await?;
            tokio::fs::write(&path_buf, &result).await?;
            self.hashline.read_file(&path_buf).await?;
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

        let before_raw = tokio::fs::read_to_string(path_buf).await?;
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
        tokio::fs::write(path_buf, &after).await?;
        self.hashline.read_file(path_buf).await?;
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
    pub(crate) async fn execute_write(
        &mut self,
        args: &serde_json::Value,
    ) -> Result<ToolResult, ToolError> {
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
        // Disabled when KITTY_BENCH_MODE=1 (benchmarks allow overwrite).
        let bench_mode = crate::config::bench_mode();
        if !bench_mode {
            // Layer 1: spec-required check — existing file not in written_files → block
            if tokio::fs::try_exists(&path_buf).await.unwrap_or(false)
                && !self.written_files.contains(&path_buf)
            {
                return Err(ToolError::WriteBlocked {
                    path: path.to_string(),
                });
            }
            // Layer 2: strict invariant — even files we created via Write in this
            // session must be edited via Edit, not overwritten via Write again.
            // This enforces "Primera vez en tmpdir fresco permite, segunda bloquea".
            if tokio::fs::try_exists(&path_buf).await.unwrap_or(false) {
                return Err(ToolError::WriteBlocked {
                    path: path.to_string(),
                });
            }
        }

        // Create parent directories if needed
        if let Some(parent) = path_buf.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        tokio::fs::write(&path_buf, content).await?;

        // Update session state so the file can be edited afterwards without extra Read
        self.read_files.insert(path_buf.clone());
        self.written_files.insert(path_buf.clone());
        // Populate hashline snapshot for read-before-edit invariant
        let _ = self.hashline.read_file(&path_buf).await;

        let mut output = format!("Wrote {path}");
        append_lint_feedback(&mut output, &path_buf).await;

        Ok(ToolResult {
            success: true,
            output,
            retry: false,
        })
    }

    /// Check if a file has been read this session.
    pub fn is_file_read(&self, path: &Path) -> bool {
        self.read_files.contains(path)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{temp_file, ToolError, ToolRegistry};

    #[tokio::test]
    async fn read_file_records_session() {
        let f = temp_file("line1\nline2");
        let mut registry = ToolRegistry::new();
        let args = serde_json::json!({"path": f.path().to_str().unwrap()});
        let result = registry.execute_read(&args).await.unwrap();
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
    async fn read_truncates_large_file() {
        // F3.1 + f8 T2.3: Read output is capped at max_read_lines, opens with
        // an honest window header, and the omission note suggests offset/limit.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.txt");
        let lines: Vec<String> = (1..=500).map(|i| format!("line {i}")).collect();
        std::fs::write(&path, lines.join("\n")).unwrap();

        let mut registry = ToolRegistry::with_timeout(60).with_read_cap(300);
        let args = serde_json::json!({"path": path.to_str().unwrap()});
        let result = registry.execute_read(&args).await.unwrap();
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

    #[tokio::test]
    async fn read_offset_limit_returns_exact_window() {
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
        let result = registry.execute_read(&args).await.unwrap();
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
            .await
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
            .await
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
            .await
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
            .await
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
            .await
            .unwrap();
        assert!(result
            .output
            .starts_with(&format!("{}: lines 1-2 of 5", path.display())));

        // Edits OUTSIDE the displayed window still succeed — the snapshot is
        // full-file and the window is display-only.
        let hash_args = serde_json::json!({"path": path.to_str().unwrap()});
        registry.execute_read(&hash_args).await.unwrap();
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
        let read_res = registry.execute_read(&read_args).await.unwrap();
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
        let read_res = registry.execute_read(&read_args).await.unwrap();
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
    async fn str_replace_success_single_match() {
        let f = temp_file("alpha\nbeta\ngamma");
        let mut registry = ToolRegistry::new();
        registry
            .execute_read(&serde_json::json!({"path": f.path().to_str().unwrap()}))
            .await
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
            .await
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

    #[tokio::test]
    async fn edit_dual_mode_args_notes_hash_ignored() {
        // E-G1a: when BOTH old_string and hash are provided, str_replace mode
        // wins and the output must say the hash was ignored.
        let f = temp_file("alpha\nbeta\ngamma");
        let mut registry = ToolRegistry::new();
        registry
            .execute_read(&serde_json::json!({"path": f.path().to_str().unwrap()}))
            .await
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
            .await
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
            .await
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
            .await
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
            .await
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
            .await
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
            .await
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
            .await
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
            .await
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
            .await
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
            .await
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
            .await
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
            .await
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
}

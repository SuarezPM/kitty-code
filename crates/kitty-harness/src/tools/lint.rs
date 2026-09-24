//! Lint gates + diff/reshow feedback builders (W2 split of `tools.rs`, por símbolo).
//!
//! Env gates, `cargo_check_lint` (blocking pool), per-extension checkers,
//! and the unified-diff/reshow builders consumed by Edit feedback in `super::io`.
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
/// Whether per-edit syntax checks are enabled (`KITTY_EDIT_CHECK=1`).
/// Off by default: checks cost a subprocess per edit; benchmarks opt in.
pub(crate) fn edit_check_enabled() -> bool {
    crate::config::edit_check()
}

/// Whether the post-edit lint gate is enabled (`KITTY_LINT_GATE=1`).
/// Off by default (same cost rationale as [`edit_check_enabled`]).
fn lint_gate_enabled() -> bool {
    crate::config::lint_gate()
}

/// Append the lint-gate section to a tool output when `KITTY_LINT_GATE=1`.
///
/// `.rs` files get `cargo check --message-format=json` on the nearest crate;
/// `.py` files get `python3 -m py_compile`. Other extensions are not gated.
pub(crate) async fn append_lint_feedback(out: &mut String, path_buf: &Path) {
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
pub(crate) struct CheckOutcome {
    pub(crate) label: String,
    pub(crate) passed: bool,
    pub(crate) detail: String,
}

/// Map a file extension to a cheap syntax checker and run it.
///
/// Returns `None` when no checker maps to the extension or the checker
/// binary cannot be spawned. For `.go` we use `gofmt -e` and treat any
/// stderr output as failure (gofmt's exit status alone is not reliable
/// across versions for parse errors).
pub(crate) fn run_extension_check(path: &Path) -> Option<CheckOutcome> {
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
pub(crate) fn first_changed_line(before: &str, after: &str) -> Option<usize> {
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
pub(crate) fn reshow_region(
    lines: &[crate::hashline::LineHash],
    center: usize,
    radius: usize,
) -> String {
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
pub(crate) fn unified_diff(before: &str, after: &str, max_lines: usize) -> String {
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
pub(crate) fn oversized_hunk(a: &[&str], b: &[&str], ctx: usize, max_lines: usize) -> String {
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
pub(crate) fn truncate_lines(text: &str, max: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= max {
        text.trim_end().to_string()
    } else {
        let mut kept: Vec<String> = lines[..max].iter().map(|s| s.to_string()).collect();
        kept.push(format!("... ({} more lines)", lines.len() - max));
        kept.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::super::ToolRegistry;
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn edit_check_disabled_by_default() {
        if crate::config::edit_check() {
            println!("SKIP: KITTY_EDIT_CHECK=1 set in ambient environment");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("script.py");
        std::fs::write(&path, "x = 1\nbroken(\n").unwrap(); // syntactically invalid
        let mut registry = ToolRegistry::new();
        registry
            .execute_read(&serde_json::json!({"path": path.to_str().unwrap()}))
            .await
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

    #[tokio::test]
    async fn extension_check_pass_and_fail_python() {
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

    #[tokio::test]
    async fn extension_check_bash_n() {
        let dir = tempfile::tempdir().unwrap();
        let ok = dir.path().join("good.sh");
        std::fs::write(&ok, "echo hi\n").unwrap();
        assert!(run_extension_check(&ok).unwrap().passed);

        let bad = dir.path().join("bad.sh");
        std::fs::write(&bad, "if then fi\n").unwrap();
        assert!(!run_extension_check(&bad).unwrap().passed);
    }

    #[tokio::test]
    async fn unified_diff_basic_and_noop() {
        assert_eq!(unified_diff("a\nb\n", "a\nb\n", 40), "");
        let d = unified_diff("a\nb\nc\n", "a\nX\nc\n", 40);
        assert!(d.contains("@@"), "got: {d}");
        assert!(d.contains("-b\n"), "got: {d}");
        assert!(d.contains("+X\n"), "got: {d}");
        assert!(d.contains(" a\n"), "context line, got: {d}");
    }

    #[tokio::test]
    async fn unified_diff_caps_output() {
        let old: String = (0..100).map(|i| format!("line{i}\n")).collect();
        let new: String = (0..100).map(|i| format!("ln{i}\n")).collect();
        let d = unified_diff(&old, &new, 40);
        let emitted = d.lines().count();
        assert!(emitted <= 41, "cap exceeded: {emitted} lines");
        assert!(d.contains("diff truncated"));
    }

    #[tokio::test]
    async fn lint_gate_disabled_by_default() {
        let _g = lint_env_lock();
        let prev = std::env::var("KITTY_LINT_GATE").ok();
        std::env::remove_var("KITTY_LINT_GATE");
        assert!(!lint_gate_enabled(), "lint gate must default to off");
        if let Some(v) = prev {
            std::env::set_var("KITTY_LINT_GATE", v);
        }
    }

    #[tokio::test]
    async fn nearest_crate_root_walks_up() {
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

    #[tokio::test]
    async fn parse_cargo_check_json_extracts_errors_and_warnings() {
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

    // -------------------------------------------------------------------
    // Lint gate (KITTY_LINT_GATE=1)
    // -------------------------------------------------------------------

    fn lint_env_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::model_profile::test_env_lock()
    }
}

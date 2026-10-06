//! Truncation-to-disk output handling (W2 split of `agent/mod.rs`, por símbolo).
//!
//! Oversized tool outputs spill to disk with head+tail excerpts.
// ---------------------------------------------------------------------------
// E3 truncation-to-disk
// ---------------------------------------------------------------------------

/// Tool outputs larger than this many BYTES are persisted to disk (E3).
pub(crate) const TRUNCATION_MAX_BYTES: usize = 50 * 1024;
/// Tool outputs with more than this many LINES are persisted to disk (E3).
pub(crate) const TRUNCATION_MAX_LINES: usize = 2000;
/// Floor (bytes) for the ctx-aware output cap, so a nearly-exhausted context
/// budget still returns a useful excerpt instead of an always-truncated stub.
pub(crate) const MIN_OUTPUT_BUDGET_BYTES: usize = 2048;
/// Head excerpt size (chars) returned for truncated outputs.
pub(crate) const TRUNCATION_HEAD_CHARS: usize = 2000;
/// Tail excerpt size (chars) returned for truncated outputs.
pub(crate) const TRUNCATION_TAIL_CHARS: usize = 1000;
/// Directory where full oversized tool outputs are spilled.
///
/// Defaults to `.kitty/truncation` under the process cwd; override with
/// `KITTY_TRUNCATION_DIR`. Created lazily on first spill.
///
/// Retention: spill files are scratch data — safe to delete after 7 days.
/// There is no automatic purge yet; cleanup is manual
/// (`find .kitty/truncation -mtime +7 -delete`).
use crate::config::truncation_dir;

/// Generate a ULID: 26-char Crockford base32, canonical layout — 48-bit ms
/// timestamp in the high bits + 80 bits of entropy sourced from the
/// nanosecond clock mixed with a process-local counter (no external crate).
fn generate_ulid() -> String {
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let ms = now.as_millis();
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed) as u128;
    // 80-bit entropy space: subsec nanos (30 bits) << 32 | monotonic counter,
    // masked to exactly 80 bits so the ULID layout stays canonical.
    let random: u128 = ((now.subsec_nanos() as u128) << 50)
        | ((counter & 0xFFFF) << 34)
        | ((ms & 0x3FF) << 24)
        | (counter & 0xFFFFFF);
    let value: u128 = (ms << 80) | (random & ((1u128 << 80) - 1));

    const ALPHABET: &[u8] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";
    let mut out = [b'0'; 26];
    let mut v = value;
    for i in (0..26).rev() {
        out[i] = ALPHABET[(v & 31) as usize];
        v >>= 5;
    }
    String::from_utf8(out.to_vec()).unwrap_or_else(|_| "0".repeat(26))
}

/// First `n` chars of `s`, char-boundary safe.
fn head_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

/// Last `n` chars of `s`, char-boundary safe.
fn tail_chars(s: &str, n: usize) -> &str {
    let total = s.chars().count();
    if total <= n {
        return s;
    }
    let skip = total - n;
    match s.char_indices().nth(skip) {
        Some((idx, _)) => &s[idx..],
        None => s,
    }
}

/// E3 truncation-to-disk: when a tool output exceeds `max_bytes` bytes or
/// 2000 lines, write the FULL output to `<truncation_dir>/<ulid>.txt` and
/// return a head+tail excerpt (first 2000 + last 1000 chars) plus a pointer
/// hint so the model can re-read the file if needed.
/// W1 async: todo-file 1-op via tokio::fs so oversized tool results never
/// park the reactor on disk persistence.
///
/// Tier 3: callers derive `max_bytes` from the remaining context budget
/// (see [`effective_output_cap`] and the call site in
/// `execute_and_push_tool_result`) so a large-but-under-50KB Read cannot
/// overflow a small window between compaction checks.
///
/// Best-effort: if the directory cannot be created or the file written, the
/// excerpt is still returned with an explicit "unavailable" marker instead
/// of failing the tool call.
pub(crate) async fn truncate_tool_output(output: String, max_bytes: usize) -> String {
    let over_bytes = output.len() > max_bytes;
    let over_lines = output.lines().count() > TRUNCATION_MAX_LINES;
    if !over_bytes && !over_lines {
        return output;
    }

    let dir = truncation_dir();
    let head = head_chars(&output, TRUNCATION_HEAD_CHARS);
    let tail = tail_chars(&output, TRUNCATION_TAIL_CHARS);

    let persisted = match tokio::fs::create_dir_all(&dir).await {
        Ok(()) => {
            let path = dir.join(format!("{}.txt", generate_ulid()));
            match tokio::fs::write(&path, &output).await {
                Ok(()) => Some(path),
                Err(_) => None,
            }
        }
        Err(_) => None,
    };

    match persisted {
        Some(path) => format!(
            "{head}\n...\n{tail}\n[truncated: full output at {}]",
            path.display()
        ),
        None => format!("{head}\n...\n{tail}\n[truncated: full output unavailable]"),
    }
}

/// Tier 3 read-guard: effective byte cap for a single tool output given the
/// remaining context budget in tokens. Uses the project-wide chars≈bytes/4
/// heuristic (consistent with `ContextBudget::estimate_tokens`) and floors
/// at [`MIN_OUTPUT_BUDGET_BYTES`].
pub(crate) fn effective_output_cap(remaining_tokens: usize) -> usize {
    remaining_tokens
        .saturating_mul(4)
        .max(MIN_OUTPUT_BUDGET_BYTES)
}

#[cfg(test)]
mod tests {
    use super::super::{env_lock, ENV_TRUNCATION_DIR};
    use super::*;
    use crate::model_profile::EnvRestoreGuard;

    #[tokio::test]
    async fn truncate_small_output_passthrough() {
        let out = truncate_tool_output("small output".to_string(), TRUNCATION_MAX_BYTES).await;
        assert_eq!(out, "small output");
    }

    #[test]
    fn head_tail_chars_are_char_boundary_safe() {
        let s = "héllo wörld 日本語のテキスト";
        let head = head_chars(s, 5);
        assert!(s.starts_with(head));
        let tail = tail_chars(s, 4);
        assert!(s.ends_with(tail));
        // No panics on multibyte boundaries; full round trip covers the string
        let total = s.chars().count();
        assert_eq!(
            format!("{}{}", head_chars(s, total - 2), tail_chars(s, 2)),
            s
        );
    }

    #[test]
    fn generate_ulid_is_canonical_shape_and_unique() {
        let a = generate_ulid();
        let b = generate_ulid();
        assert_eq!(a.len(), 26);
        assert!(
            a.chars()
                .all(|c| "0123456789ABCDEFGHJKMNPQRSTVWXYZ".contains(c)),
            "ULID must be Crockford base32, got {a}"
        );
        assert_ne!(a, b, "consecutive ULIDs must differ (counter entropy)");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // env lock serializes OS-thread tests; runtime is per-test current-thread
    async fn truncate_large_output_spills_to_disk_with_pointer() {
        let _g = env_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _e = EnvRestoreGuard::set(ENV_TRUNCATION_DIR, tmp.path());

        // >50KB triggers spill
        let big = "x".repeat(60 * 1024);
        let out = truncate_tool_output(big.clone(), TRUNCATION_MAX_BYTES).await;
        assert!(
            out.contains("[truncated: full output at "),
            "expected truncation pointer, got tail: {}",
            &out[out.len().saturating_sub(200)..]
        );
        assert!(out.starts_with(&"x".repeat(TRUNCATION_HEAD_CHARS)));
        assert!(out.contains(&"x".repeat(TRUNCATION_TAIL_CHARS)));
        // Full output persisted on disk
        let path = out
            .lines()
            .last()
            .unwrap()
            .trim_start_matches("[truncated: full output at ")
            .trim_end_matches(']');
        let persisted = std::fs::read_to_string(path).unwrap();
        assert_eq!(persisted.len(), big.len());

        // >2000 lines also triggers spill
        let many_lines = "line\n".repeat(2500);
        let out2 = truncate_tool_output(many_lines, TRUNCATION_MAX_BYTES).await;
        assert!(out2.contains("[truncated: full output at "));
    }

    #[test]
    fn effective_output_cap_scales_and_floors() {
        // Healthy budget: chars ≈ bytes/4 heuristic scales linearly.
        assert_eq!(effective_output_cap(16384), 16384 * 4);
        // Tiny / exhausted budgets floor at MIN_OUTPUT_BUDGET_BYTES.
        assert_eq!(effective_output_cap(0), MIN_OUTPUT_BUDGET_BYTES);
        assert_eq!(effective_output_cap(100), MIN_OUTPUT_BUDGET_BYTES); // 400 < 2048
                                                                        // Saturating: huge remaining tokens cannot overflow usize.
        assert_eq!(effective_output_cap(usize::MAX), usize::MAX);
    }

    #[tokio::test]
    async fn truncate_under_both_caps_passthrough() {
        // Under the injected byte cap AND the fixed line cap → untouched.
        let out = truncate_tool_output("hello world".to_string(), TRUNCATION_MAX_BYTES).await;
        assert_eq!(out, "hello world");
        let lines = "l\n".repeat(TRUNCATION_MAX_LINES); // exactly at the line cap
        let out2 = truncate_tool_output(lines.clone(), TRUNCATION_MAX_BYTES).await;
        assert_eq!(out2, lines);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // env lock serializes OS-thread tests; runtime is per-test current-thread
    async fn truncate_exceeding_small_injected_budget_spills() {
        let _g = env_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _e = EnvRestoreGuard::set(ENV_TRUNCATION_DIR, tmp.path());

        // 10KB exceeds a SMALL injected cap even though the fixed 50KB
        // threshold would not fire — exactly the Tier 3 overflow class.
        let medium = "y".repeat(10 * 1024);
        let out = truncate_tool_output(medium.clone(), 4096).await;
        assert!(
            out.contains("[truncated: full output at "),
            "expected spill pointer under small budget cap"
        );
        assert!(out.starts_with(&"y".repeat(TRUNCATION_HEAD_CHARS)));
        let path = out
            .lines()
            .last()
            .unwrap()
            .trim_start_matches("[truncated: full output at ")
            .trim_end_matches(']');
        assert_eq!(std::fs::read_to_string(path).unwrap().len(), medium.len());
    }
}

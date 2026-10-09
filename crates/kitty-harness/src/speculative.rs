//! Speculative read-only tool execution (W5-code, SDD estado-arte).
//!
//! PASTE discipline: ONLY read-only tools are ever speculated, and only by
//! co-scheduling the CURRENT turn's batch (never predicted calls with
//! invented arguments). `Glob`/`Grep` are pure queries (`&self`) and run
//! concurrently via `join_all`; results merge back in original index order
//! so downstream message order, doom-loop accounting, and retry semantics
//! are bit-for-bit identical to the sequential path.
//!
//! `Read` is classified read-only but stays SEQUENTIAL: its session
//! registration (read-before-edit) is order-sensitive, and a same-turn
//! `Read → Edit` pair must observe program order. Write-capable tools
//! (`Edit`, `Write`, `Bash`) and unknown names are never prefetched.
//!
//! Wiring: `Agent::with_speculative_readonly` opts in (default OFF —
//! compatible); `process_response` prefetches native batches through
//! [`prefetch_read_only`] and the sequential loop consumes the map.

use std::collections::HashMap;

use crate::parser::ParsedToolCall;
use crate::tools::{ToolRegistry, ToolResult};

/// Tools safe to speculate (no file mutation possible).
pub const READ_ONLY_TOOLS: &[&str] = &["Read", "Glob", "Grep"];

/// Tools prefetched CONCURRENTLY (pure `&self` queries, order-free).
const CONCURRENT_TOOLS: &[&str] = &["Glob", "Grep"];

/// Whether `name` can never mutate files.
#[must_use]
pub fn is_read_only(name: &str) -> bool {
    READ_ONLY_TOOLS.contains(&name)
}

/// Whether `name` is prefetched concurrently (vs sequential read-only).
#[must_use]
pub fn is_concurrent(name: &str) -> bool {
    CONCURRENT_TOOLS.contains(&name)
}

/// Prefetch the concurrent read-only calls of this turn's batch.
///
/// Returns `turn_index → execution outcome` for the prefetched subset only;
/// every other index stays absent (sequential path handles it, including
/// `Read`, writes, unknown names, and repair-failures — the last stored
/// directly since [`ToolRegistry::prepare_call`] is pure).
pub async fn prefetch_read_only(
    registry: &ToolRegistry,
    calls: &[ParsedToolCall],
) -> HashMap<usize, Result<ToolResult, crate::tools::ToolError>> {
    // Shared reborrow: all prefetched ops take `&self`, so their futures
    // may coexist (no `&mut` anywhere on this path).
    let shared: &ToolRegistry = registry;
    let jobs: Vec<(usize, ParsedToolCall)> = calls
        .iter()
        .enumerate()
        .filter_map(|(idx, tc)| {
            let prepared = match ToolRegistry::prepare_call(tc) {
                Ok(call) => call,
                Err(_) => return None, // sequential path returns the early result
            };
            if is_concurrent(prepared.name.as_str()) {
                Some((idx, prepared))
            } else {
                None
            }
        })
        .collect();

    let outcomes = futures::future::join_all(jobs.iter().map(|(_, call)| async move {
        match call.name.as_str() {
            "Glob" => shared.execute_glob(&call.arguments).await,
            "Grep" => shared.execute_grep(&call.arguments).await,
            // Unreachable by construction (`is_concurrent` filter above);
            // kept exhaustive so a wider filter cannot silently mis-dispatch.
            other => Err(crate::tools::ToolError::NotFound(other.to_string())),
        }
    }))
    .await;

    jobs.into_iter().map(|(idx, _)| idx).zip(outcomes).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, args: serde_json::Value) -> ParsedToolCall {
        ParsedToolCall {
            name: name.to_string(),
            arguments: args,
            id: None,
        }
    }

    #[test]
    fn classification_never_speculates_writes() {
        for tool in ["Read", "Glob", "Grep"] {
            assert!(is_read_only(tool), "{tool} must be read-only");
        }
        for tool in ["Edit", "Write", "Bash", "NotATool"] {
            assert!(!is_read_only(tool), "{tool} must NEVER be speculated");
        }
        assert!(is_concurrent("Glob") && is_concurrent("Grep"));
        assert!(
            !is_concurrent("Read"),
            "Read stays sequential (read-before-edit order)"
        );
    }

    /// Prefetch runs Glob+Grep concurrently, skips Read/writes/unknown, and
    /// keys outcomes by original turn index.
    #[tokio::test]
    async fn prefetch_covers_only_concurrent_subset() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello world").unwrap();
        let registry = ToolRegistry::new();
        let dir_s = dir.path().to_str().unwrap();
        let batch = vec![
            call(
                "Grep",
                serde_json::json!({"pattern": "hello", "path": dir_s}),
            ),
            call(
                "Read",
                serde_json::json!({"path": format!("{dir_s}/a.txt")}),
            ),
            call(
                "Glob",
                serde_json::json!({"pattern": format!("{dir_s}/*.txt")}),
            ),
            call(
                "Write",
                serde_json::json!({"path": format!("{dir_s}/x.txt"), "content": "no"}),
            ),
            call("NotATool", serde_json::json!({})),
        ];
        let map = prefetch_read_only(&registry, &batch).await;
        assert_eq!(map.len(), 2, "only Glob+Grep prefetched: {:?}", map.keys());
        assert!(map.contains_key(&0));
        assert!(map.contains_key(&2));
        for outcome in map.values() {
            let r = outcome.as_ref().expect("grep/glob must succeed");
            assert!(r.success, "unexpected failure: {}", r.output);
        }
        // Nothing was written by the prefetch.
        assert!(!dir.path().join("x.txt").exists());
    }
}

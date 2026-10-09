//! Plan mode (W3): `--plan` blocks file-mutating tools.
//!
//! Enforcement points (both in `kitty-cli`, no harness changes):
//! - `kitty run --plan "..."` never enters the agent loop (dry-run: prints
//!   what would run). Trivially zero edits.
//! - [`McpServer`](crate::mcp::McpServer) in plan mode rejects `Edit`/`Write`
//!   inside [`call_tool`](crate::mcp::McpServer::call_tool) via
//!   [`plan_blocks_tool`].
//!
//! The mutating set is exactly `{Edit, Write}`: `Bash` stays available
//! because the [`Guard`](kitty_harness::guard::Guard) already blocks
//! redirect-writes (`echo hi > file`) and dangerous commands — plan mode does
//! not need to re-block what the guard owns.

/// Tools blocked in plan mode (canonical names, case-sensitive like the
/// registry: `Read`, `Edit`, `Write`, `Bash`, `Glob`, `Grep`).
const PLAN_BLOCKED: &[&str] = &["Edit", "Write"];

/// Returns `true` when `tool` must be refused while plan mode is on.
#[must_use]
pub fn plan_blocks_tool(tool: &str) -> bool {
    PLAN_BLOCKED.contains(&tool)
}

/// Refusal message for a blocked call (surfaced as `success:false` output so
/// MCP/agent consumers handle it like any other tool error).
#[must_use]
pub fn plan_block_message(tool: &str) -> String {
    format!("plan mode: `{tool}` is disabled — re-run without `--plan` to apply edits")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_blocks_edits_only() {
        assert!(plan_blocks_tool("Edit"));
        assert!(plan_blocks_tool("Write"));
        assert!(!plan_blocks_tool("Read"));
        assert!(!plan_blocks_tool("Glob"));
        assert!(!plan_blocks_tool("Grep"));
        assert!(!plan_blocks_tool("Bash"));
        assert!(!plan_blocks_tool(""));
        assert!(!plan_blocks_tool("edit"));
    }

    #[test]
    fn block_message_names_tool() {
        assert!(plan_block_message("Edit").contains("Edit"));
        assert!(plan_block_message("Edit").contains("--plan"));
    }
}

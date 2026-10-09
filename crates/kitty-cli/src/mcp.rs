//! MCP server over stdio — spike minimal.
//!
//! Exposes Kitty Code tools (Read, Edit, Write, Bash, Glob, Grep) as
//! MCP tools over a simple line-delimited JSON-RPC on stdio.
//! Every call goes through the same security stack (Guard + write-vs-edit
//! + permission-gate) because it reuses `ToolRegistry::execute`.

use std::path::PathBuf;
use std::sync::Arc;

use kitty_harness::parser::ParsedToolCall;
use kitty_harness::tools::{standard_tool_definitions, ToolRegistry};
use kitty_runtime::ToolDefinition;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// Canonicalize a tool name for plan/journal decisions (mirrors the
/// harness `normalize_tool_name` real-name fold; synonyms stay untouched —
/// the registry resolves those after our gate).
fn canonical_tool_name(name: &str) -> &str {
    for real in ["Read", "Edit", "Write", "Bash", "Glob", "Grep"] {
        if real.eq_ignore_ascii_case(name) {
            return real;
        }
    }
    name
}

/// MCP server that advertises Kitty Code tools.
pub struct McpServer {
    tools: Vec<ToolDefinition>,
    registry: Arc<tokio::sync::Mutex<ToolRegistry>>,
    /// W3 plan mode: `Edit`/`Write` are refused (see `crate::plan`).
    plan_mode: bool,
    /// W3 edit journal: `Edit`/`Write` successes are recorded for `kitty undo`.
    /// `None` = journaling disabled (unit tests use an explicit temp path).
    journal_path: Option<PathBuf>,
}

impl McpServer {
    /// Create a server with the standard tool set.
    pub fn new() -> Self {
        Self {
            tools: standard_tool_definitions(),
            // Gate-4: sandbox allow-root resolved explicitly (project root),
            // not the registry's implicit process-cwd fallback.
            registry: Arc::new(tokio::sync::Mutex::new(
                ToolRegistry::new().with_workspace(crate::resolve_project_root()),
            )),
            plan_mode: false,
            journal_path: None,
        }
    }

    /// Enable/disable W3 plan mode (blocks `Edit`/`Write`).
    pub fn with_plan_mode(mut self, plan_mode: bool) -> Self {
        self.plan_mode = plan_mode;
        self
    }

    /// Record `Edit`/`Write` successes into `journal` for `kitty undo`.
    pub fn with_journal(mut self, journal: Option<PathBuf>) -> Self {
        self.journal_path = journal;
        self
    }

    /// Return the tool definitions as JSON value (array).
    pub fn list_tools(&self) -> Value {
        json!(self.tools)
    }

    /// Execute a tool by name with the given arguments.
    ///
    /// This goes through `ToolRegistry::execute` so Guard, write-vs-edit and
    /// Bash timeout are enforced identically to the normal agent loop.
    /// W3: plan mode refuses `Edit`/`Write` before touching the registry;
    /// successful `Edit`/`Write` calls are journaled for `kitty undo`.
    pub async fn call_tool(&self, name: &str, args: Value) -> Value {
        let canonical = canonical_tool_name(name);
        if self.plan_mode && crate::plan::plan_blocks_tool(canonical) {
            return json!({
                "success": false,
                "output": crate::plan::plan_block_message(canonical)
            });
        }
        // Pre-image for the journal (only meaningful for file mutations).
        let journal_target: Option<PathBuf> =
            if self.journal_path.is_some() && matches!(canonical, "Edit" | "Write") {
                args.get("path").and_then(|v| v.as_str()).map(PathBuf::from)
            } else {
                None
            };
        let before = journal_target
            .as_ref()
            .and_then(|p| crate::journal::snapshot(p));
        let tc = ParsedToolCall {
            name: name.to_string(),
            arguments: args,
            // MCP calls originate locally: no server-issued id exists.
            id: None,
        };
        let mut registry = self.registry.lock().await;
        match registry.execute(&tc).await {
            Ok(result) => {
                if result.success {
                    if let (Some(journal), Some(target)) = (&self.journal_path, &journal_target) {
                        let after = crate::journal::snapshot(target).unwrap_or_default();
                        let _ = crate::journal::record_file_edit(journal, target, before, after);
                    }
                }
                json!({
                    "success": result.success,
                    "output": result.output
                })
            }
            Err(e) => json!({
                "success": false,
                "output": e.to_string()
            }),
        }
    }

    /// Handle a single JSON request value and produce a JSON response value.
    ///
    /// Exposed for testing without stdio.
    pub async fn handle_request(&self, req: Value) -> Value {
        let method = req.get("method").and_then(|v| v.as_str()).unwrap_or("");
        // JSON-RPC id if present — echo back
        let id = req.get("id").cloned();

        let result = match method {
            "tools/list" | "tools/list_tools" | "list_tools" => self.list_tools(),
            "tools/call" | "tools/call_tool" | "call_tool" => {
                let params = req.get("params").cloned().unwrap_or(Value::Null);
                // Support both {name, arguments} at top-level params or directly in req
                let (name, arguments) = if params.is_object() {
                    let n = params
                        .get("name")
                        .and_then(|v| v.as_str())
                        .or_else(|| {
                            req.get("params")
                                .and_then(|p| p.get("name"))
                                .and_then(|v| v.as_str())
                        })
                        .unwrap_or_else(|| {
                            // also allow params directly containing name at req level (fallback)
                            req.get("name").and_then(|v| v.as_str()).unwrap_or("")
                        });
                    let a = params
                        .get("arguments")
                        .cloned()
                        .or_else(|| params.get("params").cloned())
                        .unwrap_or_else(|| json!({}));
                    (n.to_string(), a)
                } else {
                    // params may be absent — try top-level name/arguments
                    let n = req.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let a = req.get("arguments").cloned().unwrap_or_else(|| json!({}));
                    (n.to_string(), a)
                };
                // If params was an object with name/arguments, the above already extracted;
                // otherwise re-extract cleanly:
                let (final_name, final_args) = if !name.is_empty() || !arguments.is_object() {
                    // Already have something, but for the common MCP shape:
                    // {"method":"tools/call","params":{"name":"Bash","arguments":{...}}}
                    // we want params.name / params.arguments
                    if let Some(p) = req.get("params") {
                        let pn = p.get("name").and_then(|v| v.as_str()).unwrap_or(&name);
                        let pa = p.get("arguments").cloned().unwrap_or(arguments.clone());
                        (pn.to_string(), pa)
                    } else {
                        (name, arguments)
                    }
                } else {
                    (name, arguments)
                };
                self.call_tool(&final_name, final_args).await
            }
            // MCP initialize handshake — respond with ok
            "initialize" => json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "kitty-mcp", "version": env!("CARGO_PKG_VERSION") }
            }),
            "notifications/initialized" | "notifications/cancelled" => json!({"status":"ok"}),
            "" => json!({"error": "missing method"}),
            other => json!({"error": format!("unknown method: {}", other)}),
        };

        if let Some(id_val) = id {
            json!({"jsonrpc":"2.0","id": id_val, "result": result})
        } else {
            json!({"result": result})
        }
    }

    /// Run the stdio loop: read lines of JSON, dispatch, write JSON response per line.
    pub async fn run_stdio(&self) {
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin);
        let mut stdout = tokio::io::stdout();
        let mut line = String::new();

        loop {
            line.clear();
            let n = match reader.read_line(&mut line).await {
                Ok(n) => n,
                Err(_) => break,
            };
            if n == 0 {
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let req: Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(e) => {
                    let resp = json!({"error": format!("invalid json: {}", e)});
                    let out = serde_json::to_string(&resp)
                        .unwrap_or_else(|_| r#"{"error":"serialization failed"}"#.to_string());
                    let _ = stdout.write_all(out.as_bytes()).await;
                    let _ = stdout.write_all(b"\n").await;
                    let _ = stdout.flush().await;
                    continue;
                }
            };
            let resp = self.handle_request(req).await;
            let out = serde_json::to_string(&resp)
                .unwrap_or_else(|_| r#"{"error":"serialization failed"}"#.to_string());
            let _ = stdout.write_all(out.as_bytes()).await;
            let _ = stdout.write_all(b"\n").await;
            let _ = stdout.flush().await;
        }
    }
}

impl Default for McpServer {
    fn default() -> Self {
        Self::new()
    }
}

/// Entrypoint for `kitty mcp` subcommand.
pub async fn run(plan_mode: bool) -> anyhow::Result<()> {
    let server = McpServer::new()
        .with_plan_mode(plan_mode)
        .with_journal(Some(crate::journal::default_journal_path()));
    server.run_stdio().await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn list_tools_returns_6_tools() {
        let server = McpServer::new();
        let tools = server.list_tools();
        let arr = tools.as_array().expect("list_tools should return array");
        assert_eq!(arr.len(), 6, "expected 6 tools, got {}", arr.len());
        let names: Vec<String> = arr
            .iter()
            .map(|t| {
                t.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string()
            })
            .collect();
        for expected in ["Read", "Edit", "Write", "Bash", "Glob", "Grep"] {
            assert!(
                names.contains(&expected.to_string()),
                "missing tool {expected}, got {names:?}"
            );
        }
    }

    #[tokio::test]
    async fn call_tool_read_nonexistent_returns_error() {
        let server = McpServer::new();
        let result = server
            .call_tool(
                "Read",
                json!({"path": "/tmp/kitty_mcp_nonexistent_12345_xyz.txt"}),
            )
            .await;
        // Should be success:false with an error output
        assert_eq!(result.get("success").and_then(|v| v.as_bool()), Some(false));
        let output = result.get("output").and_then(|v| v.as_str()).unwrap_or("");
        assert!(!output.is_empty(), "output should contain error message");
    }

    #[tokio::test]
    async fn call_tool_bash_echo_works() {
        let server = McpServer::new();
        let result = server
            .call_tool("Bash", json!({"command": "echo hello"}))
            .await;
        assert_eq!(
            result.get("success").and_then(|v| v.as_bool()),
            Some(true),
            "got {result:?}"
        );
        let output = result.get("output").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            output.contains("hello"),
            "expected 'hello' in output, got: {output}"
        );
    }

    #[tokio::test]
    async fn call_tool_bash_blocked_via_guard() {
        let server = McpServer::new();
        let result = server
            .call_tool("Bash", json!({"command": "rm -rf /"}))
            .await;
        assert_eq!(result.get("success").and_then(|v| v.as_bool()), Some(false));
        let output = result.get("output").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            output.to_lowercase().contains("blocked") || output.contains("dangerous"),
            "expected blocked message, got: {output}"
        );
    }

    #[tokio::test]
    async fn handle_request_tools_list() {
        let server = McpServer::new();
        let req = json!({"method":"tools/list","id":1});
        let resp = server.handle_request(req).await;
        // Should have result array of 6
        let result = resp.get("result").expect("should have result");
        let arr = result.as_array().expect("result should be array");
        assert_eq!(arr.len(), 6);
        assert_eq!(resp.get("id").and_then(|v| v.as_u64()), Some(1));
    }

    #[tokio::test]
    async fn handle_request_tools_call() {
        let server = McpServer::new();
        let req = json!({
            "method":"tools/call",
            "params":{"name":"Bash","arguments":{"command":"echo hello"}},
            "id": 2
        });
        let resp = server.handle_request(req).await;
        let result = resp.get("result").expect("should have result");
        assert_eq!(result.get("success").and_then(|v| v.as_bool()), Some(true));
    }

    // NB: the lock intentionally spans awaits (the env must stay put for
    // the whole edit); test-only, no executor block risk that matters here.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn test_mcp_persist_read_then_edit_sequential() {
        // Successful `Edit` goes through the harness journal hook, which
        // reads process-global `KITTY_JOURNAL` — serialize against tests
        // that swap it (see `crate::journal::test_journal_env_lock`).
        let _lock = crate::journal::test_journal_env_lock();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("persist_test.txt");
        std::fs::write(&path, "hello\nworld").unwrap();

        let server = McpServer::new();

        let read_req = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "Read", "arguments": {"path": path.to_str().unwrap()}}
        });
        let read_resp = server.handle_request(read_req).await;
        let read_result = read_resp.get("result").expect("Read should have result");
        assert_eq!(
            read_result.get("success").and_then(|v| v.as_bool()),
            Some(true),
            "Read should succeed, got {read_result:?}"
        );
        let output = read_result
            .get("output")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let first_line = output
            .lines()
            .nth(1) // skip the honest-window header (f8 T2.3)
            .expect("Read output should have tagged lines after the header");
        let hash = first_line
            .split(':')
            .nth(1)
            .unwrap()
            .split('|')
            .next()
            .unwrap();

        let edit_req = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "Edit", "arguments": {"path": path.to_str().unwrap(), "hash": hash, "new": "HELLO"}}
        });
        let edit_resp = server.handle_request(edit_req).await;
        let edit_result = edit_resp.get("result").expect("Edit should have result");
        assert_eq!(
            edit_result.get("success").and_then(|v| v.as_bool()),
            Some(true),
            "Edit after Read should succeed, got {edit_result:?}"
        );
    }

    #[tokio::test]
    async fn handle_request_bash_redirect_blocked() {
        // Security: write via redirect should be blocked even via MCP
        let server = McpServer::new();
        let req = json!({
            "method":"tools/call",
            "params":{"name":"Bash","arguments":{"command":"echo hi > /tmp/evil_mcp.txt"}},
            "id": 3
        });
        let resp = server.handle_request(req).await;
        let result = resp.get("result").expect("should have result");
        assert_eq!(result.get("success").and_then(|v| v.as_bool()), Some(false));
        let output = result.get("output").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            output.contains("do not redirect")
                || output.contains("Write")
                || output.to_lowercase().contains("blocked"),
            "expected redirect block, got: {output}"
        );
    }

    #[tokio::test]
    async fn plan_mode_blocks_edit_but_allows_read() {
        let server = McpServer::new().with_plan_mode(true);
        let edit = server
            .call_tool(
                "Edit",
                json!({"path": "/tmp/x.txt", "hash": "abcdef", "new": "y"}),
            )
            .await;
        assert_eq!(edit.get("success").and_then(|v| v.as_bool()), Some(false));
        assert!(
            edit.get("output")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .contains("--plan"),
            "got {edit:?}"
        );
        // Case-insensitive canonicalization also blocks.
        let write = server
            .call_tool("write", json!({"path": "/tmp/x.txt", "content": "y"}))
            .await;
        assert_eq!(write.get("success").and_then(|v| v.as_bool()), Some(false));
        // Reads still work in plan mode.
        let read = server
            .call_tool(
                "Read",
                json!({"path": "/tmp/kitty_mcp_nonexistent_12345_xyz.txt"}),
            )
            .await;
        let output = read.get("output").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            !output.contains("--plan"),
            "reads must not be plan-blocked, got {read:?}"
        );
    }

    // NB: same intentional hold-across-await as above (test-only).
    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn journal_records_edit_for_undo() {
        // Same global-env reason as above: the `Edit` below is also seen
        // by the harness `ToolRegistry::execute` hook.
        let _lock = crate::journal::test_journal_env_lock();
        let dir = tempfile::tempdir().unwrap();
        let journal = dir.path().join("journal.jsonl");
        let target = dir.path().join("note.txt");
        std::fs::write(&target, "before").unwrap();

        let server = McpServer::new().with_journal(Some(journal.clone()));
        // Read first (read-before-edit guard), then edit via hash.
        let read = server
            .call_tool("Read", json!({"path": target.to_str().unwrap()}))
            .await;
        assert_eq!(read.get("success").and_then(|v| v.as_bool()), Some(true));
        let output = read.get("output").and_then(|v| v.as_str()).unwrap_or("");
        let hash = output
            .lines()
            .nth(1)
            .expect("tagged lines")
            .split(':')
            .nth(1)
            .unwrap()
            .split('|')
            .next()
            .unwrap();
        let edit = server
            .call_tool(
                "Edit",
                json!({"path": target.to_str().unwrap(), "hash": hash, "new": "AFTER"}),
            )
            .await;
        assert_eq!(
            edit.get("success").and_then(|v| v.as_bool()),
            Some(true),
            "got {edit:?}"
        );

        let entries = crate::journal::read_entries(&journal).unwrap();
        assert_eq!(entries.len(), 1, "got {entries:?}");
        assert_eq!(entries[0].before.as_deref(), Some("before"));

        let undone = crate::journal::undo_last(&journal).unwrap().expect("entry");
        assert_eq!(undone, target);
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "before");
    }
}

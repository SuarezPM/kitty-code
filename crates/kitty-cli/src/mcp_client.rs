//! MCP client over `rmcp` 3.5.1 (W3, plan SDD-ESTADO-ARTE §3 W-MCP).
//!
//! Production surface:
//! - [`list_tools_via_child`] — spawn an MCP server as a child process
//!   (transport `TokioChildProcess`) and return its tool names. Same pattern
//!   as the upstream `everything_stdio` example (`npx -y
//!   @modelcontextprotocol/server-everything`).
//! - [`call_tool_via_child`] — same transport, single `tools/call` round-trip.
//!
//! Self-test surface (no network, no `npx`):
//! - [`ProbeServer`] — minimal in-crate MCP server (`probe_echo`,
//!   `probe_add`) served over stdio by the hidden `probe-mcp-server`
//!   subcommand, so the child-process round-trip is exercised in tests with
//!   `CARGO_BIN_EXE_kitty`.
//!
//! The product MCP *server* stays `mcp.rs` (own line-delimited JSON-RPC); this
//! module is strictly the *client* direction.

use anyhow::{Context, Result};
use rmcp::{
    transport::{ConfigureCommandExt, TokioChildProcess},
    ServiceExt,
};
use tokio::process::Command;

/// Spawn `program args...` as an MCP server over stdio and list its tools.
///
/// Returns tool names in server order. The child is terminated on drop after
/// an explicit [`ServiceExt::cancel`].
pub async fn list_tools_via_child(program: &str, args: &[String]) -> Result<Vec<String>> {
    let transport = TokioChildProcess::new(Command::new(program).configure(|cmd| {
        cmd.args(args);
    }))
    .context("spawning MCP server child process")?;
    let client = ().serve(transport).await.context("MCP initialize")?;
    let tools = client.list_all_tools().await.context("tools/list")?;
    let names = tools.iter().map(|t| t.name.to_string()).collect::<Vec<_>>();
    let _ = client.cancel().await;
    Ok(names)
}

/// Spawn `program args...` and call one tool with JSON object arguments.
pub async fn call_tool_via_child(
    program: &str,
    args: &[String],
    tool: &str,
    arguments: serde_json::Value,
) -> Result<serde_json::Value> {
    use rmcp::model::CallToolRequestParams;

    let transport = TokioChildProcess::new(Command::new(program).configure(|cmd| {
        cmd.args(args);
    }))
    .context("spawning MCP server child process")?;
    let client = ().serve(transport).await.context("MCP initialize")?;
    let mut params = CallToolRequestParams::new(tool.to_owned());
    if let Some(map) = arguments.as_object().cloned() {
        params = params.with_arguments(map);
    }
    let result = client.call_tool(params).await.context("tools/call")?;
    let _ = client.cancel().await;
    Ok(serde_json::to_value(&result)?)
}

// ── Probe server (self-test) ──────────────────────────────────────────

/// Params for `probe_add`.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ProbeAddArgs {
    /// First summand.
    pub a: i32,
    /// Second summand.
    pub b: i32,
}

/// Minimal MCP server for tests: `probe_echo` + `probe_add`.
#[derive(Clone)]
pub struct ProbeServer {
    // Held for the `#[tool_router]` contract (routing is generated off the
    // type, not read back through this field).
    #[allow(dead_code)]
    tool_router: rmcp::handler::server::router::tool::ToolRouter<Self>,
}

#[rmcp::tool_router]
impl ProbeServer {
    /// Build a probe instance.
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    /// Echo back the raw JSON arguments as text.
    #[rmcp::tool(description = "Echo back the call arguments (self-test probe)")]
    async fn probe_echo(
        &self,
        params: rmcp::handler::server::wrapper::Parameters<rmcp::model::JsonObject>,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        Ok(rmcp::model::CallToolResult::success(vec![
            rmcp::model::ContentBlock::text(serde_json::Value::Object(params.0).to_string()),
        ]))
    }

    /// Add two integers (self-test probe).
    #[rmcp::tool(description = "Add two integers a+b (self-test probe)")]
    async fn probe_add(
        &self,
        params: rmcp::handler::server::wrapper::Parameters<ProbeAddArgs>,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        Ok(rmcp::model::CallToolResult::success(vec![
            rmcp::model::ContentBlock::text((params.0.a + params.0.b).to_string()),
        ]))
    }
}

impl Default for ProbeServer {
    fn default() -> Self {
        Self::new()
    }
}

#[rmcp::tool_handler]
impl rmcp::ServerHandler for ProbeServer {
    fn get_info(&self) -> rmcp::model::ServerConfig {
        rmcp::model::ServerConfig::new(
            rmcp::model::ServerCapabilities::builder()
                .enable_tools()
                .build(),
        )
    }
}

/// Serve the probe over stdio (hidden `probe-mcp-server` subcommand).
pub async fn serve_probe_stdio() -> Result<()> {
    let service = ProbeServer::new()
        .serve(rmcp::transport::stdio())
        .await
        .context("serving probe MCP server")?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// In-process client↔server round-trip over a duplex pair (no child, no
    /// network): proves `list_all_tools` + `tools/call` against the probe.
    #[tokio::test]
    async fn probe_lists_tools_over_duplex() {
        let (client_io, server_io) = tokio::io::duplex(64 * 1024);
        let server_task = tokio::spawn(async move {
            let svc = ProbeServer::new().serve(server_io).await?;
            svc.waiting().await?;
            Ok::<(), anyhow::Error>(())
        });
        // Yield once so the server side starts serving before initialize.
        tokio::task::yield_now().await;

        let client = ().serve(client_io).await.expect("initialize");
        let tools = client.list_all_tools().await.expect("tools/list");
        let names: Vec<String> = tools.iter().map(|t| t.name.to_string()).collect();
        assert!(
            names.contains(&"probe_echo".to_string()),
            "expected probe_echo, got {names:?}"
        );
        assert!(
            names.contains(&"probe_add".to_string()),
            "expected probe_add, got {names:?}"
        );

        let result = client
            .call_tool(
                rmcp::model::CallToolRequestParams::new("probe_add").with_arguments(
                    serde_json::json!({"a": 2, "b": 3})
                        .as_object()
                        .cloned()
                        .unwrap_or_default(),
                ),
            )
            .await
            .expect("tools/call");
        let payload = serde_json::to_value(&result).expect("serialize result");
        assert!(
            payload.to_string().contains('5'),
            "expected sum 5 in {payload}"
        );

        let _ = client.cancel().await;
        server_task.abort();
    }

    /// Spawning a nonexistent binary is a clean error, never a hang.
    #[tokio::test]
    async fn child_spawn_failure_is_error() {
        let err = list_tools_via_child("/tmp/kitty-no-such-mcp-server-xyz", &[])
            .await
            .expect_err("should fail");
        assert!(!err.to_string().is_empty());
    }
}

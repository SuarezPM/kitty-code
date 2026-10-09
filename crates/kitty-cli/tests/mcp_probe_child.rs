//! W3: rmcp child-process round-trip against the in-crate probe server.
//!
//! Spawns the `kitty` binary itself (hidden `probe-mcp-server` subcommand)
//! via `TokioChildProcess` and lists/calls tools — no network, no `npx`.
//! This test also pins the `[[bin]] name = "kitty"` contract through
//! `CARGO_BIN_EXE_kitty`.

use rmcp::{
    model::CallToolRequestParams,
    transport::{ConfigureCommandExt, TokioChildProcess},
    ServiceExt,
};
use tokio::process::Command;

#[tokio::test]
async fn probe_lists_and_calls_tools_over_child_process() {
    let exe = env!("CARGO_BIN_EXE_kitty");
    let transport = TokioChildProcess::new(Command::new(exe).configure(|cmd| {
        cmd.arg("probe-mcp-server");
    }))
    .expect("spawn kitty probe-mcp-server");
    let client = ().serve(transport).await.expect("MCP initialize");

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
            CallToolRequestParams::new("probe_add").with_arguments(
                serde_json::json!({"a": 20, "b": 22})
                    .as_object()
                    .cloned()
                    .unwrap_or_default(),
            ),
        )
        .await
        .expect("tools/call");
    let payload = serde_json::to_value(&result).expect("serialize result");
    assert!(
        payload.to_string().contains("42"),
        "expected sum 42 in {payload}"
    );

    let _ = client.cancel().await;
}

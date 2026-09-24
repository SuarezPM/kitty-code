//! Harness micro-benchmarks (Fase 3.3): parser hot path. These measure the
//! HARNESS, not the model — they close the "no profiling" gap flagged by the
//! 2026-09 audit before any further optimization is attempted.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use kitty_harness::parser::ToolCallParser;

const CLEAN_NATIVE: &str = r#"I'll read the file.
{"name": "Read", "arguments": {"path": "src/main.rs"}}"#;

const MIXED_REPAIR: &str = "Let me try:\n<tool_call>\n{\"name\": \"Edit\", \"arguments\": {\"path\": \"a.rs\", \"old_hash\": \"abc123\", \"new_content\": \"fn main() {}\"}}\n</tool_call>\nDone.";

fn bench_parser(c: &mut Criterion) {
    c.bench_function("parser_clean_native", |b| {
        let parser = ToolCallParser::new();
        b.iter(|| parser.parse(black_box(CLEAN_NATIVE)))
    });
    c.bench_function("parser_mixed_xml_stream", |b| {
        let parser = ToolCallParser::new();
        b.iter(|| parser.parse(black_box(MIXED_REPAIR)))
    });
}

// ── T1.5: tool-loop overhead SIN modelo (MockClient, N=50 turns) ──────
// Measures the HARNESS per-turn cost (dispatch + guard + budget + messages),
// never the model. Rotation Read/Bash/Glob keeps every turn successful while
// the doom streak never exceeds 1 (no guard trip). One iter = one 50-turn
// `Agent::run`; criterion reports run stats, per-turn ≈ median/50.
// k>=3 con modelo real = maratón (~43min CPU en este historial): declarado,
// no intentado.
fn tool_response(name: &str, args: serde_json::Value) -> kitty_runtime::ChatResponse {
    kitty_runtime::ChatResponse {
        model: "bench".into(),
        message: kitty_runtime::Message {
            role: "assistant".into(),
            content: None,
            thinking: None,
            tool_calls: Some(vec![kitty_runtime::ToolCall {
                id: None,
                function: kitty_runtime::ToolCallFunction {
                    name: name.into(),
                    arguments: args,
                },
            }]),
            tool_call_id: None,
        },
        done: true,
        eval_count: None,
        eval_duration: None,
        prompt_eval_count: None,
    }
}

fn text_response() -> kitty_runtime::ChatResponse {
    kitty_runtime::ChatResponse {
        model: "bench".into(),
        message: kitty_runtime::Message::assistant("done"),
        done: true,
        eval_count: None,
        eval_duration: None,
        prompt_eval_count: None,
    }
}

fn bench_tooloop_50turns(c: &mut Criterion) {
    use kitty_harness::agent::{Agent, AgentConfig, RunOverrides};

    c.bench_function("tooloop_50turns_no_model", |b| {
        b.iter_custom(|iters| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let elapsed = rt.block_on(async {
                    let tmp = tempfile::tempdir().unwrap();
                    let fp = tmp.path().join("w.txt");
                    std::fs::write(&fp, "bench content").unwrap();
                    let path = fp.to_string_lossy().to_string();
                    let mut responses = Vec::with_capacity(52);
                    for i in 0..50 {
                        match i % 3 {
                            0 => responses
                                .push(tool_response("Read", serde_json::json!({"path": path}))),
                            1 => responses.push(tool_response(
                                "Bash",
                                serde_json::json!({"command": "echo hi"}),
                            )),
                            _ => responses.push(tool_response(
                                "Glob",
                                serde_json::json!({"pattern": "*.rs"}),
                            )),
                        }
                    }
                    responses.push(text_response());
                    responses.push(text_response());
                    let mut agent = Agent::new(
                        kitty_runtime::MockClient::new(responses),
                        AgentConfig {
                            workspace: tmp.path().to_path_buf(),
                            max_iterations: 51,
                            ..Default::default()
                        },
                    );
                    let start = std::time::Instant::now();
                    // max_iterations via overrides: construction syncs the
                    // profile default (medium=20) over the struct value.
                    let out = black_box(
                        agent
                            .run_with(
                                "bench 50 turns",
                                RunOverrides {
                                    max_iterations: Some(51),
                                    ..Default::default()
                                },
                            )
                            .await,
                    );
                    assert!(out.is_ok(), "bench run must succeed, got: {out:?}");
                    assert_eq!(agent.react_steps().len(), 50, "one ReActStep per turn");
                    start.elapsed()
                });
                total += elapsed;
            }
            total
        })
    });
}

criterion_group!(benches, bench_parser, bench_tooloop_50turns);
criterion_main!(benches);

//! Instrumented spike for backend comparison (Fase 3, bench spec companion).
//!
//! Measures, per fixed prompt, what the bench core needs and nothing else:
//! wall latency, success, and completion size — through `Box<dyn ChatClient>`
//! so the SAME binary measures Ollama and embedded kitty-infer.
//!
//! Run Ollama:   `cargo run -p kitty-harness --example infer_spike`
//! Run embedded: `cargo run -p kitty-harness --example infer_spike --features kitty-infer`
//!   with `KITTY_BACKEND=kitty-infer KITTY_INFER_MODEL=<path.gguf> [KITTY_INFER_*]`
//!
//! This is NOT the bench: no exercises, no grading, no agent loop. It exists
//! to validate backend parity cheaply before the full bench-over-ChatClient
//! generalization lands (see `docs/EMBEDDED-BENCH-SPEC.md`).

use kitty_runtime::{ChatClient, Message};
use std::time::Instant;

const PROMPTS: &[&str] = &[
    "Reply with exactly: forty-two",
    "Write a Rust function that reverses a string. Reply with code only.",
    "What is 17 multiplied by 23? Reply with the number only.",
];

fn build_client() -> Box<dyn ChatClient> {
    if kitty_runtime::backend_selected(std::env::var(kitty_runtime::ENV_BACKEND).ok().as_deref()) {
        #[cfg(feature = "kitty-infer")]
        {
            let client = kitty_runtime::KittyInferClient::from_env().unwrap_or_else(|e| {
                eprintln!("embedded backend misconfigured: {e}");
                std::process::exit(2);
            });
            return Box::new(client);
        }
        #[cfg(not(feature = "kitty-infer"))]
        {
            eprintln!("KITTY_BACKEND=kitty-infer needs --features kitty-infer");
            std::process::exit(2);
        }
    }
    let url = std::env::var("KITTY_URL").unwrap_or_else(|_| "http://localhost:11434".into());
    Box::new(kitty_runtime::OllamaClient::new(&url))
}

#[tokio::main]
async fn main() {
    let model = std::env::args().nth(1).unwrap_or_else(|| "spike".into());
    let client = build_client();
    match client.health_check().await {
        Ok(()) => println!("backend=ok"),
        Err(e) => {
            println!("backend=unreachable ({e})");
            std::process::exit(1);
        }
    }
    println!("prompt_idx\tok\tlatency_ms\tchars\tresponse_head");
    for (i, prompt) in PROMPTS.iter().enumerate() {
        let start = Instant::now();
        let out = client
            .chat(&model, vec![Message::user(*prompt)], None)
            .await;
        let ms = start.elapsed().as_millis();
        match out {
            Ok(resp) => {
                let text = resp.message.content.unwrap_or_default();
                let head: String = text.chars().take(60).collect();
                println!("{i}\tok\t{ms}\t{}\t{head}", text.len());
            }
            Err(e) => println!("{i}\tERR\t{ms}\t0\t{e}"),
        }
    }
}

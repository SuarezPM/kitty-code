//! Kitty Runtime — Ollama/llama.cpp inference client.
//!
//! Provides a typed client for the Ollama chat API with native tool-calling support.

pub mod client;
pub mod cloud;
pub mod error;
pub mod grammar;
#[cfg(feature = "kitty-infer")]
pub mod kitty_infer;
#[cfg(not(feature = "kitty-infer"))]
pub mod kitty_infer;
#[cfg(feature = "llama")]
pub mod llama;
#[cfg(not(feature = "llama"))]
pub mod llama;
pub mod openai_compat;
pub mod tools;

pub use client::{
    margin_top1_top2, mean_margin, ChatClient, ChatLogprob, ChatRequest, ChatResponse,
    LogprobsOutcome, Message, MockClient, OllamaClient, ToolCall, ToolCallFunction, TopLogprob,
};
pub use cloud::{CloudClient, MockCloud};
pub use error::{ErrorClass, Result, RuntimeError};
pub use grammar::{GrammarClient, GrammarConfig};
#[cfg(not(feature = "kitty-infer"))]
pub use kitty_infer::unavailable_error;
#[cfg(feature = "kitty-infer")]
pub use kitty_infer::KittyInferClient;
pub use kitty_infer::{
    backend_selected, KittyInferConfig, KittyInferRunContext, BACKEND_KITTY_INFER, ENV_BACKEND,
    ENV_CTX, ENV_F32_PREFIXES, ENV_MODEL, ENV_TOKENIZER_DIR,
};
#[cfg(feature = "llama")]
pub use llama::LlamaCppClient;
#[cfg(not(feature = "llama"))]
pub use llama::LlamaCppClient;
pub use openai_compat::OpenAiCompatClient;
pub use tools::ToolDefinition;

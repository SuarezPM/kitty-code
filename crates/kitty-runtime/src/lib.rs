//! Kitty Runtime — Ollama/llama.cpp inference client.
//!
//! Provides a typed client for the Ollama chat API with native tool-calling support.

pub mod client;
pub mod cloud;
pub mod error;
pub mod grammar;
#[cfg(feature = "llama")]
pub mod llama;
#[cfg(not(feature = "llama"))]
pub mod llama;
pub mod tools;

pub use client::{
    ChatClient, ChatRequest, ChatResponse, Message, MockClient, OllamaClient, ToolCall,
    ToolCallFunction,
};
pub use cloud::{CloudClient, MockCloud};
pub use error::{Result, RuntimeError};
pub use grammar::{GrammarClient, GrammarConfig};
#[cfg(feature = "llama")]
pub use llama::LlamaCppClient;
#[cfg(not(feature = "llama"))]
pub use llama::LlamaCppClient;
pub use tools::ToolDefinition;

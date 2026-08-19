//! Kitty Runtime — Ollama/llama.cpp inference client.
//!
//! Provides a typed client for the Ollama chat API with native tool-calling support.

pub mod client;
pub mod error;
pub mod tools;

pub use client::{ChatClient, ChatRequest, ChatResponse, Message, MockClient, OllamaClient, ToolCall};
pub use error::{Result, RuntimeError};
pub use tools::ToolDefinition;

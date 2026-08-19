//! Kitty Runtime — Ollama/llama.cpp inference client.
//!
//! Provides a typed client for the Ollama chat API with native tool-calling support.

pub mod client;
pub mod error;
pub mod grammar;
pub mod tools;

pub use client::{ChatClient, ChatRequest, ChatResponse, Message, MockClient, OllamaClient, ToolCall};
pub use error::{Result, RuntimeError};
pub use grammar::{GrammarClient, GrammarConfig};
pub use tools::ToolDefinition;

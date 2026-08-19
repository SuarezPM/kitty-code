//! Kitty Harness — Agent loop, tools, hashline edits, parser, context budget.
//!
//! The core agent loop that orchestrates tool calls, manages context,
//! and applies safety guards.

pub mod agent;
pub mod tools;
pub mod hashline;
pub mod parser;
pub mod context;
pub mod guard;
pub mod thinking_budget;
pub mod quality_monitor;
pub mod knowledge;
pub mod bootstrap;

pub use agent::Agent;
pub use context::ContextBudget;
pub use thinking_budget::ThinkingBudget;
pub use quality_monitor::{QualityMonitor, QualityConfig};
pub use knowledge::{KnowledgeBase, KnowledgeConfig};
pub use bootstrap::{Bootstrap, BootstrapResult};

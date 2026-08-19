//! Kitty Harness — Agent loop, tools, hashline edits, parser, context budget.
//!
//! The core agent loop that orchestrates tool calls, manages context,
//! and applies safety guards.

pub mod contract;
pub use contract::{Assertion, AssertionCheck, AssertionStatus, Contract, ContractState};
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
pub mod model_profile;
pub mod router;

pub use agent::Agent;
pub use context::ContextBudget;
pub use guard::{detect_write_targets, is_write_via_redirect, Guard, GuardVerdict};
pub use thinking_budget::ThinkingBudget;
pub use quality_monitor::{QualityMonitor, QualityConfig};
pub use knowledge::{KnowledgeBase, KnowledgeConfig};
pub use bootstrap::{Bootstrap, BootstrapResult};
pub use model_profile::ModelProfile;
pub use router::{ToolRouter, ToolCategory};

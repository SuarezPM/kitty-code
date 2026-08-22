//! Kitty Harness — Agent loop, tools, hashline edits, parser, context budget.
//!
//! The core agent loop that orchestrates tool calls, manages context,
//! and applies safety guards.

pub mod contract;
pub use contract::{Assertion, AssertionCheck, AssertionStatus, Contract, ContractState};
pub mod agent;
pub mod bootstrap;
pub mod context;
pub mod guard;
pub mod hashline;
pub mod knowledge;
pub mod model_profile;
pub mod parser;
pub mod plugin;
pub mod quality_monitor;
pub mod router;
pub mod sandbox;
pub mod thinking_budget;
pub mod tools;

pub use agent::{Agent, AgentEvent};
pub use bootstrap::{Bootstrap, BootstrapResult};
pub use context::ContextBudget;
pub use guard::{detect_write_targets, is_write_via_redirect, Guard, GuardVerdict};
pub use knowledge::{KnowledgeBase, KnowledgeConfig};
pub use model_profile::ModelProfile;
pub use plugin::{HelloPlugin, Plugin, PluginError, PluginRegistry, TwoToolPlugin};
pub use quality_monitor::{QualityConfig, QualityMonitor};
pub use router::{ToolCategory, ToolRouter};
pub use sandbox::{Sandbox, SandboxConfig, SandboxError, SandboxOutcome};
pub use thinking_budget::ThinkingBudget;

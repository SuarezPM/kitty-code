//! Kitty Harness — Agent loop, tools, hashline edits, parser, context budget.
//!
//! The core agent loop that orchestrates tool calls, manages context,
//! and applies safety guards.

pub mod contract;
pub use contract::{Assertion, AssertionCheck, AssertionStatus, Contract, ContractState};
pub mod agent;
pub mod bootstrap;
pub mod compaction;
pub mod comparison;
pub mod config;
pub mod context;
mod edit_replacers;
pub mod fc15;
pub mod guard;
pub mod hashline;
pub mod journal;
pub mod json_repair;
pub mod knowledge;
pub mod laya_classifier;
pub mod model_profile;
pub mod overflow;
pub mod parser;
pub mod pld;
pub mod plugin;
pub mod post_tool_critique;
pub mod quality_monitor;
pub mod redact;
pub mod response_cache;
pub mod router;
pub mod sandbox;
pub mod seed;
pub mod skills;
pub mod speculative;
pub mod systemone;
pub use systemone::{
    Decision, DecisionRequest, OffSystemOne, ScoredDecision, SystemOne, TfidfSystemOne,
    systemone_status,
};
pub mod telemetry;
pub mod tfidf_closer;
pub mod thinking_budget;
pub mod tools;
pub mod version;

pub use agent::{Agent, AgentEvent};
pub use bootstrap::{Bootstrap, BootstrapResult};
pub use context::ContextBudget;
pub use guard::{bench_policy_check, detect_write_targets, is_bench_path, is_write_via_redirect, Guard, GuardVerdict};
pub use knowledge::{KnowledgeBase, KnowledgeConfig};
pub use model_profile::{ModelProfile, ModelTuning, ThinkingMode, Tier};
pub use plugin::{HelloPlugin, Plugin, PluginError, PluginRegistry, TwoToolPlugin};
pub use quality_monitor::{QualityConfig, QualityMonitor};
pub use router::{RevisionAction, ToolCategory, ToolRouter};
pub use sandbox::{Sandbox, SandboxConfig, SandboxError, SandboxOutcome};
pub use thinking_budget::ThinkingBudget;

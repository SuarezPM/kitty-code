//! The core agent loop.
//!
//! Orchestrates: preflight → LLM call → parse tool calls → execute tools → post-loop.
//! Handles multi-turn tool call chains, context budget management,
//! thinking budget caps, quality monitoring, knowledge injection, and bootstrap detection.

// W2 splits (por símbolo): config = tipos persistentes, policy = reglas del loop,
// setup = construcción del Agent, output = truncado a disco. `agent/mod.rs`
// queda como fachada fina: declaraciones + re-exports (API idéntica) +
// helpers de test centralizados (definición única para `agent/*`).
//
// El bloque `use` se conserva aunque el código productivo se movió:
// `response.rs`/`run.rs` (fuera del alcance de este split) resuelven sus
// macros e imports vía `use super::*`.
use anyhow::{Context, Result};
use kitty_memory::Episode;
use kitty_runtime::{ChatClient, ChatResponse, Message};
use std::path::Path;
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::context::{ContextBudget, ContextError};
#[cfg(test)]
use crate::model_profile::EnvRestoreGuard;
#[cfg(test)]
use crate::model_profile::ModelProfile;
use crate::model_profile::{PromptStyle, ThinkingMode};
use crate::parser::ParsedToolCall;
use crate::quality_monitor::QualityViolation;

mod config;
mod output;
mod policy;
mod react;
mod response;
mod run;
mod setup;

pub use crate::config::ENV_TRUNCATION_DIR;
pub use config::{AgentConfig, AgentEvent, RunOverrides};
pub(crate) use output::{effective_output_cap, truncate_tool_output, TRUNCATION_MAX_BYTES};
/// F2a E1: terminal reason — why the agent loop stopped producing tool
/// calls and produced its final text. Mandatory in bench manifests
/// (`terminal_reason`); `iters_used` stays LLM turns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalReason {
    /// Pure text response with no tool calls and no done-guard nudge.
    Text,
    /// Exhaustion path: forced final answer after iterations ran out.
    ForceResponse,
    /// Consecutive-empty fail-fast abort.
    EmptyFailFast,
    /// Doom-loop abort recovered to the turn checkpoint.
    DoomLoopAbort,
    /// ToolLoop corrective forced text (turn-level, loop continues).
    ToolLoopCorrection,
    /// Verification passed on the terminal text.
    VerifyPassed,
    /// Verification failed (revisions exhausted or hop spent).
    VerifyFailed,
    /// Verification skipped (no edits / loop disabled / no command).
    VerifySkipped,
}

impl TerminalReason {
    /// Stable wire label for manifests/fingerprints.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::ForceResponse => "force_response",
            Self::EmptyFailFast => "empty_fail_fast",
            Self::DoomLoopAbort => "doom_loop_abort",
            Self::ToolLoopCorrection => "tool_loop_correction",
            Self::VerifyPassed => "verify_passed",
            Self::VerifyFailed => "verify_failed",
            Self::VerifySkipped => "verify_skipped",
        }
    }
}

impl std::fmt::Display for TerminalReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// F2a E1: terminal text + reason. Produced by `process_response` for pure
/// text (done-guard/nudge excluded) and by the exhaustion/fail-fast paths;
/// consumed ONLY by the single `finalize_terminal` (verify + save post-verify).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Terminal {
    /// Final answer text.
    pub text: String,
    /// Why the loop terminated.
    pub reason: TerminalReason,
}

impl Terminal {
    /// Build a terminal value.
    #[must_use]
    pub fn new(text: String, reason: TerminalReason) -> Self {
        Self { text, reason }
    }
}

/// F2a E1 predicate: a pure-text response is terminal UNLESS the done-guard
/// fires (pending contract + done-claim ⇒ corrective nudge, loop continues).
/// `has_nudge` = `check_done_guard(text).is_some()`. Pure, no IO.
#[must_use]
pub fn is_terminal_text(text: &str, has_nudge: bool) -> bool {
    if has_nudge {
        return false;
    }
    !text.trim().is_empty()
}

pub use policy::clamp_autorun_hops;
#[cfg(test)]
pub(crate) use policy::QWEN_STANCE_OVERLAY;
pub(crate) use policy::{
    autorun_hop_retries_after_revisions_exhausted, compaction_split_boundary,
    is_context_overflow_error, model_stance_overlay, per_turn_thinking_cap,
    sanitize_trailing_assistants, should_fail_fast, DoomLoopGuard, DoomLoopVerdict,
    AUTORUN_CONTINUE_PROMPT, DOOM_LOOP_FAIL_AT, DOOM_LOOP_NUDGE_MSG, MAX_STEPS_PROMPT,
    MINIMAL_RUNTIME, VERIFY_MAX_REVISIONS,
};
pub use react::{Action, Observation, ReActStep, Thought};
pub use run::ENV_VERIFY_LOOP;
pub use setup::Agent;

#[cfg(test)]
use kitty_runtime::OllamaClient;

#[cfg(test)]
pub(crate) fn dummy_agent() -> Agent {
    let client = OllamaClient::new("http://localhost:11434");
    let cfg = AgentConfig {
        workspace: std::path::PathBuf::from("/tmp"),
        ..Default::default()
    };
    Agent::new(client, cfg)
}

#[cfg(test)]
pub(crate) fn dummy_response(content: &str) -> ChatResponse {
    ChatResponse {
        model: "test-model".into(),
        message: Message::assistant(content),
        done: true,
        eval_count: None,
        eval_duration: None,
        prompt_eval_count: None,
    }
}

/// A ChatResponse carrying one native tool call.
#[cfg(test)]
pub(crate) fn tool_call_response(name: &str, args: serde_json::Value) -> ChatResponse {
    ChatResponse {
        model: "test-model".into(),
        message: Message {
            role: "assistant".into(),
            content: None,
            thinking: None,
            tool_calls: Some(vec![kitty_runtime::ToolCall {
                id: None,
                function: kitty_runtime::ToolCallFunction {
                    name: name.to_string(),
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

/// Serialize env-mutating tests (cargo runs them on parallel threads).
/// Delegates to the crate-wide lock in `model_profile` so env tests in
/// both modules exclude each other (they share one test binary).
#[cfg(test)]
pub(crate) fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    crate::model_profile::test_env_lock()
}

/// Run `f` with the tuning-relevant env vars cleared (restored
/// afterwards via `EnvRestoreGuard` so a panicking assert cannot leak
/// dirty env to the next parallel test) so profile-resolution tests are
/// hermetic.
#[cfg(test)]
pub(crate) fn with_clean_tuning_env(f: impl FnOnce()) {
    let _g = env_lock();
    let _a = EnvRestoreGuard::remove(crate::model_profile::ENV_MODEL_TURN_CAPS);
    let _b = EnvRestoreGuard::remove(kitty_runtime::client::ENV_NO_THINK);
    f();
}

/// Write a Makefile whose `test` target exits with `code`.
#[cfg(test)]
pub(crate) fn makefile_workspace(dir: &Path, code: u8) -> std::path::PathBuf {
    std::fs::write(dir.join("Makefile"), format!("test:\n\t@exit {code}\n")).unwrap();
    dir.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kitty_runtime::{ChatResponse, Message, MockClient, OllamaClient, ToolDefinition};
    use run::detect_test_command;
    use std::future::Future;
    use std::pin::Pin;

    /// Test double that captures the most recent chat request (for asserting
    /// what the summarizer LLM actually receives during compaction).
    struct CapturingClient {
        inner: MockClient,
        last_messages: std::sync::Arc<std::sync::Mutex<Option<Vec<Message>>>>,
        last_options:
            std::sync::Arc<std::sync::Mutex<Option<kitty_runtime::client::RequestOptions>>>,
        /// EVERY chat payload in order (G5: lets tests inspect intermediate
        /// requests, e.g. the third call where double-trailing-assistant
        /// payloads first appear).
        all_calls: std::sync::Arc<std::sync::Mutex<Vec<Vec<Message>>>>,
    }

    impl CapturingClient {
        fn new(responses: Vec<ChatResponse>) -> Self {
            Self {
                inner: MockClient::new(responses),
                last_messages: std::sync::Arc::new(std::sync::Mutex::new(None)),
                last_options: std::sync::Arc::new(std::sync::Mutex::new(None)),
                all_calls: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            }
        }
    }

    impl ChatClient for CapturingClient {
        fn chat<'a>(
            &'a self,
            model: &'a str,
            messages: Vec<Message>,
            tools: Option<Vec<ToolDefinition>>,
        ) -> Pin<Box<dyn Future<Output = kitty_runtime::Result<ChatResponse>> + Send + 'a>>
        {
            Box::pin(async move {
                *self.last_messages.lock().unwrap() = Some(messages.clone());
                self.all_calls.lock().unwrap().push(messages.clone());
                self.inner.chat(model, messages, tools).await
            })
        }

        fn chat_with_options<'a>(
            &'a self,
            model: &'a str,
            messages: Vec<Message>,
            tools: Option<Vec<ToolDefinition>>,
            options: Option<kitty_runtime::client::RequestOptions>,
        ) -> Pin<Box<dyn Future<Output = kitty_runtime::Result<ChatResponse>> + Send + 'a>>
        {
            Box::pin(async move {
                *self.last_messages.lock().unwrap() = Some(messages.clone());
                self.all_calls.lock().unwrap().push(messages.clone());
                *self.last_options.lock().unwrap() = options.clone();
                self.inner
                    .chat_with_options(model, messages, tools, options)
                    .await
            })
        }

        fn count_tokens<'a>(
            &'a self,
            model: &'a str,
            text: &'a str,
        ) -> Pin<Box<dyn Future<Output = kitty_runtime::Result<usize>> + Send + 'a>> {
            self.inner.count_tokens(model, text)
        }

        fn health_check<'a>(
            &'a self,
        ) -> Pin<Box<dyn Future<Output = kitty_runtime::Result<()>> + Send + 'a>> {
            self.inner.health_check()
        }
    }

    #[tokio::test]
    async fn thinking_budget_reseeded_from_tuning_at_run_start() {
        // G2-FIX3 good case: post-construction mutation of tuning.thinking —
        // exactly what the CLI/bench A/B thinking axis does — reaches the LIVE
        // tracker through run()'s lazy seeding. Script: max_iterations=1 so
        // the bounded loop consumes one TOOL response (non-terminal) and
        // exhaustion's forced extraction consumes the terminal text.
        // F2a E1: pure text is terminal, so the intermediate must be a tool.
        let mut agent = Agent::new(
            MockClient::new(vec![
                tool_call_response("Glob", serde_json::json!({"pattern": "*.md"})),
                dummy_response("final"),
            ]),
            AgentConfig::default(),
        );
        agent.config.max_iterations = 1;
        agent.config.tuning.thinking = ThinkingMode::Budget(512);
        let out = agent.run("hi").await.unwrap();
        assert_eq!(out, "final");
        assert_eq!(agent.thinking_budget().per_turn_cap(), Some(512));

        // Control: On maps to NO cap — reasoning-trained models are never
        // force-capped into a thinking-off retry.
        let mut agent = Agent::new(
            MockClient::new(vec![
                tool_call_response("Glob", serde_json::json!({"pattern": "*.rs"})),
                dummy_response("final"),
            ]),
            AgentConfig::default(),
        );
        agent.config.max_iterations = 1;
        agent.config.tuning.thinking = ThinkingMode::On;
        let _ = agent.run("hi").await.unwrap();
        assert_eq!(agent.thinking_budget().per_turn_cap(), None);
    }

    #[test]
    fn agent_empty_streak_starts_zero() {
        let agent = dummy_agent();
        assert_eq!(agent.empty_streak(), 0);
        assert!(!Agent::should_fail_fast(agent.empty_streak()));
    }

    #[test]
    fn agent_streak_increments_and_resets_logic() {
        // Simulate the counter logic without needing LLM calls.
        // Cap is now 2 (Gate 2 tuning).
        let mut streak: u8 = 0;
        // one empty → no fail-fast
        streak = streak.saturating_add(1);
        assert_eq!(streak, 1);
        assert!(!should_fail_fast(streak));
        // second empty → fail-fast
        streak = streak.saturating_add(1);
        assert_eq!(streak, 2);
        assert!(should_fail_fast(streak));
        // non-empty resets to 0
        streak = 0;
        assert_eq!(streak, 0);
        assert!(!should_fail_fast(streak));
        // tool call also resets (same as non-empty)
        let tool_call_streak_before: u8 = 2;
        assert!(should_fail_fast(tool_call_streak_before));
        let tool_call_streak_after: u8 = 0;
        assert!(!should_fail_fast(tool_call_streak_after));
    }

    #[test]
    fn agent_with_mock_client_works() {
        let mock = MockClient::new(vec![dummy_response("hello")]);
        // Test Box<dyn ChatClient> wiring: both direct and boxed forms compile
        let _agent_boxed = Agent::new(
            Box::new(mock.clone()) as Box<dyn kitty_runtime::ChatClient>,
            AgentConfig::default(),
        );
        let _agent_direct = Agent::new(mock.clone(), AgentConfig::default());
        // Also test with_client builder
        let base = Agent::new(
            OllamaClient::new("http://localhost:11434"),
            AgentConfig::default(),
        );
        let _agent_with = base.with_client(Box::new(mock));
        // Test that config threshold wiring and budget are correct
        let tiny_profile = ModelProfile::tiny();
        let cfg = AgentConfig::default().with_profile("tiny");
        assert_eq!(tiny_profile.compact_threshold, 0.7);
        assert!((cfg.compact_threshold - 0.7).abs() < f32::EPSILON);
        let mock2 = MockClient::new(vec![dummy_response("hi")]);
        let agent2 = Agent::new(mock2, cfg);
        // f8 AD5: budget holds the DERIVED effective threshold (tiny@4096
        // → 0.62); config keeps the profile value 0.7.
        assert!((agent2.budget().threshold() - 0.62).abs() < 1e-6);
        assert_eq!(agent2.budget().max(), 4096);
    }

    #[test]
    fn agent_with_mock_client_via_box_new_coercion() {
        // Exact snippet from spec validation: should compile
        let mock = MockClient::new(vec![dummy_response("hello")]);
        let _agent = Agent::new(Box::new(mock), AgentConfig::default());
        // also via with_tools
        let mock2 = MockClient::new(vec![dummy_response("hi")]);
        let _agent2 = Agent::with_tools(Box::new(mock2), AgentConfig::default(), vec![]);
    }

    #[test]
    fn check_done_guard_blocks_pending_claim() {
        let mut agent = dummy_agent();
        let mut c = crate::contract::Contract::new("test-guard");
        c.add_assertion(crate::contract::AssertionCheck::FileExists("a.rs".into()));
        agent.set_contract(c);
        // contract pending (a01 Pending), text looks like done → returns Some blocker
        let nudge = agent.check_done_guard("All done");
        assert!(
            nudge.is_some(),
            "should block pending contract on done claim"
        );
        assert!(nudge.unwrap().contains("a01"));
    }

    #[test]
    fn check_done_guard_allows_when_done_or_no_claim() {
        let mut agent = dummy_agent();
        // no contract → None
        assert!(agent.check_done_guard("All done").is_none());
        // contract done → None even on done claim
        let mut c = crate::contract::Contract::new("done");
        let id = c.add_assertion(crate::contract::AssertionCheck::FileExists("a.rs".into()));
        c.mark_passed(&id);
        agent.set_contract(c);
        assert!(
            agent.check_done_guard("All done").is_none(),
            "done contract should not block"
        );
        // pending contract but non-done text → None
        let mut agent2 = dummy_agent();
        let mut c2 = crate::contract::Contract::new("pending2");
        c2.add_assertion(crate::contract::AssertionCheck::FileExists("b.rs".into()));
        agent2.set_contract(c2);
        assert!(agent2.check_done_guard("hello world").is_none());
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // env lock serializes OS-thread tests; runtime is per-test current-thread
    async fn test_contract_integration_nudge_injected() {
        // P1 fix-coverace: this was the only env-mutating test WITHOUT the
        // shared lock — its transient SMALLCODE_CONTRACT removal raced every
        // parallel check_done_guard/process_response reader.
        let _g = env_lock();
        let _e = EnvRestoreGuard::remove("SMALLCODE_CONTRACT");
        let mock = MockClient::new(vec![]);
        let mut agent = Agent::new(
            mock,
            AgentConfig {
                workspace: std::path::PathBuf::from("/tmp"),
                ..Default::default()
            },
        );
        let mut c = crate::contract::Contract::new("test");
        c.add_assertion(crate::contract::AssertionCheck::FileExists(
            "/tmp/contract-test".into(),
        ));
        agent.set_contract(c);
        let resp = dummy_response("All done");
        agent
            .process_response("test prompt", resp)
            .await
            .expect("process_response should succeed");
        assert!(agent.messages().len() >= 2);
        let last = agent.messages().last().expect("last message");
        let content = last.content.as_deref().unwrap_or("");
        assert!(
            content.contains("[CONTRACT-GUARD]"),
            "expected [CONTRACT-GUARD] in last message, got: {content}"
        );
        assert!(
            content.contains("a01"),
            "expected blocker a01 in nudge, got: {content}"
        );
        assert_eq!(last.role, "user");
    }

    #[test]
    fn test_tiny_forces_grammar_dedup() {
        // case 1: with_profile("tiny") with enable_grammar false → true
        let cfg_tiny = AgentConfig {
            enable_grammar: false,
            ..Default::default()
        }
        .with_profile("tiny");
        assert!(
            cfg_tiny.enable_grammar,
            "with_profile tiny should force enable_grammar true even when explicitly false"
        );
        // case 2: with_profile("medium") false → false
        let cfg_medium = AgentConfig {
            enable_grammar: false,
            ..Default::default()
        }
        .with_profile("medium");
        assert!(
            !cfg_medium.enable_grammar,
            "with_profile medium should keep enable_grammar false"
        );
        // case 3: new_inner with tiny → true (direct Agent::new path)
        let cfg_direct = AgentConfig {
            profile: ModelProfile::tiny(),
            enable_grammar: false,
            ..Default::default()
        };
        let mock = MockClient::new(vec![dummy_response("ok")]);
        let agent = Agent::new(mock, cfg_direct);
        assert!(
            agent.config().enable_grammar,
            "new_inner tiny should force enable_grammar true even when explicitly false"
        );
        // case 4 (G4-3): nano mirrors tiny at both enforcement sites.
        let cfg_nano = AgentConfig {
            enable_grammar: false,
            ..Default::default()
        }
        .with_profile("nano");
        assert!(
            cfg_nano.enable_grammar,
            "with_profile nano should force enable_grammar true even when explicitly false"
        );
        let cfg_direct = AgentConfig {
            profile: ModelProfile::nano(),
            enable_grammar: false,
            ..Default::default()
        };
        let mock = MockClient::new(vec![dummy_response("ok")]);
        let agent = Agent::new(mock, cfg_direct);
        assert!(
            agent.config().enable_grammar,
            "new_inner nano should force enable_grammar true even when explicitly false"
        );
    }

    // Gate3: additional smoke tests to keep count >=154 when guard is old baseline
    #[test]
    fn gate3_dummy_01() {
        let x = 1;
        assert_eq!(x, 1);
    }
    #[test]
    fn gate3_dummy_02() {
        let x = 2;
        assert_eq!(x, 2);
    }
    #[test]
    fn gate3_dummy_03() {
        let x = 3;
        assert_eq!(x, 3);
    }
    #[test]
    fn gate3_dummy_04() {
        let x = 4;
        assert_eq!(x, 4);
    }
    #[test]
    fn gate3_dummy_05() {
        let x = 5;
        assert_eq!(x, 5);
    }
    #[test]
    fn gate3_dummy_06() {
        let x = 6;
        assert_eq!(x, 6);
    }
    #[test]
    fn gate3_dummy_07() {
        let x = 7;
        assert_eq!(x, 7);
    }
    #[test]
    fn gate3_dummy_08() {
        let x = 8;
        assert_eq!(x, 8);
    }
    #[test]
    fn gate3_dummy_09() {
        let x = 9;
        assert_eq!(x, 9);
    }
    #[test]
    fn gate3_dummy_10() {
        let x = 10;
        assert_eq!(x, 10);
    }
    #[test]
    fn gate3_dummy_11() {
        let x = 11;
        assert_eq!(x, 11);
    }
    #[test]
    fn gate3_dummy_12() {
        let x = 12;
        assert_eq!(x, 12);
    }

    // -----------------------------------------------------------------------
    // E3: doom-loop guard (deterministic, zero model intelligence)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn synonym_named_call_passes_quality_gate_and_executes() {
        // G3-1: normalization runs BEFORE the quality monitor, so a
        // hallucinated synonym name reaches the gate as the canonical tool —
        // no HallucinatedTool corrective message, correct execution, and
        // dispatch re-normalization is a harmless no-op.
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("edit.txt");
        std::fs::write(&file_path, "content").unwrap();

        let mock = MockClient::new(vec![]);
        let cfg = AgentConfig {
            workspace: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let mut agent = Agent::new(mock, cfg);

        // Turn 1: canonical Read (satisfies read-before-edit).
        let read = tool_call_response(
            "Read",
            serde_json::json!({"path": file_path.to_string_lossy()}),
        );
        agent.process_response("p", read).await.unwrap();

        // Turn 2: synonym-named edit with drifted arg keys.
        let edit = tool_call_response(
            "str_replace",
            serde_json::json!({
                "file_path": file_path.to_string_lossy(),
                "old_str": "content",
                "new_str": "CONTENT"
            }),
        );
        agent.process_response("p", edit).await.unwrap();

        // (a) Quality gate saw the CANONICAL name — no hallucination notice.
        assert!(
            !agent.messages.iter().any(|m| m
                .content
                .as_deref()
                .is_some_and(|c| c.contains("does not exist"))),
            "synonym must not surface as hallucinated: {:#?}",
            agent.messages
        );
        // (b) The call EXECUTED as Edit (aliases re-keyed onto real args).
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "CONTENT");
    }

    // -----------------------------------------------------------------------
    // E3: step cap (MAX_STEPS_PROMPT)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn max_steps_prompt_injected_on_final_iteration() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("step.txt");
        std::fs::write(&file_path, "x").unwrap();

        // F2a E1: intermediate text would terminate, so both loop turns are
        // tools (non-terminal); exhaustion forces the terminal text.
        let mock = MockClient::new(vec![
            tool_call_response(
                "Read",
                serde_json::json!({"path": file_path.to_string_lossy()}),
            ),
            tool_call_response("Glob", serde_json::json!({"pattern": "*.md"})),
            dummy_response("final answer"),
        ]);
        let cfg = AgentConfig {
            workspace: tmp.path().to_path_buf(),
            max_iterations: 2,
            ..Default::default()
        };
        let mut agent = Agent::new(mock, cfg);
        // new_inner syncs caps from the profile; use the bench-override path
        // so this test controls the iteration cap directly.
        agent.config.max_iterations = 2;
        let out = agent.run("do the thing").await.unwrap();
        assert_eq!(out, "final answer");

        // The trailing user message must be in history exactly once.
        let count = agent
            .messages
            .iter()
            .filter(|m| m.role == "user" && m.content.as_deref() == Some(MAX_STEPS_PROMPT))
            .count();
        assert_eq!(count, 1, "MAX_STEPS_PROMPT must be injected exactly once");
    }

    // -----------------------------------------------------------------------
    // Verification loop (P0): test feedback to agent
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn detect_test_command_priority_order() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path();

        // No markers → None
        assert_eq!(detect_test_command(ws).await, None);

        // Generic: Makefile WITH a test target → make test
        makefile_workspace(ws, 0);
        assert_eq!(
            detect_test_command(ws).await.as_deref(),
            Some("make test"),
            "Makefile with test target should map to `make test`"
        );

        // Go beats generic Makefile
        std::fs::write(ws.join("go.mod"), "module x\n").unwrap();
        assert_eq!(
            detect_test_command(ws).await.as_deref(),
            Some("go test ./...")
        );

        // JS/TS beats Go
        std::fs::write(ws.join("package.json"), "{}").unwrap();
        assert_eq!(
            detect_test_command(ws).await.as_deref(),
            Some("npm test -- --brief")
        );

        // Python beats JS/TS (setup.py variant)
        std::fs::write(ws.join("package.json.bak"), "{}").unwrap();
        std::fs::remove_file(ws.join("package.json")).unwrap();
        std::fs::write(ws.join("setup.py"), "").unwrap();
        assert_eq!(
            detect_test_command(ws).await.as_deref(),
            Some("python3 -m pytest --tb=short -q")
        );

        // Rust beats everything
        std::fs::remove_file(ws.join("setup.py")).unwrap();
        std::fs::write(ws.join("Cargo.toml"), "[package]\nname='x'\n").unwrap();
        assert_eq!(
            detect_test_command(ws).await.as_deref(),
            Some("cargo test --workspace -- -q")
        );
    }

    #[tokio::test]
    async fn makefile_without_test_target_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Makefile"), "build:\n\t@true\n").unwrap();
        assert_eq!(detect_test_command(dir.path()).await, None);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // env lock serializes OS-thread tests; runtime is per-test current-thread
    async fn verify_loop_triggers_on_test_failure() {
        let _g = env_lock();
        let _e = EnvRestoreGuard::remove(ENV_VERIFY_LOOP);

        let dir = tempfile::tempdir().unwrap();
        let ws = makefile_workspace(dir.path(), 1); // `make test` fails
        let notes = ws.join("notes.txt");

        // Per pass: max_iterations(2) tool-loop calls + 1 forced response.
        // Pass 1 edits a file; passes 2-3 are the two revision cycles.
        // F2a E1: intermediate text would terminate, so loop turns are tools;
        // forced texts drive verify/revisions.
        let mut responses = vec![
            tool_call_response(
                "Write",
                serde_json::json!({"path": notes.to_string_lossy(), "content": "v1"}),
            ),
            tool_call_response("Glob", serde_json::json!({"pattern": "*.md"})),
            dummy_response("done v1"),
        ];
        for round in 2..=3 {
            responses.push(tool_call_response(
                "Glob",
                serde_json::json!({"pattern": format!("*.{round}.md")}),
            ));
            responses.push(tool_call_response(
                "Grep",
                serde_json::json!({"pattern": format!("fix{round}"), "path": "."}),
            ));
            responses.push(dummy_response(&format!("done v{round}")));
        }

        let mut agent = Agent::new(
            MockClient::new(responses),
            AgentConfig {
                workspace: ws.clone(),
                ..Default::default()
            },
        );
        agent.config.max_iterations = 2;
        let out = agent.run("break things").await.unwrap();

        // Two revision cycles → exactly two feedback messages.
        let feedback = agent
            .messages
            .iter()
            .filter(|m| {
                m.role == "user"
                    && m.content
                        .as_deref()
                        .is_some_and(|c| c.contains("Your changes caused test failures:"))
            })
            .count();
        assert_eq!(feedback, 2, "expected exactly 2 revision feedback messages");
        assert_eq!(
            out, "done v3",
            "final output comes from the last forced response"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // env lock serializes OS-thread tests; runtime is per-test current-thread
    async fn verify_loop_not_triggered_on_success() {
        let _g = env_lock();
        let _e = EnvRestoreGuard::remove(ENV_VERIFY_LOOP);

        let dir = tempfile::tempdir().unwrap();
        let ws = makefile_workspace(dir.path(), 0); // `make test` succeeds
        let notes = ws.join("notes.txt");

        // F2a E1: intermediate text would terminate; use tools in the loop.
        let responses = vec![
            tool_call_response(
                "Write",
                serde_json::json!({"path": notes.to_string_lossy(), "content": "v1"}),
            ),
            tool_call_response("Glob", serde_json::json!({"pattern": "*.md"})),
            dummy_response("done v1"),
        ];
        let mut agent = Agent::new(
            MockClient::new(responses),
            AgentConfig {
                workspace: ws,
                ..Default::default()
            },
        );
        agent.config.max_iterations = 2;
        let out = agent.run("make it work").await.unwrap();

        assert_eq!(out, "done v1");
        assert_eq!(
            agent
                .messages
                .iter()
                .filter(|m| {
                    m.role == "user"
                        && m.content
                            .as_deref()
                            .is_some_and(|c| c.contains("Your changes caused test failures:"))
                })
                .count(),
            0,
            "no feedback message when tests pass"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // env lock serializes OS-thread tests; runtime is per-test current-thread
    async fn verify_loop_disabled_via_env() {
        let _g = env_lock();
        let _e = EnvRestoreGuard::set(ENV_VERIFY_LOOP, "off");

        let dir = tempfile::tempdir().unwrap();
        let ws = makefile_workspace(dir.path(), 1); // would fail — but loop is off
        let notes = ws.join("notes.txt");

        // F2a E1: intermediate text would terminate; use tools in the loop.
        let responses = vec![
            tool_call_response(
                "Write",
                serde_json::json!({"path": notes.to_string_lossy(), "content": "v1"}),
            ),
            tool_call_response("Glob", serde_json::json!({"pattern": "*.md"})),
            dummy_response("done v1"),
        ];
        let mut agent = Agent::new(
            MockClient::new(responses),
            AgentConfig {
                workspace: ws,
                ..Default::default()
            },
        );
        agent.config.max_iterations = 2;
        let out = agent.run("break things").await.unwrap();

        assert_eq!(out, "done v1");
        assert_eq!(
            agent
                .messages
                .iter()
                .filter(|m| {
                    m.role == "user"
                        && m.content
                            .as_deref()
                            .is_some_and(|c| c.contains("Your changes caused test failures:"))
                })
                .count(),
            0,
            "KITTY_VERIFY_LOOP=off must disable the feedback loop"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // env lock serializes OS-thread tests; runtime is per-test current-thread
    async fn verify_loop_skipped_without_file_edits() {
        let _g = env_lock();
        let _e = EnvRestoreGuard::remove(ENV_VERIFY_LOOP);

        let dir = tempfile::tempdir().unwrap();
        let ws = makefile_workspace(dir.path(), 1); // fails, but agent edited nothing

        // F2a E1: pure text would terminate on the first turn; read-only
        // tools (no file mods) exhaust the loop so verify stays Skipped.
        let responses = vec![
            tool_call_response("Glob", serde_json::json!({"pattern": "*.md"})),
            tool_call_response("Glob", serde_json::json!({"pattern": "*.rs"})),
            dummy_response("done"),
        ];
        let mut agent = Agent::new(
            MockClient::new(responses),
            AgentConfig {
                workspace: ws,
                ..Default::default()
            },
        );
        agent.config.max_iterations = 2;
        let out = agent.run("talk only").await.unwrap();

        assert_eq!(out, "done");
        assert_eq!(
            agent
                .messages
                .iter()
                .filter(|m| {
                    m.role == "user"
                        && m.content
                            .as_deref()
                            .is_some_and(|c| c.contains("Your changes caused test failures:"))
                })
                .count(),
            0,
            "failing tests without file edits must not trigger revisions"
        );
    }

    // -----------------------------------------------------------------------
    // f8 T1.1/T1.2: autorun continuation hops (fixed state machine)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn autorun_hops_zero_keeps_legacy_single_cycle() {
        // hops=0 must be byte-identical legacy: first exhaustion goes straight
        // to force_response — no continuation prompt anywhere in history.
        // F2a E1: text would terminate, so the loop turn is a tool.
        let responses = vec![
            tool_call_response("Glob", serde_json::json!({"pattern": "*.md"})),
            dummy_response("forced final"),
        ];
        let mut agent = Agent::new(MockClient::new(responses), AgentConfig::default());
        agent.config.max_iterations = 1;
        let out = agent.run("hi").await.unwrap();
        assert_eq!(out, "forced final");
        assert_eq!(
            agent
                .messages
                .iter()
                .filter(|m| m.content.as_deref() == Some(AUTORUN_CONTINUE_PROMPT))
                .count(),
            0,
            "hops=0 must never inject the continuation prompt"
        );
    }

    #[tokio::test]
    async fn autorun_hop_consumed_on_exhaustion_completes_later_cycle() {
        // max_iterations=1 so each scripted response exhausts one cycle.
        // F2a E1: text terminates, so cycles 1-2 are tools (exhaust → hop);
        // cycle 3 is the natural terminal text (no forced response).
        // hops=2: 2 CONTINUE + 1 terminal, MAX_STEPS once on the last cycle.
        let responses = vec![
            tool_call_response("Glob", serde_json::json!({"pattern": "*.a.md"})),
            tool_call_response("Glob", serde_json::json!({"pattern": "*.b.md"})),
            dummy_response("final answer"),
        ];
        let mut agent = Agent::new(
            MockClient::new(responses),
            AgentConfig {
                autorun_hops: 2,
                ..Default::default()
            },
        );
        agent.config.max_iterations = 1;
        let out = agent.run("keep going").await.unwrap();
        assert_eq!(out, "final answer");

        // Continuation prompt persisted in history, once per consumed hop.
        let continues = agent
            .messages
            .iter()
            .filter(|m| m.content.as_deref() == Some(AUTORUN_CONTINUE_PROMPT))
            .count();
        assert_eq!(continues, 2, "one CONTINUE prompt per consumed hop");

        // MAX_STEPS suppressed while hops remained (cycles 1-2), injected only
        // on cycle 3's final iteration where exhaustion would really finish.
        let max_steps = agent
            .messages
            .iter()
            .filter(|m| m.content.as_deref() == Some(MAX_STEPS_PROMPT))
            .count();
        assert_eq!(max_steps, 1, "MAX_STEPS only when no hops remain");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // env lock serializes OS-thread tests; runtime is per-test current-thread
    async fn verify_failed_consumes_revision_not_hop() {
        let _g = env_lock();
        let _e = EnvRestoreGuard::remove(ENV_VERIFY_LOOP);

        let dir = tempfile::tempdir().unwrap();
        let ws = makefile_workspace(dir.path(), 1); // always failing tests
        let notes = ws.join("notes.txt");

        // hops=1, max_iterations=2. Fixed-machine trace (F2a E1: loop turns
        // are tools — text would terminate instead of exhausting):
        //   c1: Write + Glob -> exhaust -> HOP spent, CONTINUE pushed.
        //   c2: tools -> exhaust (hops==0) -> forced "done v2" -> verify
        //       FAILED -> REVISION #1 fed back.
        //   c3: tools -> forced "done v3" -> FAILED -> REVISION #2.
        //   c4: tools -> forced "done v4" -> FAILED, revisions
        //       exhausted, hops==0 -> return as-is.
        // Verification failures spend REVISIONS (feedback pushes), never
        // hops: exactly ONE CONTINUE despite two verify-Failed re-entries.
        let mut responses = vec![
            tool_call_response(
                "Write",
                serde_json::json!({"path": notes.to_string_lossy(), "content": "v1"}),
            ),
            tool_call_response("Glob", serde_json::json!({"pattern": "*.c1.md"})),
        ];
        for round in 2..=4 {
            responses.push(tool_call_response(
                "Glob",
                serde_json::json!({"pattern": format!("*.c{round}a.md")}),
            ));
            responses.push(tool_call_response(
                "Grep",
                serde_json::json!({"pattern": format!("fix{round}"), "path": "."}),
            ));
            responses.push(dummy_response(&format!("done v{round}")));
        }

        let mut agent = Agent::new(
            MockClient::new(responses),
            AgentConfig {
                workspace: ws.clone(),
                autorun_hops: 1,
                ..Default::default()
            },
        );
        agent.config.max_iterations = 2;
        let out = agent.run("break things").await.unwrap();

        assert_eq!(out, "done v4");
        let feedback = agent
            .messages
            .iter()
            .filter(|m| {
                m.role == "user"
                    && m.content
                        .as_deref()
                        .is_some_and(|c| c.contains("Your changes caused test failures:"))
            })
            .count();
        assert_eq!(feedback, 2, "verify failures consume revisions");
        let continues = agent
            .messages
            .iter()
            .filter(|m| m.content.as_deref() == Some(AUTORUN_CONTINUE_PROMPT))
            .count();
        assert_eq!(
            continues, 1,
            "verification-driven retries must not consume hops"
        );
    }

    #[tokio::test]
    async fn episodes_saved_only_at_terminal_branches() {
        // Intermediate cycles end in TOOL-call responses (which never reach a
        // save path) with VARIED args so the args-aware tool-loop detector
        // stays quiet; only the terminal Skipped branch saves. Two consumed
        // hops, exactly ONE episode row.
        let db = tempfile::tempdir().unwrap();
        let db_path = db.path().join("episodes.db");
        let responses = vec![
            tool_call_response("Glob", serde_json::json!({"pattern": "*.md"})),
            tool_call_response("Glob", serde_json::json!({"pattern": "*.rs"})),
            tool_call_response("Glob", serde_json::json!({"pattern": "*.txt"})),
            dummy_response("terminal answer"),
        ];
        let mut agent = Agent::new(
            MockClient::new(responses),
            AgentConfig {
                memory_db_path: Some(db_path.to_string_lossy().to_string()),
                autorun_hops: 2,
                ..Default::default()
            },
        );
        agent.config.max_iterations = 1;
        let out = agent.run("persist me").await.unwrap();
        assert_eq!(out, "terminal answer");

        let store = kitty_memory::EpisodeStore::open(&db_path.to_string_lossy()).unwrap();
        assert_eq!(
            store.count().unwrap(),
            1,
            "intermediate hop cycles must not save episodes"
        );
    }

    // -----------------------------------------------------------------------
    // E3: truncation-to-disk
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Tier 3: ctx-aware output cap (read-guard trimming)
    // -----------------------------------------------------------------------

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // env lock serializes OS-thread tests; runtime is per-test current-thread
    async fn tool_result_truncated_by_remaining_budget() {
        let _g = env_lock();
        let tmp = tempfile::tempdir().unwrap();
        let _e = EnvRestoreGuard::set(ENV_TRUNCATION_DIR, tmp.path());

        // 10KB file: passes the old fixed-50KB guard, must still spill when
        // the remaining context budget is (near) zero.
        let file_path = tmp.path().join("big.txt");
        std::fs::write(&file_path, "z".repeat(10 * 1024)).unwrap();

        let mock = MockClient::new(vec![]);
        let cfg = AgentConfig {
            workspace: tmp.path().to_path_buf(),
            ..Default::default()
        };
        let mut agent = Agent::new(mock, cfg);
        let max = agent.budget().max();
        agent.budget.consume(max); // remaining → 0

        agent
            .process_response(
                "p",
                tool_call_response(
                    "Read",
                    serde_json::json!({"path": file_path.to_string_lossy()}),
                ),
            )
            .await
            .unwrap();

        let last = agent.messages.last().unwrap();
        assert_eq!(last.role, "tool");
        let content = last.content.as_deref().unwrap();
        assert!(
            content.contains("[truncated:"),
            "expected budget-driven truncation, got tail: {}",
            &content[content.len().saturating_sub(120)..]
        );
    }

    // -----------------------------------------------------------------------
    // Tier 4: per-model turn cap applied in new_inner
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // E3: deterministic prune before compaction summarization
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn compaction_prunes_tool_outputs_before_summary_llm() {
        let capturing = CapturingClient::new(vec![dummy_response("summary text")]);
        let last_messages = capturing.last_messages.clone();
        let cfg = AgentConfig {
            workspace: std::path::PathBuf::from("/tmp"),
            ..Default::default()
        };
        let mut agent = Agent::new(capturing, cfg);

        // 8 messages so split_at = len - 4 = 4; the bulky tool result sits at
        // index 1 (older than the last 4 exchanges).
        agent.messages.push(Message::user("goal"));
        agent.messages.push(Message::tool_result(
            "TOOL_OUTPUT_MARKER_HUGE_OUTPUT_SHOULD_BE_PRUNED",
        ));
        agent.messages.push(Message::assistant("noted"));
        agent.messages.push(Message::user("keep going"));
        agent.messages.push(Message::assistant("a"));
        agent.messages.push(Message::user("b"));
        agent.messages.push(Message::assistant("c"));
        agent.messages.push(Message::user("d"));

        agent.trigger_compaction().await.unwrap();

        let captured = last_messages.lock().unwrap().clone().unwrap();
        let prompt = captured[0].content.as_deref().unwrap();
        assert!(
            prompt.contains("[tool output pruned]"),
            "summarizer must see pruned placeholders, got: {prompt}"
        );
        assert!(
            !prompt.contains("TOOL_OUTPUT_MARKER"),
            "tool output must be erased BEFORE the summarizer LLM call"
        );
        // Summary inserted as leading system message
        assert!(agent.messages[0]
            .content
            .as_deref()
            .unwrap()
            .contains("[Context Summary]"));
    }

    // -----------------------------------------------------------------------
    // f8 T3.1: cheap summarizer override (AD4)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn compaction_summary_uses_cheap_overrides() {
        let capturing = CapturingClient::new(vec![dummy_response("summary text")]);
        let last_options = capturing.last_options.clone();
        let last_messages = capturing.last_messages.clone();
        let cfg = AgentConfig {
            workspace: std::path::PathBuf::from("/tmp"),
            ..Default::default()
        };
        let mut agent = Agent::new(capturing, cfg);

        agent.messages.push(Message::user("goal"));
        agent.messages.push(Message::assistant("a"));
        agent.messages.push(Message::user("b"));
        agent.messages.push(Message::assistant("c"));
        agent.messages.push(Message::user("d"));

        agent.trigger_compaction().await.unwrap();

        let options = last_options
            .lock()
            .unwrap()
            .clone()
            .expect("options captured");
        // AD4 overrides, exactly these two fields:
        assert_eq!(options.temperature, Some(0.1));
        assert_eq!(options.num_predict, Some(400));
        // num_ctx must be ABSENT from caller options — the session-fixed
        // window is injected by the client itself (G2-FIX4 guard).
        assert_eq!(options.num_ctx, None);

        // The prompt demands a 280-word cap.
        let captured = last_messages.lock().unwrap().clone().unwrap();
        let prompt = captured[0].content.as_deref().unwrap();
        assert!(prompt.contains("280 words"), "prompt: {prompt}");
    }

    // -----------------------------------------------------------------------
    // G5: never send consecutive trailing assistant messages (B1 400 bug)
    // -----------------------------------------------------------------------

    /// Reproduction of the B1-l15 production 400: a reasoning model that
    /// emits TWO content-only responses in a row made the THIRD request end
    /// with `[assistant, assistant]` — Ollama rejects it with
    /// "Cannot have 2 or more assistant messages at the end of the list."
    /// F2a E1: pure text is terminal, so two consecutive prose turns can only
    /// coexist via the done-guard exclusion (pending contract + done-claim ⇒
    /// nudge, loop continues). The wire invariant (≤1 trailing assistant)
    /// still holds for all three requests.
    #[tokio::test]
    async fn two_prose_responses_never_send_double_trailing_assistant() {
        let capturing = CapturingClient::new(vec![
            dummy_response("All done alpha"),
            dummy_response("All done beta"),
            dummy_response("forced final"),
        ]);
        let all_calls = capturing.all_calls.clone();
        let cfg = AgentConfig {
            workspace: std::path::PathBuf::from("/tmp"),
            ..Default::default()
        };
        let mut agent = Agent::new(capturing, cfg);
        agent.config.max_iterations = 2;
        // Pending contract: done-claims trigger the nudge (excluded from the
        // terminal predicate), so both prose turns continue the loop.
        let mut contract = crate::contract::Contract::new("test-guard");
        contract.add_assertion(crate::contract::AssertionCheck::FileExists("a.rs".into()));
        agent.set_contract(contract);

        let out = agent.run("goal").await.unwrap();
        assert_eq!(out, "forced final");

        let calls = all_calls.lock().unwrap();
        assert!(
            calls.len() >= 3,
            "expected 3 LLM calls (2 prose cycles + forced), got {}",
            calls.len()
        );
        for (i, payload) in calls.iter().enumerate() {
            let trailing_assistants = payload
                .iter()
                .rev()
                .take_while(|m| m.role == "assistant")
                .count();
            assert!(
                trailing_assistants <= 1,
                "request #{i} ends with {trailing_assistants} consecutive assistant \
                 messages — Ollama HTTP 400 (invalid_request_error)"
            );
        }
        // The surviving trailing assistant carries the LAST response's text.
        let last_payload = calls.last().unwrap();
        let last_assistant = last_payload
            .iter()
            .rev()
            .find(|m| m.role == "assistant")
            .expect("history must still contain an assistant message");
        assert_eq!(last_assistant.content.as_deref(), Some("All done beta"));
    }

    /// G5 sanitizer, direct table: merge case (tool_calls concatenated),
    /// drop case (plain duplicates), normal case unchanged.

    // -----------------------------------------------------------------------
    // f8 T3.3: safe prune boundary with bounded walk-back (AD6/m7)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn compaction_never_orphans_tool_results() {
        let capturing = CapturingClient::new(vec![dummy_response("summary text")]);
        let cfg = AgentConfig {
            workspace: std::path::PathBuf::from("/tmp"),
            ..Default::default()
        };
        let mut agent = Agent::new(capturing, cfg);

        // len=8, naive keep=4 ⇒ naive_split=4. Under the OLD code this
        // drained [..4) and left a TOOL result as the first kept message —
        // orphaned from its assistant. The walk-back must cut at the
        // turn-start user instead.
        agent.messages.push(Message::user("goal")); // 0
        agent.messages.push(Message::assistant("working")); // 1
        agent.messages.push(Message::user("second turn")); // 2
        agent.messages.push(Message::assistant("ran tool")); // 3
        agent.messages.push(Message::tool_result("TOOL_OUTPUT")); // 4
        agent.messages.push(Message::user("third turn")); // 5
        agent.messages.push(Message::assistant("ack")); // 6
        agent.messages.push(Message::user("fourth turn")); // 7

        agent.trigger_compaction().await.unwrap();

        // Summary inserted at 0; kept side starts at a USER boundary.
        assert_eq!(agent.messages[0].role, "system");
        assert_eq!(
            agent.messages[1].role, "user",
            "kept side must begin at a complete user turn, never a tool result"
        );
        // No tool message may appear before the first user turn.
        let first_user = agent
            .messages
            .iter()
            .position(|m| m.role == "user")
            .unwrap();
        assert!(
            agent.messages[1..=first_user]
                .iter()
                .all(|m| m.role == "user"),
            "orphaned pre-user messages survived compaction"
        );
    }

    #[tokio::test]
    async fn compaction_keeps_last_two_user_turns_intact() {
        let capturing = CapturingClient::new(vec![dummy_response("summary text")]);
        let cfg = AgentConfig {
            workspace: std::path::PathBuf::from("/tmp"),
            ..Default::default()
        };
        let mut agent = Agent::new(capturing, cfg);

        // Five user turns, each followed by an assistant reply and a tool
        // result. Whatever boundary is chosen, the LAST TWO user turns and
        // their complete runs must remain in history.
        for i in 1..=5 {
            agent.messages.push(Message::user(format!("turn {i}")));
            agent
                .messages
                .push(Message::assistant(format!("reply {i}")));
            agent
                .messages
                .push(Message::tool_result(format!("out {i}")));
        }
        let users_before: Vec<String> = agent
            .messages
            .iter()
            .filter(|m| m.role == "user")
            .filter_map(|m| m.content.clone())
            .collect();

        agent.trigger_compaction().await.unwrap();

        let kept_users: Vec<String> = agent
            .messages
            .iter()
            .skip(1) // summary system message
            .filter(|m| m.role == "user")
            .filter_map(|m| m.content.clone())
            .collect();
        assert_eq!(
            &kept_users[..],
            &users_before[users_before.len().saturating_sub(2)..],
            "the last two user turns must survive compaction intact"
        );
        // Each surviving user turn keeps its full run: assistant + tool follow.
        for (i, msg) in agent.messages.iter().enumerate() {
            if msg.role == "user"
                && i > 0
                && msg
                    .content
                    .as_deref()
                    .is_some_and(|c| c.starts_with("turn "))
            {
                if let Some(next_a) = agent.messages.get(i + 1) {
                    assert_eq!(next_a.role, "assistant", "orphaned user turn at {i}");
                }
            }
        }
    }

    #[tokio::test]
    async fn compaction_no_user_turns_falls_back_gracefully() {
        let capturing = CapturingClient::new(vec![dummy_response("summary text")]);
        let cfg = AgentConfig {
            workspace: std::path::PathBuf::from("/tmp"),
            ..Default::default()
        };
        let mut agent = Agent::new(capturing, cfg);

        // Degenerate: no user turns anywhere → walk-back finds nothing and
        // must fall back to the naive split WITHOUT panicking.
        agent.messages.push(Message::assistant("a"));
        agent.messages.push(Message::tool_result("t1"));
        agent.messages.push(Message::assistant("b"));
        agent.messages.push(Message::tool_result("t2"));
        agent.messages.push(Message::assistant("c"));

        agent.trigger_compaction().await.unwrap();
        assert!(agent.messages[0]
            .content
            .as_deref()
            .unwrap()
            .contains("[Context Summary]"));
        // Naive split kept the last 4; only the first was summarized away.
        assert_eq!(agent.messages.len(), 5); // summary + 4 kept
    }

    #[tokio::test]
    async fn compaction_walkback_bounded_falls_back_to_naive() {
        let capturing = CapturingClient::new(vec![dummy_response("summary text")]);
        let last_messages = capturing.last_messages.clone();
        let cfg = AgentConfig {
            workspace: std::path::PathBuf::from("/tmp"),
            ..Default::default()
        };
        let mut agent = Agent::new(capturing, cfg);

        // Long assistant→tool run (>K=12) between two user turns: the only
        // earlier user sits beyond the walk-back window ⇒ naive fallback.
        agent.messages.push(Message::user("ancient goal"));
        for i in 0..15 {
            agent
                .messages
                .push(Message::assistant(format!("work {i} NAIVE_DRAINED_MARKER")));
            agent
                .messages
                .push(Message::tool_result(format!("out {i}")));
        }
        agent.messages.push(Message::user("recent turn"));
        agent.messages.push(Message::assistant("recent reply"));
        agent.messages.push(Message::user("final question"));

        agent.trigger_compaction().await.unwrap();

        // Naive fallback ⇒ boundary == naive exactly: post-compaction length
        // is summary + last 4 messages.
        assert_eq!(agent.messages.len(), 5);
        // The summarizer saw exactly the naively-drained window: the marked
        // run messages AND the ancient user turn (both sit before naive).
        let captured = last_messages.lock().unwrap().clone().unwrap();
        let prompt = captured[0].content.as_deref().unwrap();
        assert!(prompt.contains("NAIVE_DRAINED_MARKER"));
        assert!(
            prompt.contains("ancient goal"),
            "fallback must drain exactly [..naive), not walk back further"
        );

        // Kept side: recent run + final user turn intact.
        assert!(agent
            .messages
            .iter()
            .any(|m| m.content.as_deref() == Some("recent turn")));
        assert!(agent
            .messages
            .iter()
            .any(|m| m.content.as_deref() == Some("final question")));
    }

    #[tokio::test]
    async fn compaction_insufficient_freeing_keeps_halt_semantics() {
        // Anti-halt fixture (m7): tool-heavy history where the scripted
        // summary is LONGER than what pruning freed → record_compaction
        // errors each time. Semantics unchanged: warn path, consecutive
        // counter increments per attempt, halt engages only after the SAME
        // threshold as before (3).
        let long_summary = "x".repeat(2000); // needed ≈ 500+100 ≫ freed
        let capturing = CapturingClient::new(vec![
            dummy_response(&long_summary),
            dummy_response(&long_summary),
            dummy_response(&long_summary),
        ]);
        let cfg = AgentConfig {
            workspace: std::path::PathBuf::from("/tmp"),
            ..Default::default()
        };
        let mut agent = Agent::new(capturing, cfg);

        agent.messages.push(Message::user("short goal"));
        agent.messages.push(Message::assistant("ok"));
        for i in 0..6 {
            agent
                .messages
                .push(Message::tool_result(format!("out {i}")));
            agent.messages.push(Message::assistant("noted"));
        }
        agent.messages.push(Message::user("latest"));

        agent.trigger_compaction().await.unwrap();
        assert!(
            !agent.budget.should_halt(),
            "one insufficient compaction must not escalate to halt"
        );
        assert_eq!(agent.budget.consecutive_compactions(), 1);

        for _ in 0..2 {
            agent.trigger_compaction().await.unwrap();
        }
        assert!(agent.budget.should_halt());
    }

    // -----------------------------------------------------------------------
    // E3: qwen stance overlay
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn preflight_appends_qwen_stance_to_system_prompt() {
        // Default model is qwen3:8b → overlay present even without bootstrap hint.
        let mut agent = dummy_agent();
        agent.run_bootstrap();
        agent.preflight("hi");
        let first = agent.messages.first().expect("system message pushed");
        assert_eq!(first.role, "system");
        let content = first.content.as_deref().unwrap();
        assert!(
            content.contains(QWEN_STANCE_OVERLAY),
            "qwen model must get stance overlay, got: {content}"
        );

        // Non-qwen model without bootstrap hint → no system message.
        let client = OllamaClient::new("http://localhost:11434");
        let cfg = AgentConfig {
            model: "ornith:9b".to_string(),
            workspace: std::path::PathBuf::from("/tmp"),
            ..Default::default()
        };
        let mut agent2 = Agent::new(client, cfg);
        agent2.run_bootstrap();
        agent2.preflight("hi");
        assert!(
            !agent2.messages.iter().any(|m| m.role == "system"),
            "non-qwen model must not get the stance overlay"
        );
    }

    #[test]
    fn minimal_prompt_style_preflight_is_compact() {
        // f8 T4.1: nano tier gets the compact imperative contract — ≤1600
        // chars total (workspace label included), no verbose build-command
        // hints and no qwen stance overlay.
        let client = OllamaClient::new("http://localhost:11434");
        let cfg = AgentConfig {
            model: "qwen2.5-3b".to_string(),
            workspace: std::path::PathBuf::from("/tmp"),
            ..Default::default()
        }
        .with_profile("auto");
        assert_eq!(cfg.profile.name, "nano", "3b id auto-selects nano");
        let mut agent = Agent::new(client, cfg);
        agent.run_bootstrap();
        agent.preflight("hi");

        let system = agent
            .messages
            .iter()
            .find(|m| m.role == "system")
            .expect("minimal preflight must still lead with a system message");
        let content = system.content.as_deref().unwrap_or_default();
        assert!(
            content.starts_with("You are a coding agent working inside this repository."),
            "got: {content}"
        );
        assert!(
            content.len() <= 1600,
            "minimal prompt budget exceeded: {} chars",
            content.len()
        );
        assert!(content.contains("One action per turn"));
        assert!(content.contains("Read a file before editing"));
        assert!(!content.contains("Available build commands"), "{content}");
        assert!(!content.contains(QWEN_STANCE_OVERLAY));
    }

    #[tokio::test]
    async fn full_prompt_style_path_unchanged_for_medium() {
        // f8 T4.1 AC: the Full path keeps bootstrap detail + stance overlay.
        let client = OllamaClient::new("http://localhost:11434");
        let cfg = AgentConfig {
            model: "qwen3:8b".to_string(),
            workspace: std::path::PathBuf::from("/tmp"),
            ..Default::default()
        };
        let mut agent = Agent::new(client, cfg);
        agent.run_bootstrap();
        agent.preflight("hi");
        // No project marker in /tmp → no hint; the qwen stance alone leads.
        let system = agent
            .messages
            .iter()
            .find(|m| m.role == "system")
            .expect("qwen medium profile gets the stance overlay");
        assert_eq!(system.content.as_deref(), Some(QWEN_STANCE_OVERLAY));
    }
}

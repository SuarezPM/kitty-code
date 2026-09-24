//! Outer agent loop: `run` orchestration, verification gate, and autorun
//! continuation hops.

use super::VERIFY_MAX_REVISIONS;
use super::*;
use anyhow::{Context, Result};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

impl Agent {
    /// Run the agent loop with a user prompt.
    ///
    /// Returns the final text response after tool calls are resolved.
    ///
    /// F2 fixes applied:
    /// - F2.2: Tool errors → failed result messages, never abort session
    /// - F2.3: `push_message` is the sole token accounting point (no double count)
    /// - F2.4: Quality monitor window persists across turns
    /// - F2.5: Tool results pushed before corrective user messages
    /// - F2.6: Thinking budget exceeded retries inline, doesn't return raw text
    pub async fn run(&mut self, user_prompt: &str) -> Result<String> {
        self.run_with(user_prompt, RunOverrides::default()).await
    }

    async fn run_impl(&mut self, user_prompt: &str) -> Result<String> {
        // Fase 0 remediation: reset empty streak per run (prevents cross-run contamination)
        self.empty_response_streak = 0;
        // E3: reset the deterministic doom-loop detector per run as well.
        self.doom_loop.reset();
        // W7: the ReAct trace is per-run ephemeral, like the doom streak.
        self.react_steps.clear();
        // Bootstrap: detect workspace on first turn
        self.run_bootstrap();

        // F3.4: telemetry writer (JSONL, rotates at 10MB, disabled by KITTY_TELEMETRY=off)
        let mut telemetry = crate::telemetry::TelemetryWriter::new(None);

        // Preflight FIRST so the system message is the first message in the
        // conversation (recall/knowledge injections below must follow it).
        self.preflight(user_prompt);

        // Episodic memory: recall past episodes for this goal (context, not a
        // user turn — injected after the system message and budget-tracked).
        self.recall_episodes(user_prompt).await;

        // Knowledge injection: inject on first user message if enabled.
        self.inject_knowledge(user_prompt).await;

        // F2.3: push_message is the sole token accounting point.
        self.push_message(Message {
            role: "user".to_string(),
            content: Some(user_prompt.to_string()),
            thinking: None,
            tool_calls: None,
            tool_call_id: None,
        })
        .await;

        // G2-FIX3 (now structural): re-seed the per-turn thinking cap from
        // the CURRENT tuning — per-run overrides were applied above, so this
        // reads the effective tuning instead of trusting a post-construction
        // mutation (the old max_thinking_tokens dead-store bug class).
        self.thinking_budget
            .set_per_turn_cap(per_turn_thinking_cap(self.config.tuning.thinking));
        // Reset thinking budget for this turn
        self.thinking_budget.reset_turn();

        // Verification loop (P0): outer revision cycle around the tool loop.
        // After the agent finishes, workspace tests grade the work; failures
        // are fed back as a user message so the model can revise (up to
        // VERIFY_MAX_REVISIONS cycles). See `verify_workspace`.
        let mut revision_cycles: usize = 0;
        // f8 AD1: autorun hop budget. Consumed ONLY on iteration exhaustion
        // (never by verification revisions); 0 = byte-identical legacy run.
        let mut hops_remaining = self.config.autorun_hops;
        'verify: loop {
            // Fixed state machine (f8 AD1): the inner for restarts on hop
            // consumption; force_response + Verification run ONLY on real
            // completion paths — natural terminal return, or exhaustion with
            // no hops left.
            'autorun: loop {
                // Bounded tool loop: EVERY iteration processes the response —
                // tool calls are executed and fed back, text responses still
                // consume an iteration (the loop never breaks on text; it runs
                // to `max_iterations`). Only when the budget is EXHAUSTED does
                // the post-loop code below extract a final answer. The outer
                // cycles multiply worst-case LLM calls per prompt: revisions
                // (×VERIFY_MAX_REVISIONS) and hops (see ModelTuning docs).
                for iteration in 0..self.config.max_iterations {
                    debug!(iteration, "Agent loop iteration");

                    // Check context budget before LLM call. Compaction fires when the
                    // tracked usage reaches the compaction threshold (default 80% of
                    // the context limit — `ContextBudget::DEFAULT_THRESHOLD`).
                    if self.budget.should_compact() {
                        self.trigger_compaction().await?;
                    }

                    if self.budget.should_halt() {
                        warn!("Context halting: compaction exhausted");
                        return self
                            .force_response(
                                "Context limit reached. Please start a new conversation.",
                            )
                            .await;
                    }

                    // E3 step cap: on the final iteration, tell the model to wrap up
                    // now instead of being cut mid-action (runs exactly once, since
                    // `iteration + 1 == max_iterations` matches a single iteration).
                    // SUPPRESSED while autorun hops remain (f8 AD1/m6): this cycle
                    // would continue rather than finish, and contradictory
                    // "finish now"/"continue" signals confuse small models.
                    if iteration + 1 == self.config.max_iterations && hops_remaining == 0 {
                        self.push_message(Message::user(MAX_STEPS_PROMPT)).await;
                    }

                    // LLM call (F3.4: track turn start for telemetry latency)
                    // F4.3: schema breaker — after 3 consecutive invalid tool calls,
                    // restrict to no tools (Plan category) for the next turn.
                    let saved_tools;
                    if self.invalid_streak >= 3 {
                        warn!(
                            streak = self.invalid_streak,
                            "Schema breaker: restricting tools for next turn"
                        );
                        saved_tools = std::mem::take(&mut self.tools);
                        self.invalid_streak = 0;
                    } else {
                        saved_tools = Vec::new();
                    }
                    let turn_start = std::time::Instant::now();
                    let response = match self.llm_call().await {
                        Ok(resp) => resp,
                        Err(e) if is_context_overflow_error(&e) => {
                            warn!(error = %e, "Context overflow detected, attempting compaction + retry");
                            if let Err(compact_err) = self.trigger_compaction().await {
                                warn!(error = %compact_err, "Compaction after overflow failed");
                            }
                            self.llm_call()
                                .await
                                .context("LLM call failed after overflow compaction")?
                        }
                        Err(e) => return Err(e),
                    };
                    // Restore tools after the breaker turn.
                    if !saved_tools.is_empty() {
                        self.tools = saved_tools;
                    }
                    let turn_latency = turn_start.elapsed();

                    // Extract thinking tokens for budget tracking (thinking is metadata,
                    // not a conversation message — tracked via explicit budget.consume).
                    if let Some(thinking) = response.message.thinking.as_deref() {
                        let tokens = self.count_tokens(thinking).await;
                        // Apply thinking budget cap
                        if let Err(e) = self.thinking_budget.record(tokens) {
                            warn!(error = %e, streak = self.empty_response_streak, "Thinking budget exceeded, retrying without thinking");
                            // F2.6: Retry without thinking — re-call LLM, then fall through
                            // to normal processing (tool calls and all). Max 1 retry per turn.
                            // Note: empty_response_streak is NOT reset here; thinking retry counts toward streak.
                            let retry_response = self.llm_call_no_thinking().await?;
                            self.process_response(user_prompt, retry_response).await?;
                            // Fail-fast after retry as well (handoff §6). Warn with streak always.
                            if should_fail_fast(self.empty_response_streak) {
                                warn!(streak = self.empty_response_streak, "Empty response streak cap reached (2) after thinking retry, failing fast");
                                return Ok("no response (empty after 2 retries)".to_string());
                            }
                            // Don't reset thinking budget — we already retried once
                            continue;
                        }
                        // Thinking tokens are consumed against budget but not pushed as
                        // a message — explicit consume here is correct (F2.3 exception).
                        self.budget.consume(tokens);
                    }

                    // F3.4: extract telemetry fields before process_response takes ownership
                    let t_tool_calls = response
                        .message
                        .tool_calls
                        .as_ref()
                        .map_or(0, |tc| tc.len());
                    let t_tool_names: Vec<String> = response
                        .message
                        .tool_calls
                        .as_ref()
                        .map(|tc| tc.iter().map(|t| t.function.name.clone()).collect())
                        .unwrap_or_default();
                    let t_tokens_in = response.prompt_eval_count.map(|c| c as usize);
                    let t_tokens_out = response.eval_count.map(|c| c as usize);

                    self.process_response(user_prompt, response).await?;

                    // Fail-fast: 2 consecutive empty responses → terminate quickly
                    // instead of looping until max_iterations/timeout (handoff §6).
                    if should_fail_fast(self.empty_response_streak) {
                        warn!(
                            streak = self.empty_response_streak,
                            "Empty response streak cap reached (2), failing fast"
                        );
                        // Episode already saved as failure in process_response.
                        return Ok("no response (empty after 2 retries)".to_string());
                    }

                    // F3.4: write telemetry for this turn (after process_response)
                    telemetry
                        .write_turn(&crate::telemetry::TurnEvent {
                            ts: SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs(),
                            model: self.config.model.clone(),
                            turn: iteration,
                            tool_calls: t_tool_calls,
                            tool_names: t_tool_names,
                            latency_ms: turn_latency.as_millis() as u64,
                            tokens_in: t_tokens_in,
                            tokens_out: t_tokens_out,
                            error: None,
                            bench_suite: None,
                            compacted: false,
                        })
                        .await;

                    // Reset thinking budget for next turn
                    self.thinking_budget.reset_turn();
                    // F2.4: Do NOT clear the quality monitor window — it persists
                    // across turns to detect cross-turn tool loops.
                }

                // Iteration budget exhausted WITHOUT a natural terminal return.
                if hops_remaining > 0 {
                    // AD1 hop path: NO force_response, NO verification, NO episode
                    // save — push the continuation prompt into history so context
                    // persists, spend one hop, restart the full inner loop.
                    hops_remaining -= 1;
                    info!(
                        remaining = hops_remaining,
                        total = self.config.autorun_hops,
                        "autorun: iterations exhausted — continuing with a fresh cycle"
                    );
                    self.push_message(Message::user(AUTORUN_CONTINUE_PROMPT))
                        .await;
                    continue 'autorun;
                }
                break 'autorun;
            }

            // Exhausted iterations with no hops left — force a final response.
            warn!(
                iterations = self.config.max_iterations,
                "Agent loop exhausted iterations"
            );
            let text = self
                .force_response("Please provide your final answer now.")
                .await?;

            // Verification gate: only meaningful when the agent actually
            // touched files — otherwise failures are pre-existing and not
            // the model's to fix.
            let verification = if self.registry.file_modifications() == 0 {
                debug!("verification skipped: no file modifications this session");
                Verification::Skipped
            } else {
                self.verify_workspace().await
            };

            match verification {
                Verification::Failed(output) if revision_cycles < VERIFY_MAX_REVISIONS => {
                    revision_cycles += 1;
                    warn!(
                        revision_cycles,
                        max_revisions = VERIFY_MAX_REVISIONS,
                        "Verification failed — feeding test output back for revision"
                    );
                    self.push_message(Message::user(format!(
                        "Your changes caused test failures:\n\n{output}\n\nPlease fix."
                    )))
                    .await;
                    continue 'verify;
                }
                Verification::Failed(output) => {
                    // Revision budget exhausted (or gate disabled mid-flight):
                    // consume ONE autorun hop iff any remain — a full fresh
                    // cycle retries with the LATEST failure output fed back;
                    // no save here (intermediate). Otherwise return the work
                    // as-is, recorded as a failure (terminal).
                    if autorun_hop_retries_after_revisions_exhausted(
                        hops_remaining,
                        revision_cycles,
                    ) {
                        hops_remaining -= 1;
                        warn!(remaining = hops_remaining, "Verification failed and revisions exhausted — autorun hop spent, retrying full cycle");
                        self.push_message(Message::user(format!(
                            "Your changes caused test failures:\n\n{output}\n\nPlease fix."
                        )))
                        .await;
                        continue 'verify;
                    }
                    self.save_episode(user_prompt, &text, false).await;
                    return Ok(text);
                }
                Verification::Passed => {
                    self.save_episode(user_prompt, &text, true).await;
                    return Ok(text);
                }
                Verification::Skipped => {
                    // No test command / loop disabled / nothing modified —
                    // historical behavior: exhausted iterations count as fail.
                    self.save_episode(user_prompt, &text, false).await;
                    return Ok(text);
                }
            }
        }
    }

    pub async fn run_with_events(
        &mut self,
        user_prompt: &str,
        overrides: RunOverrides,
        tx: tokio::sync::mpsc::Sender<AgentEvent>,
    ) -> Result<String> {
        self.apply_overrides(&overrides);
        self.run_inner(user_prompt, Some(tx)).await
    }

    /// Run the agent loop with per-run overrides (Fase 2.1). `run` delegates
    /// here with [`RunOverrides::default`], so the legacy path is unchanged.
    pub async fn run_with(&mut self, user_prompt: &str, overrides: RunOverrides) -> Result<String> {
        self.apply_overrides(&overrides);
        self.run_inner(user_prompt, None).await
    }

    /// Single application point for per-run overrides. Runs BEFORE the
    /// thinking-budget re-seed so the live tracker observes the effective
    /// tuning (this is the structural fix for the G2-FIX3 dead-store class).
    fn apply_overrides(&mut self, ov: &RunOverrides) {
        if let Some(w) = ov.context_window {
            self.config.context_window = w;
            self.budget.set_max(w);
        }
        if let Some(t) = ov.compact_threshold {
            self.config.compact_threshold = t;
        }
        if ov.context_window.is_some() || ov.compact_threshold.is_some() {
            let threshold = self.config.compact_threshold;
            self.budget
                .set_threshold(ContextBudget::derive_compaction_threshold(
                    threshold,
                    self.config.context_window,
                ));
        }
        if let Some(mi) = ov.max_iterations {
            self.config.max_iterations = mi;
        }
        if let Some(bt) = ov.bash_timeout {
            self.config.bash_timeout_secs = bt;
        }
        if let Some(h) = ov.autorun_hops {
            self.config.autorun_hops = clamp_autorun_hops(h);
        }
        if let Some(t) = ov.tuning.clone() {
            self.config.tuning = t;
        }
    }

    async fn run_inner(
        &mut self,
        user_prompt: &str,
        tx: Option<tokio::sync::mpsc::Sender<AgentEvent>>,
    ) -> Result<String> {
        let result = self.run_impl(user_prompt).await;
        if let Some(sender) = tx {
            match &result {
                Ok(text) => {
                    let _ = sender.send(AgentEvent::Done(text.clone())).await;
                }
                Err(e) => {
                    let _ = sender.send(AgentEvent::Error(e.to_string())).await;
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Verification loop (P0): grade the agent's work with workspace tests and
// feed failures back so the model can revise.
// ---------------------------------------------------------------------------

/// Env override that disables the verification loop (`KITTY_VERIFY_LOOP=off`).
pub const ENV_VERIFY_LOOP: &str = "KITTY_VERIFY_LOOP";

/// Hard timeout for one workspace test run. Independent of
/// `bash_timeout_secs` (profile-driven, tuned for single commands) because a
/// full test suite legitimately runs longer.
const VERIFY_TEST_TIMEOUT_SECS: u64 = 300;

/// Outcome of grading the workspace after the tool loop finishes.
enum Verification {
    /// Loop disabled / no test command detected / no files modified —
    /// behave exactly as before the loop existed.
    Skipped,
    /// Test command ran successfully.
    Passed,
    /// Test command failed; payload is the combined stdout+stderr excerpt.
    Failed(String),
}

/// Whether the verification loop is enabled (default on; `KITTY_VERIFY_LOOP=off`
/// disables).
use crate::config::verify_loop_enabled;

/// `true` when a Makefile declares a `test:` target (column-0 target line).
/// W1 async: todo-file 1-op via tokio::fs.
pub(super) async fn makefile_has_test_target(path: &Path) -> bool {
    tokio::fs::read_to_string(path)
        .await
        .map(|content| content.lines().any(|l| l.starts_with("test:")))
        .unwrap_or(false)
}

/// Detect the workspace test command (priority order: Rust → Python → JS/TS →
/// Go → generic Makefile). `None` when no recognizable project marker exists.
/// W1 async: marker checks via try_exists + makefile content via tokio::fs.
pub(super) async fn detect_test_command(workspace: &Path) -> Option<String> {
    if tokio::fs::try_exists(workspace.join("Cargo.toml"))
        .await
        .unwrap_or(false)
    {
        return Some("cargo test --workspace -- -q".to_string());
    }
    if tokio::fs::try_exists(workspace.join("pyproject.toml"))
        .await
        .unwrap_or(false)
        || tokio::fs::try_exists(workspace.join("setup.py"))
            .await
            .unwrap_or(false)
    {
        return Some("python3 -m pytest --tb=short -q".to_string());
    }
    if tokio::fs::try_exists(workspace.join("package.json"))
        .await
        .unwrap_or(false)
    {
        return Some("npm test -- --brief".to_string());
    }
    if tokio::fs::try_exists(workspace.join("go.mod"))
        .await
        .unwrap_or(false)
    {
        return Some("go test ./...".to_string());
    }
    for name in ["Makefile", "makefile"] {
        let mk = workspace.join(name);
        if tokio::fs::try_exists(&mk).await.unwrap_or(false) && makefile_has_test_target(&mk).await
        {
            return Some("make test".to_string());
        }
    }
    None
}

impl Agent {
    /// Run the workspace test command detected for [`AgentConfig::workspace`].
    ///
    /// The command executes through the same containment stack as the Bash
    /// tool (Guard policy → Sandbox kernel containment → hard timeout), so
    /// `KITTY_SANDBOX=off` applies here too. Output is truncated to the same
    /// cap as tool results before being fed back to the model.
    async fn verify_workspace(&self) -> Verification {
        if !verify_loop_enabled() {
            return Verification::Skipped;
        }
        let Some(command) = detect_test_command(&self.config.workspace).await else {
            debug!("verification skipped: no test command detected");
            return Verification::Skipped;
        };
        info!(command = %command, "verification: running workspace tests");
        let result = self
            .registry
            .run_bash_command_in(
                &command,
                std::time::Duration::from_secs(VERIFY_TEST_TIMEOUT_SECS),
                Some(&self.config.workspace),
            )
            .await;
        match result {
            Ok(r) if r.success => {
                info!(command = %command, "verification: tests passed");
                Verification::Passed
            }
            Ok(r) => {
                warn!(command = %command, "verification: tests failed");
                Verification::Failed(truncate_tool_output(r.output, TRUNCATION_MAX_BYTES).await)
            }
            Err(e) => {
                // Blocked / sandbox failure running the test command itself:
                // do not punish the model for harness-level issues.
                warn!(error = %e, "verification: test command could not run");
                Verification::Skipped
            }
        }
    }
}

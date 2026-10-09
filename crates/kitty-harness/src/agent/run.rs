//! Outer agent loop: `run` orchestration, verification gate, and autorun
//! continuation hops.

use super::VERIFY_MAX_REVISIONS;
use super::*;
use anyhow::{Context, Result};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

/// W1B: esfuerzo consumido en un `run` (hops = restarts ejecutados, iters =
/// turnos LLM totales). Lo devuelve [`Agent::last_run_effort`] para los
/// manifiestos bench.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunEffort {
    /// Hop-restarts ejecutados (ciclos extra tras agotar iteraciones).
    pub hops_used: usize,
    /// Turnos LLM totales del run (todos los ciclos y revisiones).
    pub iters_used: usize,
    /// El `EffortRouter` quedó OFF en ese run (bench o kill-switch).
    pub router_off: bool,
}

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
        // W1B: reset per-run effort stats (read via `last_run_effort`).
        self.last_run_hops_used = 0;
        self.last_run_iters_used = 0;
        self.last_run_router_off = false;
        // E3: reset the deterministic doom-loop detector per run as well.
        self.doom_loop.reset();
        // W7: the ReAct trace is per-run ephemeral, like the doom streak.
        self.react_steps.clear();
        // W5-code: the n-gram predictor observes one run's tool sequence;
        // reset per run so predictions never leak across tasks.
        self.predictor = crate::pld::NgramPredictor::new();
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
        // W1B: esfuerzo consumido en este run (señal + manifiesto). Los
        // contadores viven aquí (locales) y se publican en `self.last_run_*`
        // cada turno para que cualquier retorno ya los lleve al día.
        let mut total_turns: usize = 0;
        let mut vacuous_turns: usize = 0;
        let mut hops_spent: usize = 0;
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
                    // W7-code: turn-start checkpoint for doom-loop recovery
                    // (rewind point if this turn aborts the loop).
                    let checkpoint = self.messages.len();

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
                    if self.invalid_streak >= crate::router::INVALID_STREAK_BREAKER_AT {
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
                            // W7-code: same doom-loop recovery as the main
                            // path (terminal `Some` returns the recovered
                            // text; `None` falls through to the checks below).
                            if let Some(recovered) = self
                                .process_response_recoverable(
                                    user_prompt,
                                    retry_response,
                                    checkpoint,
                                )
                                .await?
                            {
                                return Ok(recovered);
                            }
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

                    // W1B: vacuity counters for the EffortRouter signal (a turn is
                    // vacuous when it carries neither text nor tool calls).
                    total_turns += 1;
                    let vacuous = response
                        .message
                        .content
                        .as_deref()
                        .is_none_or(|c| c.trim().is_empty())
                        && response
                            .message
                            .tool_calls
                            .as_ref()
                            .is_none_or(|tc| tc.is_empty());
                    if vacuous {
                        vacuous_turns += 1;
                    }
                    // Publica aquí (no al fin del turno): el fail-fast retorna
                    // antes y debe llevar los contadores al día.
                    self.last_run_iters_used = total_turns;
                    self.last_run_hops_used = hops_spent;

                    // W0 (SDD estado-arte): per-turn latency desglose
                    // (wall = modelo/guard/sandbox/tools). The registry
                    // accumulators reset here so this turn's event only
                    // carries its own guard/sandbox time; `tools_ms` wraps
                    // the whole `process_response` (tool execution +
                    // result handling). Read+reset via `take_turn_timing`
                    // so verification-loop Bash runs (outside any turn)
                    // never leak into the next event.
                    self.registry.reset_turn_timing();
                    let tools_start = std::time::Instant::now();
                    // W7-code: a doom-loop abort recovers to `checkpoint`
                    // with a terminal text instead of failing the run.
                    if let Some(recovered) = self
                        .process_response_recoverable(user_prompt, response, checkpoint)
                        .await?
                    {
                        return Ok(recovered);
                    }
                    let tools_ms = tools_start.elapsed().as_millis() as u64;
                    let (guard_ms, sandbox_ms) = self.registry.take_turn_timing();
                    let wall_model_ms = turn_latency.as_millis() as u64;
                    self.last_turn_breakdown = Some(crate::telemetry::TurnBreakdown {
                        wall_model_ms,
                        guard_ms,
                        sandbox_ms,
                        tools_ms,
                    });

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
                            // W0: hops/iters a telemetría (margen None: sin
                            // plumbing de logits en esta lane, ver W3).
                            // W1B: `hops` = restarts ejecutados, `iters` =
                            // turnos totales del run (no del ciclo).
                            hops: Some(hops_spent),
                            iters: Some(total_turns),
                            margin: None,
                            // W0 (SDD estado-arte): wall =
                            // modelo/guard/sandbox/tools. Medidos arriba;
                            // `latency_ms` se conserva como wall del modelo
                            // (compat) y `wall_model_ms` lo nombra explícito.
                            wall_model_ms: Some(wall_model_ms),
                            guard_ms: Some(guard_ms),
                            sandbox_ms: Some(sandbox_ms),
                            tools_ms: Some(tools_ms),
                        })
                        .await;
                    // W0: turn span as a tracing event (carries the current
                    // span context; no `entered()` guard is held across
                    // awaits). Visible in tokio-console once the binary
                    // enables it (cli lane) and in fmt logs at debug level.
                    tracing::debug!(
                        turn = iteration,
                        wall_model_ms,
                        guard_ms,
                        sandbox_ms,
                        tools_ms,
                        tool_calls = t_tool_calls,
                        "turn breakdown"
                    );

                    // Reset thinking budget for next turn
                    self.thinking_budget.reset_turn();
                    // F2.4: Do NOT clear the quality monitor window — it persists
                    // across turns to detect cross-turn tool loops.
                }

                // Iteration budget exhausted WITHOUT a natural terminal return.
                if hops_remaining > 0 {
                    // W1B: EffortRouter consultation (fail-open). Los drivers
                    // bench marcan el proceso (`KITTY_BENCH_MODE=1` /
                    // `KITTY_VERIFY_LOOP=off`) → `for_bench()`: decide existe
                    // pero deja pasar intacto, y el manifiesto lo registra vía
                    // `last_run_router_off`. `Normal` es el legacy AD1 exacto.
                    // W4: el closer-Done corrobora como evidencia de parar.
                    let router = Self::effort_router();
                    let mut signal = crate::router::EffortSignal::new(
                        crate::router::EffortSignal::vacuity_from_counts(
                            vacuous_turns,
                            total_turns,
                        ),
                        None, // margin: sin logprobs en esta lane (W3)
                        self.empty_response_streak,
                        self.invalid_streak,
                        self.quality_monitor.events().len(),
                        false, // verify_fail: este hop es pre-verificación
                    );
                    signal.closer_done = self.closer_signal();
                    let decision = router.decide(&signal, hops_remaining);
                    self.last_run_router_off = decision.off_in_bench;
                    match decision.route {
                        crate::router::EffortRoute::Normal => {
                            hops_remaining -= 1;
                            hops_spent += 1;
                            self.last_run_hops_used = hops_spent;
                            info!(
                                remaining = hops_remaining,
                                total = self.config.autorun_hops,
                                "autorun: iterations exhausted — continuing with a fresh cycle"
                            );
                            self.push_message(Message::user(AUTORUN_CONTINUE_PROMPT))
                                .await;
                            continue 'autorun;
                        }
                        crate::router::EffortRoute::ReduceHops => {
                            hops_remaining = decision.hops_budget;
                            if hops_remaining == 0 {
                                warn!(
                                    "effort router: evidence against more hops — forcing final answer"
                                );
                                break 'autorun;
                            }
                            hops_spent += 1;
                            self.last_run_hops_used = hops_spent;
                            info!(
                                remaining = hops_remaining,
                                total = self.config.autorun_hops,
                                "autorun: effort router reduced hops — continuing with a fresh cycle"
                            );
                            self.push_message(Message::user(AUTORUN_CONTINUE_PROMPT))
                                .await;
                            continue 'autorun;
                        }
                        crate::router::EffortRoute::Fallback => {
                            warn!("effort router: compounded evidence — fallback, no more hops");
                            hops_remaining = 0;
                            break 'autorun;
                        }
                    }
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
                Verification::Failed(output) => {
                    // W4 router-donde-decide: el router también decide en el
                    // presupuesto de revisiones. Sin evidencia → `Continue`
                    // (1 revisión, legacy exacto); evidencia débil →
                    // `ConsumeHalf`; compuesta → `Stop` (cae al manejo de
                    // agotadas de abajo). En bench el router está OFF y deja
                    // pasar intacto (trampa conocida: gates-en-bench no).
                    if revision_cycles < VERIFY_MAX_REVISIONS {
                        let mut signal = crate::router::EffortSignal::new(
                            crate::router::EffortSignal::vacuity_from_counts(
                                vacuous_turns,
                                total_turns,
                            ),
                            None, // margin: sin logprobs en esta lane (W3)
                            self.empty_response_streak,
                            self.invalid_streak,
                            self.quality_monitor.events().len(),
                            false, // verify_fail: aún quedan revisiones
                        );
                        signal.closer_done = self.closer_signal();
                        let left = VERIFY_MAX_REVISIONS - revision_cycles;
                        match signal.revision_action() {
                            crate::router::RevisionAction::Stop => {
                                warn!(
                                    "effort router: compounded evidence — skipping remaining revisions"
                                );
                            }
                            action => {
                                revision_cycles += action.consume(left);
                                debug_assert!(revision_cycles <= VERIFY_MAX_REVISIONS);
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
                        }
                    }
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
                        // W1B: este hop post-verificación también cuenta.
                        hops_spent += 1;
                        self.last_run_hops_used = hops_spent;
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

    /// W1B: esfuerzo consumido en el último `run` (para `BenchResult`;
    /// ceros si aún no hubo run). No requiere nombrar [`RunEffort`] en el
    /// call-site: basta leer sus campos públicos sobre el valor devuelto.
    pub fn last_run_effort(&self) -> RunEffort {
        RunEffort {
            hops_used: self.last_run_hops_used,
            iters_used: self.last_run_iters_used,
            router_off: self.last_run_router_off,
        }
    }

    /// W7-code: `process_response` with doom-loop recovery (HU-PODER:
    /// colapso = abandono/error-no-recuperado). On a doom-loop abort
    /// (streak at `DOOM_LOOP_FAIL_AT`), rewinds messages to the turn
    /// `checkpoint`, records the episode as failed, and returns the
    /// recovered terminal text (`Some`). `None` = turn processed normally,
    /// continue the loop. Any other error propagates untouched (legacy).
    async fn process_response_recoverable(
        &mut self,
        user_prompt: &str,
        response: ChatResponse,
        checkpoint: usize,
    ) -> Result<Option<String>> {
        if let Err(e) = self.process_response(user_prompt, response).await {
            if self.doom_loop.streak >= DOOM_LOOP_FAIL_AT {
                warn!(
                    streak = self.doom_loop.streak,
                    "Doom loop abort — recovering to turn checkpoint"
                );
                self.messages.truncate(checkpoint);
                let text = "Stopped: the same tool was repeated with identical arguments. Recovered with partial progress — please continue with smaller steps.";
                self.save_episode(user_prompt, text, false).await;
                return Ok(Some(text.to_string()));
            }
            return Err(e);
        }
        Ok(None)
    }

    /// W4: shared router construction for hop AND revision decisions.
    /// Bench/verify-off drivers get `for_bench()` (decide existe pero deja
    /// pasar intacto — trampa conocida: nunca gates-en-bench).
    fn effort_router() -> crate::router::EffortRouter {
        if crate::config::bench_mode() || !crate::config::verify_loop_enabled() {
            crate::router::EffortRouter::for_bench()
        } else {
            crate::router::EffortRouter::new()
        }
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

#[cfg(test)]
mod run_effort_tests {
    use super::super::{dummy_response, env_lock, tool_call_response, Agent, AgentConfig};
    use super::*;

    /// W5-test helper: one response carrying TWO native tool calls (one turn,
    /// one speculative batch).
    fn tool_call_response_two(
        name1: &str,
        args1: serde_json::Value,
        name2: &str,
        args2: serde_json::Value,
    ) -> kitty_runtime::ChatResponse {
        kitty_runtime::ChatResponse {
            model: "test-model".into(),
            message: kitty_runtime::Message {
                role: "assistant".into(),
                content: None,
                thinking: None,
                tool_calls: Some(vec![
                    kitty_runtime::ToolCall {
                        id: None,
                        function: kitty_runtime::ToolCallFunction {
                            name: name1.to_string(),
                            arguments: args1,
                        },
                    },
                    kitty_runtime::ToolCall {
                        id: None,
                        function: kitty_runtime::ToolCallFunction {
                            name: name2.to_string(),
                            arguments: args2,
                        },
                    },
                ]),
                tool_call_id: None,
            },
            done: true,
            eval_count: None,
            eval_duration: None,
            prompt_eval_count: None,
        }
    }
    use crate::model_profile::EnvRestoreGuard;
    use kitty_runtime::MockClient;

    fn empty_response() -> kitty_runtime::ChatResponse {
        kitty_runtime::ChatResponse {
            model: "test-model".into(),
            message: kitty_runtime::Message {
                role: "assistant".to_string(),
                content: None,
                thinking: None,
                tool_calls: None,
                tool_call_id: None,
            },
            done: true,
            eval_count: None,
            eval_duration: None,
            prompt_eval_count: None,
        }
    }

    fn continues(agent: &Agent) -> usize {
        agent
            .messages
            .iter()
            .filter(|m| m.content.as_deref() == Some(AUTORUN_CONTINUE_PROMPT))
            .count()
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // env lock serializes OS-thread tests; runtime is per-test current-thread
    async fn router_normal_is_legacy_by_default() {
        let _g = env_lock();
        let _e = EnvRestoreGuard::remove(ENV_VERIFY_LOOP);
        // Sin evidencia el router deja pasar: 2 hops gastados como el legacy.
        let responses = vec![
            dummy_response("cycle1"),
            dummy_response("cycle2"),
            dummy_response("cycle3"),
            dummy_response("forced final"),
        ];
        let mut agent = Agent::new(
            MockClient::new(responses),
            AgentConfig {
                autorun_hops: 2,
                ..Default::default()
            },
        );
        agent.config.max_iterations = 1;
        let out = agent.run("hi").await.unwrap();
        assert_eq!(out, "forced final");
        assert_eq!(continues(&agent), 2);
        assert_eq!(
            agent.last_run_effort(),
            RunEffort {
                hops_used: 2,
                iters_used: 3,
                router_off: false,
            }
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // ver arriba
    async fn router_reduce_hops_cuts_useless_cycles() {
        let _g = env_lock();
        let _e = EnvRestoreGuard::remove(ENV_VERIFY_LOOP);
        // Vacuidad 0.5 por ciclo → ReduceHops: 2→1→0, un solo CONTINUE.
        let responses = vec![
            dummy_response("work"),
            empty_response(),
            dummy_response("work"),
            empty_response(),
            dummy_response("forced final"),
        ];
        let mut agent = Agent::new(
            MockClient::new(responses),
            AgentConfig {
                autorun_hops: 2,
                ..Default::default()
            },
        );
        agent.config.max_iterations = 2;
        let out = agent.run("hi").await.unwrap();
        assert_eq!(out, "forced final");
        assert_eq!(continues(&agent), 1, "ReduceHops debe recortar un hop");
        assert_eq!(
            agent.last_run_effort(),
            RunEffort {
                hops_used: 1,
                iters_used: 4,
                router_off: false,
            }
        );
    }

    /// W4: closer stop-signal — Done sobre la última salida ⇒ `Some(true)`;
    /// NotDone/abstain/sin closer/deshabilitado ⇒ `None` (fail-open).
    #[test]
    fn w4_closer_signal_done_abstain_and_disabled() {
        use crate::tfidf_closer::{CloserConfig, TfidfCloser, FIXED_TASK};
        let docs: Vec<(String, bool)> = FIXED_TASK
            .iter()
            .map(|(t, b)| (t.to_string(), *b))
            .collect();
        let closer = TfidfCloser::fit(&docs);
        let cfg = CloserConfig {
            enabled: true,
            ..Default::default()
        };
        let mut agent = Agent::new(MockClient::new(vec![]), AgentConfig::default());
        agent.set_closer(closer, cfg);
        // Sin salida del asistente → sin señal.
        assert_eq!(agent.closer_signal(), None);
        // Frase done exacta del training → Done.
        agent.messages.push(Message::assistant(
            "task complete all tests pass solution saved",
        ));
        assert_eq!(agent.closer_signal(), Some(true));
        // Frase not-done → None (sin anti-evidencia: fail-open).
        agent.messages.push(Message::assistant(
            "error failed traceback exception occurred",
        ));
        assert_eq!(agent.closer_signal(), None);
        // Deshabilitado → None aunque el texto sea done.
        let mut agent = Agent::new(MockClient::new(vec![]), AgentConfig::default());
        agent.set_closer(
            TfidfCloser::fit(&docs),
            CloserConfig {
                enabled: false,
                ..Default::default()
            },
        );
        agent.messages.push(Message::assistant(
            "task complete all tests pass solution saved",
        ));
        assert_eq!(agent.closer_signal(), None);
        // Sin modelo (fit vacío) → Abstain → None.
        let mut agent = Agent::new(MockClient::new(vec![]), AgentConfig::default());
        agent.set_closer(
            TfidfCloser::fit(&[]),
            CloserConfig {
                enabled: true,
                ..Default::default()
            },
        );
        agent.messages.push(Message::assistant(
            "task complete all tests pass solution saved",
        ));
        assert_eq!(agent.closer_signal(), None);
    }

    /// W4 tiers: triaje compacta en QR (queries verbatim), ejecución en
    /// formato completo (legacy exacto).
    #[tokio::test]
    async fn w4_triage_compacts_qr_execution_compacts_full() {
        for (profile, expect_qr) in [("tiny", true), ("medium", false)] {
            let mut agent = Agent::new(
                MockClient::new(vec![dummy_response("summary text")]),
                AgentConfig {
                    workspace: std::path::PathBuf::from("/tmp"),
                    ..Default::default()
                }
                .with_profile(profile),
            );
            agent
                .messages
                .push(Message::user("goal: fix the login bug"));
            agent.messages.push(Message::assistant("working"));
            agent.messages.push(Message::user("keep going"));
            agent.messages.push(Message::assistant("a"));
            agent.messages.push(Message::user("b"));
            agent.messages.push(Message::assistant("c"));
            agent.messages.push(Message::user("d"));
            agent.messages.push(Message::assistant("e"));
            agent.trigger_compaction().await.unwrap();
            let summary = agent.messages[0].content.as_deref().unwrap();
            assert!(
                summary.contains("[Context Summary]"),
                "{profile}: {summary}"
            );
            assert_eq!(
                summary.contains("## Open Queries"),
                expect_qr,
                "{profile}: {summary}"
            );
            if expect_qr {
                assert!(
                    summary.contains("fix the login bug"),
                    "QR must keep the query verbatim: {summary}"
                );
            }
        }
    }

    /// W5-code: la caché idempotente sirve un retry idéntico sin quemar
    /// otra llamada al modelo (la cola del Mock solo trae una respuesta: el
    /// segundo `llm_call` idéntico SOLO puede salir de la caché).
    #[tokio::test]
    async fn w5_response_cache_serves_identical_retry() {
        let agent = Agent::new(
            MockClient::new(vec![dummy_response("cached")]),
            AgentConfig::default(),
        )
        .with_response_cache(8);
        let first = agent.llm_call().await.unwrap();
        assert_eq!(first.message.content.as_deref(), Some("cached"));
        let second = agent.llm_call().await.unwrap();
        assert_eq!(second.message.content.as_deref(), Some("cached"));
        let (hits, misses) = agent
            .response_cache
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .stats();
        assert_eq!((hits, misses), (1, 1));
    }

    /// W5-code: sin caché, el mismo retry sí quema otra llamada (compat:
    /// dos respuestas encoladas, dos llamadas).
    #[tokio::test]
    async fn w5_no_cache_no_reuse() {
        let agent = Agent::new(
            MockClient::new(vec![dummy_response("one"), dummy_response("two")]),
            AgentConfig::default(),
        );
        assert!(agent.response_cache.is_none());
        assert_eq!(
            agent.llm_call().await.unwrap().message.content.as_deref(),
            Some("one")
        );
        assert_eq!(
            agent.llm_call().await.unwrap().message.content.as_deref(),
            Some("two")
        );
    }

    /// W5-code: el batch especulativo ejecuta Glob+Grep concurrentes y
    /// conserva orden y resultados idénticos al path secuencial.
    #[tokio::test]
    async fn w5_speculative_batch_keeps_order_and_results() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.txt"), "hello").unwrap();
        let dir = tmp.path().to_str().unwrap().to_string();
        let batch = tool_call_response_two(
            "Grep",
            serde_json::json!({"pattern": "hello", "path": dir}),
            "Glob",
            serde_json::json!({"pattern": format!("{dir}/*.txt")}),
        );
        for speculative in [false, true] {
            let mut agent = Agent::new(
                MockClient::new(vec![
                    batch.clone(),
                    dummy_response("d"),
                    dummy_response("f"),
                ]),
                AgentConfig {
                    workspace: tmp.path().to_path_buf(),
                    ..Default::default()
                },
            )
            .with_speculative_readonly(speculative);
            agent.config.max_iterations = 2;
            let out = agent.run("search").await.unwrap();
            assert_eq!(out, "f");
            let tools: Vec<&str> = agent
                .messages
                .iter()
                .filter(|m| m.role == "tool")
                .filter_map(|m| m.content.as_deref())
                .collect();
            assert_eq!(tools.len(), 2, "speculative={speculative}: {tools:?}");
            assert!(
                tools[0].contains("hello"),
                "Grep result first (turn order kept): {tools:?}"
            );
            assert!(tools[1].contains("a.txt"), "Glob result second: {tools:?}");
        }
    }

    /// W5-code: el predictor observa la secuencia ejecutada del batch.
    #[tokio::test]
    async fn w5_predictor_observes_executed_sequence() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.txt"), "hello").unwrap();
        let dir = tmp.path().to_str().unwrap().to_string();
        let batch = tool_call_response_two(
            "Grep",
            serde_json::json!({"pattern": "hello", "path": dir}),
            "Glob",
            serde_json::json!({"pattern": format!("{dir}/*.txt")}),
        );
        let mut agent = Agent::new(
            MockClient::new(vec![batch, dummy_response("d"), dummy_response("f")]),
            AgentConfig {
                workspace: tmp.path().to_path_buf(),
                ..Default::default()
            },
        );
        agent.config.max_iterations = 2;
        agent.run("search").await.unwrap();
        assert_eq!(
            agent.predict_next_tool(&["Grep".to_string()]),
            Some("Glob".to_string())
        );
        assert_eq!(agent.predict_next_tool(&["Bash".to_string()]), None);
    }

    /// W7 anti-loop: el texto forzado tras un ToolLoop entra en el
    /// historial (antes solo iba a memoria episódica y el loop repetía
    /// hasta `max_iterations` sin registro de la corrección).
    #[tokio::test]
    async fn w7_tooloop_forced_text_enters_history() {
        let tool = || tool_call_response("Bash", serde_json::json!({"command": "echo hi"}));
        // Solo el force_response del ToolLoop consume del Mock (las 3
        // llamadas llegan por argumento, no por cola).
        let mut agent = Agent::new(
            MockClient::new(vec![dummy_response("forced answer")]),
            AgentConfig::default(),
        );
        for _ in 0..3 {
            agent.process_response("p", tool()).await.unwrap();
        }
        let texts: Vec<&str> = agent
            .messages
            .iter()
            .filter(|m| m.role == "assistant")
            .filter_map(|m| m.content.as_deref())
            .collect();
        assert!(
            texts.contains(&"forced answer"),
            "forced correction must be in history: {texts:?}"
        );
    }

    /// W7-code: 6 ejecuciones idénticas consecutivas abortaban el run con
    /// `Err` (colapso); ahora recovery a checkpoint con texto terminal `Ok`.
    #[tokio::test]
    async fn w7_doom_loop_abort_recovers_to_checkpoint() {
        let responses =
            vec![tool_call_response("Bash", serde_json::json!({"command": "echo hi"}),); 8];
        // Solo el doom guard dispara: el ToolLoop de quality se desactiva
        // en la CONFIG previa (el monitor clona su config al construir, así
        // que mutar `agent.config` después no lo afectaría).
        let mut cfg = AgentConfig::default();
        cfg.quality_config.loop_threshold = 100;
        let mut agent = Agent::new(MockClient::new(responses), cfg);
        agent.config.max_iterations = 8;
        let out = agent.run("repeat").await.expect("recovery must return Ok");
        assert!(
            out.contains("repeated with identical arguments"),
            "recovered terminal text, got: {out}"
        );
        assert!(
            agent.doom_loop.streak >= DOOM_LOOP_FAIL_AT,
            "recovery fires on doom abort, streak={}",
            agent.doom_loop.streak
        );
    }

    /// W0 (SDD estado-arte): a turn that executes a real Bash tool records
    /// the full `wall = modelo/guard/sandbox/tools` desglose, with the
    /// nesting invariant `tools_ms >= guard_ms + sandbox_ms` (nested
    /// intervals on the same clock; floor-truncation preserves it).
    #[tokio::test]
    async fn w0_turn_breakdown_recorded_with_bash_tool() {
        let tmp = tempfile::tempdir().unwrap();
        // max_iterations=1: the single tool turn IS the telemetered turn
        // (a second text turn would overwrite `last_turn_breakdown`).
        let responses = vec![
            tool_call_response("Bash", serde_json::json!({"command": "echo hi"})),
            dummy_response("forced final"),
        ];
        let mut agent = Agent::new(
            MockClient::new(responses),
            AgentConfig {
                workspace: tmp.path().to_path_buf(),
                ..Default::default()
            },
        );
        agent.config.max_iterations = 1;
        let out = agent.run("run echo").await.unwrap();
        assert_eq!(out, "forced final");
        let b = agent
            .last_turn_breakdown()
            .expect("a telemetered turn must record the breakdown");
        assert!(
            b.tools_ms >= b.guard_ms + b.sandbox_ms,
            "nested intervals keep tools >= guard+sandbox: {b:?}"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // ver arriba
    async fn router_never_decides_in_bench() {
        let _g = env_lock();
        let _e = EnvRestoreGuard::set(ENV_VERIFY_LOOP, "off"); // marca bench
                                                               // Misma evidencia que recorta en interactivo: en bench se gastan
                                                               // los 2 hops igual que el legacy (3 ciclos + forced).
        let responses = vec![
            dummy_response("work"),
            empty_response(),
            dummy_response("work"),
            empty_response(),
            dummy_response("work"),
            empty_response(),
            dummy_response("forced final"),
        ];
        let mut agent = Agent::new(
            MockClient::new(responses),
            AgentConfig {
                autorun_hops: 2,
                ..Default::default()
            },
        );
        agent.config.max_iterations = 2;
        let out = agent.run("hi").await.unwrap();
        assert_eq!(out, "forced final");
        assert_eq!(continues(&agent), 2, "en bench el router no decide");
        assert_eq!(
            agent.last_run_effort(),
            RunEffort {
                hops_used: 2,
                iters_used: 6,
                router_off: true,
            }
        );
    }
}

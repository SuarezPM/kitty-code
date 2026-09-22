//! Response processing: tool execution and result ordering, LLM calls with
//! schema/overflow recovery, force-response, preflight, memory recall/save,
//! knowledge/bootstrap injection, and context compaction.

use super::*;

impl Agent {
    /// Process a single LLM response within the agent loop.
    ///
    /// Handles tool execution, quality checks, and message ordering.
    /// Returns Ok(true) when a final text response was produced, Ok(false) to
    /// continue the loop (tool calls were processed).
    pub(super) async fn process_response(
        &mut self,
        user_prompt: &str,
        response: ChatResponse,
    ) -> Result<()> {
        // Check for tool calls — any tool call resets the empty streak
        // (progress was made, handoff §6).
        if let Some(tool_calls) = &response.message.tool_calls {
            if !tool_calls.is_empty() {
                self.empty_response_streak = 0;
                // Add assistant message with tool calls (F2.3: push handles tokens)
                self.push_message(response.message.clone()).await;

                // Also parse text content for XML-style tool calls
                let mut all_tool_calls: Vec<ParsedToolCall> = Vec::new();

                // Native tool calls from Ollama
                for tc in tool_calls {
                    all_tool_calls.push(ParsedToolCall {
                        name: tc.function.name.clone(),
                        arguments: tc.function.arguments.clone(),
                        // Thread the server-issued id so tool results
                        // correlate back (Fase 3); text-parsed calls keep None.
                        id: tc.id.clone(),
                    });
                }

                // Also parse content for XML-style tool calls (with diagnostics for malformed)
                if let Some(content) = response.message.content.as_deref() {
                    let (parsed, diagnostics) = self.parser.parse_with_diagnostics(content);
                    if !diagnostics.is_empty() {
                        for diag in diagnostics {
                            warn!(diagnostic = %diag, "Malformed tool call detected");
                            self.push_message(Message::user(diag)).await;
                        }
                    }
                    all_tool_calls.extend(parsed);
                }

                // G3-1: fold synonym/aliased tool names and argument keys onto
                // the canonical forms UPSTREAM of every consumer that inspects
                // tool-call identity — quality monitoring, corrective
                // diagnostics, episode memory, and dispatch. ToolRegistry::
                // execute re-normalizes harmlessly (the operation is
                // idempotent), so hallucinated names never leak through.
                for tc in &mut all_tool_calls {
                    *tc = crate::tools::normalize_tool_call(tc);
                }

                // Quality monitor: check for hallucinated tools before execution.
                // F2.4: window is NOT cleared — persists across turns.
                let violations = self
                    .quality_monitor
                    .check_response(response.message.content.as_deref(), &all_tool_calls);

                // F2.5: Push tool results FIRST, then corrective user messages.
                // Ollama message order: assistant(tool_calls) → tool* → next.
                for tc in &all_tool_calls {
                    // Skip hallucinated tools
                    if violations.iter().any(|v| {
                        matches!(
                            v,
                            QualityViolation::HallucinatedTool { tool_name }
                                if tool_name == &tc.name
                        )
                    }) {
                        continue;
                    }

                    // If retry requested, the file changed — break to re-read
                    if self.execute_and_push_tool_result(tc).await? {
                        break;
                    }
                }

                // F2.5: NOW push corrective user messages (after all tool results).
                for violation in &violations {
                    warn!(violation = ?violation, "Quality violation detected");
                    match violation {
                        QualityViolation::HallucinatedTool { tool_name } => {
                            let msg = format!(
                                "The tool '{tool_name}' does not exist. Available tools are: \
                                 Read, Edit, Write, Bash, Glob, Grep. Please use one of these tools."
                            );
                            self.push_message(Message::user(msg)).await;
                        }
                        QualityViolation::ToolLoop { tool_name, count } => {
                            let msg = format!(
                                "You have called {tool_name} {count} times in a row with the \
                                 same arguments. This appears to be a loop. Please try a \
                                 different approach or provide a text response."
                            );
                            self.push_message(Message::user(msg)).await;
                            // Force text response instead of continuing the loop
                            let text = self
                                .force_response(
                                    "Please provide a text response instead of repeating the same tool call.",
                                )
                                .await?;
                            self.save_episode(user_prompt, &text, true).await;
                            return Ok(());
                        }
                        QualityViolation::EmptyResponse => {
                            // Handled in the no-tool-calls branch below
                        }
                    }
                }

                return Ok(()); // Next iteration for tool results
            }
        }

        // No tool calls — extract text response
        let text = response.message.content.unwrap_or_default();

        // Push the assistant's response (even if empty) so the model sees it
        // in the conversation history and corrects on the next iteration.
        self.push_message(Message {
            role: "assistant".to_string(),
            content: Some(text.clone()),
            thinking: None,
            tool_calls: None,
            tool_call_id: None,
        })
        .await;

        // Fail-fast on consecutive empty responses (handoff §6):
        // push already done above so history stays consistent, no orphaned
        // LLM round-trips. Streak resets on any non-empty content or tool call.
        if text.trim().is_empty() {
            self.empty_response_streak = self.empty_response_streak.saturating_add(1);
            warn!(
                streak = self.empty_response_streak,
                "Empty response detected, streak={}", self.empty_response_streak
            );
            if should_fail_fast(self.empty_response_streak) {
                warn!(
                    streak = self.empty_response_streak,
                    "Empty response streak cap reached (2), failing fast"
                );
                self.save_episode(user_prompt, "no response (empty after 2 retries)", false)
                    .await;
                return Ok(());
            }
            warn!(
                streak = self.empty_response_streak,
                "Empty response detected, continuing loop for retry"
            );
            return Ok(());
        }
        // Non-empty text resets the streak
        self.empty_response_streak = 0;

        // Check if the text contains XML tool calls that weren't in tool_calls (with diagnostics)
        let (parsed, diagnostics) = self.parser.parse_with_diagnostics(&text);
        if !diagnostics.is_empty() && parsed.is_empty() {
            for diag in diagnostics {
                warn!(diagnostic = %diag, "Malformed tool call in text response");
                self.push_message(Message::user(diag)).await;
            }
            return Ok(());
        }
        if !parsed.is_empty() {
            // There are tool calls in the text, process them
            for tc in &parsed {
                self.execute_and_push_tool_result(tc).await?;
            }
            return Ok(()); // Process tool results next iteration
        }

        // Pure text response — we're done
        // (Assistant message already pushed above before the empty check)
        if let Some(nudge) = self.check_done_guard(&text) {
            self.push_message(Message::user(nudge)).await;
            return Ok(());
        }
        // Save successful episode to memory
        self.save_episode(user_prompt, &text, true).await;

        Ok(())
    }

    /// Execute one tool call and push its result message (shared by the
    /// native tool_calls path and the XML-in-text path).
    ///
    /// Applies the E3 deterministic guards around every execution:
    /// - doom-loop detection (3× identical → system nudge, 6× → fail-fast),
    /// - truncation-to-disk for oversized outputs.
    ///
    /// F2.2: tool errors become failed result messages, never aborts (except
    /// the doom-loop fail-fast, which is an explicit loop-level abort).
    /// Returns `true` when the caller should break and re-read (retry).
    async fn execute_and_push_tool_result(&mut self, tc: &ParsedToolCall) -> Result<bool> {
        // F2.2: Tool errors become failed results, never abort.
        // F4.3: Track invalid streak for schema breaker.
        let result = match self.registry.execute(tc).await {
            Ok(r) => {
                self.invalid_streak = 0;
                r
            }
            Err(e) => {
                // F4.3: increment invalid streak on NotFound/InvalidArgs.
                if matches!(
                    e,
                    crate::tools::ToolError::NotFound(_)
                        | crate::tools::ToolError::InvalidArgs { .. }
                ) {
                    self.invalid_streak += 1;
                }
                warn!(error = %e, name = tc.name, streak = self.invalid_streak, "Tool execution failed");
                crate::tools::ToolResult {
                    success: false,
                    output: format!("Tool '{}' failed: {e}", tc.name),
                    retry: false,
                }
            }
        };
        debug!(name = tc.name, success = result.success, "Tool executed");

        // E3 doom-loop guard — deterministic repetition detector. The result
        // message is pushed FIRST so history stays consistent even on the
        // fail-fast verdict (F2.5 ordering: results before corrective msgs).
        let doom = self.doom_loop.record(&tc.name, &tc.arguments);

        // E3 truncation-to-disk: oversized outputs are persisted in full and
        // replaced by a head+tail excerpt with a pointer to the file.
        // Tier 3 read-guard: cap at min(fixed 50KB, remaining-budget-derived
        // cap) so a big tool result cannot overflow the window between
        // compaction checks.
        let budget_cap = effective_output_cap(self.budget.remaining());
        let output = truncate_tool_output(result.output, TRUNCATION_MAX_BYTES.min(budget_cap));

        self.push_message(Message {
            role: "tool".to_string(),
            content: Some(output),
            thinking: None,
            tool_calls: None,
            // Thread the server-issued id so backends correlate the result;
            // text-parsed calls carry None (fallback documented per backend).
            tool_call_id: tc.id.clone(),
        })
        .await;

        match doom {
            DoomLoopVerdict::Fail => {
                anyhow::bail!(
                    "Doom loop: tool '{}' executed {}× consecutively with identical arguments",
                    tc.name,
                    DOOM_LOOP_FAIL_AT
                );
            }
            DoomLoopVerdict::Nudge => {
                warn!(
                    name = tc.name,
                    streak = self.doom_loop.streak,
                    "Doom loop detected, injecting corrective nudge"
                );
                self.push_message(Message {
                    role: "system".to_string(),
                    content: Some(DOOM_LOOP_NUDGE_MSG.to_string()),
                    thinking: None,
                    tool_calls: None,
                    tool_call_id: None,
                })
                .await;
            }
            DoomLoopVerdict::Fresh => {}
        }

        Ok(result.retry)
    }

    /// Count tokens of `text` using the model's real tokenizer when the
    /// Ollama server is reachable, falling back to the local heuristic.
    pub(super) fn run_bootstrap(&mut self) {
        if self.bootstrap_done {
            return;
        }
        let result = self.bootstrap.detect();
        if !result.hint.is_empty() {
            info!(hint = %result.hint, "Bootstrap detection complete");
        }
        self.bootstrap_done = true;
    }

    /// Inject knowledge at the conversation tail if relevant cards are found.
    pub(super) async fn inject_knowledge(&mut self, user_prompt: &str) {
        if !self.knowledge_base.is_enabled() {
            return;
        }

        if let Some(injection) = self.knowledge_base.build_injection(user_prompt) {
            debug!(
                cards = self.knowledge_base.card_count(),
                "Knowledge injection prepared"
            );
            // F2.3: Use push_message for token accounting (not manual consume).
            self.messages.push(Message {
                role: "user".to_string(),
                content: Some(injection.clone()),
                thinking: None,
                tool_calls: None,
                tool_call_id: None,
            });
            let tokens = self.count_tokens(&injection).await;
            self.budget.consume(tokens);
        }
    }

    /// Recall top-3 past episodes by goal similarity and inject at conversation tail.
    pub(super) async fn recall_episodes(&mut self, goal: &str) {
        // Fase 3.1: SQLite recall runs on the blocking pool so the async
        // executor never parks on rusqlite. The store is taken and returned —
        // the agent owns it exclusively and the task flow is sequential.
        let recalled = if let Some(store) = self.memory_store.take() {
            let goal = goal.to_string();
            let (store, result) = tokio::task::spawn_blocking(move || {
                let result = store.recall_similar(&goal, 3);
                (Some(store), result)
            })
            .await
            .unwrap_or((None, Ok(Vec::new())));
            if let Some(store) = store {
                self.memory_store = Some(store);
            }
            result
        } else {
            Ok(Vec::new())
        };
        if let Ok(episodes) = recalled {
            if !episodes.is_empty() {
                let context = episodes
                    .iter()
                    .map(|ep| {
                        format!(
                            "- Goal: {}\n  Outcome: {}\n  Files: {:?}",
                            ep.goal, ep.outcome, ep.files_touched
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let injection = format!("[Past Episodes]\n{}\n[End Past Episodes]\n", context);
                debug!(count = episodes.len(), "Episodic memory recall");
                // F2.3: Use push_message for token accounting.
                self.messages.push(Message {
                    role: "user".to_string(),
                    content: Some(injection.clone()),
                    thinking: None,
                    tool_calls: None,
                    tool_call_id: None,
                });
                let tokens = self.count_tokens(&injection).await;
                self.budget.consume(tokens);
            }
        }
    }

    /// Save an episode after task completion (best-effort).
    /// Fase 3.1: the blocking SQLite write runs on the blocking pool; best
    /// effort — never fails the agent on memory errors.
    pub(super) async fn save_episode(&mut self, goal: &str, outcome: &str, success: bool) {
        if let Some(store) = self.memory_store.take() {
            let files_touched = self.list_workspace_files();
            let ep = Episode::new(
                goal.to_string(),
                files_touched,
                Some(if success { "pass" } else { "fail" }.into()),
                outcome.to_string(),
            );
            let (store, result) = tokio::task::spawn_blocking(move || {
                let result = store.save(&ep);
                (Some(store), result)
            })
            .await
            .unwrap_or((None, Ok(())));
            if let Some(store) = store {
                self.memory_store = Some(store);
            }
            let _ = result;
        }
    }

    /// List top-level files in the workspace directory.
    fn list_workspace_files(&self) -> Vec<String> {
        let mut files = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.config.workspace) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    files.push(name.to_string());
                }
            }
        }
        files
    }

    /// Preflight: inject workspace context + model stance overlay.
    ///
    /// The system message leads the conversation. When the model id contains
    /// "qwen", the E3 [`QWEN_STANCE_OVERLAY`] is appended (or emitted alone)
    /// so Qwen-family models act instead of hedging.
    ///
    /// f8 T4.1: profiles with [`PromptStyle::Minimal`] (nano tier) get a
    /// compact imperative contract instead — terse identity, one-action-
    /// per-turn, read-before-edit and smallest-unique-span guidance, plus at
    /// most the project-type labels. Verbose bootstrap command hints and the
    /// stance overlay are skipped (the Minimal text is already imperative).
    pub(super) fn preflight(&mut self, _user_prompt: &str) {
        if self.config.profile.prompt_style == PromptStyle::Minimal {
            let mut system_msg = String::from(MINIMAL_RUNTIME);
            if let Some(result) = self.bootstrap.result() {
                let labels = result
                    .project_types
                    .iter()
                    .map(|p| p.label())
                    .collect::<Vec<_>>()
                    .join(", ");
                if !labels.is_empty() {
                    system_msg.push_str(&format!("\nWorkspace type: {labels}."));
                }
            }
            self.messages.push(Message {
                role: "system".to_string(),
                content: Some(system_msg),
                thinking: None,
                tool_calls: None,
                tool_call_id: None,
            });
            return;
        }

        let mut system_msg = String::new();
        if let Some(result) = self.bootstrap.result() {
            if !result.hint.is_empty() {
                system_msg = format!(
                    "Workspace detected: {}. Available build commands: {}",
                    result
                        .project_types
                        .iter()
                        .map(|p| p.label())
                        .collect::<Vec<_>>()
                        .join(", "),
                    result
                        .project_types
                        .iter()
                        .flat_map(|p| p.build_commands())
                        .collect::<Vec<_>>()
                        .join("; ")
                );
            }
        }
        if let Some(overlay) = model_stance_overlay(&self.config.model) {
            if !system_msg.is_empty() {
                system_msg.push_str("\n\n");
            }
            system_msg.push_str(overlay);
        }
        if !system_msg.is_empty() {
            // Insert as first system context
            self.messages.push(Message {
                role: "system".to_string(),
                content: Some(system_msg),
                thinking: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }
    }

    /// Retry the LLM call with thinking disabled (F2.6).
    ///
    /// Instead of returning raw text, this re-calls the LLM with thinking
    /// disabled and returns the full `ChatResponse` so the caller can process
    /// tool calls and quality checks normally.
    pub(super) async fn llm_call_no_thinking(&self) -> Result<ChatResponse> {
        info!("Retrying LLM call with thinking disabled");
        // B1-400: sanitize the cloned history before wiring (tail-only, idempotent).
        let mut messages = sanitize_trailing_assistants(self.messages.clone());
        messages.insert(
            0,
            Message {
                role: "system".to_string(),
                content: Some(
                    "Do not use thinking/reasoning mode. Respond directly with text or tool calls."
                        .to_string(),
                ),
                thinking: None,
                tool_calls: None,
                tool_call_id: None,
            },
        );

        let mut options = self.request_options(&messages);
        // F2.6 hard guarantee: the retry must not think again, regardless of
        // model-id/env heuristics — force `"think": false` on this request.
        options.think = Some(false);
        let options = Some(options);
        self.client
            .chat_with_options(
                &self.config.model,
                messages,
                Some(self.tools.clone()),
                options,
            )
            .await
            .context("Retry LLM call without thinking failed")
    }

    /// E3 dynamic max-output cap + F2 per-model sampling/thinking:
    /// request only as much output as still fits the context window
    /// (`context_limit − estimated_input − 1024`, floor 256), never more than
    /// the profile's `num_predict`; sampling parameters and the thinking
    /// override come from the resolved per-model tuning. Input estimated
    /// conservatively at chars/4 via
    /// [`kitty_runtime::client::estimate_input_tokens`]. The session-fixed
    /// `num_ctx` is injected by [`kitty_runtime::OllamaClient`] itself.
    pub(super) fn request_options(
        &self,
        messages: &[Message],
    ) -> kitty_runtime::client::RequestOptions {
        let est = kitty_runtime::client::estimate_input_tokens(messages);
        let dynamic = i64::from(kitty_runtime::client::dynamic_max_output(
            self.config.context_window,
            est,
        ));
        let num_predict = i32::try_from(dynamic.min(i64::from(self.config.tuning.num_predict)))
            .unwrap_or(kitty_runtime::client::MIN_OUTPUT_TOKENS);
        kitty_runtime::client::RequestOptions {
            num_predict: Some(num_predict),
            temperature: Some(self.config.tuning.temperature),
            top_p: self.config.tuning.top_p,
            top_k: self.config.tuning.top_k.map(|k| k as i32),
            think: if self.config.tuning.thinking == ThinkingMode::Off {
                Some(false)
            } else {
                None
            },
            ..Default::default()
        }
    }

    /// Make an LLM call with bounded retry for transient errors (F3.3).
    ///
    /// Retries on transient failures (rate limits, 5xx, connection resets) with
    /// exponential backoff. Non-transient errors (bad request, auth, etc.) fail
    /// immediately. Maximum 3 retries with 500ms → 1s → 2s backoff.
    pub(super) async fn llm_call(&self) -> Result<ChatResponse> {
        // B1-400: transport sanitizer — never wire a trailing [assistant, assistant] (Ollama 400).
        let messages = sanitize_trailing_assistants(self.messages.clone());
        let model = &self.config.model;
        let tools = Some(self.tools.clone());
        let options = Some(self.request_options(&messages));

        const MAX_RETRIES: usize = 3;
        let mut last_err: Option<anyhow::Error> = None;

        for attempt in 0..=MAX_RETRIES {
            match self
                .client
                .chat_with_options(model, messages.clone(), tools.clone(), options.clone())
                .await
            {
                Ok(resp) => return Ok(resp),
                Err(e) if e.is_retryable() && attempt < MAX_RETRIES => {
                    let backoff = Duration::from_millis(500 * 2u64.pow(attempt as u32));
                    warn!(
                        attempt,
                        backoff_ms = backoff.as_millis() as u64,
                        error = %e,
                        "Transient LLM error, retrying"
                    );
                    tokio::time::sleep(backoff).await;
                    last_err = Some(e.into());
                }
                Err(e) => return Err(e).context("LLM call failed"),
            }
        }
        Err(last_err.unwrap())
    }

    /// Force the model to respond with text (no tools).
    pub(super) async fn force_response(&self, instruction: &str) -> Result<String> {
        // B1-400: sanitize the base history tail BEFORE pushing the forcing user turn
        // (keeps interior duplicates intact; wire ends with user so stays valid).
        let mut messages = sanitize_trailing_assistants(self.messages.clone());
        messages.push(Message {
            role: "user".to_string(),
            content: Some(instruction.to_string()),
            thinking: None,
            tool_calls: None,
            tool_call_id: None,
        });

        // Call without tools to force text response
        let options = Some(self.request_options(&messages));
        let response = self
            .client
            .chat_with_options(
                &self.config.model,
                messages,
                None, // No tools — forces text
                options,
            )
            .await
            .context("Force response LLM call failed")?;

        Ok(response
            .message
            .content
            .unwrap_or_else(|| "No response generated.".to_string()))
    }

    /// Trigger context compaction: summarize old messages and free tokens.
    ///
    /// Uses [`CompactionSummary`] for structured output — the LLM provides
    /// prose, and file operations are extracted deterministically from tool
    /// results to give small models a clear session-state snapshot.
    pub(super) async fn trigger_compaction(&mut self) -> Result<()> {
        info!("Triggering context compaction");

        // Split messages: keep recent ones, summarize old ones.
        let len = self.messages.len();
        let keep_count = 4.min(len);
        let naive_split_at = len.saturating_sub(keep_count);

        if naive_split_at == 0 {
            // Nothing left to compact (e.g. context full from a single huge
            // message). Register the no-op so the consecutive-compaction halt
            // guard can engage instead of spinning until max_iterations.
            let _ = self.budget.record_compaction(0, 1);
            return Ok(());
        }

        // f8 AD6: move the boundary back to the nearest USER turn so tool
        // results are never orphaned and kept turns stay complete.
        let final_split = compaction_split_boundary(&self.messages, naive_split_at);

        // m7 accounting consistency: freed_text AND summary_source BOTH
        // derive from the SAME final boundary, computed BEFORE the drain.
        let freed_text: String = self.messages[..final_split]
            .iter()
            .filter_map(|m| m.content.as_deref())
            .collect::<Vec<_>>()
            .join("\n");

        // E3 deterministic prune: everything drained gets its tool-output
        // content erased BEFORE the summarizer LLM sees it — survives even a
        // weak/dumb summarizer model, because the erasure is deterministic,
        // not model-dependent.
        let summary_source: String = self.messages[..final_split]
            .iter()
            .map(|m| {
                if m.role == "tool" {
                    "[tool output pruned]"
                } else {
                    m.content.as_deref().unwrap_or("")
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        // P2: extract file operations deterministically from tool results
        let tool_outputs: Vec<String> = self.messages[..final_split]
            .iter()
            .filter_map(|m| {
                if m.role == "tool" {
                    m.content.clone()
                } else {
                    None
                }
            })
            .collect();
        let (read_files, modified_files) =
            crate::compaction::extract_file_operations(&tool_outputs);

        let old_messages: Vec<Message> = self.messages.drain(..final_split).collect();
        debug_assert_eq!(old_messages.len(), final_split);

        let summary_prompt = format!(
            "Summarize the key decisions, code changes, and context from this conversation \
             segment in at most 280 words of plain prose. Be concise — only include \
             information needed for ongoing work:\n\n{}",
            summary_source
        );

        // Ask the model to summarize
        let summary_messages = vec![Message {
            role: "user".to_string(),
            content: Some(summary_prompt),
            thinking: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        // f8 AD4: summarization is mechanical — override ONLY the sampling
        // fields for a cheap, near-deterministic, bounded reply. num_ctx is
        // intentionally NOT set here: the session window is client-frozen
        // (G2-FIX4 guard traps caller injection in debug builds).
        let mut options = self.request_options(&summary_messages);
        options.temperature = Some(0.1);
        options.num_predict = Some(400);
        let summary = self
            .client
            .chat_with_options(&self.config.model, summary_messages, None, Some(options))
            .await
            .context("Compaction summary LLM call failed")?;

        let summary_text = summary
            .message
            .content
            .unwrap_or_else(|| "[compaction summary unavailable]".to_string());

        // P2: build structured summary with file operations
        let structured = crate::compaction::CompactionSummary {
            goal: summary_text,
            read_files,
            modified_files,
            ..crate::compaction::CompactionSummary::empty()
        };
        let formatted_summary = structured.format_markdown();

        let freed = ContextBudget::estimate_tokens(&freed_text);
        let needed = ContextBudget::estimate_tokens(&formatted_summary) + 100;

        // Record compaction
        if let Err(ContextError::InsufficientCompaction {
            freed: f,
            needed: n,
        }) = self.budget.record_compaction(freed, needed)
        {
            warn!(
                freed = f,
                needed = n,
                "Compaction freed insufficient tokens"
            );
        }

        // Insert summary as system message at the start
        self.messages.insert(
            0,
            Message {
                role: "system".to_string(),
                content: Some(format!("[Context Summary]\n{formatted_summary}")),
                thinking: None,
                tool_calls: None,
                tool_call_id: None,
            },
        );

        Ok(())
    }
}

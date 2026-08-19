//! The core agent loop.
//!
//! Orchestrates: preflight → LLM call → parse tool calls → execute tools → post-loop.
//! Handles multi-turn tool call chains, context budget management,
//! thinking budget caps, quality monitoring, knowledge injection, and bootstrap detection.

use anyhow::{Context, Result};
use kitty_memory::{Episode, EpisodeStore};
use kitty_runtime::{ChatResponse, Message, OllamaClient, ToolDefinition};
use tracing::{debug, info, warn};

use crate::bootstrap::Bootstrap;
use crate::context::{ContextBudget, ContextError};
use crate::knowledge::{KnowledgeBase, KnowledgeConfig};
use crate::parser::{ParsedToolCall, ToolCallParser};
use crate::quality_monitor::{QualityConfig, QualityMonitor, QualityViolation};
use crate::thinking_budget::ThinkingBudget;
use crate::tools::ToolRegistry;

/// Configuration for the agent.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Model name to use.
    pub model: String,
    /// Maximum number of tool-call iterations before forcing a text response.
    pub max_iterations: usize,
    /// Context window size in tokens.
    pub context_window: usize,
    /// Compaction threshold (0.0-1.0). Default: 0.8.
    pub compact_threshold: f32,
    /// Maximum reasoning tokens (thinking budget cap). None = no cap.
    pub max_thinking_tokens: Option<usize>,
    /// Whether knowledge injection is enabled.
    pub enable_knowledge: bool,
    /// Knowledge configuration.
    pub knowledge_config: KnowledgeConfig,
    /// Quality monitor configuration.
    pub quality_config: QualityConfig,
    /// Workspace directory for bootstrap detection.
    pub workspace: std::path::PathBuf,
    /// Path to episodic memory database. None = memory disabled.
    pub memory_db_path: Option<String>,
    /// Bash command timeout in seconds (F1.3). Default: 60.
    pub bash_timeout_secs: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: "qwen3:8b".to_string(),
            max_iterations: 20,
            context_window: 16384,
            compact_threshold: 0.8,
            max_thinking_tokens: Some(2048),
            enable_knowledge: false,
            knowledge_config: KnowledgeConfig::default(),
            quality_config: QualityConfig::default(),
            workspace: std::path::PathBuf::from("."),
            memory_db_path: None,
            bash_timeout_secs: 60,
        }
    }
}

/// Events emitted by the agent loop for observability.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// LLM call started.
    LlmCall { iteration: usize },
    /// Tool call being executed.
    ToolCall {
        name: String,
        arguments: String,
    },
    /// Tool execution result.
    ToolResult {
        name: String,
        success: bool,
        output_preview: String,
    },
    /// Context compaction triggered.
    Compaction { freed: usize, summary_preview: String },
    /// Iteration limit reached.
    IterationLimit { iterations: usize },
    /// Agent finished with a text response.
    Done { response_preview: String },
    /// Thinking budget exceeded, retrying without thinking.
    ThinkingBudgetRetry { turn_tokens: usize },
    /// Quality violation detected.
    QualityViolation { violation: QualityViolation },
    /// Bootstrap detection result.
    Bootstrap { hint: String },
}

/// The core agent loop.
pub struct Agent {
    client: OllamaClient,
    config: AgentConfig,
    messages: Vec<Message>,
    tools: Vec<ToolDefinition>,
    budget: ContextBudget,
    registry: ToolRegistry,
    parser: ToolCallParser,
    thinking_budget: ThinkingBudget,
    quality_monitor: QualityMonitor,
    knowledge_base: KnowledgeBase,
    bootstrap: Bootstrap,
    /// Whether bootstrap has run for the current session.
    bootstrap_done: bool,
    /// Episodic memory store. None when memory is disabled.
    memory_store: Option<EpisodeStore>,
}

impl Agent {
    /// Create a new agent with the given config.
    pub fn new(client: OllamaClient, config: AgentConfig) -> Self {
        let tools = crate::tools::standard_tool_definitions();
        Self::new_inner(client, config, tools)
    }

    /// Create with custom tool definitions.
    pub fn with_tools(
        client: OllamaClient,
        config: AgentConfig,
        tools: Vec<ToolDefinition>,
    ) -> Self {
        Self::new_inner(client, config, tools)
    }

    /// Shared constructor: wires budget, monitors, knowledge, bootstrap and
    /// memory from `config` (F4.1 — single implementation, no duplication).
    fn new_inner(
        client: OllamaClient,
        config: AgentConfig,
        tools: Vec<ToolDefinition>,
    ) -> Self {
        let budget = ContextBudget::new(config.context_window);
        let tool_names: Vec<String> = tools.iter().map(|t| t.function.name.clone()).collect();
        let thinking_budget = ThinkingBudget::new(
            config.max_thinking_tokens,
            None, // No session cap by default
        );
        let quality_monitor = QualityMonitor::new(config.quality_config.clone(), tool_names);
        let knowledge_base = KnowledgeBase::new(config.knowledge_config.clone());
        let bootstrap = Bootstrap::new(&config.workspace);
        let memory_store = config.memory_db_path.as_ref().and_then(|path| {
            if let Some(dir) = std::path::Path::new(path).parent() {
                std::fs::create_dir_all(dir).ok();
            }
            EpisodeStore::open(path).ok()
        });
        // Read before moving `config` into the struct (no Copy).
        let bash_timeout_secs = config.bash_timeout_secs;

        Self {
            client,
            config,
            messages: Vec::new(),
            tools,
            budget,
            registry: ToolRegistry::with_timeout(bash_timeout_secs),
            parser: ToolCallParser::new(),
            thinking_budget,
            quality_monitor,
            knowledge_base,
            bootstrap,
            bootstrap_done: false,
            memory_store,
        }
    }

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
        // Bootstrap: detect workspace on first turn
        self.run_bootstrap();

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
        })
        .await;

        // Reset thinking budget for this turn
        self.thinking_budget.reset_turn();

        // Main loop: iterate until we get a text response
        for iteration in 0..self.config.max_iterations {
            debug!(iteration, "Agent loop iteration");

            // Check context budget before LLM call
            if self.budget.should_compact() {
                self.trigger_compaction().await?;
            }

            if self.budget.should_halt() {
                warn!("Context halting: compaction exhausted");
                return self
                    .force_response("Context limit reached. Please start a new conversation.")
                    .await;
            }

            // LLM call
            let response = self.llm_call().await?;

            // Extract thinking tokens for budget tracking (thinking is metadata,
            // not a conversation message — tracked via explicit budget.consume).
            if let Some(thinking) = response.message.thinking.as_deref() {
                let tokens = self.count_tokens(thinking).await;
                // Apply thinking budget cap
                if let Err(e) = self.thinking_budget.record(tokens) {
                    warn!(error = %e, "Thinking budget exceeded, retrying without thinking");
                    // F2.6: Retry without thinking — re-call LLM, then fall through
                    // to normal processing (tool calls and all). Max 1 retry per turn.
                    let retry_response = self.llm_call_no_thinking().await?;
                    self.process_response(user_prompt, retry_response).await?;
                    // Don't reset thinking budget — we already retried once
                    continue;
                }
                // Thinking tokens are consumed against budget but not pushed as
                // a message — explicit consume here is correct (F2.3 exception).
                self.budget.consume(tokens);
            }

            self.process_response(user_prompt, response).await?;

            // Reset thinking budget for next turn
            self.thinking_budget.reset_turn();
            // F2.4: Do NOT clear the quality monitor window — it persists
            // across turns to detect cross-turn tool loops.
        }

        // Exhausted iterations — force a final response
        warn!(
            iterations = self.config.max_iterations,
            "Agent loop exhausted iterations"
        );
        let text = self
            .force_response("Please provide your final answer now.")
            .await?;
        // The loop did not finish naturally — record the episode as a failure
        // so memory reflects reality (otherwise only "pass" episodes accumulate).
        self.save_episode(user_prompt, &text, false);
        Ok(text)
    }

    /// Process a single LLM response within the agent loop.
    ///
    /// Handles tool execution, quality checks, and message ordering.
    /// Returns Ok(true) when a final text response was produced, Ok(false) to
    /// continue the loop (tool calls were processed).
    async fn process_response(
        &mut self,
        user_prompt: &str,
        response: ChatResponse,
    ) -> Result<()> {
        // Check for tool calls
        if let Some(tool_calls) = &response.message.tool_calls {
            if !tool_calls.is_empty() {
                // Add assistant message with tool calls (F2.3: push handles tokens)
                self.push_message(response.message.clone()).await;

                // Also parse text content for XML-style tool calls
                let mut all_tool_calls: Vec<ParsedToolCall> = Vec::new();

                // Native tool calls from Ollama
                for tc in tool_calls {
                    all_tool_calls.push(ParsedToolCall {
                        name: tc.function.name.clone(),
                        arguments: tc.function.arguments.clone(),
                    });
                }

                // Also parse content for XML-style tool calls
                if let Some(content) = response.message.content.as_deref() {
                    let parsed = self.parser.parse(content);
                    all_tool_calls.extend(parsed);
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
                    if violations.iter().any(|v| matches!(
                        v,
                        QualityViolation::HallucinatedTool { tool_name }
                            if tool_name == &tc.name
                    )) {
                        continue;
                    }

                    // F2.2: Tool errors become failed results, never abort.
                    let result = match self.registry.execute(tc).await {
                        Ok(r) => r,
                        Err(e) => {
                            warn!(error = %e, name = tc.name, "Tool execution failed");
                            crate::tools::ToolResult {
                                success: false,
                                output: format!("Tool '{}' failed: {e}", tc.name),
                                retry: false,
                            }
                        }
                    };
                    debug!(name = tc.name, success = result.success, "Tool executed");

                    // Push tool result (F2.3: push handles tokens)
                    self.push_message(Message {
                        role: "tool".to_string(),
                        content: Some(result.output.clone()),
                        thinking: None,
                        tool_calls: None,
                    })
                    .await;

                    // If retry requested, the file changed — break to re-read
                    if result.retry {
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
                            self.save_episode(user_prompt, &text, true);
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

        // Check for empty response
        let violations = self.quality_monitor.check_response(Some(&text), &[]);
        if violations.contains(&QualityViolation::EmptyResponse) {
            warn!("Empty response detected, forcing retry");
            self.force_response("Your response was empty. Please provide text content.")
                .await?;
            return Ok(());
        }

        // Check if the text contains XML tool calls that weren't in tool_calls
        let parsed = self.parser.parse(&text);
        if !parsed.is_empty() {
            // There are tool calls in the text, process them
            self.push_message(Message {
                role: "assistant".to_string(),
                content: Some(text.clone()),
                thinking: None,
                tool_calls: None,
            })
            .await;

            for tc in &parsed {
                // F2.2: Handle tool errors gracefully
                let result = match self.registry.execute(tc).await {
                    Ok(r) => r,
                    Err(e) => {
                        warn!(error = %e, name = tc.name, "Tool execution failed (XML)");
                        crate::tools::ToolResult {
                            success: false,
                            output: format!("Tool '{}' failed: {e}", tc.name),
                            retry: false,
                        }
                    }
                };
                self.push_message(Message {
                    role: "tool".to_string(),
                    content: Some(result.output.clone()),
                    thinking: None,
                    tool_calls: None,
                })
                .await;
            }
            return Ok(()); // Process tool results next iteration
        }

        // Pure text response — we're done
        self.push_message(Message {
            role: "assistant".to_string(),
            content: Some(text.clone()),
            thinking: None,
            tool_calls: None,
        })
        .await;

        // Save successful episode to memory
        self.save_episode(user_prompt, &text, true);

        Ok(())
    }

    /// Count tokens of `text` using the model's real tokenizer when the
    /// Ollama server is reachable, falling back to the local heuristic.
    async fn count_tokens(&self, text: &str) -> usize {
        match self.client.count_tokens(&self.config.model, text).await {
            Ok(n) => n,
            Err(e) => {
                debug!(error = %e, "tokenize failed; using heuristic estimate");
                ContextBudget::estimate_tokens(text)
            }
        }
    }

    /// Run bootstrap detection (cached, runs once).
    fn run_bootstrap(&mut self) {
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
    async fn inject_knowledge(&mut self, user_prompt: &str) {
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
            });
            let tokens = self.count_tokens(&injection).await;
            self.budget.consume(tokens);
        }
    }

    /// Recall top-3 past episodes by goal similarity and inject at conversation tail.
    async fn recall_episodes(&mut self, goal: &str) {
        if let Some(ref store) = self.memory_store {
            if let Ok(episodes) = store.recall_similar(goal, 3) {
                if !episodes.is_empty() {
                    let context = episodes.iter().map(|ep| {
                        format!(
                            "- Goal: {}\n  Outcome: {}\n  Files: {:?}",
                            ep.goal, ep.outcome, ep.files_touched
                        )
                    }).collect::<Vec<_>>().join("\n");
                    let injection = format!(
                        "[Past Episodes]\n{}\n[End Past Episodes]\n", context
                    );
                    debug!(count = episodes.len(), "Episodic memory recall");
                    // F2.3: Use push_message for token accounting.
                    self.messages.push(Message {
                        role: "user".to_string(),
                        content: Some(injection.clone()),
                        thinking: None,
                        tool_calls: None,
                    });
                    let tokens = self.count_tokens(&injection).await;
                    self.budget.consume(tokens);
                }
            }
        }
    }

    /// Save an episode after task completion (best-effort).
    fn save_episode(&self, goal: &str, outcome: &str, success: bool) {
        if let Some(ref store) = self.memory_store {
            let files_touched = self.list_workspace_files();
            let ep = Episode::new(
                goal.to_string(),
                files_touched,
                Some(if success { "pass" } else { "fail" }.into()),
                outcome.to_string(),
            );
            store.save(&ep).ok(); // Best effort — don't fail the agent on memory errors
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

    /// Preflight: inject workspace context
    fn preflight(&mut self, _user_prompt: &str) {
        if let Some(result) = self.bootstrap.result() {
            if !result.hint.is_empty() {
                let system_msg = format!(
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
                // Insert as first system context
                self.messages.push(Message {
                    role: "system".to_string(),
                    content: Some(system_msg),
                    thinking: None,
                    tool_calls: None,
                });
            }
        }
    }

    /// Retry the LLM call with thinking disabled (F2.6).
    ///
    /// Instead of returning raw text, this re-calls the LLM with thinking
    /// disabled and returns the full `ChatResponse` so the caller can process
    /// tool calls and quality checks normally.
    async fn llm_call_no_thinking(&self) -> Result<ChatResponse> {
        info!("Retrying LLM call with thinking disabled");
        let mut messages = self.messages.clone();
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
            },
        );

        self.client
            .chat(
                &self.config.model,
                messages,
                Some(self.tools.clone()),
                None,
            )
            .await
            .context("Retry LLM call without thinking failed")
    }

    /// Make an LLM call and handle the response.
    async fn llm_call(&self) -> Result<ChatResponse> {
        self.client
            .chat(
                &self.config.model,
                self.messages.clone(),
                Some(self.tools.clone()),
                None,
            )
            .await
            .context("LLM call failed")
    }

    /// Force the model to respond with text (no tools).
    async fn force_response(&self, instruction: &str) -> Result<String> {
        let mut messages = self.messages.clone();
        messages.push(Message {
            role: "user".to_string(),
            content: Some(instruction.to_string()),
            thinking: None,
            tool_calls: None,
        });

        // Call without tools to force text response
        let response = self
            .client
            .chat(
                &self.config.model,
                messages,
                None, // No tools — forces text
                None,
            )
            .await
            .context("Force response LLM call failed")?;

        Ok(response
            .message
            .content
            .unwrap_or_else(|| "No response generated.".to_string()))
    }

    /// Trigger context compaction: summarize old messages and free tokens.
    async fn trigger_compaction(&mut self) -> Result<()> {
        info!("Triggering context compaction");

        // Split messages: keep recent ones, summarize old ones
        let keep_count = 4.min(self.messages.len());
        let split_at = self.messages.len().saturating_sub(keep_count);

        if split_at == 0 {
            // Nothing left to compact (e.g. context full from a single huge
            // message). Register the no-op so the consecutive-compaction halt
            // guard can engage instead of spinning until max_iterations.
            let _ = self.budget.record_compaction(0, 1);
            return Ok(());
        }

        let old_messages: Vec<Message> = self.messages.drain(..split_at).collect();

        // Build a summary prompt
        let old_text: String = old_messages
            .iter()
            .filter_map(|m| m.content.as_deref())
            .collect::<Vec<_>>()
            .join("\n");

        let summary_prompt = format!(
            "Summarize the key decisions, code changes, and context from this conversation \
             segment. Be concise — only include information needed for ongoing work:\n\n{}",
            old_text
        );

        // Ask the model to summarize
        let summary = self
            .client
            .chat(
                &self.config.model,
                vec![Message {
                    role: "user".to_string(),
                    content: Some(summary_prompt),
                    thinking: None,
                    tool_calls: None,
                }],
                None,
                None,
            )
            .await
            .context("Compaction summary LLM call failed")?;

        let summary_text = summary
            .message
            .content
            .unwrap_or_else(|| "[compaction summary unavailable]".to_string());

        let freed = ContextBudget::estimate_tokens(&old_text);
        let needed = ContextBudget::estimate_tokens(&summary_text) + 100;

        // Record compaction
        if let Err(ContextError::InsufficientCompaction { freed: f, needed: n }) =
            self.budget.record_compaction(freed, needed)
        {
            warn!(freed = f, needed = n, "Compaction freed insufficient tokens");
        }

        // Insert summary as system message at the start
        self.messages.insert(
            0,
            Message {
                role: "system".to_string(),
                content: Some(format!("[Context Summary]\n{summary_text}")),
                thinking: None,
                tool_calls: None,
            },
        );

        Ok(())
    }

    /// Push a message and track token usage (F2.3 — sole accounting point).
    ///
    /// Uses real tokenization when the server is reachable; falls back to the
    /// heuristic (chars/4). Every token consumed by the conversation flows
    /// through this one call — no separate `budget.consume()` elsewhere.
    async fn push_message(&mut self, msg: Message) {
        if let Some(content) = msg.content.as_deref() {
            let tokens = self.count_tokens(content).await;
            self.budget.consume(tokens);
        }
        self.messages.push(msg);
    }

    /// Get the current message history.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Get the context budget status.
    pub fn budget(&self) -> &ContextBudget {
        &self.budget
    }

    /// Get the thinking budget tracker.
    pub fn thinking_budget(&self) -> &ThinkingBudget {
        &self.thinking_budget
    }

    /// Get the quality monitor.
    pub fn quality_monitor(&self) -> &QualityMonitor {
        &self.quality_monitor
    }

    /// Get the bootstrap result.
    pub fn bootstrap_result(&self) -> Option<&crate::bootstrap::BootstrapResult> {
        self.bootstrap.result()
    }
}

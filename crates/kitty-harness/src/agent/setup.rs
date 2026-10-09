//! Agent construction (W2 split of `agent/mod.rs`, por símbolo).
//!
//! [`Agent`] struct + constructors. The run/response loops live in
//! `super::run` / `super::response` and see fields via `pub(crate)`.
use kitty_memory::EpisodeStore;
use kitty_runtime::{ChatClient, GrammarClient, GrammarConfig, Message, ToolDefinition};
use tracing::debug;

use super::react::ReActStep;
use super::{
    clamp_autorun_hops, per_turn_thinking_cap, should_fail_fast, AgentConfig, DoomLoopGuard,
};
use crate::bootstrap::Bootstrap;
use crate::context::ContextBudget;
use crate::contract::{contract_guard_enabled, looks_like_done_claim, Contract};
use crate::knowledge::KnowledgeBase;
use crate::model_profile::ModelTuning;
use crate::parser::ToolCallParser;
use crate::quality_monitor::QualityMonitor;
use crate::router::ToolRouter;
use crate::tfidf_closer::{CloserConfig, CloserLabel, TfidfCloser};
use crate::thinking_budget::ThinkingBudget;
use crate::tools::ToolRegistry;
/// The core agent loop.
pub struct Agent {
    pub(crate) client: Box<dyn ChatClient>,
    pub(crate) config: AgentConfig,
    pub(crate) messages: Vec<Message>,
    pub(crate) tools: Vec<ToolDefinition>,
    pub(crate) budget: ContextBudget,
    pub(crate) registry: ToolRegistry,
    pub(crate) parser: ToolCallParser,
    pub(crate) thinking_budget: ThinkingBudget,
    pub(crate) quality_monitor: QualityMonitor,
    pub(crate) knowledge_base: KnowledgeBase,
    pub(crate) bootstrap: Bootstrap,
    /// Whether bootstrap has run for the current session.
    pub(crate) bootstrap_done: bool,
    /// Episodic memory store. None when memory is disabled.
    pub(crate) memory_store: Option<EpisodeStore>,
    /// Consecutive empty responses in the current turn (fail-fast §6).
    pub(crate) empty_response_streak: u8,
    /// E3 deterministic doom-loop detector over executed tools.
    pub(crate) doom_loop: DoomLoopGuard,
    /// W7 ReAct trace: one [`ReActStep`] per executed tool, in loop order.
    /// Ephemeral per run (cleared in `run_impl`); thoughts never persisted.
    pub(crate) react_steps: Vec<ReActStep>,
    /// 2-stage tool router (affirmation guard, stateful).
    pub(crate) router: ToolRouter,
    pub(crate) contract: Option<Contract>,
    /// Consecutive invalid tool calls (NotFound, InvalidArgs) — breaker mechanism (F4.3).
    pub(crate) invalid_streak: usize,
    /// W1B: hops consumidos en el último `run` (para `BenchResult.hops`).
    pub(crate) last_run_hops_used: usize,
    /// W1B: iteraciones consumidas en el último `run` (`BenchResult.iters`).
    pub(crate) last_run_iters_used: usize,
    /// W1B: el `EffortRouter` quedó OFF en el último `run` (bench o
    /// kill-switch; alimenta `BenchResult.effort_off_in_bench`).
    pub(crate) last_run_router_off: bool,
    /// W0 (SDD estado-arte): desglose de latencia del último turno con
    /// telemetría (`wall_model_ms/guard_ms/sandbox_ms/tools_ms`). `None`
    /// si el run aún no emitió ningún turno (p.ej. fail-fast temprano).
    pub(crate) last_turn_breakdown: Option<crate::telemetry::TurnBreakdown>,
    /// W4: closer TF-IDF opcional (veredicto Done → evidencia de parar ante
    /// el `EffortRouter`). `None` = sin closer (compat: el router decide
    /// solo con sus señales propias). Respetado `CloserConfig.enabled`
    /// (OFF en bench por trampa conocida).
    pub(crate) closer: Option<(TfidfCloser, CloserConfig)>,
    /// W7-code: semilla de determinismo (`None` = legacy exacto: backoff
    /// fijo, orden de carga en empates). `Some` = jitter determinista +
    /// desempates sembrados (misma semilla + mismas entradas ⇒ mismo
    /// resultado, entre runs y máquinas).
    pub(crate) seed: Option<u64>,
    /// W5-code: prefetch especulativo read-only en batches nativos
    /// (`false` por defecto — compat: ejecución secuencial legacy).
    pub(crate) speculative: bool,
    /// W5-code: caché idempotente de respuestas (`None` = desactivada).
    pub(crate) response_cache: Option<std::sync::Mutex<crate::response_cache::ResponseCache>>,
    /// W5-code: predictor n-gram sobre nombres de tools ejecutados
    /// (observa cada batch; expone `predict_next_tool`).
    pub(crate) predictor: crate::pld::NgramPredictor,
}

impl Agent {
    /// Create a new agent with the given config (accepts any ChatClient impl, boxes it).
    pub fn new(client: impl ChatClient + 'static, config: AgentConfig) -> Self {
        let tools = crate::tools::standard_tool_definitions();
        Self::new_inner(Box::new(client), config, tools)
    }

    /// Create with custom tool definitions (accepts any ChatClient impl, boxes it).
    pub fn with_tools(
        client: impl ChatClient + 'static,
        config: AgentConfig,
        tools: Vec<ToolDefinition>,
    ) -> Self {
        Self::new_inner(Box::new(client), config, tools)
    }

    /// Replace the inner client (useful for tests with MockClient).
    pub fn with_client(mut self, client: Box<dyn ChatClient>) -> Self {
        self.client = client;
        self
    }

    /// Shared constructor: wires budget, monitors, knowledge, bootstrap and
    /// memory from `config` (F4.1 — single implementation, no duplication).
    /// W1: stays sync by design — startup runs once per agent (dir create +
    /// SQLite open + knowledge load), never per tool call, so it is not
    /// reactor-blocking. Hot-loop leaves are async; constructors are not
    /// (firma rule: async solo en hojas I/O). Async call sites offload via
    /// `spawn_blocking` (cli `build_bench_agent` + Run path), no async ctor.
    fn new_inner(
        client: Box<dyn ChatClient>,
        mut config: AgentConfig,
        tools: Vec<ToolDefinition>,
    ) -> Self {
        // Clamp the autorun surface at construction (covers CLI/env/bench
        // plumbing, which all set the field before Agent::new).
        config.autorun_hops = clamp_autorun_hops(config.autorun_hops);
        // Sync derived fields from profile (single source of truth).
        // Keeps AgentConfig::default() (medium) unchanged for existing tests,
        // while --profile tiny correctly drives caps.
        {
            let p = config.profile;
            config.context_window = p.context_window;
            config.max_iterations = p.max_iterations;
            config.bash_timeout_secs = p.bash_timeout_secs;
            config.compact_threshold = p.compact_threshold;
        }
        // F2: per-model inference settings — sampling, thinking mode, and the
        // UNIFIED iteration cap (absorbs the former standalone Tier-4
        // turn-cap override; env KITTY_MODEL_TURN_CAPS + KITTY_NO_THINK apply
        // inside the resolver). Per-run overrides (Fase 2.1) still win:
        // they apply later through the single auditable path.
        //
        // G2-FIX2: only overwrite max_iterations when a per-model turn-cap
        // entry ACTUALLY matched (exact table hit or env spec hit). Family/
        // default resolution carries no cap authority and must not stomp
        // deliberate profile presets (tiny's invariant 4K/5iter/4tools).
        let (tuning, cap_matched) = ModelTuning::resolve_from_env_with_cap_hit(&config.model);
        config.tuning = tuning;
        if cap_matched {
            config.max_iterations = config.tuning.max_iterations;
        }
        // Filter tools by profile subset (tiny → 4 tools, medium/large → 6)
        let tools: Vec<ToolDefinition> = {
            let subset = config.profile.tool_subset;
            // If tool set equals standard, filter; custom tools are left as-is.
            // Heuristic: if len == standard len, apply subset filter.
            let standard_len = crate::tools::standard_tool_definitions().len();
            if tools.len() == standard_len {
                tools
                    .into_iter()
                    .filter(|t| subset.contains(&t.function.name.as_str()))
                    .collect()
            } else {
                tools
            }
        };
        // Grammar enforcement: auto-enable for tiny/nano profiles (spike).
        // dedup: with_profile and new_inner both enforce tiny/nano→grammar for CLI vs direct Agent::new callers
        if matches!(config.profile.name, "tiny" | "nano") {
            config.enable_grammar = true;
        }
        // Grammar enforcement: wrap client if enabled (spike, default false).
        let client: Box<dyn ChatClient> = if config.enable_grammar {
            let grammar = GrammarConfig::from_tools(tools.clone());
            Box::new(GrammarClient::new(client, grammar))
        } else {
            client
        };
        let mut budget = ContextBudget::new(config.context_window);
        // f8 AD5: derive the EFFECTIVE compaction threshold for the actual
        // window (never raises above the profile value; equals it on ≥16k
        // windows against 0.8 profiles). `config.compact_threshold` keeps the
        // PROFILE value as the explicit-override surface.
        budget.set_threshold(ContextBudget::derive_compaction_threshold(
            config.compact_threshold,
            config.context_window,
        ));
        let tool_names: Vec<String> = tools.iter().map(|t| t.function.name.clone()).collect();
        // F2/G2-FIX3: per-turn thinking policy comes from the resolved model
        // profile. This is only the INITIAL seed — `run()` re-seeds from the
        // CURRENT `config.tuning.thinking` on every call, so post-construction
        // mutations (CLI/bench A/B thinking axis) take effect.
        let thinking_budget = ThinkingBudget::new(
            per_turn_thinking_cap(config.tuning.thinking),
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
        // f8 T2.3: the active profile governs the default Read window.
        let read_cap = config.profile.read_cap;
        // F4.1: the active profile governs the Bash output cap.
        let bash_output_cap = config.profile.bash_output_cap;
        // Gate-4: the Bash-sandbox FS allow-root must follow the configured
        // workspace explicitly — never silently inherit the process cwd
        // (launching from `$HOME` would otherwise make all of HOME writable).
        let workspace = config.workspace.clone();

        Self {
            client,
            config,
            messages: Vec::new(),
            tools,
            budget,
            registry: ToolRegistry::with_timeout(bash_timeout_secs)
                .with_workspace(workspace)
                .with_read_cap(read_cap)
                .with_bash_output_cap(bash_output_cap),
            parser: ToolCallParser::new(),
            thinking_budget,
            quality_monitor,
            knowledge_base,
            bootstrap,
            bootstrap_done: false,
            memory_store,
            empty_response_streak: 0,
            doom_loop: DoomLoopGuard::default(),
            react_steps: Vec::new(),
            router: ToolRouter::new(),
            contract: None,
            invalid_streak: 0,
            last_run_hops_used: 0,
            last_run_iters_used: 0,
            last_run_router_off: false,
            last_turn_breakdown: None,
            closer: None,
            seed: None,
            speculative: false,
            response_cache: None,
            predictor: crate::pld::NgramPredictor::new(),
        }
    }

    /// Get filtered tool definitions for a prompt via the 2-stage router.
    ///
    /// When `use_routing` is false (default), returns the full profile-filtered
    /// tool set. When true, classifies `prompt` and returns only the
    /// category subset (empty for Plan → saves ~800 tokens). Statefully
    /// preserves affirmation guard via internal `ToolRouter`.
    ///
    /// W8 enforcement: the tier subset (not the construction-time profile
    /// subset) is the authority — category decides WHAT, tier caps HOW MUCH.
    /// Identical output for standard profiles (pinned by
    /// `tier_routing_matches_profile_routing`).
    pub fn get_tools_for_prompt(&mut self, prompt: &str) -> Vec<ToolDefinition> {
        if !self.config.use_routing {
            return self.tools.clone();
        }
        let tier = self.tier();
        self.router.tools_for_prompt_tiered(prompt, tier)
    }

    /// Stateless helper: classify and return tools for `prompt` without mutating router.
    pub fn tools_for_prompt(prompt: &str) -> Vec<ToolDefinition> {
        crate::router::get_tools_for_prompt(prompt)
    }

    /// Expose current router last category (for testing).
    pub fn router_last(&self) -> Option<crate::router::ToolCategory> {
        self.router.last_category()
    }

    /// Immutable access to config.
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    pub(crate) async fn count_tokens(&self, text: &str) -> usize {
        match self.client.count_tokens(&self.config.model, text).await {
            Ok(n) => n,
            Err(e) => {
                debug!(error = %e, "tokenize failed; using heuristic estimate");
                ContextBudget::estimate_tokens(text)
            }
        }
    }

    /// Run bootstrap detection (cached, runs once).
    /// Push a message and track token usage (F2.3 — sole accounting point).
    ///
    /// Uses real tokenization when the server is reachable; falls back to the
    /// heuristic (chars/4). Every token consumed by the conversation flows
    /// through this one call — no separate `budget.consume()` elsewhere.
    // B1-400 decision: keep push_message as-is — wire sanitizer (idempotent tail-only) is sufficient and avoids touching episode/tool ordering invariants.
    pub(crate) async fn push_message(&mut self, msg: Message) {
        if let Some(content) = msg.content.as_deref() {
            let tokens = self.count_tokens(content).await;
            self.budget.consume(tokens);
        }
        self.messages.push(msg);
    }

    /// W7 ReAct trace for the current run, in loop order (ephemeral).
    pub fn react_steps(&self) -> &[ReActStep] {
        &self.react_steps
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

    /// Current consecutive empty-response count (for testing / introspection).
    pub fn empty_streak(&self) -> u8 {
        self.empty_response_streak
    }

    /// W0 (SDD estado-arte): latency breakdown of the last telemetered turn
    /// (`None` until the first `TurnEvent` is written). Test/bench hook so
    /// the desglose is observable without scraping JSONL files.
    pub fn last_turn_breakdown(&self) -> Option<crate::telemetry::TurnBreakdown> {
        self.last_turn_breakdown
    }

    /// W4: execution tier of the active profile (triaje → compact-QR
    /// summaries; ejecución → full summaries).
    pub fn tier(&self) -> crate::model_profile::Tier {
        self.config.profile.tier()
    }

    /// W4: attach a fitted TF-IDF closer. Its `Done` verdicts become
    /// stop-evidence for the `EffortRouter` at hop/revision decisions
    /// (fail-open: `Abstain` or disabled config contributes nothing).
    pub fn set_closer(&mut self, closer: TfidfCloser, cfg: CloserConfig) {
        self.closer = Some((closer, cfg));
    }

    /// W7-code: set the determinism seed (also propagated to the knowledge
    /// base tie-breaks). `None` (default) = legacy exact behavior.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self.knowledge_base.with_seed(seed);
        self
    }

    /// W5-code: enable speculative read-only prefetch on tool batches
    /// (default OFF — sequential legacy execution).
    pub fn with_speculative_readonly(mut self, on: bool) -> Self {
        self.speculative = on;
        self
    }

    /// W5-code: enable the idempotent response cache with `cap` entries
    /// (same-process, FIFO). Serves byte-identical retries without burning
    /// another model call.
    pub fn with_response_cache(mut self, cap: usize) -> Self {
        self.response_cache = Some(std::sync::Mutex::new(
            crate::response_cache::ResponseCache::new(cap),
        ));
        self
    }

    /// W5-code: most likely next tool after `ctx` (trailing executed names),
    /// per the observed n-gram table (`None` = unseen context).
    pub fn predict_next_tool(&self, ctx: &[String]) -> Option<String> {
        self.predictor.predict(ctx)
    }

    /// W4: closer stop-signal for the current history (`Some(true)` = the
    /// closer predicts Done on the last assistant text with enough
    /// confidence/margin; `None` = no closer, disabled, or abstain).
    pub(crate) fn closer_signal(&self) -> Option<bool> {
        let (closer, cfg) = self.closer.as_ref()?;
        if !cfg.enabled {
            return None;
        }
        let text = self
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "assistant")
            .and_then(|m| m.content.as_deref())?;
        let verdict = closer.predict(text, cfg);
        match verdict.label {
            CloserLabel::Done => Some(true),
            CloserLabel::NotDone | CloserLabel::Abstain => None,
        }
    }

    /// Associated helper mirroring the free `should_fail_fast` (for `Agent::` call sites).
    pub fn should_fail_fast(streak: u8) -> bool {
        should_fail_fast(streak)
    }

    pub fn set_contract(&mut self, c: Contract) {
        self.contract = Some(c);
    }
    pub fn clear_contract(&mut self) {
        self.contract = None;
    }
    pub fn contract(&self) -> Option<&Contract> {
        self.contract.as_ref()
    }
    pub(crate) fn check_done_guard(&self, text: &str) -> Option<String> {
        if !contract_guard_enabled() {
            return None;
        }
        let c = self.contract.as_ref()?;
        if c.is_done() {
            return None;
        }
        if !looks_like_done_claim(text) {
            return None;
        }
        Some(format!("[CONTRACT-GUARD] Blockers:\n{}", c.to_markdown()))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{env_lock, Agent, AgentConfig};
    use crate::model_profile::EnvRestoreGuard;
    use crate::model_profile::ModelProfile;
    use crate::router::{EXECUTION_TOOL_SUBSET, TRIAGE_TOOL_SUBSET};
    use kitty_runtime::{MockClient, OllamaClient};

    /// W8: `get_tools_for_prompt` con routing respeta el subset del tier
    /// (enforcement a nivel agente, para cualquier categoría clasificada).
    #[test]
    fn get_tools_for_prompt_respects_tier_subset() {
        for (profile, allowed) in [
            ("tiny", TRIAGE_TOOL_SUBSET),
            ("medium", EXECUTION_TOOL_SUBSET),
        ] {
            let mut agent = Agent::new(
                MockClient::new(vec![]),
                AgentConfig {
                    workspace: std::path::PathBuf::from("/tmp"),
                    ..Default::default()
                }
                .with_profile(profile),
            );
            agent.config.use_routing = true;
            for prompt in [
                "search the codebase for foo",
                "run the test suite",
                "write a new file",
                "read foo.rs",
                "hello there",
            ] {
                let tools = agent.get_tools_for_prompt(prompt);
                assert!(
                    tools
                        .iter()
                        .all(|t| allowed.contains(&t.function.name.as_str())),
                    "{profile}/{prompt}: {tools:?} fuera del subset del tier"
                );
            }
        }
    }

    #[test]
    fn turn_cap_applied_per_model_in_new_inner() {
        let _g = env_lock();
        let _e = EnvRestoreGuard::remove(crate::model_profile::ENV_MODEL_TURN_CAPS);

        // Matching model id → static-table cap overrides the profile default.
        let client = OllamaClient::new("http://localhost:11434");
        let cfg = AgentConfig {
            model: "ornith:9b".to_string(),
            workspace: std::path::PathBuf::from("/tmp"),
            ..Default::default()
        };
        let agent = Agent::new(client, cfg);
        assert_eq!(agent.config().max_iterations, 24);

        // Prefix semantics hold through Agent construction too.
        let client = OllamaClient::new("http://localhost:11434");
        let cfg = AgentConfig {
            model: "ornith-1.5-instruct".to_string(),
            workspace: std::path::PathBuf::from("/tmp"),
            ..Default::default()
        };
        let agent = Agent::new(client, cfg);
        assert_eq!(agent.config().max_iterations, 16);

        // Non-matching model keeps the profile default (medium → 20).
        let client = OllamaClient::new("http://localhost:11434");
        let cfg = AgentConfig {
            model: "qwen3:8b".to_string(),
            workspace: std::path::PathBuf::from("/tmp"),
            ..Default::default()
        };
        let agent = Agent::new(client, cfg);
        assert_eq!(
            agent.config().max_iterations,
            ModelProfile::medium().max_iterations
        );
    }
}

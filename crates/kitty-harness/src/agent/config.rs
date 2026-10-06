//! Agent configuration (W2 split of `agent/mod.rs`, por símbolo).
//!
//! [`AgentEvent`], [`AgentConfig`] (+ Default/impl) and [`RunOverrides`].
use crate::knowledge::KnowledgeConfig;
use crate::model_profile::{ModelProfile, ModelTuning};
use crate::quality_monitor::QualityConfig;
/// Streaming events emitted during `run_with_events`.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    Thinking(String),
    ToolCall {
        name: String,
        args: serde_json::Value,
    },
    ToolResult {
        name: String,
        output: String,
        success: bool,
    },
    AssistantMessage(String),
    Error(String),
    Done(String),
}

/// Configuration for the agent.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Model name to use.
    pub model: String,
    /// Maximum number of tool-call iterations before forcing a text response.
    pub max_iterations: usize,
    /// Autorun continuation hops (f8 AD1): extra FULL tool-loop cycles granted
    /// ONLY on iteration exhaustion — force_response is skipped, an
    /// [`AUTORUN_CONTINUE_PROMPT`] is pushed and the inner loop restarts.
    /// Default 0 keeps behavior byte-identical to the legacy single-cycle run;
    /// clamped to 0..=[`MAX_AUTORUN_HOPS`] wherever set. Independent of
    /// `VERIFY_MAX_REVISIONS` (which counts verification-driven re-entries).
    pub autorun_hops: usize,
    /// Context window size in tokens.
    pub context_window: usize,
    /// Compaction threshold (0.0-1.0). Default: 0.8.
    pub compact_threshold: f32,
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
    /// Model profile preset (tiny|medium|large). Determines defaults for context, thinking, caps, etc.
    pub profile: &'static ModelProfile,
    /// Per-model inference settings (F2): sampling, thinking mode, and the
    /// unified per-model iteration cap. Resolved from the model id in
    /// [`Agent::new`] (exact → family → default); post-construction
    /// per-run overrides (Fase 2.1) still win.
    pub tuning: ModelTuning,
    /// Whether to use 2-stage tool routing (router ready but gated behind this flag).
    pub use_routing: bool,
    /// Whether grammar-constrained decoding is enabled (tiny profile / spike).
    /// Default false; enabled via `--profile tiny` or future CLI flag.
    pub enable_grammar: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: "qwen3:8b".to_string(),
            max_iterations: 20,
            autorun_hops: 0,
            context_window: 16384,
            compact_threshold: 0.8,
            enable_knowledge: false,
            knowledge_config: KnowledgeConfig::default(),
            quality_config: QualityConfig::default(),
            workspace: std::path::PathBuf::from("."),
            memory_db_path: None,
            bash_timeout_secs: 60,
            profile: ModelProfile::medium(),
            tuning: ModelTuning::default(),
            use_routing: false,
            enable_grammar: false,
        }
    }
}

impl AgentConfig {
    /// Build a config from a CLI profile string, syncing derived fields from
    /// the profile. The MODEL ID participates when the profile value does not
    /// name a preset ("auto" or unknown): f8 T4.1 auto-detects nano-tier
    /// models by id shape.
    pub fn with_profile(mut self, profile_name: &str) -> Self {
        let p = ModelProfile::from_cli(profile_name, &self.model);
        self.profile = p;
        self.context_window = p.context_window;
        self.max_iterations = p.max_iterations;
        self.bash_timeout_secs = p.bash_timeout_secs;
        self.compact_threshold = p.compact_threshold;
        // dedup: with_profile and new_inner both enforce tiny/nano→grammar for CLI vs direct Agent::new callers
        if matches!(p.name, "tiny" | "nano") {
            self.enable_grammar = true;
        }
        self
    }
}
/// Per-run overrides applied at a single point at the start of
/// [`Agent::run_with`] (audit 2026-09 Fase 2.1). Absent fields mean "use the
/// configured defaults"; there are no sentinel values, so an invalid state is
/// unrepresentable.
///
/// This replaces the former `config_mut()`/`budget_mut()` escape hatches
/// whose post-construction mutations were the root cause of the G2-FIX3
/// dead-store bug class: overrides now flow through one auditable path and
/// the thinking-budget re-seed reads the post-override tuning.
#[derive(Debug, Clone, Default)]
pub struct RunOverrides {
    /// Full tuning replacement (sampling, thinking mode, turn caps).
    pub tuning: Option<ModelTuning>,
    /// Compaction threshold; combined with `context_window` it re-derives the
    /// effective threshold via `ContextBudget::derive_compaction_threshold`.
    pub compact_threshold: Option<f32>,
    /// Autorun continuation hops (clamped into 0..=MAX_AUTORUN_HOPS).
    pub autorun_hops: Option<usize>,
    /// Context window; resizes the budget to match.
    pub context_window: Option<usize>,
    /// Iteration cap for the tool loop.
    pub max_iterations: Option<usize>,
    /// Bash command timeout in seconds.
    pub bash_timeout: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::super::{env_lock, with_clean_tuning_env, Agent, AgentConfig};
    use crate::model_profile::{EnvRestoreGuard, ModelProfile, PromptStyle, ThinkingMode};
    use kitty_runtime::MockClient;

    #[test]
    fn profile_preset_iterations_survive_family_default() {
        with_clean_tuning_env(|| {
            // G2-FIX2 good case: unknown model → family/default resolution has
            // NO cap authority → tiny's deliberate 5-iteration invariant must
            // survive (the old code stomped it to 20).
            let cfg = AgentConfig {
                model: "unknown-model:7b".into(),
                profile: ModelProfile::tiny(),
                ..Default::default()
            };
            let agent = Agent::new(MockClient::new(vec![]), cfg);
            assert_eq!(agent.config().max_iterations, 5);
            // Tuning still carries its family default internally…
            assert_eq!(agent.config().tuning.max_iterations, 20);
            // …and sampling/thinking fields apply regardless.
            assert_eq!(agent.config().tuning.thinking, ThinkingMode::Budget(2048));
        });
    }

    #[test]
    fn exact_turn_cap_still_overrides_profile_preset() {
        with_clean_tuning_env(|| {
            // Good case: exact Tier-4 table hit RAISES tiny's 5 → 24 for
            // ornith:9b (the original unified behavior must keep working).
            let cfg = AgentConfig {
                model: "ornith:9b".into(),
                profile: ModelProfile::tiny(),
                ..Default::default()
            };
            let agent = Agent::new(MockClient::new(vec![]), cfg);
            assert_eq!(agent.config().max_iterations, 24);
            assert_eq!(agent.config().tuning.max_iterations, 24);
            assert_eq!(agent.config().tuning.thinking, ThinkingMode::On);
        });
    }

    #[test]
    fn no_think_env_forces_thinking_off_and_request_flag() {
        let _g = env_lock();
        let key = kitty_runtime::client::ENV_NO_THINK;
        let _e = EnvRestoreGuard::set(key, "1");
        let agent = Agent::new(
            MockClient::new(vec![]),
            AgentConfig {
                model: "ornith:9b".into(),
                ..Default::default()
            },
        );
        // KITTY_NO_THINK wins over the ornith profile's ThinkingMode::On…
        assert_eq!(agent.config().tuning.thinking, ThinkingMode::Off);
        // …and reaches the wire as think:false on every request.
        assert_eq!(agent.request_options(&[]).think, Some(false));
    }

    #[test]
    fn explicit_profile_beats_nano_autodetection() {
        // f8 T4.1 AC: --profile medium on a 3B id stays medium.
        let cfg = AgentConfig {
            model: "qwen2.5-3b".to_string(),
            ..Default::default()
        }
        .with_profile("medium");
        assert_eq!(cfg.profile.name, "medium");
        assert_eq!(cfg.profile.prompt_style, PromptStyle::Full);
        // And nano is reachable explicitly.
        let cfg = AgentConfig::default().with_profile("nano");
        assert_eq!(cfg.profile.name, "nano");
        assert_eq!(cfg.profile.prompt_style, PromptStyle::Minimal);
    }
}

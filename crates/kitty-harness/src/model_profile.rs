//! Model profiles — parametrize harness behaviour per model size.
//!
//! Three presets: tiny (≤2B), medium (8-13B, default), large (27-35B).
//! Each preset determines context window, thinking budget, iteration cap,
//! tool subset, native-tool flag, read cap, bash timeout and compaction threshold.

/// Profile that parametrizes harness behaviour (context, thinking, tools, caps).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelProfile {
    /// Human-readable name: "tiny" | "medium" | "large".
    pub name: &'static str,
    /// Context window in tokens.
    pub context_window: usize,
    /// Max thinking/reasoning tokens. `None` = no thinking budget (tiny).
    pub max_thinking_tokens: Option<usize>,
    /// Max tool-call iterations before forcing a text response.
    pub max_iterations: usize,
    /// Subset of tool names exposed for this profile.
    pub tool_subset: &'static [&'static str],
    /// Whether the model supports native tool calling (Ollama).
    pub use_native_tools: bool,
    /// Max lines returned by Read.
    pub read_cap: usize,
    /// Bash command timeout in seconds.
    pub bash_timeout_secs: u64,
    /// Compaction threshold (0.0-1.0).
    pub compact_threshold: f32,
}

// ── Static presets ─────────────────────────────────────────────────

static TINY: ModelProfile = ModelProfile {
    name: "tiny",
    context_window: 4096,
    max_thinking_tokens: None,
    max_iterations: 5,
    tool_subset: &["Read", "Write", "Edit", "Bash"],
    use_native_tools: false,
    read_cap: 100,
    bash_timeout_secs: 30,
    compact_threshold: 0.7,
};

static MEDIUM: ModelProfile = ModelProfile {
    name: "medium",
    context_window: 16384,
    max_thinking_tokens: Some(1024),
    max_iterations: 20,
    tool_subset: &["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
    use_native_tools: true,
    read_cap: 300,
    bash_timeout_secs: 60,
    compact_threshold: 0.8,
};

static LARGE: ModelProfile = ModelProfile {
    name: "large",
    context_window: 32768,
    max_thinking_tokens: Some(4096),
    max_iterations: 20,
    tool_subset: &["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
    use_native_tools: true,
    read_cap: 500,
    bash_timeout_secs: 120,
    compact_threshold: 0.8,
};

impl ModelProfile {
    /// Preset for sub-2B models (4K ctx, no thinking, 5 iter, 4 tools, no native).
    pub fn tiny() -> &'static Self {
        &TINY
    }
    /// Preset for 8-13B models (16K, 1024 thinking — bench ab 50 knee 1024>2048 96% vs 90% p0.44, 20 iter, 6 tools, native).
    pub fn medium() -> &'static Self {
        &MEDIUM
    }
    /// Preset for 27-35B models (32K, 4096 thinking, 20 iter, all tools, native).
    pub fn large() -> &'static Self {
        &LARGE
    }

    /// Resolve a CLI string to a profile. Unknown values fall back to `medium`.
    pub fn from_cli(s: &str) -> &'static Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "tiny" => Self::tiny(),
            "large" => Self::large(),
            "medium" => Self::medium(),
            _ => Self::medium(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_preset_values() {
        let p = ModelProfile::tiny();
        assert_eq!(p.name, "tiny");
        assert_eq!(p.context_window, 4096);
        assert_eq!(p.max_thinking_tokens, None);
        assert_eq!(p.max_iterations, 5);
        assert_eq!(p.tool_subset, &["Read", "Write", "Edit", "Bash"]);
        assert!(!p.use_native_tools);
        assert_eq!(p.read_cap, 100);
        assert_eq!(p.bash_timeout_secs, 30);
    }

    #[test]
    fn medium_preset_values() {
        let p = ModelProfile::medium();
        assert_eq!(p.name, "medium");
        assert_eq!(p.context_window, 16384);
        assert_eq!(p.max_thinking_tokens, Some(1024));
        assert_eq!(p.max_iterations, 20);
        assert_eq!(p.tool_subset.len(), 6);
        assert!(p.use_native_tools);
        assert_eq!(p.read_cap, 300);
        assert_eq!(p.bash_timeout_secs, 60);
        assert!((p.compact_threshold - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn large_preset_values() {
        let p = ModelProfile::large();
        assert_eq!(p.name, "large");
        assert_eq!(p.context_window, 32768);
        assert_eq!(p.max_thinking_tokens, Some(4096));
        assert_eq!(p.max_iterations, 20);
        assert!(p.use_native_tools);
        assert_eq!(p.read_cap, 500);
    }

    #[test]
    fn from_cli_dispatch() {
        assert_eq!(ModelProfile::from_cli("tiny").name, "tiny");
        assert_eq!(ModelProfile::from_cli("TINY").name, "tiny");
        assert_eq!(ModelProfile::from_cli("medium").name, "medium");
        assert_eq!(ModelProfile::from_cli("large").name, "large");
        assert_eq!(ModelProfile::from_cli("LARGE").name, "large");
        assert_eq!(ModelProfile::from_cli("unknown").name, "medium");
        assert_eq!(ModelProfile::from_cli("").name, "medium");
        assert_eq!(ModelProfile::from_cli("  tiny  ").name, "tiny");
    }

    #[test]
    fn from_cli_is_static() {
        let a = ModelProfile::from_cli("tiny") as *const _;
        let b = ModelProfile::tiny() as *const _;
        assert_eq!(a, b);
    }
}

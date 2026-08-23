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

// ── Per-model turn caps (Tier 4) ───────────────────────────────────

/// Per-model-id iteration caps, matched by plain `starts_with` prefix
/// (first matching entry wins).
///
/// PROVISIONAL PLACEHOLDERS pending bench-ab validation: the values below
/// (`ornith:9b=24`, `ornith-1.5=16`) are mechanism-first guesses, NOT tuned
/// numbers. Re-tune via bench-ab (or override at runtime with
/// [`ENV_MODEL_TURN_CAPS`]) before treating them as authoritative.
static MODEL_TURN_CAPS: &[(&str, usize)] = &[("ornith:9b", 24), ("ornith-1.5", 16)];

/// Env override for per-model turn caps.
///
/// Format: comma-separated `prefix=N` pairs, e.g.
/// `KITTY_MODEL_TURN_CAPS="ornith:9b=24,ornith-1.5=16"`. Entries are parsed
/// leniently (invalid ones skipped silently); the first entry whose prefix
/// matches the model id wins; when the variable is set but nothing matches,
/// the static [`MODEL_TURN_CAPS`] table applies.
pub const ENV_MODEL_TURN_CAPS: &str = "KITTY_MODEL_TURN_CAPS";

/// Iteration cap for `model_id`, if one is configured.
///
/// Resolution order:
/// 1. [`ENV_MODEL_TURN_CAPS`] (read at call time — this runs once per agent
///    construction, so the env read is cheap): first prefix match wins.
/// 2. Static [`MODEL_TURN_CAPS`] table: first prefix match wins.
/// 3. `None` — caller keeps the profile default.
pub fn turn_cap_for_model(model_id: &str) -> Option<usize> {
    if let Ok(spec) = std::env::var(ENV_MODEL_TURN_CAPS) {
        for entry in spec.split(',') {
            let Some((prefix, cap)) = entry.split_once('=') else {
                continue;
            };
            let prefix = prefix.trim();
            if prefix.is_empty() {
                continue;
            }
            let Ok(cap) = cap.trim().parse::<usize>() else {
                continue;
            };
            if model_id.starts_with(prefix) {
                return Some(cap);
            }
        }
    }
    for (prefix, cap) in MODEL_TURN_CAPS {
        if model_id.starts_with(prefix) {
            return Some(*cap);
        }
    }
    None
}

/// Crate-wide lock serializing env-mutating unit tests (cargo runs test
/// threads in parallel inside a single test binary, and every module's test
/// mod shares that binary). Used by this module's tests; `agent.rs` tests
/// delegate their `env_lock` here so both modules exclude each other.
#[cfg(test)]
pub(crate) fn test_env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap()
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

    // ── Tier 4: per-model turn caps ────────────────────────────────

    /// Run `f` with `ENV_MODEL_TURN_CAPS` set to `val` (or removed when
    /// `None`), restoring the previous value afterwards.
    fn with_env_turn_caps(val: Option<&str>, f: impl FnOnce()) {
        let _g = test_env_lock();
        let prev = std::env::var(ENV_MODEL_TURN_CAPS).ok();
        match val {
            Some(v) => std::env::set_var(ENV_MODEL_TURN_CAPS, v),
            None => std::env::remove_var(ENV_MODEL_TURN_CAPS),
        }
        f();
        match prev {
            Some(v) => std::env::set_var(ENV_MODEL_TURN_CAPS, v),
            None => std::env::remove_var(ENV_MODEL_TURN_CAPS),
        }
    }

    #[test]
    fn turn_cap_static_table_hit() {
        with_env_turn_caps(None, || {
            assert_eq!(turn_cap_for_model("ornith:9b"), Some(24));
            assert_eq!(turn_cap_for_model("ornith-1.5"), Some(16));
        });
    }

    #[test]
    fn turn_cap_prefix_semantics() {
        with_env_turn_caps(None, || {
            // Prefix match, not exact equality: suffixes still resolve.
            assert_eq!(turn_cap_for_model("ornith:9b-custom"), Some(24));
            assert_eq!(turn_cap_for_model("ornith-1.5-instruct-q4"), Some(16));
            // Non-matching ids → None (caller keeps profile default).
            assert_eq!(turn_cap_for_model("qwen3:8b"), None);
            assert_eq!(turn_cap_for_model("ornith"), None);
            assert_eq!(turn_cap_for_model(""), None);
        });
    }

    #[test]
    fn turn_cap_env_override_valid() {
        with_env_turn_caps(Some("qwen3:8b=7"), || {
            assert_eq!(turn_cap_for_model("qwen3:8b"), Some(7));
            // Env present but no matching prefix → falls through to static table.
            assert_eq!(turn_cap_for_model("ornith:9b"), Some(24));
        });
    }

    #[test]
    fn turn_cap_env_mixed_valid_and_invalid_entries() {
        with_env_turn_caps(Some("garbage,qwen3:8b=abc,=42,ornith-1.5= 16 ,qwen3:8b=7"), || {
            // "garbage" (no '='), "qwen3:8b=abc" (bad number) and "=42"
            // (empty prefix) are skipped silently; valid entries still apply.
            assert_eq!(turn_cap_for_model("ornith-1.5-x"), Some(16));
            assert_eq!(turn_cap_for_model("qwen3:8b"), Some(7));
        });
    }

    #[test]
    fn turn_cap_env_empty_string_falls_through_to_static() {
        with_env_turn_caps(Some(""), || {
            assert_eq!(turn_cap_for_model("ornith:9b"), Some(24));
            assert_eq!(turn_cap_for_model("qwen3:8b"), None);
        });
    }

    #[test]
    fn turn_cap_env_precedence_over_static_table() {
        with_env_turn_caps(Some("ornith:9b=99"), || {
            assert_eq!(turn_cap_for_model("ornith:9b"), Some(99));
        });
        // First matching env prefix wins over later ones.
        with_env_turn_caps(Some("ornith=5,ornith:9b=8"), || {
            assert_eq!(turn_cap_for_model("ornith:9b"), Some(5));
        });
    }
}

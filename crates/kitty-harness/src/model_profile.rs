//! Model profiles — parametrize harness behaviour per model size.
//!
//! Four presets: nano (<2B small-family models, minimal prompt), tiny (≤2B),
//! medium (8-13B, default), large (27-35B). Each preset determines context
//! window, iteration cap, tool subset, native-tool flag, read cap, bash
//! timeout, compaction threshold and prompt style.

/// How verbose the agent's runtime system prompt should be (f8 T4.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptStyle {
    /// Full bootstrap detail + qwen stance overlay (tiny/medium/large).
    Full,
    /// Compact imperative runtime contract for nano models; skips verbose
    /// bootstrap hints and stance overlays.
    Minimal,
}

/// Profile that parametrizes harness behaviour (context, thinking, tools, caps).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelProfile {
    /// Human-readable name: "nano" | "tiny" | "medium" | "large".
    pub name: &'static str,
    /// Context window in tokens.
    pub context_window: usize,
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
    /// Maximum characters returned by any single tool call.
    pub bash_output_cap: usize,
    /// Compaction threshold (0.0-1.0).
    pub compact_threshold: f32,
    /// System-prompt verbosity for this tier (f8 T4.1).
    pub prompt_style: PromptStyle,
}

// ── Static presets ─────────────────────────────────────────────────

static NANO: ModelProfile = ModelProfile {
    name: "nano",
    context_window: 4096,
    max_iterations: 5,
    tool_subset: &["Read", "Write", "Edit", "Bash"],
    use_native_tools: false,
    read_cap: 80,
    bash_timeout_secs: 30,
    bash_output_cap: 2_000,
    compact_threshold: 0.7,
    prompt_style: PromptStyle::Minimal,
};

static TINY: ModelProfile = ModelProfile {
    name: "tiny",
    context_window: 4096,
    max_iterations: 5,
    tool_subset: &["Read", "Write", "Edit", "Bash"],
    use_native_tools: false,
    read_cap: 100,
    bash_timeout_secs: 30,
    bash_output_cap: 3_000,
    compact_threshold: 0.7,
    prompt_style: PromptStyle::Full,
};

static MEDIUM: ModelProfile = ModelProfile {
    name: "medium",
    context_window: 16384,
    max_iterations: 20,
    tool_subset: &["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
    use_native_tools: true,
    read_cap: 300,
    bash_timeout_secs: 60,
    bash_output_cap: 5_000,
    compact_threshold: 0.8,
    prompt_style: PromptStyle::Full,
};

static LARGE: ModelProfile = ModelProfile {
    name: "large",
    context_window: 32768,
    max_iterations: 20,
    tool_subset: &["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
    use_native_tools: true,
    read_cap: 500,
    bash_timeout_secs: 120,
    bash_output_cap: 10_000,
    compact_threshold: 0.8,
    prompt_style: PromptStyle::Full,
};

impl ModelProfile {
    /// Preset for sub-2B small-family models (4K ctx, 5 iter, 4 tools,
    /// read cap 80, minimal prompt).
    pub fn nano() -> &'static Self {
        &NANO
    }
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

    /// Resolve the session profile from the CLI `--profile` value and the
    /// model id (f8 T4.1).
    ///
    /// An explicitly recognized profile name always wins (`--profile medium`
    /// on a 3B model stays medium). Otherwise ("auto" or any unrecognized
    /// value) the MODEL NAME decides via [`is_nano_sized_model`]: conservative
    /// sub-3B family ids select `nano`, everything else stays `medium`.
    pub fn from_cli(profile: &str, model_id: &str) -> &'static Self {
        match profile.trim().to_ascii_lowercase().as_str() {
            "nano" => Self::nano(),
            "tiny" => Self::tiny(),
            "large" => Self::large(),
            "medium" => Self::medium(),
            _ if is_nano_sized_model(model_id) => Self::nano(),
            _ => Self::medium(),
        }
    }
}

/// Conservative auto-detection of nano-tier models by id shape (f8 T4.1).
///
/// Matches ONLY the enumerated parameter sizes {0.5, 1, 1.5, 2, 3} billion at
/// a token boundary (`start`/`-`/`.`/`_`/`:` before — colon covers canonical
/// ollama tags like `qwen2.5:3b`; end/`-`/`.`/`_`/`@` after), case-insensitive:
/// `qwen2.5-3b`, `llama3.2:1b`, `smollm2-1.5B@q4` match; `qwen3-1.7b`,
/// `ornith-1.5:9b` and `qwen3:8b` deliberately do NOT (1.7b and 8b are outside
/// the enumerated set) and stay on their explicit/default profile.
#[must_use]
pub fn is_nano_sized_model(model_id: &str) -> bool {
    static NANO_RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    NANO_RE
        .get_or_init(|| {
            regex::Regex::new(r"(?i)(^|[-._:])(0\.5|1|1\.5|2|3)b($|[-._@])")
                .expect("static regex should compile")
        })
        .is_match(model_id)
}

// ── Per-model turn caps (Tier 4) ───────────────────────────────────

/// Per-model-id iteration caps, matched by plain `starts_with` prefix
/// (first matching entry wins).
///
/// PROVISIONAL PLACEHOLDERS pending bench-ab validation: the values below
/// (`ornith:9b=24`, `ornith-1.5=16`) are mechanism-first guesses, NOT tuned
/// numbers. Re-tune via bench-ab (or override at runtime with
/// [`ENV_MODEL_TURN_CAPS`]) before treating them as authoritative.
///
/// A matching entry replaces the profile default even when it RAISES it
/// (e.g. tiny profile's deliberate 5 → 24 for `ornith:9b`); use the env
/// override to escape that without recompiling.
///
/// SINGLE SOURCE OF TRUTH (F2): these caps are consumed exclusively through
/// [`ModelTuning::resolve`] / [`turn_cap_for_model`] — never read directly
/// elsewhere.
static MODEL_TURN_CAPS: &[(&str, usize)] = &[("ornith:9b", 24), ("ornith-1.5", 16)];

// ── Per-model inference settings (F2) ──────────────────────────────

/// How a model should use its reasoning/thinking capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingMode {
    /// Keep thinking enabled with NO forced per-turn cap. For
    /// reasoning-trained models: exceeding any cap retries without thinking,
    /// which is exactly the force-off we must avoid by default.
    On,
    /// Keep thinking enabled but cap it at N tokens per turn; exceeding the
    /// cap retries once with thinking disabled (FASE D budget mechanism).
    Budget(u32),
    /// Hard-disable thinking (`"think": false` on every request). Only
    /// reachable via explicit override (`KITTY_NO_THINK=1` or programmatic
    /// config mutation) — profiles never select Off on their own.
    Off,
}

/// Per-model inference settings resolved from the model id (F2).
///
/// Named `ModelTuning` because `ModelProfile` (above) is the existing
/// size-preset struct; this type carries what the size presets cannot:
/// sampling parameters and thinking policy keyed off the model id itself.
///
/// # Iteration-budget accounting (Oracle finding M5)
///
/// `max_iterations` is the tool-loop budget PER PROMPT ATTEMPT. The outer
/// verification revision cycle (`VERIFY_MAX_REVISIONS = 2`) can re-enter the
/// full loop up to 2 more times after workspace-test failures, so the
/// WORST-CASE LLM-call count per user prompt is roughly
/// `max_iterations × 3`. Keep the default modest (≤24 unless an explicit
/// bench profile raises it): raising it multiplies across revision cycles.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelTuning {
    /// Unified per-model tool-loop iteration cap (absorbs the Tier-4 turn
    /// caps — see the accounting warning above).
    pub max_iterations: usize,
    /// Sampling temperature sent as `options.temperature`.
    pub temperature: f32,
    /// Nucleus sampling cutoff (`options.top_p`).
    pub top_p: Option<f32>,
    /// Top-K sampling cutoff (`options.top_k`).
    pub top_k: Option<u32>,
    /// Max output tokens requested (`options.num_predict`); the E3 dynamic
    /// output cap may lower it further so prompt+output fit the window.
    pub num_predict: u32,
    /// Thinking policy wired into the request path.
    pub thinking: ThinkingMode,
}

impl Default for ModelTuning {
    fn default() -> Self {
        Self::default_profile()
    }
}

/// Static per-family defaults (F2). Resolution order in
/// [`ModelTuning::resolve`]: exact model-id match → family match (id contains
/// "ornith" / "qwen") → default.
impl ModelTuning {
    /// Fallback for unknown models: conservative sampling, capped thinking.
    #[must_use]
    pub fn default_profile() -> Self {
        Self {
            max_iterations: 20,
            temperature: 0.3,
            top_p: None,
            top_k: None,
            num_predict: 8192,
            thinking: ThinkingMode::Budget(2048),
        }
    }

    /// Qwen family: deterministic-ish sampling with a ~2048-token thinking
    /// budget (winning bench config keeps thinking ON but bounded).
    #[must_use]
    pub fn qwen_family() -> Self {
        Self::default_profile()
    }

    /// Ornith family (reasoning-trained): thinking stays ON uncapped — NEVER
    /// forced off by default — with the higher-entropy sampling it was
    /// trained for.
    #[must_use]
    pub fn ornith_family() -> Self {
        Self {
            max_iterations: 20,
            temperature: 0.6,
            top_p: Some(0.95),
            top_k: None,
            num_predict: 8192,
            thinking: ThinkingMode::On,
        }
    }

    /// Resolve inference settings for `model_id`.
    ///
    /// Order: exact model-id table hit → family substring match ("ornith",
    /// "qwen", case-insensitive) → default. Overrides applied last:
    ///
    /// 1. `KITTY_MODEL_TURN_CAPS` env entries (`turn_caps_env`) override
    ///    `max_iterations` when their prefix matches.
    /// 2. `no_think_env == Some("1")` forces `ThinkingMode::Off` — it wins
    ///    over EVERY profile.
    ///
    /// Pure function: pass `std::env::var(...)` results from the caller
    /// ([`resolve_from_env`]) to keep resolution testable.
    #[must_use]
    pub fn resolve(
        model_id: &str,
        no_think_env: Option<&str>,
        turn_caps_env: Option<&str>,
    ) -> Self {
        Self::resolve_inner(model_id, no_think_env, turn_caps_env).0
    }

    /// Like [`Self::resolve`] but also reports whether a per-model turn-cap
    /// entry (env spec hit or static exact-table hit) actually matched.
    ///
    /// G2 finding 2: family/default resolution must NOT stomp deliberate
    /// profile presets — the agent loop only overwrites
    /// `config.max_iterations` when this returns `true`.
    ///
    /// Returns `(tuning, cap_matched)`.
    #[must_use]
    pub fn resolve_with_cap_hit(
        model_id: &str,
        no_think_env: Option<&str>,
        turn_caps_env: Option<&str>,
    ) -> (Self, bool) {
        Self::resolve_inner(model_id, no_think_env, turn_caps_env)
    }

    /// Shared implementation behind [`Self::resolve`] /
    /// [`Self::resolve_with_cap_hit`].
    fn resolve_inner(
        model_id: &str,
        no_think_env: Option<&str>,
        turn_caps_env: Option<&str>,
    ) -> (Self, bool) {
        let lower = model_id.to_ascii_lowercase();
        let mut tuning = if lower.contains("ornith") {
            Self::ornith_family()
        } else if lower.contains("qwen") {
            Self::qwen_family()
        } else {
            Self::default_profile()
        };
        // Unified Tier-4 caps: env override first, static prefix table next;
        // both beat the family/default value computed above. The match flag
        // lets callers preserve profile presets otherwise.
        let matched = turn_cap_from_env_spec(turn_caps_env, model_id)
            .or_else(|| turn_cap_static_for(model_id));
        if let Some(cap) = matched {
            tuning.max_iterations = cap;
        }
        // KITTY_NO_THINK=1 overrides any profile's thinking mode.
        if no_think_env == Some("1") {
            tuning.thinking = ThinkingMode::Off;
        }
        (tuning, matched.is_some())
    }

    /// Convenience wrapper reading both env vars at call time, also reporting
    /// whether a turn-cap entry matched ([`Self::resolve_with_cap_hit`]).
    #[must_use]
    pub fn resolve_from_env_with_cap_hit(model_id: &str) -> (Self, bool) {
        Self::resolve_inner(
            model_id,
            std::env::var(kitty_runtime::client::ENV_NO_THINK)
                .ok()
                .as_deref(),
            std::env::var(ENV_MODEL_TURN_CAPS).ok().as_deref(),
        )
    }

    /// Convenience wrapper reading both env vars at call time.
    #[must_use]
    pub fn resolve_from_env(model_id: &str) -> Self {
        Self::resolve_from_env_with_cap_hit(model_id).0
    }
}

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
///
/// F2: this function remains the standalone Tier-4 lookup (used by tests and
/// external callers); [`ModelTuning::resolve`] composes the same pieces into
/// the full per-model tuning.
pub fn turn_cap_for_model(model_id: &str) -> Option<usize> {
    turn_cap_from_env_spec(std::env::var(ENV_MODEL_TURN_CAPS).ok().as_deref(), model_id)
        .or_else(|| turn_cap_static_for(model_id))
}

/// Pure env-spec half of [`turn_cap_for_model`]: parse a
/// `KITTY_MODEL_TURN_CAPS`-style spec (`prefix=N,...`) and return the cap of
/// the first prefix matching `model_id`. Invalid entries are skipped
/// silently; `spec == None` yields `None`.
#[must_use]
pub fn turn_cap_from_env_spec(spec: Option<&str>, model_id: &str) -> Option<usize> {
    let spec = spec?;
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
    None
}

/// Static-table half of [`turn_cap_for_model`]: first [`MODEL_TURN_CAPS`]
/// prefix match, or `None`.
#[must_use]
fn turn_cap_static_for(model_id: &str) -> Option<usize> {
    MODEL_TURN_CAPS
        .iter()
        .find(|(prefix, _)| model_id.starts_with(prefix))
        .map(|(_, cap)| *cap)
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
        assert_eq!(p.max_iterations, 5);
        assert_eq!(p.tool_subset, &["Read", "Write", "Edit", "Bash"]);
        assert!(!p.use_native_tools);
        assert_eq!(p.read_cap, 100);
        assert_eq!(p.bash_timeout_secs, 30);
        assert_eq!(p.prompt_style, PromptStyle::Full);
    }

    #[test]
    fn medium_preset_values() {
        let p = ModelProfile::medium();
        assert_eq!(p.name, "medium");
        assert_eq!(p.context_window, 16384);
        assert_eq!(p.max_iterations, 20);
        assert_eq!(p.tool_subset.len(), 6);
        assert!(p.use_native_tools);
        assert_eq!(p.read_cap, 300);
        assert_eq!(p.bash_timeout_secs, 60);
        assert!((p.compact_threshold - 0.8).abs() < f32::EPSILON);
        assert_eq!(p.prompt_style, PromptStyle::Full);
    }

    #[test]
    fn large_preset_values() {
        let p = ModelProfile::large();
        assert_eq!(p.name, "large");
        assert_eq!(p.context_window, 32768);
        assert_eq!(p.max_iterations, 20);
        assert!(p.use_native_tools);
        assert_eq!(p.read_cap, 500);
        assert_eq!(p.prompt_style, PromptStyle::Full);
    }

    #[test]
    fn nano_preset_values() {
        // f8 T4.1: minimal-prompt tier for conservative sub-3B models.
        let p = ModelProfile::nano();
        assert_eq!(p.name, "nano");
        assert_eq!(p.context_window, 4096);
        assert_eq!(p.max_iterations, 5);
        assert_eq!(p.tool_subset, &["Read", "Write", "Edit", "Bash"]);
        assert!(!p.use_native_tools);
        assert_eq!(p.read_cap, 80);
        assert_eq!(p.bash_timeout_secs, 30);
        assert!((p.compact_threshold - 0.7).abs() < f32::EPSILON);
        assert_eq!(p.prompt_style, PromptStyle::Minimal);
    }

    #[test]
    fn from_cli_dispatch_explicit_names_win() {
        assert_eq!(ModelProfile::from_cli("tiny", "any-model").name, "tiny");
        assert_eq!(ModelProfile::from_cli("TINY", "qwen2.5-3b").name, "tiny");
        assert_eq!(
            ModelProfile::from_cli("medium", "qwen2.5-3b").name,
            "medium",
            "explicit profile beats auto-detection"
        );
        assert_eq!(ModelProfile::from_cli("large", "m").name, "large");
        assert_eq!(ModelProfile::from_cli("LARGE", "m").name, "large");
        assert_eq!(ModelProfile::from_cli("nano", "anything").name, "nano");
        assert_eq!(ModelProfile::from_cli("", "unknown-model").name, "medium");
        assert_eq!(ModelProfile::from_cli("  tiny  ", "x").name, "tiny");
    }

    #[test]
    fn from_cli_auto_detects_nano_by_model_name() {
        // Enumerated sizes at token boundaries → nano.
        for model in [
            "qwen2.5-3b",
            "llama3.2-1b",
            "qwen2.5:3b",
            "llama3.2:1b",
            "smollm2-1.7B-placeholder-not-matching",
            "qwen3-0.5b-instruct",
            "gemma-2b",
            "_1.5b",
        ] {
            let expected = if model.contains("1.7") {
                "medium"
            } else {
                "nano"
            };
            assert_eq!(
                ModelProfile::from_cli("auto", model).name,
                expected,
                "{model} should auto-resolve to {expected}"
            );
        }
        // Outside the enumerated set → NOT nano (stays medium).
        for model in ["qwen3-1.7b", "qwen3:8b", "llama3.2-12b", "ornith:9b"] {
            assert_eq!(
                ModelProfile::from_cli("auto", model).name,
                "medium",
                "{model} must not auto-select nano"
            );
        }
    }

    #[test]
    fn nano_detection_regex_boundaries() {
        // Dash/dot separators.
        assert!(is_nano_sized_model("qwen2.5-3b"));
        assert!(is_nano_sized_model("llama3.2-1b"));
        assert!(is_nano_sized_model("QWEN3-0.5B-Instruct"));
        assert!(is_nano_sized_model("smollm-1.5b@q4"));
        assert!(is_nano_sized_model("2b"));
        // Colon-tagged ollama ids (G4-4).
        assert!(is_nano_sized_model("qwen2.5:3b"), "colon pre-boundary");
        assert!(
            is_nano_sized_model("llama3.2:1b"),
            "colon-tagged canonical form"
        );
        // Boundary discipline: no false positives.
        assert!(!is_nano_sized_model("qwen3-1.7b"), "1.7b outside the set");
        assert!(!is_nano_sized_model("13b"), "13b is not in the set");
        assert!(!is_nano_sized_model("xb"), "not a size token");
        assert!(!is_nano_sized_model("qwen3:8b"));
        assert!(
            !is_nano_sized_model("ornith-1.5:9b"),
            "size must be followed by b"
        );
    }

    #[test]
    fn from_cli_is_static() {
        let a = ModelProfile::from_cli("tiny", "m") as *const _;
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
        with_env_turn_caps(
            Some("garbage,qwen3:8b=abc,=42,ornith-1.5= 16 ,qwen3:8b=7"),
            || {
                // "garbage" (no '='), "qwen3:8b=abc" (bad number) and "=42"
                // (empty prefix) are skipped silently; valid entries still apply.
                assert_eq!(turn_cap_for_model("ornith-1.5-x"), Some(16));
                assert_eq!(turn_cap_for_model("qwen3:8b"), Some(7));
            },
        );
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

    // ── F2: per-model tuning resolution ────────────────────────────────

    #[test]
    fn tuning_resolution_exact_beats_family_beats_default() {
        // Exact match (via unified Tier-4 caps): ornith:9b keeps its 24-cap
        // AND the ornith family sampling/thinking policy.
        let exact = ModelTuning::resolve("ornith:9b", None, None);
        assert_eq!(exact.max_iterations, 24);
        assert_eq!(exact.thinking, ThinkingMode::On);
        assert!((exact.temperature - 0.6).abs() < f32::EPSILON);
        assert_eq!(exact.top_p, Some(0.95));
        assert_eq!(exact.num_predict, 8192);

        // Family match (substring, case-insensitive): ornith without an
        // exact cap entry falls back to the family default 20.
        let family = ModelTuning::resolve("Ornith-2X-Instruct", None, None);
        assert_eq!(family.thinking, ThinkingMode::On);
        assert_eq!(family.max_iterations, 20);

        // qwen family.
        let qwen = ModelTuning::resolve("qwen3:8b", None, None);
        assert_eq!(qwen.thinking, ThinkingMode::Budget(2048));
        assert!((qwen.temperature - 0.3).abs() < f32::EPSILON);
        assert_eq!(qwen.num_predict, 8192);

        // Unknown model → default profile.
        let unknown = ModelTuning::resolve("smollm2:1.7b", None, None);
        assert_eq!(unknown, ModelTuning::default_profile());
    }

    #[test]
    fn tuning_no_think_override_wins_over_all_profiles() {
        for model in ["ornith:9b", "qwen3:8b", "unknown-model"] {
            let t = ModelTuning::resolve(model, Some("1"), None);
            assert_eq!(
                t.thinking,
                ThinkingMode::Off,
                "KITTY_NO_THINK must win for {model}"
            );
        }
        // Values other than exactly "1" do NOT override.
        let kept = ModelTuning::resolve("ornith:9b", Some("0"), None);
        assert_eq!(kept.thinking, ThinkingMode::On);
        let kept = ModelTuning::resolve("ornith:9b", Some("yes"), None);
        assert_eq!(kept.thinking, ThinkingMode::On);
        let kept = ModelTuning::resolve("qwen3:8b", None, None);
        assert_ne!(kept.thinking, ThinkingMode::Off);
    }

    #[test]
    fn tuning_caps_env_overrides_max_iterations_last() {
        // Env caps beat even the static exact-match table entry.
        let t = ModelTuning::resolve("ornith:9b", None, Some("ornith:9b=7"));
        assert_eq!(t.max_iterations, 7);
        // Env entries for unrelated models leave the resolved value alone;
        // invalid entries are skipped silently (bad input tolerated).
        let t = ModelTuning::resolve("ornith:9b", None, Some("garbage,qwen3=abc,=4"));
        assert_eq!(t.max_iterations, 24);
        let t = ModelTuning::resolve("qwen3:8b", None, Some("garbage,qwen3=abc,=4"));
        assert_eq!(t.max_iterations, 20);
    }

    #[test]
    fn tuning_cap_hit_flag_reports_exact_and_env_matches_only() {
        // Good: exact table hit and env hit both report true.
        let (_, hit) = ModelTuning::resolve_with_cap_hit("ornith:9b", None, None);
        assert!(hit);
        let (_, hit) = ModelTuning::resolve_with_cap_hit("qwen3:8b", None, Some("qwen=11"));
        assert!(hit);
        // Bad: family/default resolution is NOT a cap match — callers must
        // keep the size-profile preset (e.g. tiny's 5 iterations).
        let (t, hit) = ModelTuning::resolve_with_cap_hit("unknown-model", None, None);
        assert!(!hit);
        // Family default stays inside tuning; callers keep the preset.
        assert_eq!(t.max_iterations, 20);
        // Invalid env entries yield nothing AND the model has no static
        // entry → no match. A static-entry model (ornith:9b) would still
        // legitimately report a hit via the static fallback.
        let (_, hit) =
            ModelTuning::resolve_with_cap_hit("qwen3:8b", None, Some("garbage,qwen3=abc"));
        assert!(!hit, "invalid env entries must not count as a match");
        let (_, hit) = ModelTuning::resolve_with_cap_hit("ornith:9b", None, Some("garbage"));
        assert!(hit, "static-table fallback is a genuine match");
    }

    #[test]
    fn tuning_defaults_stay_modest_for_revision_cycles() {
        // Oracle M5: worst case ≈ max_iterations × 3 LLM calls per prompt
        // (VERIFY_MAX_REVISIONS = 2 revision cycles re-enter the full loop).
        // Every resolvable default must stay ≤ 24 so the worst case stays
        // bounded (~72 calls).
        for model in ["ornith:9b", "ornith-1.5", "qwen3:8b", "mystery-model"] {
            let t = ModelTuning::resolve(model, None, None);
            assert!(
                t.max_iterations <= 24,
                "{model} default too high: {}",
                t.max_iterations
            );
        }
    }
}

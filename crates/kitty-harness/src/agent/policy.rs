//! Agent loop policy (W2 split of `agent/mod.rs`, por símbolo).
//!
//! Fail-fast, autorun hops, doom-loop guard, stance overlays, thinking caps.
use crate::model_profile::ThinkingMode;
use crate::overflow;
use kitty_runtime::Message;
/// Maximum consecutive empty LLM responses before failing fast.
///
/// After this many empty responses the agent terminates the turn quickly
/// instead of looping until `max_iterations`/timeout (handoff §6).
/// Gate 2 tuning: 3→2 for faster fail-fast on small models.
const MAX_EMPTY_STREAK: u8 = 2;

/// Returns `true` when the empty-response streak has reached the fail-fast cap.
///
/// Exposed for unit testing the streak logic without a live LLM.
pub fn should_fail_fast(streak: u8) -> bool {
    streak >= MAX_EMPTY_STREAK
}

// ---------------------------------------------------------------------------
// E3 deterministic guards (zero model intelligence required)
// ---------------------------------------------------------------------------

/// Doom-loop guard: identical tool+args executions before a corrective nudge.
pub(crate) const DOOM_LOOP_NUDGE_AT: usize = 3;
/// Doom-loop guard: identical tool+args executions before fail-fast abort.
pub(crate) const DOOM_LOOP_FAIL_AT: usize = 6;
/// System nudge injected when the doom-loop streak reaches [`DOOM_LOOP_NUDGE_AT`].
pub(crate) const DOOM_LOOP_NUDGE_MSG: &str =
    "Loop detected: same tool+args 3×. Change approach or provide a different strategy.";
/// E3 step cap: injected as a trailing user message on the final iteration so
/// the model wraps up instead of being cut mid-action (opencode
/// MAX_STEPS_PROMPT pattern).
pub(crate) const MAX_STEPS_PROMPT: &str =
    "FINAL ITERATION: summarize progress and deliver your best result now.";

/// Autorun continuation prompt (f8 AD1): pushed when an iteration budget is
/// exhausted while [`AgentConfig::autorun_hops`] remain. Style-matches
/// [`MAX_STEPS_PROMPT`] but signals the opposite — keep working the SAME task
/// with tools; stop calling tools only when done. The two are never injected
/// in the same cycle: MAX_STEPS is suppressed while hops remain.
pub(crate) const AUTORUN_CONTINUE_PROMPT: &str =
    "CONTINUE: you have more budget — keep working on the same task using tools. \
     When the task is complete, stop calling tools and give your final answer.";

/// Hard upper bound for [`AgentConfig::autorun_hops`] (clamp helper below).
/// Each hop multiplies worst-case LLM calls by another full tool-loop cycle,
/// so runaway values are rejected at every set site.
pub const MAX_AUTORUN_HOPS: usize = 12;

/// Clamp `autorun_hops` into 0..=[`MAX_AUTORUN_HOPS`] (usize cannot be
/// negative). Applied wherever the value is set: config default stays 0
/// (byte-identical legacy behavior), CLI/env/bench values are bounded here.
#[must_use]
pub fn clamp_autorun_hops(hops: usize) -> usize {
    hops.min(MAX_AUTORUN_HOPS)
}

/// Fixed-machine rule (f8 AD1): after verification FAILS and the revision
/// budget is exhausted, one autorun hop may buy one more full cycle.
///
/// Reachability (G1 rider): UNREACHABLE on every current live path. The
/// `'autorun` loop provably exits ONLY through its `break`, and that break is
/// guarded by `hops_remaining == 0` (any positive balance is spent one hop
/// earlier to restart the inner loop instead of falling through). The
/// verification section runs strictly after that break, so it always observes
/// `hops_remaining == 0` and this predicate is constant-false in practice.
/// The arm is retained deliberately: it encodes the AD1 machine rule for any
/// future path where hops may legitimately survive into verification, and it
/// stays unit-tested so the invariant is pinned.
/// Maximum revision cycles after a failed verification (spec: up to 2).
pub(crate) const VERIFY_MAX_REVISIONS: usize = 2;

#[must_use]
pub(crate) fn autorun_hop_retries_after_revisions_exhausted(
    hops_remaining: usize,
    revision_cycles: usize,
) -> bool {
    hops_remaining > 0 && revision_cycles >= VERIFY_MAX_REVISIONS
}

/// f8 AD6/m7: walk-back window (in messages) when searching for a safe
/// user-turn boundary before the naive compaction split point. Bounded so a
/// pathologically long assistant→tool run cannot scan the whole history.
const COMPACTION_WALKBACK_MAX: usize = 12;

/// Pure boundary selection for compaction (f8 AD6 + G3-FIX2): if the naive
/// split ALREADY lands on a user turn, keep it exactly (no under-freeing).
/// Otherwise walk BACK up to [`COMPACTION_WALKBACK_MAX`] messages to the
/// nearest index `i > 0` where `messages[i]` begins a USER turn. Cutting at
/// a user boundary can never orphan tool results or split an
/// assistant→tool run (rule 3): every kept message from there on belongs to
/// a complete turn. If no user boundary exists within the window, fall back
/// to the naive split (degenerate histories stay graceful).
#[must_use]
pub(crate) fn compaction_split_boundary(messages: &[Message], naive_split_at: usize) -> usize {
    if naive_split_at == 0 {
        return 0;
    }
    // G3-FIX2: the naive point is tested FIRST — when it already sits on a
    // user boundary it is used as-is instead of being skipped for an older,
    // strictly-less-freeing cut.
    if messages[naive_split_at].role == "user" {
        return naive_split_at;
    }
    // Remaining candidates are naive-1 down to max(naive-K, 1) — index 0 is
    // excluded (a zero split would compact nothing).
    let lower = naive_split_at
        .saturating_sub(COMPACTION_WALKBACK_MAX)
        .max(1);
    let mut idx = naive_split_at;
    while idx > lower {
        idx -= 1;
        if messages[idx].role == "user" {
            return idx;
        }
    }
    naive_split_at
}

/// Collapse consecutive trailing `assistant` messages to at most one (B1-400
/// wire sanitizer). Idempotent, tail-only — interior duplicates untouched.
/// MERGE: if any tail assistant has tool_calls, concat all onto the LAST;
/// DROP: else keep only the LAST. Cheap O(tail) scan, no allocation when tail≤1.
pub fn sanitize_trailing_assistants(mut history: Vec<Message>) -> Vec<Message> {
    let trailing = history
        .iter()
        .rev()
        .take_while(|m| m.role == "assistant")
        .count();
    if trailing <= 1 {
        return history;
    }
    let len = history.len();
    let start = len - trailing;
    let mut merged: Vec<kitty_runtime::ToolCall> = Vec::new();
    for msg in &history[start..] {
        if let Some(tcs) = &msg.tool_calls {
            merged.extend(tcs.clone());
        }
    }
    let last = history[start + trailing - 1].clone();
    history.truncate(start);
    let collapsed = if merged.is_empty() {
        last
    } else {
        Message {
            role: "assistant".to_string(),
            content: last.content.clone(),
            thinking: last.thinking.clone(),
            tool_calls: Some(merged),
            tool_call_id: None,
        }
    };
    history.push(collapsed);
    history
}

/// Verdict returned by [`DoomLoopGuard::record`] after one tool execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DoomLoopVerdict {
    /// No repetition pattern (or streak below the nudge threshold).
    Fresh,
    /// Streak reached [`DOOM_LOOP_NUDGE_AT`] — inject the corrective nudge.
    Nudge,
    /// Streak reached [`DOOM_LOOP_FAIL_AT`] — fail fast.
    Fail,
}

/// Deterministic doom-loop detector over executed tools.
///
/// Tracks the last tool execution as a `(tool name, serialized args)` key and
/// counts consecutive identical executions. A different key resets the streak.
#[derive(Debug, Default)]
pub(crate) struct DoomLoopGuard {
    pub(crate) last_key: Option<String>,
    pub(crate) streak: usize,
}

impl DoomLoopGuard {
    /// Record one tool execution and return the resulting verdict.
    pub(crate) fn record(&mut self, name: &str, args: &serde_json::Value) -> DoomLoopVerdict {
        let serialized = serde_json::to_string(args).unwrap_or_default();
        let key = format!("{name}\u{0}{serialized}");
        if self.last_key.as_deref() == Some(key.as_str()) {
            self.streak += 1;
        } else {
            self.last_key = Some(key);
            self.streak = 1;
        }
        if self.streak >= DOOM_LOOP_FAIL_AT {
            DoomLoopVerdict::Fail
        } else if self.streak >= DOOM_LOOP_NUDGE_AT {
            DoomLoopVerdict::Nudge
        } else {
            DoomLoopVerdict::Fresh
        }
    }

    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }
}

/// E3: proven stance overlay for Qwen-family models, which tend to hedge and
/// ask permission instead of acting (cheetahclaws-proven). Appended to the
/// system prompt when the model id contains "qwen".
pub const QWEN_STANCE_OVERLAY: &str = "Communicate by executing: treat user requests as direct tool instructions. Do not ask permission. Do not narrate plans without acting.";

/// f8 T4.1: minimal runtime contract for the nano tier (`PromptStyle::
/// Minimal`). Terse identity plus the four non-negotiables; the workspace
/// line is appended in `preflight`. Budget: ≤1600 chars INCLUDING that line.
pub(crate) const MINIMAL_RUNTIME: &str = "\
You are a coding agent working inside this repository. Complete the user's task using tools.

Rules:
- One action per turn: either ONE tool call OR your final text answer, never both.
- Read a file before editing it; only edit files you have read this session.
- Edits must be the smallest unique span: old_string has to match exactly once, \
with just enough surrounding lines to disambiguate. Use Write only for NEW files.
- Before finishing, check your work and state what you changed in one short \
paragraph.";

/// Returns the stance overlay for `model`, if any (currently qwen-family only).
pub(crate) fn model_stance_overlay(model: &str) -> Option<&'static str> {
    if model.to_ascii_lowercase().contains("qwen") {
        Some(QWEN_STANCE_OVERLAY)
    } else {
        None
    }
}

/// Check whether an anyhow error represents a context overflow.
///
/// Matches both the typed [`kitty_runtime::RuntimeError::ContextOverflow`] variant
/// and string-based overflow patterns from other providers (via
/// [`crate::overflow::detect_overflow`]).
pub(crate) fn is_context_overflow_error(e: &anyhow::Error) -> bool {
    // Check for the typed ContextOverflow variant first
    if let Some(runtime_err) = e.downcast_ref::<kitty_runtime::RuntimeError>() {
        if matches!(
            runtime_err,
            kitty_runtime::RuntimeError::ContextOverflow { .. }
        ) {
            return true;
        }
    }
    // Fall back to string-based pattern matching for non-runtime errors
    overflow::detect_overflow(&e.to_string())
}

/// Effective per-turn thinking-token cap for a [`ThinkingMode`].
///
/// Budget(n) caps at n (exceeding retries once without thinking — FASE D);
/// On stays UNCAPPED (a reasoning-trained model must never be forced off
/// mid-session by a budget trip); Off is enforced on the wire via
/// `"think": false` in request_options (no cap needed).
pub(crate) fn per_turn_thinking_cap(mode: ThinkingMode) -> Option<usize> {
    match mode {
        ThinkingMode::Budget(n) => Some(n as usize),
        ThinkingMode::On | ThinkingMode::Off => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{tool_call_response, Agent, AgentConfig};
    use super::*;
    use crate::quality_monitor::QualityConfig;
    use kitty_runtime::{Message, MockClient};

    #[test]
    fn should_fail_fast_false_before_cap() {
        assert!(!should_fail_fast(0));
        assert!(!should_fail_fast(1));
        assert!(!Agent::should_fail_fast(1));
    }

    #[test]
    fn should_fail_fast_true_at_and_beyond_cap() {
        assert!(should_fail_fast(2));
        assert!(should_fail_fast(3));
        assert!(should_fail_fast(255));
        assert!(Agent::should_fail_fast(2));
        assert_eq!(MAX_EMPTY_STREAK, 2);
    }

    #[test]
    fn doom_loop_guard_nudges_at_three_and_fails_at_six() {
        let mut guard = DoomLoopGuard::default();
        let args = serde_json::json!({"path": "a.rs"});
        // 1st and 2nd identical executions — no verdict
        assert_eq!(guard.record("Read", &args), DoomLoopVerdict::Fresh);
        assert_eq!(guard.record("Read", &args), DoomLoopVerdict::Fresh);
        // 3rd consecutive identical → nudge
        assert_eq!(guard.record("Read", &args), DoomLoopVerdict::Nudge);
        // 4th / 5th → still nudge territory (nudge fires at exactly 3)
        assert_eq!(guard.record("Read", &args), DoomLoopVerdict::Nudge);
        assert_eq!(guard.record("Read", &args), DoomLoopVerdict::Nudge);
        // 6th consecutive identical → fail fast
        assert_eq!(guard.record("Read", &args), DoomLoopVerdict::Fail);
    }

    #[test]
    fn doom_loop_guard_resets_on_different_tool_or_args() {
        let mut guard = DoomLoopGuard::default();
        let a = serde_json::json!({"path": "a.rs"});
        let b = serde_json::json!({"path": "b.rs"});
        assert_eq!(guard.record("Read", &a), DoomLoopVerdict::Fresh);
        assert_eq!(guard.record("Read", &a), DoomLoopVerdict::Fresh);
        // Different args reset the streak
        assert_eq!(guard.record("Read", &b), DoomLoopVerdict::Fresh);
        assert_eq!(guard.record("Read", &b), DoomLoopVerdict::Fresh);
        // Two more with `b` → 3rd consecutive of `b` nudges
        assert_eq!(guard.record("Read", &b), DoomLoopVerdict::Nudge);
        // Different tool resets too
        assert_eq!(guard.record("Grep", &b), DoomLoopVerdict::Fresh);
    }

    #[tokio::test]
    async fn process_response_doom_loop_nudges_then_fails_fast() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("doom.txt");
        std::fs::write(&file_path, "content").unwrap();

        let mock = MockClient::new(vec![]);
        let cfg = AgentConfig {
            workspace: tmp.path().to_path_buf(),
            quality_config: QualityConfig {
                loop_threshold: 100, // disable quality ToolLoop so only the doom guard fires
            },
            ..Default::default()
        };
        let mut agent = Agent::new(mock, cfg);

        let response = tool_call_response(
            "Read",
            serde_json::json!({"path": file_path.to_string_lossy()}),
        );

        // Executions 1-2: no nudge
        for _ in 0..2 {
            agent.process_response("p", response.clone()).await.unwrap();
        }
        assert!(!agent
            .messages
            .iter()
            .any(|m| m.content.as_deref() == Some(DOOM_LOOP_NUDGE_MSG)));

        // Execution 3: system nudge injected after the tool result
        agent.process_response("p", response.clone()).await.unwrap();
        let last = agent.messages.last().unwrap();
        assert_eq!(last.role, "system");
        assert_eq!(last.content.as_deref(), Some(DOOM_LOOP_NUDGE_MSG));

        // Executions 4-5: still nudge-only
        for _ in 0..2 {
            agent.process_response("p", response.clone()).await.unwrap();
        }

        // Execution 6: fail-fast error aborts the loop
        let err = agent.process_response("p", response).await.unwrap_err();
        assert!(
            err.to_string().contains("Doom loop"),
            "expected doom-loop fail-fast, got: {err}"
        );
    }

    #[test]
    fn clamp_autorun_hops_bounds_surface() {
        // Good inputs pass through.
        assert_eq!(clamp_autorun_hops(0), 0);
        assert_eq!(clamp_autorun_hops(1), 1);
        assert_eq!(clamp_autorun_hops(12), 12);
        // Bad inputs clamp to the cap (usize cannot go negative).
        assert_eq!(clamp_autorun_hops(13), 12);
        assert_eq!(clamp_autorun_hops(usize::MAX), 12);
        assert_eq!(MAX_AUTORUN_HOPS, 12);
    }

    #[test]
    fn autorun_hop_retry_rule_after_revisions_exhausted() {
        // Good: hop available AND revisions exhausted -> retry via hop.
        assert!(autorun_hop_retries_after_revisions_exhausted(
            1,
            VERIFY_MAX_REVISIONS
        ));
        // Bad: no hop left -> terminal return even with revisions exhausted.
        assert!(!autorun_hop_retries_after_revisions_exhausted(
            0,
            VERIFY_MAX_REVISIONS
        ));
        // Bad: revisions NOT exhausted yet -> revision branch handles it.
        assert!(!autorun_hop_retries_after_revisions_exhausted(
            1,
            VERIFY_MAX_REVISIONS - 1
        ));
    }

    #[test]
    fn sanitize_trailing_assistants_merge_drop_and_noop() {
        let m = |role: &str, content: &str| Message {
            role: role.to_string(),
            content: Some(content.to_string()),
            thinking: None,
            tool_calls: None,
            tool_call_id: None,
        };
        let a_tc = |name: &str| Message {
            role: "assistant".to_string(),
            content: None,
            thinking: None,
            tool_calls: Some(vec![kitty_runtime::ToolCall {
                id: None,
                function: kitty_runtime::ToolCallFunction {
                    name: name.to_string(),
                    arguments: serde_json::json!({}),
                },
            }]),
            tool_call_id: None,
        };

        // MERGE case: [user, assistant(tool), assistant(text)] → tool_calls
        // carried onto the kept LAST assistant.
        let history = vec![
            m("user", "goal"),
            a_tc("Read"),
            m("assistant", "later text"),
        ];
        let out = sanitize_trailing_assistants(history);
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].content.as_deref(), Some("later text"));
        let tcs = out[1].tool_calls.as_ref().expect("tool_calls carried over");
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].function.name, "Read");

        // DROP case: plain consecutive assistants keep only the LAST content.
        let history = vec![
            m("user", "goal"),
            m("assistant", "first"),
            m("assistant", "second"),
        ];
        let out = sanitize_trailing_assistants(history);
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].content.as_deref(), Some("second"));
        assert!(out[1].tool_calls.is_none());

        // NORMAL cases untouched.
        let history = vec![m("user", "q"), m("assistant", "a"), m("user", "q2")];
        let out = sanitize_trailing_assistants(history.clone());
        assert_eq!(out, history);
        let single = vec![m("assistant", "only")];
        assert_eq!(sanitize_trailing_assistants(single.clone()), single);
        assert!(sanitize_trailing_assistants(Vec::new()).is_empty());
        // Interior duplicates are NOT touched (wire constraint is about the tail).
        let history = vec![m("assistant", "x"), m("assistant", "y"), m("user", "z")];
        let out = sanitize_trailing_assistants(history);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn compaction_split_boundary_pure_table() {
        let m = |role: &str| Message {
            role: role.to_string(),
            content: Some(role.to_string()),
            thinking: None,
            tool_calls: None,
            tool_call_id: None,
        };
        // Nearest user within the window wins.
        let history = vec![
            m("user"),
            m("assistant"),
            m("tool"),
            m("user"),
            m("assistant"),
        ];
        assert_eq!(compaction_split_boundary(&history, 4), 3);

        // Tool at naive split → walk back to its turn-start user.
        let history = vec![
            m("user"),
            m("assistant"),
            m("user"),
            m("assistant"),
            m("tool"),
            m("user"),
        ];
        assert_eq!(compaction_split_boundary(&history, 4), 2);

        // No user in reach within K=12 → naive fallback.
        let mut history = vec![m("system"), m("assistant"), m("tool")];
        for _ in 0..12 {
            history.push(m("assistant"));
            history.push(m("tool"));
        }
        history.push(m("user"));
        history.push(m("assistant"));
        // naive_split_at points deep into the long assistant→tool run; the
        // only user sits >12 back ⇒ naive fallback.
        let naive = history.len() - 2;
        assert_eq!(compaction_split_boundary(&history, naive), naive);

        // Degenerate: no user at all → naive fallback.
        let history = vec![m("system"), m("assistant"), m("tool"), m("assistant")];
        assert_eq!(compaction_split_boundary(&history, 2), 2);
        // Zero split stays zero.
        assert_eq!(compaction_split_boundary(&history, 0), 0);

        // G3-FIX2: naive lands EXACTLY on a user ⇒ returned unchanged (the
        // old decrement-first scan skipped it for an older turn).
        let history = vec![
            m("user"),
            m("assistant"),
            m("user"),
            m("assistant"),
            m("user"),
            m("assistant"),
        ];
        assert_eq!(
            compaction_split_boundary(&history, 4),
            4,
            "exact user-boundary landing must not under-free"
        );
        // Mixed: exact landing beats walking back even when older users exist.
        let history = vec![
            m("user"),
            m("assistant"),
            m("tool"),
            m("user"),
            m("assistant"),
            m("user"),
            m("assistant"),
        ];
        assert_eq!(compaction_split_boundary(&history, 5), 5);
    }

    #[test]
    fn qwen_stance_overlay_pure_predicate() {
        assert_eq!(model_stance_overlay("qwen3:8b"), Some(QWEN_STANCE_OVERLAY));
        assert_eq!(
            model_stance_overlay("Qwen2.5-Coder-7B"),
            Some(QWEN_STANCE_OVERLAY)
        );
        assert_eq!(model_stance_overlay("ornith:9b"), None);
        assert_eq!(model_stance_overlay("llama3"), None);
    }
}

//! ReAct loop typing — Thought → Action → Observation.
//!
//! W7: makes the agent loop's implicit step explicit without migrating any
//! existing data flow. [`Thought`] is EPHEMERAL by design: it carries the
//! model's reasoning for the current turn only, is never persisted to memory
//! or telemetry, and intentionally implements NO `Serialize` (so a future
//! `#[derive(Serialize)]` on a parent cannot silently persist it).
//!
//! [`DoomLoopGuard`](super::DoomLoopGuard) interop is direct: the doom test
//! below drives the REAL guard through [`ReActStep`] actions (no mirrored
//! streak logic).
//!
//! NOTE: these types are exported but the agent loop does not construct
//! `ReActStep` yet — no loop migration, no persistence change (W7 scope).

use crate::parser::ParsedToolCall;
use crate::tools::ToolResult;

/// Ephemeral model reasoning for one turn.
///
/// Fresh per step; never stored. Construction truncates to a bounded excerpt
/// so an unbounded chain-of-thought cannot blow the context window when a
/// caller embeds the summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Thought {
    summary: String,
}

/// Maximum chars kept from raw thinking text.
pub const THOUGHT_SUMMARY_CAP: usize = 500;

impl Thought {
    /// Build from optional raw thinking/content text (both `None` → empty).
    pub fn new(raw: Option<&str>) -> Self {
        let summary = raw
            .unwrap_or("")
            .chars()
            .take(THOUGHT_SUMMARY_CAP)
            .collect();
        Self { summary }
    }

    /// Bounded excerpt of the turn's reasoning.
    pub fn summary(&self) -> &str {
        &self.summary
    }

    /// `true` when no reasoning text was supplied.
    pub fn is_empty(&self) -> bool {
        self.summary.is_empty()
    }
}

/// A typed tool invocation — the ReAct "Action".
#[derive(Debug, Clone, PartialEq)]
pub struct Action {
    /// Canonical tool name (post-normalization casing not assumed).
    pub name: String,
    /// Arguments object passed to the tool.
    pub arguments: serde_json::Value,
    /// Server-issued call id, when the backend supplied one.
    pub id: Option<String>,
}

impl Action {
    /// Build directly from parts.
    pub fn new(name: impl Into<String>, arguments: serde_json::Value, id: Option<String>) -> Self {
        Self {
            name: name.into(),
            arguments,
            id,
        }
    }

    /// Lift a parsed tool call (pre- or post-normalization) into an `Action`.
    pub fn from_call(call: &ParsedToolCall) -> Self {
        Self {
            name: call.name.clone(),
            arguments: call.arguments.clone(),
            id: call.id.clone(),
        }
    }
}

/// A typed tool outcome — the ReAct "Observation".
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    /// Tool output text (possibly a truncation excerpt with pointer).
    pub output: String,
    /// Whether the tool reported success.
    pub success: bool,
    /// Whether `output` is a head+tail excerpt pointing at a persisted file.
    pub truncated: bool,
}

impl Observation {
    /// Lift a [`ToolResult`]; `truncated` is inferred from the truncation
    /// pointer marker so callers need not thread it separately.
    pub fn from_result(result: &ToolResult) -> Self {
        Self {
            output: result.output.clone(),
            success: result.success,
            truncated: result.output.contains("[truncated: full output at "),
        }
    }
}

/// One explicit ReAct turn: ephemeral thought, typed action, optional
/// observation (absent when the action was blocked before execution).
#[derive(Debug, Clone, PartialEq)]
pub struct ReActStep {
    /// This turn's reasoning. Ephemeral — never persisted.
    pub thought: Thought,
    /// The tool invocation decided for this turn.
    pub action: Action,
    /// The tool outcome, once executed.
    pub observation: Option<Observation>,
}

impl ReActStep {
    /// Start a step from thinking text + a parsed call, pre-execution.
    pub fn new(thinking: Option<&str>, call: &ParsedToolCall) -> Self {
        Self {
            thought: Thought::new(thinking),
            action: Action::from_call(call),
            observation: None,
        }
    }

    /// Attach the execution outcome (builder style).
    pub fn with_observation(mut self, observation: Observation) -> Self {
        self.observation = Some(observation);
        self
    }

    /// Doom-loop detector key for this step's action.
    ///
    /// Contract: identical format to the loop guard's
    /// `(tool name, serialized args)` streak key, so a sequence of identical
    /// steps drives the guard to `Fail` (pinned by test, no signature
    /// coupling to the guard itself).
    pub fn doom_key(&self) -> String {
        let serialized = serde_json::to_string(&self.action.arguments).unwrap_or_default();
        format!("{}\u{0}{serialized}", self.action.name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, args: serde_json::Value) -> ParsedToolCall {
        ParsedToolCall {
            name: name.to_string(),
            arguments: args,
            id: None,
        }
    }

    #[test]
    fn thought_is_ephemeral_and_bounded() {
        assert!(Thought::new(None).is_empty());
        assert_eq!(Thought::new(Some("")).summary(), "");
        let long = "x".repeat(THOUGHT_SUMMARY_CAP + 100);
        let t = Thought::new(Some(&long));
        assert_eq!(t.summary().chars().count(), THOUGHT_SUMMARY_CAP);
    }

    #[test]
    fn action_lifts_parsed_call() {
        let c = call("Read", serde_json::json!({"path": "a.rs"}));
        let a = Action::from_call(&c);
        assert_eq!(a.name, "Read");
        assert_eq!(a.arguments["path"], "a.rs");
        assert!(a.id.is_none());
    }

    #[test]
    fn observation_marks_truncation_pointer() {
        let full = ToolResult {
            success: true,
            output: "Wrote x".to_string(),
            retry: false,
        };
        assert!(!Observation::from_result(&full).truncated);
        let cut = ToolResult {
            success: true,
            output: "head\n...\n[truncated: full output at /tmp/t.txt]".to_string(),
            retry: false,
        };
        let o = Observation::from_result(&cut);
        assert!(o.truncated && o.success);
    }

    #[test]
    fn step_builds_then_attaches_observation() {
        let c = call("Bash", serde_json::json!({"command": "echo hi"}));
        let step = ReActStep::new(Some("need shell output"), &c);
        assert!(!step.thought.is_empty());
        assert!(step.observation.is_none());
        let done = step.with_observation(Observation::from_result(&ToolResult {
            success: true,
            output: "hi".to_string(),
            retry: false,
        }));
        assert_eq!(done.observation.unwrap().output, "hi");
    }

    /// W7 wiring: the loop emits one `ReActStep` per executed tool and the
    /// REAL `DoomLoopGuard` consumes the steps (nudge at 3). Fails without
    /// the `process_response → ReActStep → doom_loop` threading (trace empty).
    #[tokio::test]
    async fn loop_emits_steps_consumed_by_real_guard() {
        use super::super::{tool_call_response, Agent, AgentConfig, DOOM_LOOP_NUDGE_MSG};
        use crate::quality_monitor::QualityConfig;
        use kitty_runtime::MockClient;

        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("wired.txt");
        std::fs::write(&file_path, "content").unwrap();
        let args = serde_json::json!({"path": file_path.to_string_lossy()});

        let mut agent = Agent::new(
            MockClient::new(vec![]),
            AgentConfig {
                workspace: tmp.path().to_path_buf(),
                quality_config: QualityConfig {
                    loop_threshold: 100, // disable quality ToolLoop: only the doom guard fires
                },
                ..Default::default()
            },
        );
        assert!(agent.react_steps().is_empty());

        let response = tool_call_response("Read", args);
        agent.process_response("p", response.clone()).await.unwrap();
        agent.process_response("p", response.clone()).await.unwrap();

        // Two turns → two steps, action + observation populated.
        let steps = agent.react_steps();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].action.name, "Read");
        assert_eq!(steps[1].action.name, "Read");
        let obs = steps[0].observation.as_ref().expect("observation attached");
        assert!(obs.success);
        assert!(obs.output.contains("content"));
        assert!(steps[0].thought.is_empty());

        // Third identical execution → the REAL guard nudges via the steps.
        agent.process_response("p", response).await.unwrap();
        assert_eq!(agent.react_steps().len(), 3);
        assert!(
            agent
                .messages
                .iter()
                .any(|m| m.content.as_deref() == Some(DOOM_LOOP_NUDGE_MSG)),
            "doom guard must nudge on the 3rd identical step"
        );
    }
    /// W7 doom-loop: six identical ReAct steps must drive the REAL
    /// [`DoomLoopGuard`](super::DoomLoopGuard) from Fresh to Fail (nudge at
    /// 3), proving `doom_key` parity without mirrored streak logic.
    #[test]
    fn react_steps_drive_doom_guard_to_fail() {
        use super::super::{DoomLoopGuard, DoomLoopVerdict};

        let args = serde_json::json!({"path": "loop.rs"});
        let keys: Vec<String> = (0..6)
            .map(|_| ReActStep::new(None, &call("Read", args.clone())).doom_key())
            .collect();
        assert!(
            keys.iter().all(|k| *k == keys[0]),
            "identical steps, identical keys"
        );

        let mut guard = DoomLoopGuard::default();
        let mut verdicts = Vec::new();
        for _ in 0..6 {
            let step = ReActStep::new(None, &call("Read", args.clone()));
            verdicts.push(guard.record(&step.action.name, &step.action.arguments));
        }
        assert_eq!(
            verdicts,
            vec![
                DoomLoopVerdict::Fresh,
                DoomLoopVerdict::Fresh,
                DoomLoopVerdict::Nudge,
                DoomLoopVerdict::Nudge,
                DoomLoopVerdict::Nudge,
                DoomLoopVerdict::Fail,
            ]
        );

        // A different action resets the streak.
        let other = ReActStep::new(None, &call("Bash", serde_json::json!({"command": "x"})));
        assert_ne!(other.doom_key(), keys[0]);
        assert_eq!(
            guard.record(&other.action.name, &other.action.arguments),
            DoomLoopVerdict::Fresh
        );
    }
}

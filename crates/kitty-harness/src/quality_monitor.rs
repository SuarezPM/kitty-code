//! Quality monitor — detects and mitigates poor agent behavior.
//!
//! Detects:
//! - Empty responses (no content, no tool calls)
//! - Tool loops (same tool call 3+ times in a row)
//! - Hallucinated tools (tool name not in the registered set)
//!
//! When detected, injects corrective messages or forces a text response.

use std::collections::{HashMap, VecDeque};

use crate::parser::ParsedToolCall;

/// Types of quality violations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualityViolation {
    /// No content and no tool calls in the response.
    EmptyResponse,
    /// Same tool called `count` times consecutively.
    ToolLoop {
        tool_name: String,
        count: usize,
    },
    /// Tool name is not in the registered set.
    HallucinatedTool { tool_name: String },
}

/// A quality event recorded for the session.
#[derive(Debug, Clone)]
pub struct QualityEvent {
    pub turn: usize,
    pub violation: QualityViolation,
    pub corrective_action: String,
}

/// Configuration for the quality monitor.
#[derive(Debug, Clone)]
pub struct QualityConfig {
    /// How many consecutive identical tool calls before triggering a loop.
    pub loop_threshold: usize,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            loop_threshold: 3,
        }
    }
}

/// Tracks and detects quality issues across the session.
pub struct QualityMonitor {
    config: QualityConfig,
    /// Registered tool names (for hallucination detection).
    registered_tools: std::collections::HashSet<String>,
    /// Recent tool calls for loop detection (sliding window) — name + hash of args.
    recent_tool_calls: VecDeque<(String, u64)>,
    /// Quality events recorded this session.
    events: Vec<QualityEvent>,
    /// Current turn number.
    turn: usize,
}

impl QualityMonitor {
    /// Hash tool arguments deterministically for loop detection.
    fn args_hash(args: &serde_json::Value) -> u64 {
        let s = serde_json::to_string(args).unwrap_or_default();
        let hash = blake3::hash(s.as_bytes());
        let bytes = hash.as_bytes();
        u64::from_le_bytes(bytes[0..8].try_into().unwrap_or([0u8; 8]))
    }

    /// Create a new quality monitor.
    pub fn new(config: QualityConfig, registered_tools: Vec<String>) -> Self {
        Self {
            config,
            registered_tools: registered_tools.into_iter().collect(),
            recent_tool_calls: VecDeque::new(),
            events: Vec::new(),
            turn: 0,
        }
    }

    /// Advance to the next turn.
    ///
    /// F2.4: The sliding window is NOT cleared between turns so that
    /// cross-turn tool loops (same tool+args across 3+ consecutive turns)
    /// are detected. The window is only reset when a different tool appears
    /// or a pure text response breaks the pattern.
    pub fn next_turn(&mut self) {
        self.turn += 1;
        // Intentionally NOT clearing recent_tool_calls here (F2.4).
    }

    /// Check an LLM response for quality issues.
    /// Returns the list of violations found and suggested corrective actions.
    ///
    /// F2.4: The sliding window persists across turns but resets when a
    /// different tool name appears or a pure text response breaks the pattern.
    pub fn check_response(
        &mut self,
        content: Option<&str>,
        tool_calls: &[ParsedToolCall],
    ) -> Vec<QualityViolation> {
        let mut violations = Vec::new();

        // Check for empty response
        let has_content = content
            .as_ref()
            .map(|c| !c.trim().is_empty())
            .unwrap_or(false);
        let has_tools = !tool_calls.is_empty();

        if !has_content && !has_tools {
            violations.push(QualityViolation::EmptyResponse);
        }

        // F2.4: Reset window when a pure text response breaks the tool pattern
        if !has_tools && !self.recent_tool_calls.is_empty() {
            self.recent_tool_calls.clear();
        }

        // Check for hallucinated tools
        for tc in tool_calls {
            if !self.registered_tools.contains(&tc.name) {
                violations.push(QualityViolation::HallucinatedTool {
                    tool_name: tc.name.clone(),
                });
            }
        }

        // Check for tool loops — args-aware dedup (spec small-model-tool-loop)
        for tc in tool_calls {
            let arg_hash = Self::args_hash(&tc.arguments);
            // F2.4 + args-aware: Reset window if tool name OR args differ (breaks the pattern)
            if let Some((last_name, last_hash)) = self.recent_tool_calls.back() {
                if last_name != &tc.name || last_hash != &arg_hash {
                    self.recent_tool_calls.clear();
                }
            }

            self.recent_tool_calls.push_back((tc.name.clone(), arg_hash));
            // Keep only the last N entries
            while self.recent_tool_calls.len() > self.config.loop_threshold {
                self.recent_tool_calls.pop_front();
            }

            // Check if all recent calls are the same tool WITH same args
            if self.recent_tool_calls.len() >= self.config.loop_threshold {
                let all_same = self
                    .recent_tool_calls
                    .iter()
                    .all(|(n, h)| n == &tc.name && h == &arg_hash);
                if all_same {
                    violations.push(QualityViolation::ToolLoop {
                        tool_name: tc.name.clone(),
                        count: self.recent_tool_calls.len(),
                    });
                }
            }
        }

        // Record events
        for violation in &violations {
            let action = match violation {
                QualityViolation::EmptyResponse => {
                    "Injected corrective message to prompt for content".to_string()
                }
                QualityViolation::ToolLoop { tool_name, .. } => {
                    format!("Injected message to break {tool_name} loop with same arguments")
                }
                QualityViolation::HallucinatedTool { tool_name } => {
                    format!("Rejected hallucinated tool: {tool_name}")
                }
            };
            self.events.push(QualityEvent {
                turn: self.turn,
                violation: violation.clone(),
                corrective_action: action,
            });
        }

        violations
    }

    /// Get a corrective user message for a given violation.
    /// This message is injected into the conversation to guide the model.
    pub fn corrective_message(violation: &QualityViolation) -> String {
        match violation {
            QualityViolation::EmptyResponse => {
                "Your response was empty. Please provide either text content or tool calls."
                    .to_string()
            }
            QualityViolation::ToolLoop {
                tool_name, count, ..
            } => {
                format!(
                    "You have called {tool_name} {count} times in a row with the same arguments. \
                     This appears to be a loop. Please try a different approach or provide a text response."
                )
            }
            QualityViolation::HallucinatedTool { tool_name } => {
                format!(
                    "The tool '{tool_name}' does not exist. Available tools are: Read, Edit, Write, Bash, Glob, Grep. \
                     Please use one of these tools."
                )
            }
        }
    }

    /// Get all quality events this session.
    pub fn events(&self) -> &[QualityEvent] {
        &self.events
    }

    /// Get violation counts by type.
    pub fn violation_summary(&self) -> HashMap<String, usize> {
        let mut summary = HashMap::new();
        for event in &self.events {
            let key = format!("{:?}", event.violation);
            *summary.entry(key).or_insert(0) += 1;
        }
        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_tools() -> Vec<String> {
        vec![
            "Read".to_string(),
            "Edit".to_string(),
            "Write".to_string(),
            "Bash".to_string(),
            "Glob".to_string(),
            "Grep".to_string(),
        ]
    }

    #[test]
    fn empty_response_detected() {
        let mut monitor = QualityMonitor::new(QualityConfig::default(), default_tools());
        let violations = monitor.check_response(None, &[]);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0], QualityViolation::EmptyResponse);
    }

    #[test]
    fn empty_content_with_tools_is_fine() {
        let mut monitor = QualityMonitor::new(QualityConfig::default(), default_tools());
        let tc = ParsedToolCall {
            name: "Read".to_string(),
            arguments: serde_json::json!({"path": "foo.rs"}),
        };
        let violations = monitor.check_response(None, &[tc]);
        assert!(violations.is_empty());
    }

    #[test]
    fn content_without_tools_is_fine() {
        let mut monitor = QualityMonitor::new(QualityConfig::default(), default_tools());
        let violations = monitor.check_response(Some("Here is the answer."), &[]);
        assert!(violations.is_empty());
    }

    #[test]
    fn hallucinated_tool_detected() {
        let mut monitor = QualityMonitor::new(QualityConfig::default(), default_tools());
        let tc = ParsedToolCall {
            name: "NonexistentTool".to_string(),
            arguments: serde_json::json!({}),
        };
        let violations = monitor.check_response(None, &[tc]);
        assert!(violations.iter().any(|v| matches!(
            v,
            QualityViolation::HallucinatedTool { tool_name }
                if tool_name == "NonexistentTool"
        )));
    }

    #[test]
    fn tool_loop_detected() {
        let config = QualityConfig {
            loop_threshold: 3,
        };
        let mut monitor = QualityMonitor::new(config, default_tools());
        let tc = ParsedToolCall {
            name: "Read".to_string(),
            arguments: serde_json::json!({"path": "foo.rs"}),
        };

        // First two calls — no loop yet
        monitor.check_response(None, std::slice::from_ref(&tc));
        monitor.check_response(None, std::slice::from_ref(&tc));
        let violations = monitor.check_response(None, &[tc]);
        assert!(violations.iter().any(|v| matches!(
            v,
            QualityViolation::ToolLoop { tool_name, count }
                if tool_name == "Read" && *count == 3
        )));
    }

    #[test]
    fn different_tools_no_loop() {
        let config = QualityConfig {
            loop_threshold: 3,
        };
        let mut monitor = QualityMonitor::new(config, default_tools());
        monitor.check_response(
            None,
            &[ParsedToolCall {
                name: "Read".to_string(),
                arguments: serde_json::json!({}),
            }],
        );
        monitor.check_response(
            None,
            &[ParsedToolCall {
                name: "Edit".to_string(),
                arguments: serde_json::json!({}),
            }],
        );
        let violations = monitor.check_response(
            None,
            &[ParsedToolCall {
                name: "Read".to_string(),
                arguments: serde_json::json!({}),
            }],
        );
        assert!(violations
            .iter()
            .all(|v| !matches!(v, QualityViolation::ToolLoop { .. })));
    }

    #[test]
    fn turn_reset_does_not_clear_loop_window() {
        // F2.4: next_turn() no longer clears the window — cross-turn loops
        // are detected. The window resets only when a different tool appears.
        let config = QualityConfig {
            loop_threshold: 3,
        };
        let mut monitor = QualityMonitor::new(config, default_tools());
        let tc = ParsedToolCall {
            name: "Read".to_string(),
            arguments: serde_json::json!({}),
        };
        monitor.check_response(None, std::slice::from_ref(&tc));
        monitor.check_response(None, std::slice::from_ref(&tc));
        monitor.next_turn();
        // After turn reset, window persists — third call triggers loop
        let violations = monitor.check_response(None, &[tc]);
        assert!(violations.iter().any(|v| matches!(
            v,
            QualityViolation::ToolLoop { tool_name, count }
                if tool_name == "Read" && *count == 3
        )));
    }

    #[test]
    fn different_tool_resets_loop_window() {
        // F2.4: A different tool name resets the sliding window.
        let config = QualityConfig {
            loop_threshold: 3,
        };
        let mut monitor = QualityMonitor::new(config, default_tools());
        let read_tc = ParsedToolCall {
            name: "Read".to_string(),
            arguments: serde_json::json!({}),
        };
        let edit_tc = ParsedToolCall {
            name: "Edit".to_string(),
            arguments: serde_json::json!({}),
        };
        monitor.check_response(None, std::slice::from_ref(&read_tc));
        monitor.check_response(None, std::slice::from_ref(&read_tc));
        // Different tool resets the window
        monitor.check_response(None, std::slice::from_ref(&edit_tc));
        // Now two more reads — still under threshold since window was reset
        let violations = monitor.check_response(None, std::slice::from_ref(&read_tc));
        assert!(violations
            .iter()
            .all(|v| !matches!(v, QualityViolation::ToolLoop { .. })));
    }

    #[test]
    fn corrective_message_content() {
        let msg = QualityMonitor::corrective_message(&QualityViolation::EmptyResponse);
        assert!(msg.contains("empty"));

        let msg = QualityMonitor::corrective_message(&QualityViolation::ToolLoop {
            tool_name: "Read".to_string(),
            count: 3,
        });
        assert!(msg.contains("Read"));
        assert!(msg.contains("3"));

        let msg = QualityMonitor::corrective_message(&QualityViolation::HallucinatedTool {
            tool_name: "FakeTool".to_string(),
        });
        assert!(msg.contains("FakeTool"));
    }

    #[test]
    fn events_tracked() {
        let mut monitor = QualityMonitor::new(QualityConfig::default(), default_tools());
        monitor.check_response(None, &[]);
        monitor.check_response(Some("text"), &[]);
        assert_eq!(monitor.events().len(), 1);
        assert_eq!(monitor.events()[0].turn, 0);
    }

    #[test]
    fn violation_summary() {
        let mut monitor = QualityMonitor::new(QualityConfig::default(), default_tools());
        monitor.check_response(None, &[]);
        monitor.check_response(None, &[]);
        let summary = monitor.violation_summary();
        assert_eq!(summary.len(), 1);
        assert_eq!(summary.values().next().unwrap(), &2);
    }

    #[test]
    fn same_tool_same_args_triggers_loop() {
        let config = QualityConfig { loop_threshold: 3 };
        let mut monitor = QualityMonitor::new(config, default_tools());
        let tc = ParsedToolCall {
            name: "Read".to_string(),
            arguments: serde_json::json!({"path": "foo.rs"}),
        };
        monitor.check_response(None, std::slice::from_ref(&tc));
        monitor.check_response(None, std::slice::from_ref(&tc));
        let violations = monitor.check_response(None, std::slice::from_ref(&tc));
        assert!(violations.iter().any(|v| matches!(
            v,
            QualityViolation::ToolLoop { tool_name, count }
                if tool_name == "Read" && *count == 3
        )));
        // corrective message must mention same arguments
        let msg = QualityMonitor::corrective_message(&QualityViolation::ToolLoop {
            tool_name: "Read".to_string(),
            count: 3,
        });
        assert!(msg.contains("same arguments"));
    }

    #[test]
    fn same_tool_different_args_no_loop() {
        let config = QualityConfig { loop_threshold: 3 };
        let mut monitor = QualityMonitor::new(config, default_tools());
        let tc_a = ParsedToolCall {
            name: "Read".to_string(),
            arguments: serde_json::json!({"path": "a.rs"}),
        };
        let tc_b = ParsedToolCall {
            name: "Read".to_string(),
            arguments: serde_json::json!({"path": "b.rs"}),
        };
        let tc_c = ParsedToolCall {
            name: "Read".to_string(),
            arguments: serde_json::json!({"path": "c.rs"}),
        };
        monitor.check_response(None, std::slice::from_ref(&tc_a));
        monitor.check_response(None, std::slice::from_ref(&tc_b));
        let violations = monitor.check_response(None, &[tc_c]);
        assert!(
            violations
                .iter()
                .all(|v| !matches!(v, QualityViolation::ToolLoop { .. })),
            "same tool with different args should not trigger loop"
        );
    }
}

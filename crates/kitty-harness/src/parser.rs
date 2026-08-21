//! Forgiving parser for tool calls from model output.
//!
//! Parses tool calls from multiple formats:
//! - Native Ollama `tool_calls` in the message object
//! - `<tool_call>` XML blocks in text content
//! - JSON objects embedded in text (```json ... ```)
//! - Reasoning models that embed tool calls in `<think>` blocks
//!
//! Includes auto-repair for malformed tool calls.

use serde_json::Value;

/// A parsed tool call with name and arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: Value,
}

/// Parser for tool calls from model output.
pub struct ToolCallParser;

/// Canonical error emitted when a tool call intent is detected but parsing fails.
const MALFORMED_ERROR: &str = "Malformed tool call detected. Expected format: {\"name\":\"ToolName\",\"arguments\":{...}} or <tool_call>{\"name\":\"...\",\"arguments\":{...}}</tool_call>. Available tools: Read, Edit, Write, Bash, Glob, Grep.";

impl ToolCallParser {
    pub fn new() -> Self {
        Self
    }

    /// Parse all tool calls from the model's output text.
    /// Returns empty vec if none found (caller decides what to do).
    pub fn parse(&self, content: &str) -> Vec<ParsedToolCall> {
        let results = self.parse_internal(content);

        if results.is_empty() && self.has_tool_call_intent(content) {
            tracing::warn!(content = %content, "malformed tool call detected (parse returned empty but intent heuristic matched)");
        }

        results
    }

    /// Parse with diagnostics: returns calls plus a list of diagnostic error strings.
    ///
    /// Heuristic: if content contains a tool call intent marker (`<tool_call>`,
    /// ` ```json `, `{"name"`, `{"function"`, `{"tool"`) but `parse()` yields
    /// no calls, a single diagnostic string is returned explaining the expected format.
    pub fn parse_with_diagnostics(&self, content: &str) -> (Vec<ParsedToolCall>, Vec<String>) {
        let calls = self.parse_internal(content);
        if calls.is_empty() && self.has_tool_call_intent(content) {
            (calls, vec![MALFORMED_ERROR.to_string()])
        } else {
            (calls, Vec::new())
        }
    }

    /// Try to parse, returning `Err` with a diagnostic if a malformed intent is detected.
    ///
    /// Returns `Ok(calls)` when parsing succeeds — including the legitimate empty case
    /// (no intent). Returns `Err(message)` only when intent heuristics fire but no
    /// call could be parsed.
    pub fn try_parse_with_error(&self, content: &str) -> Result<Vec<ParsedToolCall>, String> {
        let (calls, diagnostics) = self.parse_with_diagnostics(content);
        if let Some(err) = diagnostics.into_iter().next() {
            Err(err)
        } else {
            Ok(calls)
        }
    }

    /// Heuristic detector for "model tried to call a tool but syntax was broken".
    fn has_tool_call_intent(&self, content: &str) -> bool {
        content.contains("<tool_call>")
            || content.contains("```json")
            || content.contains(r#"{"name""#)
            || content.contains(r#"{"function""#)
            || content.contains(r#"{"tool""#)
    }

    /// Internal parsing logic without diagnostics side-effects.
    fn parse_internal(&self, content: &str) -> Vec<ParsedToolCall> {
        let mut results = Vec::new();

        // 1. Try XML <tool_call> blocks
        results.extend(self.parse_xml_tool_calls(content));

        // 2. Try JSON blocks (```json ... ```)
        results.extend(self.parse_json_code_blocks(content));

        // 3. Try standalone JSON objects that look like tool calls
        if results.is_empty() {
            results.extend(self.parse_standalone_json(content));
        }

        // 4. Try extracting from <think> blocks (reasoning models)
        if results.is_empty() {
            if let Some(thinking_content) = self.extract_thinking(content) {
                results.extend(self.parse_xml_tool_calls(thinking_content));
                results.extend(self.parse_json_code_blocks(thinking_content));
            }
        }

        results
    }

    /// Parse `<tool_call>{"name":"...","arguments":{...}}</tool_call>` XML blocks.
    fn parse_xml_tool_calls(&self, content: &str) -> Vec<ParsedToolCall> {
        let mut results = Vec::new();
        let mut search_start = 0;

        while let Some(start) = content[search_start..].find("<tool_call>") {
            let abs_start = search_start + start;
            let inner_start = abs_start + 8; // len("<tool_call>")

            if let Some(end) = content[inner_start..].find("</tool_call>") {
                let inner = &content[inner_start..inner_start + end];
                if let Some(tc) = self.parse_tool_call_json(inner.trim()) {
                    results.push(tc);
                }
                search_start = inner_start + end + 9; // len("</tool_call>")
            } else {
                // Unclosed tag — try to parse rest of content
                let inner = &content[inner_start..];
                if let Some(tc) = self.parse_tool_call_json(inner.trim()) {
                    results.push(tc);
                }
                break;
            }
        }

        results
    }

    /// Parse JSON code blocks: ```json\n{"name":"...", ...}\n```
    fn parse_json_code_blocks(&self, content: &str) -> Vec<ParsedToolCall> {
        let mut results = Vec::new();
        let mut search_start = 0;

        while let Some(start) = content[search_start..].find("```json") {
            let abs_start = search_start + start;
            let json_start = abs_start + 7; // len("```json")

            if let Some(end) = content[json_start..].find("```") {
                let json_str = content[json_start..json_start + end].trim();
                if let Some(tc) = self.parse_tool_call_json(json_str) {
                    results.push(tc);
                }
                search_start = json_start + end + 3;
            } else {
                let json_str = content[json_start..].trim();
                if let Some(tc) = self.parse_tool_call_json(json_str) {
                    results.push(tc);
                }
                break;
            }
        }

        results
    }

    /// Try to parse standalone JSON that looks like a tool call.
    fn parse_standalone_json(&self, content: &str) -> Vec<ParsedToolCall> {
        let trimmed = content.trim();

        // Try parsing the entire content as a single tool call
        if let Some(tc) = self.parse_tool_call_json(trimmed) {
            return vec![tc];
        }

        // Try finding JSON objects in the text
        let mut results = Vec::new();
        let mut depth = 0;
        let mut start = None;

        for (i, ch) in trimmed.char_indices() {
            match ch {
                '{' => {
                    if depth == 0 {
                        start = Some(i);
                    }
                    depth += 1;
                }
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(s) = start {
                            let candidate = &trimmed[s..=i];
                            if let Some(tc) = self.parse_tool_call_json(candidate) {
                                results.push(tc);
                            }
                        }
                        start = None;
                    }
                }
                _ => {}
            }
        }

        results
    }

    /// Extract content from <think>...</think> blocks.
    fn extract_thinking<'a>(&self, content: &'a str) -> Option<&'a str> {
        let start = content.find("<think>")?;
        let inner_start = start + 8;
        let end = content[inner_start..].find("</think>")?;
        Some(content[inner_start..inner_start + end].trim())
    }

    /// Try to parse a JSON string as a tool call.
    /// Handles: {"name":"...","arguments":{...}} and {"function":{"name":"...","arguments":{...}}}
    fn parse_tool_call_json(&self, json_str: &str) -> Option<ParsedToolCall> {
        let val: Value = serde_json::from_str(json_str).ok()?;

        // Format 1: {"name": "tool_name", "arguments": {...}}
        if let (Some(name), Some(args)) = (
            val.get("name").and_then(|v| v.as_str()),
            val.get("arguments"),
        ) {
            return Some(ParsedToolCall {
                name: name.to_string(),
                arguments: args.clone(),
            });
        }

        // Format 2: {"function": {"name": "tool_name", "arguments": {...}}}
        if let Some(func) = val.get("function") {
            if let (Some(name), Some(args)) = (
                func.get("name").and_then(|v| v.as_str()),
                func.get("arguments"),
            ) {
                return Some(ParsedToolCall {
                    name: name.to_string(),
                    arguments: args.clone(),
                });
            }
        }

        // Format 3: {"tool": "name", "parameters": {...}}
        if let (Some(name), Some(params)) = (
            val.get("tool").and_then(|v| v.as_str()),
            val.get("parameters"),
        ) {
            return Some(ParsedToolCall {
                name: name.to_string(),
                arguments: params.clone(),
            });
        }

        None
    }
}

impl Default for ToolCallParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_xml_tool_call() {
        let parser = ToolCallParser::new();
        let input = r#"<tool_call>{"name":"Read","arguments":{"path":"foo.rs"}}</tool_call>"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[0].arguments["path"], "foo.rs");
    }

    #[test]
    fn parse_json_code_block() {
        let parser = ToolCallParser::new();
        let input = r#"```json
{"name":"Bash","arguments":{"command":"ls -la"}}
```"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Bash");
    }

    #[test]
    fn parse_function_format() {
        let parser = ToolCallParser::new();
        let input =
            r#"{"function":{"name":"Edit","arguments":{"path":"a.rs","old":"x","new":"y"}}}"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Edit");
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let parser = ToolCallParser::new();
        let input = r#"<tool_call>{"name":"Read","arguments":{"path":"a.rs"}}</tool_call>
some text here
<tool_call>{"name":"Edit","arguments":{"path":"a.rs","old":"x","new":"y"}}"</tool_call>"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[1].name, "Edit");
    }

    #[test]
    fn parse_from_thinking_block() {
        let parser = ToolCallParser::new();
        let input = r#"<think>I need to read the file first.</think><tool_call>{"name":"Read","arguments":{"path":"main.rs"}}</tool_call>"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
    }

    #[test]
    fn parse_empty_content() {
        let parser = ToolCallParser::new();
        let calls = parser.parse("");
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_malformed_json_gracefully() {
        let parser = ToolCallParser::new();
        let input = "<tool_call>{not json}</tool_call>";
        let calls = parser.parse(input);
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_unclosed_xml_tag() {
        let parser = ToolCallParser::new();
        let input = r#"<tool_call>{"name":"Read","arguments":{"path":"foo.rs"}}"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
    }

    #[test]
    fn parse_tool_format() {
        let parser = ToolCallParser::new();
        let input = r#"{"tool":"Grep","parameters":{"pattern":"fn main"}}"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Grep");
        assert_eq!(calls[0].arguments["pattern"], "fn main");
    }

    // --- diagnostics hardening tests ---

    #[test]
    fn diagnostics_malformed_xml_returns_error() {
        let parser = ToolCallParser::new();
        let input = "<tool_call>{not json}</tool_call>";
        let calls = parser.parse(input);
        assert!(
            calls.is_empty(),
            "parse should still return empty for malformed"
        );

        let (calls2, diagnostics) = parser.parse_with_diagnostics(input);
        assert!(calls2.is_empty());
        assert_eq!(diagnostics.len(), 1);
        assert!(
            diagnostics[0].contains("Malformed"),
            "diagnostic should contain Malformed, got: {}",
            diagnostics[0]
        );
        assert!(
            diagnostics[0].contains("Available tools"),
            "diagnostic should list available tools"
        );

        let err = parser.try_parse_with_error(input).unwrap_err();
        assert!(err.contains("Malformed"));
    }

    #[test]
    fn diagnostics_valid_xml_no_error() {
        let parser = ToolCallParser::new();
        let input = r#"<tool_call>{"name":"Read","arguments":{"path":"foo.rs"}}</tool_call>"#;
        let (calls, diagnostics) = parser.parse_with_diagnostics(input);
        assert_eq!(calls.len(), 1);
        assert!(
            diagnostics.is_empty(),
            "valid call should produce no diagnostics"
        );
        assert!(parser.try_parse_with_error(input).is_ok());
    }

    #[test]
    fn diagnostics_plain_text_no_error() {
        let parser = ToolCallParser::new();
        let input = "Hello world, this is just plain text without any tool intent.";
        let (calls, diagnostics) = parser.parse_with_diagnostics(input);
        assert!(calls.is_empty());
        assert!(
            diagnostics.is_empty(),
            "plain text should not trigger diagnostic"
        );
        assert!(parser.try_parse_with_error(input).is_ok());
        assert!(parser.try_parse_with_error(input).unwrap().is_empty());
    }

    #[test]
    fn diagnostics_malformed_json_block() {
        let parser = ToolCallParser::new();
        let input = "```json\n{not valid json}\n```";
        let (calls, diagnostics) = parser.parse_with_diagnostics(input);
        assert!(calls.is_empty());
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].contains("Malformed"));
    }

    #[test]
    fn diagnostics_malformed_standalone_name() {
        let parser = ToolCallParser::new();
        // Contains {"name" heuristic but invalid JSON / missing arguments field
        let input = r#"{"name": "Read", "arguments": not_json}"#;
        let (calls, diagnostics) = parser.parse_with_diagnostics(input);
        assert!(calls.is_empty());
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].contains("Malformed"));
    }

    #[test]
    fn diagnostics_empty_no_error() {
        let parser = ToolCallParser::new();
        let (calls, diagnostics) = parser.parse_with_diagnostics("");
        assert!(calls.is_empty());
        assert!(diagnostics.is_empty());
        assert!(parser.try_parse_with_error("").is_ok());
    }
}

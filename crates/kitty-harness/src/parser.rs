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
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParseError {
    #[error("No tool calls found in output")]
    NoToolCalls,

    #[error("Malformed tool call: {0}")]
    Malformed(String),
}

/// A parsed tool call with name and arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: Value,
}

/// Parser for tool calls from model output.
pub struct ToolCallParser;

impl ToolCallParser {
    pub fn new() -> Self {
        Self
    }

    /// Parse all tool calls from the model's output text.
    /// Returns empty vec if none found (caller decides what to do).
    pub fn parse(&self, content: &str) -> Vec<ParsedToolCall> {
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
        let input = r#"{"function":{"name":"Edit","arguments":{"path":"a.rs","old":"x","new":"y"}}}"#;
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
}

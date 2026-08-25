//! Grammar enforcement spike — llama.cpp GBNF adapter stub.
//!
//! Generates a GBNF string that restricts tool-calls to `{"name": enum, "arguments": object}`
//! and validates via `serde_json` + required-field check. This is a spike, not a
//! production llama.cpp bindgen.
//! SPIKE defer: no llama.cpp bind — validated post-hoc via serde_json + required-field check.
//! Real enforcement requires llama-cpp-2 crate (LlamaSampler::grammar, json_schema_to_grammar) with decode-time enforcement.
//! Decision: defer to F6.4 ADR. See docs/STRATEGY.md F6.4.

use std::future::Future;
use std::pin::Pin;

use crate::client::{ChatClient, ChatResponse};
use crate::tools::ToolDefinition;

/// Configuration for grammar-constrained tool calling.
#[derive(Debug, Clone)]
pub struct GrammarConfig {
    /// Active tool schemas (profile + category routing filtered).
    pub tool_schemas: Vec<ToolDefinition>,
    /// When true, invalid tool calls are filtered out.
    pub strict: bool,
}

impl GrammarConfig {
    /// Create a new config.
    pub fn new(tool_schemas: Vec<ToolDefinition>, strict: bool) -> Self {
        Self {
            tool_schemas,
            strict,
        }
    }

    /// Create from a tool set with `strict = true` (default for the spike).
    pub fn from_tools(tools: Vec<ToolDefinition>) -> Self {
        Self::new(tools, true)
    }

    /// Generate a GBNF grammar that restricts output to a valid tool-call.
    ///
    /// The grammar encodes `{"name": <enum of tool names>, "arguments": <object>}`.
    /// It is valid GBNF for reference; actual enforcement in this spike is via
    /// [`Self::validate`] (serde_json + required fields).
    pub fn to_gbnf(&self) -> String {
        if self.tool_schemas.is_empty() {
            return "root ::= \"null\"\n".to_string();
        }
        let names: Vec<String> = self
            .tool_schemas
            .iter()
            .map(|t| format!("\"{}\"", t.function.name))
            .collect();
        let name_rule = format!("name ::= {}", names.join(" | "));

        // Keep GBNF valid and deterministic: reference-style grammar for
        // {"name": enum, "arguments": object} with generic JSON value rules.
        format!(
            "root ::= tool-call\n\
             tool-call ::= \"{{\" ws \"\\\"name\\\"\" ws \":\" ws name ws \",\" ws \"\\\"arguments\\\"\" ws \":\" ws arguments ws \"}}\"\n\
             {name_rule}\n\
             arguments ::= object\n\
             object ::= \"{{\" ws (string \":\" ws value (\",\" ws string \":\" ws value)*)? ws \"}}\"\n\
             string ::= \"\\\"\" ([^\"\\\\] | \"\\\\\" [\"\\\\/bfnrt] | \"\\\\u\" [0-9a-fA-F]{{4}})* \"\\\"\"\n\
             value ::= string | number | object | array | \"true\" | \"false\" | \"null\"\n\
             array ::= \"[\" ws (value (\",\" ws value)*)? ws \"]\"\n\
             number ::= \"-\"? ([0-9] | [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [+-]? [0-9]+)?\n\
             ws ::= [ \\t\\n]*\n"
        )
    }

    /// Validate that `json` represents a tool-call matching one of the schemas.
    ///
    /// Expected shape: `{{"name": "<tool>", "arguments": {{ ... }} }}` or
    /// `{{"function": {{"name": "<tool>", "arguments": {{ ... }} }}}}`
    /// (both accepted for convenience). Returns `true` iff:
    /// - `name` is in `tool_schemas`
    /// - `arguments` is an object
    /// - all `required` fields from the tool's JSON schema are present in `arguments`
    pub fn validate(&self, json: &str) -> bool {
        let value: serde_json::Value = match serde_json::from_str(json) {
            Ok(v) => v,
            Err(_) => return false,
        };
        self.validate_value(&value)
    }

    fn validate_value(&self, value: &serde_json::Value) -> bool {
        // Extract (name, arguments) from either flat or nested function form.
        let (name_opt, args_opt) = if let Some(obj) = value.as_object() {
            if obj.contains_key("name") && obj.contains_key("arguments") {
                (
                    obj.get("name").and_then(|v| v.as_str()),
                    obj.get("arguments"),
                )
            } else if let Some(func) = obj.get("function").and_then(|v| v.as_object()) {
                (
                    func.get("name").and_then(|v| v.as_str()),
                    func.get("arguments"),
                )
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        let Some(name) = name_opt else {
            return false;
        };
        let Some(args) = args_opt else {
            return false;
        };
        if !args.is_object() {
            return false;
        }

        // Find matching schema
        let Some(def) = self.tool_schemas.iter().find(|t| t.function.name == name) else {
            return false;
        };

        // Check required fields from JSON schema
        if let Some(required) = def
            .function
            .parameters
            .get("required")
            .and_then(|v| v.as_array())
        {
            for field in required {
                if let Some(key) = field.as_str() {
                    if args.get(key).is_none() {
                        return false;
                    }
                }
            }
        }

        true
    }
}

/// Adapter that wraps any [`ChatClient`] and validates tool_calls via [`GrammarConfig`].
///
/// When `grammar.strict` is true, invalid tool calls are filtered out before
/// returning the response. When false, the inner response is passed through
/// unchanged (but `validate` remains available for inspection).
pub struct GrammarClient<C: ChatClient> {
    inner: C,
    grammar: GrammarConfig,
}

impl<C: ChatClient> GrammarClient<C> {
    /// Create a new grammar-constrained client.
    pub fn new(inner: C, grammar: GrammarConfig) -> Self {
        Self { inner, grammar }
    }

    /// Access the grammar config.
    pub fn grammar(&self) -> &GrammarConfig {
        &self.grammar
    }

    /// Access the inner client.
    pub fn inner(&self) -> &C {
        &self.inner
    }

    /// Post-hoc GBNF validation of a response's tool_calls (strict mode only;
    /// non-strict passes through unchanged). Shared by BOTH transport paths —
    /// [`ChatClient::chat`] and [`ChatClient::chat_with_options`] — so the
    /// options path cannot bypass the filter (G2 finding 1: the agent calls
    /// exclusively through chat_with_options).
    fn apply_strict_filter(&self, mut resp: ChatResponse) -> ChatResponse {
        if self.grammar.strict {
            if let Some(calls) = resp.message.tool_calls.take() {
                let filtered: Vec<crate::client::ToolCall> = calls
                    .into_iter()
                    .filter(|tc| {
                        let json = serde_json::json!({
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        })
                        .to_string();
                        self.grammar.validate(&json)
                    })
                    .collect();
                resp.message.tool_calls = if filtered.is_empty() {
                    None
                } else {
                    Some(filtered)
                };
            }
        }
        resp
    }
}

impl<C: ChatClient> ChatClient for GrammarClient<C> {
    fn chat<'a>(
        &'a self,
        model: &'a str,
        messages: Vec<crate::client::Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<ChatResponse>> + Send + 'a>> {
        Box::pin(async move {
            let resp = self.inner.chat(model, messages, tools).await?;
            Ok(self.apply_strict_filter(resp))
        })
    }

    /// F2/G2-FIX1: forward request options (num_ctx, num_predict, think, …)
    /// to the inner client — the default trait impl would silently drop them
    /// — while STILL applying the strict filter via [`Self::apply_strict_filter`].
    fn chat_with_options<'a>(
        &'a self,
        model: &'a str,
        messages: Vec<crate::client::Message>,
        tools: Option<Vec<ToolDefinition>>,
        options: Option<crate::client::RequestOptions>,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<ChatResponse>> + Send + 'a>> {
        Box::pin(async move {
            let resp = self
                .inner
                .chat_with_options(model, messages, tools, options)
                .await?;
            Ok(self.apply_strict_filter(resp))
        })
    }

    fn count_tokens<'a>(
        &'a self,
        model: &'a str,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<usize>> + Send + 'a>> {
        Box::pin(async move { self.inner.count_tokens(model, text).await })
    }

    fn health_check<'a>(
        &'a self,
    ) -> Pin<Box<dyn Future<Output = crate::error::Result<()>> + Send + 'a>> {
        Box::pin(async move { self.inner.health_check().await })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolDefinition;

    fn sample_tools() -> Vec<ToolDefinition> {
        vec![
            ToolDefinition::new(
                "Read",
                "Read a file",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"}
                    },
                    "required": ["path"]
                }),
            ),
            ToolDefinition::new(
                "Write",
                "Write a file",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "content": {"type": "string", "description": "Content"}
                    },
                    "required": ["path", "content"]
                }),
            ),
        ]
    }

    #[test]
    fn to_gbnf_contains_tool_names() {
        let cfg = GrammarConfig::from_tools(sample_tools());
        let gbnf = cfg.to_gbnf();
        assert!(gbnf.contains("\"Read\""), "GBNF missing Read: {gbnf}");
        assert!(gbnf.contains("\"Write\""), "GBNF missing Write: {gbnf}");
        assert!(gbnf.contains("name ::="), "GBNF missing name rule: {gbnf}");
        assert!(gbnf.contains("root ::="), "GBNF missing root rule: {gbnf}");
    }

    #[test]
    fn validate_accepts_valid_tool_call() {
        let cfg = GrammarConfig::from_tools(sample_tools());
        let valid_read = r#"{"name": "Read", "arguments": {"path": "/tmp/foo.rs"}}"#;
        assert!(cfg.validate(valid_read));

        let valid_write =
            r#"{"name": "Write", "arguments": {"path": "/tmp/foo.rs", "content": "hello"}}"#;
        assert!(cfg.validate(valid_write));
    }

    #[test]
    fn validate_rejects_invalid_tool() {
        let cfg = GrammarConfig::from_tools(sample_tools());
        // Unknown tool name
        let unknown = r#"{"name": "NotATool", "arguments": {"path": "/tmp/x"}}"#;
        assert!(!cfg.validate(unknown));

        // Missing required field
        let missing_required = r#"{"name": "Read", "arguments": {}}"#;
        assert!(!cfg.validate(missing_required));

        let missing_one = r#"{"name": "Write", "arguments": {"path": "/tmp/x"}}"#;
        assert!(!cfg.validate(missing_one));

        // Invalid JSON
        assert!(!cfg.validate("not json"));

        // Wrong shape (no arguments key)
        assert!(!cfg.validate(r#"{"name": "Read"}"#));
    }

    #[test]
    fn validate_accepts_nested_function_shape() {
        let cfg = GrammarConfig::from_tools(sample_tools());
        let nested = r#"{"function": {"name": "Read", "arguments": {"path": "/tmp/a"}}}"#;
        assert!(cfg.validate(nested));
    }

    #[test]
    fn to_gbnf_empty_returns_null() {
        let cfg = GrammarConfig::new(vec![], true);
        let gbnf = cfg.to_gbnf();
        assert!(gbnf.contains("null"));
    }

    #[tokio::test]
    async fn grammar_client_strict_filters_invalid() {
        use crate::client::{ChatResponse, Message, MockClient, ToolCall, ToolCallFunction};

        let tools = sample_tools();
        let grammar = GrammarConfig::from_tools(tools.clone());
        // Mock returns one valid and one invalid tool call
        let valid_call = ToolCall {
            id: None,
            function: ToolCallFunction {
                name: "Read".to_string(),
                arguments: serde_json::json!({"path": "/tmp/x"}),
            },
        };
        let invalid_call = ToolCall {
            id: None,
            function: ToolCallFunction {
                name: "Read".to_string(),
                arguments: serde_json::json!({}), // missing path
            },
        };
        let resp = ChatResponse {
            model: "test".into(),
            message: Message {
                role: "assistant".into(),
                content: None,
                thinking: None,
                tool_calls: Some(vec![valid_call, invalid_call]),
            },
            done: true,
            eval_count: None,
            eval_duration: None,
            prompt_eval_count: None,
        };
        let mock = MockClient::new(vec![resp]);
        let client = GrammarClient::new(mock, grammar);
        let out = client
            .chat("test", vec![Message::user("hi")], Some(tools))
            .await
            .unwrap();
        let calls = out.message.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "Read");
    }

    #[test]
    fn test_gbnf_spike_documented() {
        assert!(
            include_str!("grammar.rs").contains(concat!("SPIKE", " defer")),
            "grammar.rs header must document spike defer — see docs/STRATEGY.md F6.4"
        );
    }

    #[tokio::test]
    async fn grammar_client_non_strict_passes_through() {
        use crate::client::{ChatResponse, Message, MockClient, ToolCall, ToolCallFunction};

        let tools = sample_tools();
        let mut grammar = GrammarConfig::from_tools(tools.clone());
        grammar.strict = false;
        let invalid_call = ToolCall {
            id: None,
            function: ToolCallFunction {
                name: "Read".to_string(),
                arguments: serde_json::json!({}),
            },
        };
        let resp = ChatResponse {
            model: "test".into(),
            message: Message {
                role: "assistant".into(),
                content: None,
                thinking: None,
                tool_calls: Some(vec![invalid_call]),
            },
            done: true,
            eval_count: None,
            eval_duration: None,
            prompt_eval_count: None,
        };
        let mock = MockClient::new(vec![resp]);
        let client = GrammarClient::new(mock, grammar);
        let out = client
            .chat("test", vec![Message::user("hi")], Some(tools))
            .await
            .unwrap();
        // non-strict should not filter
        assert!(out.message.tool_calls.is_some());
        assert_eq!(out.message.tool_calls.unwrap().len(), 1);
    }

    /// Regression test (G2 finding 1): the agent calls exclusively through
    /// `chat_with_options`; that path must still apply the strict GBNF filter.
    #[tokio::test]
    async fn chat_with_options_path_filters_invalid_tool_calls() {
        use crate::client::{Message, MockClient, RequestOptions, ToolCall, ToolCallFunction};

        let tools = sample_tools();
        let grammar = GrammarConfig::from_tools(tools.clone());
        // BAD fixture: unknown tool + missing required arg — both must go.
        let calls = vec![
            ToolCall {
                id: None,
                function: ToolCallFunction {
                    name: "NotATool".to_string(),
                    arguments: serde_json::json!({"path": "/tmp/x"}),
                },
            },
            ToolCall {
                id: None,
                function: ToolCallFunction {
                    name: "Read".to_string(),
                    arguments: serde_json::json!({}),
                },
            },
        ];
        let resp = ChatResponse {
            model: "test".into(),
            message: Message {
                role: "assistant".into(),
                content: None,
                thinking: None,
                tool_calls: Some(calls),
            },
            done: true,
            eval_count: None,
            eval_duration: None,
            prompt_eval_count: None,
        };
        let client = GrammarClient::new(MockClient::new(vec![resp]), grammar);
        let out = client
            .chat_with_options(
                "test",
                vec![Message::user("hi")],
                Some(tools),
                Some(RequestOptions::default()),
            )
            .await
            .unwrap();
        assert!(
            out.message.tool_calls.is_none(),
            "invalid tool_calls must be filtered on the options path"
        );
    }

    /// Good fixture: a valid tool call passes the options path unharmed.
    #[tokio::test]
    async fn chat_with_options_path_passes_valid_tool_calls() {
        use crate::client::{Message, MockClient, RequestOptions, ToolCall, ToolCallFunction};

        let tools = sample_tools();
        let grammar = GrammarConfig::from_tools(tools.clone());
        let resp = ChatResponse {
            model: "test".into(),
            message: Message {
                role: "assistant".into(),
                content: None,
                thinking: None,
                tool_calls: Some(vec![ToolCall {
                    id: None,
                    function: ToolCallFunction {
                        name: "Write".to_string(),
                        arguments: serde_json::json!({"path": "/tmp/x", "content": "hi"}),
                    },
                }]),
            },
            done: true,
            eval_count: None,
            eval_duration: None,
            prompt_eval_count: None,
        };
        let client = GrammarClient::new(MockClient::new(vec![resp]), grammar);
        let out = client
            .chat_with_options(
                "test",
                vec![Message::user("hi")],
                Some(tools),
                Some(RequestOptions::default()),
            )
            .await
            .unwrap();
        let calls = out.message.tool_calls.expect("valid call must survive");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "Write");
    }
}

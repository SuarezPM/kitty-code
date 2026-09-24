//! OpenAI-compatible chat client (SGLang / vLLM / llama.cpp server).
//!
//! For models Ollama cannot serve — currently K2-Horizon-3.7B, whose GGUF
//! requires the vendor llama.cpp fork and whose reference serving recipe is
//! SGLang with `trust_remote_code` + the `k2_horizon` parsers. Implements
//! [`ChatClient`] against `POST {base}/v1/chat/completions` so the harness
//! loop, budgets, and parser treat it exactly like any other backend.
//!
//! Wire mapping notes:
//! - Request tools reuse [`ToolDefinition`]: it already serializes as
//!   `{"type":"function","function":{...}}`, which IS the OpenAI shape.
//! - Assistant `thinking` is sent as `reasoning_content` (SGLang/DeepSeek
//!   convention; ignored by servers that do not understand it) and parsed
//!   back from `reasoning_content` into [`Message::thinking`] — the same
//!   normalization the Ollama client applies, so harness thinking-budget
//!   accounting keeps working.
//! - `tool` messages carry the `tool_call_id` of the assistant call they
//!   answer (positional pairing against the preceding assistant message;
//!   generated `kitty_{n}` ids when the server sent none).
//! - `count_tokens` has no `/api/tokenize` equivalent here: it uses the
//!   conservative chars/4 estimate ([`estimate_input_tokens`]). Budget
//!   accounting stays single-point in the harness `push_message` path.
//!
//! Security: server-returned tool-call names are validated against
//! `^[A-Za-z_][A-Za-z0-9_]{0,63}$` before delivery — a compromised or
//! confused server cannot inject arbitrary tool names into the harness.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::client::{
    ChatClient, ChatResponse, Message, RequestOptions, ToolCall, ToolCallFunction,
    DEFAULT_OLLAMA_TIMEOUT_SECS,
};
use crate::error::{Result, RuntimeError};
use crate::tools::ToolDefinition;

/// Env var overriding the base URL (`http://127.0.0.1:30000` default).
pub const ENV_OPENAI_COMPAT_BASE_URL: &str = "KITTY_OPENAI_BASE_URL";

/// Default base URL: local SGLang (vendor K2 recipe serves on :30000).
pub const DEFAULT_OPENAI_COMPAT_BASE_URL: &str = "http://127.0.0.1:30000";

/// Tool-call names must be identifier-shaped and bounded: anything else is
/// dropped before it reaches the harness (anti prompt-injection via server).
fn is_safe_tool_name(name: &str) -> bool {
    !name.is_empty() && name.len() <= 64 && {
        let mut chars = name.chars();
        match chars.next() {
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {
                chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
            }
            _ => false,
        }
    }
}

/// OpenAI-style chat message as sent on the wire.
#[derive(Debug, Clone, Serialize)]
struct CompatMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    /// Reasoning trace (SGLang `reasoning_content` convention).
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<CompatToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

/// OpenAI-style tool call as sent on the wire (`arguments` is a JSON string).
#[derive(Debug, Clone, Serialize)]
struct CompatToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: CompatFunction,
}

/// OpenAI-style function payload as sent on the wire.
#[derive(Debug, Clone, Serialize)]
struct CompatFunction {
    name: String,
    arguments: String,
}

/// OpenAI-style chat completion request.
#[derive(Debug, Clone, Serialize)]
struct CompatChatRequest {
    model: String,
    messages: Vec<CompatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<i32>,
    /// Vendor reasoning selector (K2 recipe: `{"reasoning_effort":"high"}`).
    /// Plain `extra_body` — servers that do not understand it ignore it.
    #[serde(skip_serializing_if = "Option::is_none")]
    extra_body: Option<serde_json::Value>,
    stream: bool,
}

/// OpenAI-style chat completion response (non-streaming subset we use).
#[derive(Debug, Clone, Deserialize)]
struct CompatChatResponse {
    #[serde(default)]
    choices: Vec<CompatChoice>,
    #[serde(default)]
    usage: Option<CompatUsage>,
}

#[derive(Debug, Clone, Deserialize)]
struct CompatChoice {
    #[serde(default)]
    message: CompatResponseMessage,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct CompatResponseMessage {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<serde_json::Value>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<CompatResponseToolCall>>,
}

#[derive(Debug, Clone, Deserialize)]
struct CompatResponseToolCall {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<CompatResponseFunction>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct CompatResponseFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct CompatUsage {
    #[serde(default)]
    prompt_tokens: Option<i64>,
}

/// OpenAI-compatible chat client (SGLang / vLLM).
///
/// Resolution: `new()` reads [`ENV_OPENAI_COMPAT_BASE_URL`] once;
/// [`OpenAiCompatClient::with_base_url`] pins an explicit endpoint.
pub struct OpenAiCompatClient {
    base_url: String,
    http: reqwest::Client,
    /// Value sent as `extra_body.reasoning_effort` (K2 vendor recipe:
    /// `"high"`). `None` omits `extra_body` entirely.
    reasoning_effort: Option<String>,
}

impl OpenAiCompatClient {
    /// Build against `base_url` (e.g. `http://127.0.0.1:30000`).
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(DEFAULT_OLLAMA_TIMEOUT_SECS))
            .build()
            .expect("reqwest client with rustls should build");
        Self {
            base_url: base_url.into(),
            http,
            reasoning_effort: None,
        }
    }

    /// Build from [`ENV_OPENAI_COMPAT_BASE_URL`] (or the SGLang default).
    pub fn new() -> Self {
        let base = std::env::var(ENV_OPENAI_COMPAT_BASE_URL)
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_OPENAI_COMPAT_BASE_URL.to_string());
        Self::with_base_url(base)
    }

    /// Set the vendor reasoning selector (K2: `"high"`).
    pub fn with_reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    /// Base URL (for diagnostics).
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Serialize harness history into OpenAI-style messages, pairing each
    /// `tool` message with the `tool_call_id` of the assistant call it
    /// answers (positional order; generated `kitty_{n}` ids when the
    /// server sent none, which OpenAI-style servers accept).
    fn to_compat_messages(messages: &[Message]) -> Vec<CompatMessage> {
        let mut out = Vec::with_capacity(messages.len());
        let mut pending_ids: Vec<String> = Vec::new();
        let mut generated = 0usize;
        for m in messages {
            match m.role.as_str() {
                "tool" => {
                    // Explicit id wins (harness-threaded via
                    // `Message::tool_call_id`); positional pairing stays as
                    // the fallback for results that predate the field.
                    let id = match m.tool_call_id.clone() {
                        Some(explicit) => {
                            if let Some(pos) = pending_ids.iter().position(|p| p == &explicit) {
                                pending_ids.remove(pos);
                            }
                            explicit
                        }
                        None => {
                            let id = pending_ids.first().cloned().unwrap_or_else(|| {
                                generated += 1;
                                // Unknown id: positional fallback, loud —
                                // never a silent verbatim correlation.
                                tracing::warn!(
                                    "tool result without id and empty pending queue; \
                                     generated fallback id (multi-tool turns need ids)"
                                );
                                format!("kitty_{generated}")
                            });
                            if !pending_ids.is_empty() {
                                pending_ids.remove(0);
                            }
                            id
                        }
                    };
                    out.push(CompatMessage {
                        role: "tool".to_string(),
                        content: m.content.clone(),
                        reasoning_content: None,
                        tool_calls: None,
                        tool_call_id: Some(id),
                    });
                }
                _ => {
                    let tool_calls = m.tool_calls.as_ref().map(|calls| {
                        pending_ids = calls
                            .iter()
                            .enumerate()
                            .map(|(i, c)| {
                                c.id.clone().unwrap_or_else(|| {
                                    generated += 1;
                                    format!("kitty_{generated}_{i}")
                                })
                            })
                            .collect();
                        calls
                            .iter()
                            .zip(pending_ids.iter())
                            .map(|(c, id)| CompatToolCall {
                                id: id.clone(),
                                call_type: "function".to_string(),
                                function: CompatFunction {
                                    name: c.function.name.clone(),
                                    arguments: serde_json::to_string(&c.function.arguments)
                                        .unwrap_or_else(|_| "{}".to_string()),
                                },
                            })
                            .collect()
                    });
                    out.push(CompatMessage {
                        role: m.role.clone(),
                        content: m.content.clone(),
                        reasoning_content: m.thinking.clone(),
                        tool_calls,
                        tool_call_id: None,
                    });
                }
            }
        }
        out
    }

    /// Build the wire request, honoring caller [`RequestOptions`] (the
    /// harness `ModelTuning` values). `num_ctx` has no OpenAI equivalent
    /// and is ignored (server-side `--max-model-len` owns the window);
    /// `think: Some(false)` drops `reasoning_effort` (best-effort off).
    fn build_request(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        options: Option<RequestOptions>,
    ) -> CompatChatRequest {
        let opts = options.unwrap_or_default();
        let thinking_off = opts.think == Some(false);
        let extra_body = if thinking_off {
            None
        } else {
            self.reasoning_effort
                .clone()
                .map(|effort| serde_json::json!({ "reasoning_effort": effort }))
        };
        CompatChatRequest {
            model: model.to_string(),
            messages: Self::to_compat_messages(&messages),
            tools,
            temperature: opts.temperature,
            top_p: opts.top_p,
            max_tokens: opts.num_predict,
            extra_body,
            stream: false,
        }
    }

    /// Convert a wire response into [`ChatResponse`]. Server tool calls with
    /// non-identifier names are dropped (see [`is_safe_tool_name`]); when
    /// every call is dropped the text content still delivers.
    fn to_chat_response(model: &str, wire: CompatChatResponse) -> Result<ChatResponse> {
        let usage = wire.usage.clone();
        let choice = wire.choices.into_iter().next().ok_or_else(|| {
            RuntimeError::UnexpectedResponse("compat server returned no choices".into())
        })?;
        let msg = choice.message;
        let content = match msg.content {
            Some(serde_json::Value::String(s)) => Some(s),
            Some(other) => Some(serde_json::to_string(&other).unwrap_or_default()),
            None => None,
        };
        let mut tool_calls = Vec::new();
        for call in msg.tool_calls.unwrap_or_default() {
            let func = call.function.unwrap_or_default();
            let name = func.name.unwrap_or_default();
            if !is_safe_tool_name(&name) {
                tracing::warn!(name = %name, "dropping server tool call with unsafe name");
                continue;
            }
            let raw_args = func.arguments.unwrap_or_else(|| "{}".to_string());
            let arguments = match serde_json::from_str::<serde_json::Value>(&raw_args) {
                Ok(v) if v.is_object() => v,
                _ => serde_json::json!({ "_raw": raw_args }),
            };
            tool_calls.push(ToolCall {
                id: call.id,
                function: ToolCallFunction { name, arguments },
            });
        }
        Ok(ChatResponse {
            model: model.to_string(),
            message: Message {
                role: msg.role.unwrap_or_else(|| "assistant".to_string()),
                content,
                thinking: msg.reasoning_content,
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                tool_call_id: None,
            },
            done: true,
            eval_count: None,
            eval_duration: None,
            prompt_eval_count: usage.and_then(|u| u.prompt_tokens),
        })
    }

    async fn chat_impl(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        options: Option<RequestOptions>,
    ) -> Result<ChatResponse> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = self.build_request(model, messages, tools, options);
        debug!(url = %url, model = %model, "sending compat chat request");
        let resp = self.http.post(&url).json(&body).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(RuntimeError::Api {
                status: status.as_u16(),
                message: text,
            });
        }
        let wire: CompatChatResponse = resp.json().await?;
        Self::to_chat_response(model, wire)
    }
}

impl Default for OpenAiCompatClient {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatClient for OpenAiCompatClient {
    fn chat<'a>(
        &'a self,
        model: &'a str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>> {
        Box::pin(async move { self.chat_impl(model, messages, tools, None).await })
    }

    fn chat_with_options<'a>(
        &'a self,
        model: &'a str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        options: Option<RequestOptions>,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>> {
        // Unlike the pre-f8 boxed-client bug (options silently dropped), the
        // compat client MUST forward options: num_predict caps and sampling
        // overrides are the whole K2 vendor recipe.
        Box::pin(async move { self.chat_impl(model, messages, tools, options).await })
    }

    fn count_tokens<'a>(
        &'a self,
        _model: &'a str,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<usize>> + Send + 'a>> {
        // No /api/tokenize equivalent: conservative chars/4 estimate.
        let estimate = text.chars().count().div_ceil(4).max(1);
        Box::pin(async move { Ok(estimate) })
    }

    fn health_check<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let url = format!("{}/v1/models", self.base_url);
            let resp = self.http.get(&url).send().await.map_err(|e| {
                RuntimeError::ConnectionRefused(format!("compat server unreachable: {e}"))
            })?;
            if resp.status().is_success() {
                Ok(())
            } else {
                Err(RuntimeError::ConnectionRefused(format!(
                    "compat server returned {}",
                    resp.status()
                )))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn chat_response_body(
        content: &str,
        reasoning: Option<&str>,
        tool_calls: serde_json::Value,
        prompt_tokens: i64,
    ) -> serde_json::Value {
        serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning,
                    "tool_calls": tool_calls,
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 10,
                "total_tokens": prompt_tokens + 10,
            },
        })
    }

    async fn mock_chat_server(body: serde_json::Value) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;
        server
    }

    #[test]
    fn safe_tool_names() {
        for good in ["Read", "Bash", "_private", "a1"] {
            assert!(is_safe_tool_name(good), "{good} should be safe");
        }
        for bad in [
            "",
            "rm -rf",
            "a".repeat(65).as_str(),
            "tool;evil",
            "x.y",
            "a-b",
        ] {
            assert!(!is_safe_tool_name(bad), "{bad:?} should be rejected");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn maps_reasoning_content_and_tool_calls() {
        let server = mock_chat_server(chat_response_body(
            "working on it",
            Some("let me think"),
            serde_json::json!([{
                "id": "call_1",
                "type": "function",
                "function": {"name": "Read", "arguments": "{\"path\":\"a.rs\"}"},
            }]),
            128,
        ))
        .await;
        let client = OpenAiCompatClient::with_base_url(server.uri());
        let resp = client
            .chat("k2-test", vec![Message::user("hi")], None)
            .await
            .expect("chat should succeed");
        assert_eq!(resp.message.content.as_deref(), Some("working on it"));
        assert_eq!(resp.message.thinking.as_deref(), Some("let me think"));
        let calls = resp.message.tool_calls.expect("tool calls mapped");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(calls[0].function.name, "Read");
        assert_eq!(calls[0].function.arguments["path"], "a.rs");
        assert_eq!(resp.prompt_eval_count, Some(128));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn drops_unsafe_tool_names_but_keeps_text() {
        let server = mock_chat_server(chat_response_body(
            "done",
            None,
            serde_json::json!([
                {"id": "c1", "type": "function",
                 "function": {"name": "Read", "arguments": "{}"}},
                {"id": "c2", "type": "function",
                 "function": {"name": "rm -rf /", "arguments": "{}"}},
            ]),
            10,
        ))
        .await;
        let client = OpenAiCompatClient::with_base_url(server.uri());
        let resp = client
            .chat("k2-test", vec![Message::user("hi")], None)
            .await
            .expect("chat should succeed");
        let calls = resp.message.tool_calls.expect("safe call survives");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "Read");
        assert_eq!(resp.message.content.as_deref(), Some("done"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn malformed_arguments_salvaged_as_raw() {
        let server = mock_chat_server(chat_response_body(
            "",
            None,
            serde_json::json!([{
                "id": "c1", "type": "function",
                "function": {"name": "Bash", "arguments": "{not json"},
            }]),
            10,
        ))
        .await;
        let client = OpenAiCompatClient::with_base_url(server.uri());
        let resp = client
            .chat("k2-test", vec![Message::user("hi")], None)
            .await
            .expect("chat should succeed");
        let calls = resp.message.tool_calls.expect("call salvaged");
        assert_eq!(calls[0].function.arguments["_raw"], "{not json");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn non_2xx_becomes_api_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
            .mount(&server)
            .await;
        let client = OpenAiCompatClient::with_base_url(server.uri());
        let err = client
            .chat("k2-test", vec![Message::user("hi")], None)
            .await
            .expect_err("400 must error");
        assert!(matches!(err, RuntimeError::Api { status: 400, .. }));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn empty_choices_is_unexpected_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [],
                "usage": {"prompt_tokens": 1},
            })))
            .mount(&server)
            .await;
        let client = OpenAiCompatClient::with_base_url(server.uri());
        let err = client
            .chat("k2-test", vec![Message::user("hi")], None)
            .await
            .expect_err("empty choices must error");
        assert!(matches!(err, RuntimeError::UnexpectedResponse(_)));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn chat_with_options_forwards_recipe() {
        let server =
            mock_chat_server(chat_response_body("ok", None, serde_json::json!([]), 5)).await;
        let client = OpenAiCompatClient::with_base_url(server.uri()).with_reasoning_effort("high");
        let options = RequestOptions {
            temperature: Some(1.0),
            top_p: Some(0.95),
            num_predict: Some(32768),
            ..Default::default()
        };
        client
            .chat_with_options("k2-test", vec![Message::user("hi")], None, Some(options))
            .await
            .expect("chat should succeed");
        let received = server.received_requests().await.expect("requests recorded");
        assert_eq!(received.len(), 1);
        let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
        assert_eq!(body["temperature"], 1.0);
        assert_eq!(body["top_p"], 0.95);
        assert_eq!(body["max_tokens"], 32768);
        assert_eq!(body["extra_body"]["reasoning_effort"], "high");
        assert_eq!(body["stream"], false);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn think_false_drops_reasoning_effort() {
        let server =
            mock_chat_server(chat_response_body("ok", None, serde_json::json!([]), 5)).await;
        let client = OpenAiCompatClient::with_base_url(server.uri()).with_reasoning_effort("high");
        let options = RequestOptions {
            think: Some(false),
            ..Default::default()
        };
        client
            .chat_with_options("k2-test", vec![Message::user("hi")], None, Some(options))
            .await
            .expect("chat should succeed");
        let received = server.received_requests().await.expect("requests recorded");
        let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
        assert!(body.get("extra_body").is_none());
    }

    #[test]
    fn tool_messages_carry_paired_call_ids() {
        let assistant = Message {
            role: "assistant".to_string(),
            content: None,
            thinking: None,
            tool_calls: Some(vec![
                ToolCall {
                    id: None,
                    function: ToolCallFunction {
                        name: "Read".into(),
                        arguments: serde_json::json!({}),
                    },
                },
                ToolCall {
                    id: Some("srv_9".into()),
                    function: ToolCallFunction {
                        name: "Bash".into(),
                        arguments: serde_json::json!({}),
                    },
                },
            ]),
            tool_call_id: None,
        };
        let msgs = vec![
            assistant,
            Message::tool_result("file"),
            Message::tool_result("out"),
            // Explicit id beats positional pairing (and doesn't consume it).
            Message::tool_result_with_id("explicit", Some("srv_9".into())),
        ];
        let wire = OpenAiCompatClient::to_compat_messages(&msgs);
        assert_eq!(wire.len(), 4);
        let calls = wire[0].tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 2);
        // Server ids preserved or generated — and stable across the pair.
        assert_eq!(calls[1].id, "srv_9");
        assert_eq!(wire[1].tool_call_id.as_deref(), Some(calls[0].id.as_str()));
        assert_eq!(wire[2].tool_call_id.as_deref(), Some("srv_9"));
        // Explicit id on the result wins over positional pairing.
        assert_eq!(wire[3].tool_call_id.as_deref(), Some("srv_9"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn health_check_ok_and_fail() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [],
            })))
            .mount(&server)
            .await;
        let client = OpenAiCompatClient::with_base_url(server.uri());
        client.health_check().await.expect("healthy");
        // Unreachable port fails fast (connection refused, not success).
        let bad = OpenAiCompatClient::with_base_url("http://127.0.0.1:1");
        assert!(bad.health_check().await.is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn count_tokens_estimates() {
        let client = OpenAiCompatClient::with_base_url("http://127.0.0.1:1");
        let n = client.count_tokens("m", "abcd").await.unwrap();
        assert_eq!(n, 1);
        let n = client.count_tokens("m", &"x".repeat(400)).await.unwrap();
        assert_eq!(n, 100);
    }
}

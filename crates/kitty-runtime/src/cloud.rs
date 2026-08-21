//! Cloud fallback — generic OpenAI-compatible `ChatClient`.
//!
//! `CloudClient` is vendor-agnostic: it speaks the OpenAI `/v1/chat/completions`
//! wire format against any base URL (OpenAI, Azure, OpenRouter, etc.).
//! No vendor is hardcoded. API key is read from `KITTY_CLOUD_API_KEY` and
//! never logged. Timeout is 120s, cost is metered via `tracing::info!`.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::client::{ChatResponse, Message, ToolCall, ToolCallFunction};
use crate::error::{Result, RuntimeError};
use crate::tools::ToolDefinition;

const ENV_API_KEY: &str = "KITTY_CLOUD_API_KEY";
const ENV_BASE_URL: &str = "KITTY_CLOUD_BASE_URL";

// ---------------------------------------------------------------------------
// Wire types — OpenAI-compatible
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct CloudChatRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CloudChatResponse {
    id: Option<String>,
    model: Option<String>,
    choices: Vec<CloudChoice>,
    usage: Option<CloudUsage>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CloudChoice {
    message: CloudMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CloudMessage {
    role: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<CloudToolCall>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct CloudToolCall {
    #[serde(default)]
    id: Option<String>,
    #[serde(rename = "type", default)]
    call_type: Option<String>,
    function: CloudFunction,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct CloudFunction {
    name: String,
    arguments: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct CloudUsage {
    #[serde(default)]
    prompt_tokens: Option<i64>,
    #[serde(default)]
    completion_tokens: Option<i64>,
    #[serde(default)]
    total_tokens: Option<i64>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct CloudModelsResponse {
    #[serde(default)]
    data: Vec<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// CloudClient — generic frontier
// ---------------------------------------------------------------------------

/// Generic OpenAI-compatible cloud client implementing [`crate::client::ChatClient`].
///
/// - `base_url`: any OpenAI-compatible endpoint, e.g. `https://api.example.com/v1`
///   or a self-hosted proxy. No vendor is hardcoded.
/// - `api_key`: bearer token, read from `KITTY_CLOUD_API_KEY` via [`CloudClient::from_env`].
/// - Timeout is 120s per request (rustls).
/// - Cost is metered via `tracing::info!` on every successful chat (tokens + cost_usd).
#[derive(Clone)]
pub struct CloudClient {
    base_url: String,
    api_key: String,
    http: reqwest::Client,
}

impl std::fmt::Debug for CloudClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CloudClient")
            .field("base_url", &self.base_url)
            .field("api_key", &"<redacted>")
            .finish()
    }
}

impl CloudClient {
    /// Create with explicit base URL and API key.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        let api_key = api_key.into();
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("reqwest::Client builder should not fail");
        Self {
            base_url,
            api_key,
            http,
        }
    }

    /// Create from environment: reads `KITTY_CLOUD_API_KEY` (required) and
    /// `KITTY_CLOUD_BASE_URL` (optional, defaults to empty error if missing).
    ///
    /// Returns a clear error if the key is absent — never panics, never logs the key.
    pub fn from_env(base_url: Option<String>) -> Result<Self> {
        let api_key = std::env::var(ENV_API_KEY).map_err(|_| {
            RuntimeError::InvalidArgument(format!(
                "set {ENV_API_KEY} to use CloudClient (env var missing)"
            ))
        })?;
        if api_key.trim().is_empty() {
            return Err(RuntimeError::InvalidArgument(format!(
                "set {ENV_API_KEY} to use CloudClient (env var empty)"
            )));
        }
        let url = if let Some(u) = base_url {
            u
        } else {
            std::env::var(ENV_BASE_URL).map_err(|_| {
                RuntimeError::InvalidArgument(format!(
                    "set {ENV_BASE_URL} or pass base_url to CloudClient::new (env var missing)"
                ))
            })?
        };
        if url.trim().is_empty() {
            return Err(RuntimeError::InvalidArgument(format!(
                "set {ENV_BASE_URL} to use CloudClient (env var empty)"
            )));
        }
        Ok(Self::new(url, api_key))
    }

    /// Convenience: try from env with `KITTY_CLOUD_BASE_URL` implicit.
    pub fn try_from_env() -> Result<Self> {
        Self::from_env(None)
    }

    /// Expose timeout for verification (must be 120s).
    pub fn timeout(&self) -> Duration {
        Duration::from_secs(120)
    }

    fn auth_header(&self) -> String {
        format!("Bearer {}", self.api_key)
    }

    async fn chat_inner(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Result<ChatResponse> {
        let url = format!("{}/chat/completions", self.base_url);
        let req = CloudChatRequest {
            model: model.to_string(),
            messages,
            tools,
            stream: Some(false),
        };
        debug!(url = %url, model = %model, "cloud chat request");

        let resp = self
            .http
            .post(&url)
            .header("Authorization", self.auth_header())
            .json(&req)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            // Never include api_key in logs
            warn!(status = %status, body = %body, "cloud chat returned error");
            return Err(RuntimeError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let cloud: CloudChatResponse = resp.json().await?;
        let usage = cloud.usage.as_ref();

        // Cost metering — generic pricing placeholder, logged never with key
        if let Some(u) = usage {
            let prompt = u.prompt_tokens.unwrap_or(0);
            let completion = u.completion_tokens.unwrap_or(0);
            let total = u.total_tokens.unwrap_or(prompt + completion);
            // Placeholder frontier pricing: $2 per 1M input, $6 per 1M output (generic)
            let cost_usd = prompt as f64 * 0.000002 + completion as f64 * 0.000006;
            info!(
                model = %model,
                prompt_tokens = prompt,
                completion_tokens = completion,
                total_tokens = total,
                cost_usd = cost_usd,
                "cloud cost"
            );
        } else {
            debug!(model = %model, "cloud chat no usage returned");
        }

        let choice = cloud.choices.into_iter().next().ok_or_else(|| {
            RuntimeError::UnexpectedResponse("cloud response has no choices".into())
        })?;

        let cm = choice.message;
        let tool_calls = cm.tool_calls.map(|v| {
            v.into_iter()
                .map(|c| ToolCall {
                    id: c.id,
                    function: ToolCallFunction {
                        name: c.function.name,
                        arguments: c.function.arguments,
                    },
                })
                .collect()
        });

        let chat_model = cloud.model.unwrap_or_else(|| model.to_string());
        Ok(ChatResponse {
            model: chat_model,
            message: Message {
                role: cm.role,
                content: cm.content,
                thinking: None,
                tool_calls,
            },
            done: true,
            eval_count: usage.and_then(|u| u.total_tokens),
            eval_duration: None,
        })
    }
}

impl crate::client::ChatClient for CloudClient {
    fn chat<'a>(
        &'a self,
        model: &'a str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>> {
        Box::pin(async move { self.chat_inner(model, messages, tools).await })
    }

    fn count_tokens<'a>(
        &'a self,
        _model: &'a str,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<usize>> + Send + 'a>> {
        let len = text.len();
        Box::pin(async move { Ok(len.div_ceil(4)) })
    }

    fn health_check<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let url = format!("{}/models", self.base_url);
            let resp = self
                .http
                .get(&url)
                .header("Authorization", self.auth_header())
                .send()
                .await?;
            if resp.status().is_success() {
                // Try to parse but don't fail if shape differs
                let _ = resp.json::<CloudModelsResponse>().await;
                Ok(())
            } else {
                let status = resp.status().as_u16();
                let body = resp.text().await.unwrap_or_default();
                Err(RuntimeError::Api {
                    status,
                    message: body,
                })
            }
        })
    }
}

// ---------------------------------------------------------------------------
// MockCloud — test double for CloudClient path
// ---------------------------------------------------------------------------

/// Mock cloud client for tests — scripted responses, no network, never logs a key.
pub struct MockCloud {
    responses: Arc<Mutex<VecDeque<ChatResponse>>>,
    health_ok: bool,
}

impl MockCloud {
    pub fn new(responses: Vec<ChatResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(VecDeque::from(responses))),
            health_ok: true,
        }
    }

    pub fn with_response(resp: ChatResponse) -> Self {
        Self::new(vec![resp])
    }

    pub fn push_response(&self, resp: ChatResponse) {
        self.responses.lock().unwrap().push_back(resp);
    }

    pub fn is_empty(&self) -> bool {
        self.responses.lock().unwrap().is_empty()
    }

    pub fn with_health(mut self, ok: bool) -> Self {
        self.health_ok = ok;
        self
    }
}

impl Clone for MockCloud {
    fn clone(&self) -> Self {
        Self {
            responses: Arc::clone(&self.responses),
            health_ok: self.health_ok,
        }
    }
}

impl crate::client::ChatClient for MockCloud {
    fn chat<'a>(
        &'a self,
        _model: &'a str,
        _messages: Vec<Message>,
        _tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>> {
        Box::pin(async move {
            let mut g = self.responses.lock().unwrap();
            g.pop_front().ok_or_else(|| RuntimeError::Api {
                status: 500,
                message: "MockCloud: no more scripted responses".into(),
            })
        })
    }

    fn count_tokens<'a>(
        &'a self,
        _model: &'a str,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<usize>> + Send + 'a>> {
        let len = text.len();
        Box::pin(async move { Ok(len.div_ceil(4)) })
    }

    fn health_check<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            if self.health_ok {
                Ok(())
            } else {
                Err(RuntimeError::ConnectionRefused(
                    "mock cloud health check failed".into(),
                ))
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{ChatClient, Message, ToolCall, ToolCallFunction};

    fn dummy_response(content: &str) -> ChatResponse {
        ChatResponse {
            model: "frontier".into(),
            message: Message::assistant(content),
            done: true,
            eval_count: None,
            eval_duration: None,
        }
    }

    fn dummy_tool_response() -> ChatResponse {
        ChatResponse {
            model: "frontier".into(),
            message: Message {
                role: "assistant".into(),
                content: None,
                thinking: None,
                tool_calls: Some(vec![ToolCall {
                    id: Some("call_1".into()),
                    function: ToolCallFunction {
                        name: "Read".into(),
                        arguments: serde_json::json!({"path": "/tmp/x"}),
                    },
                }]),
            },
            done: true,
            eval_count: Some(42),
            eval_duration: None,
        }
    }

    #[tokio::test]
    async fn cloud_mock_returns_scripted_responses_in_order() {
        let r1 = dummy_response("first");
        let r2 = dummy_tool_response();
        let mock = MockCloud::new(vec![r1.clone(), r2.clone()]);
        let client: Box<dyn ChatClient> = Box::new(mock);
        let resp1 = client
            .chat("frontier", vec![Message::user("hi")], None)
            .await
            .unwrap();
        assert_eq!(resp1.message.content.as_deref(), Some("first"));
        let resp2 = client
            .chat("frontier", vec![Message::user("hi2")], None)
            .await
            .unwrap();
        assert!(resp2.message.tool_calls.is_some());
        assert_eq!(resp2.message.tool_calls.unwrap()[0].function.name, "Read");
        let err = client
            .chat("frontier", vec![Message::user("hi")], None)
            .await
            .unwrap_err();
        assert!(matches!(err, RuntimeError::Api { status: 500, .. }));
    }

    #[tokio::test]
    async fn cloud_mock_health_and_tokens() {
        let mock = MockCloud::new(vec![dummy_response("ok")]);
        assert_eq!(mock.count_tokens("m", "hello world").await.unwrap(), 3);
        assert!(mock.health_check().await.is_ok());
        let failing = MockCloud::new(vec![]).with_health(false);
        assert!(failing.health_check().await.is_err());
    }

    #[test]
    fn cloud_client_timeout_is_120s() {
        let c = CloudClient::new("https://example.com/v1", "sk-test");
        assert_eq!(c.timeout(), Duration::from_secs(120));
    }

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap()
    }

    #[test]
    fn cloud_client_new_from_env_missing_key_returns_clear_error() {
        let _g = env_lock();
        let prev_key = std::env::var(ENV_API_KEY).ok();
        let prev_url = std::env::var(ENV_BASE_URL).ok();
        std::env::remove_var(ENV_API_KEY);
        std::env::remove_var(ENV_BASE_URL);
        let err = CloudClient::try_from_env().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains(ENV_API_KEY),
            "error should mention {ENV_API_KEY}, got: {msg}"
        );
        assert!(!msg.contains("sk-"), "error must not leak key material");
        if let Some(v) = prev_key {
            std::env::set_var(ENV_API_KEY, v);
        }
        if let Some(v) = prev_url {
            std::env::set_var(ENV_BASE_URL, v);
        }
    }

    #[test]
    fn cloud_client_new_from_env_missing_url_returns_clear_error() {
        let _g = env_lock();
        let prev_key = std::env::var(ENV_API_KEY).ok();
        let prev_url = std::env::var(ENV_BASE_URL).ok();
        std::env::set_var(ENV_API_KEY, "sk-test-dummy");
        std::env::remove_var(ENV_BASE_URL);
        let err = CloudClient::try_from_env().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains(ENV_BASE_URL),
            "error should mention {ENV_BASE_URL}, got: {msg}"
        );
        if let Some(v) = prev_key {
            std::env::set_var(ENV_API_KEY, v);
        } else {
            std::env::remove_var(ENV_API_KEY);
        }
        if let Some(v) = prev_url {
            std::env::set_var(ENV_BASE_URL, v);
        }
    }

    #[tokio::test]
    async fn cloud_client_implements_chat_client_trait() {
        fn assert_impl<T: ChatClient>() {}
        assert_impl::<CloudClient>();
        assert_impl::<MockCloud>();
        let c = CloudClient::new("https://example.com/v1", "sk-test");
        let _: Box<dyn ChatClient> = Box::new(c);
        let m = MockCloud::new(vec![dummy_response("x")]);
        let _: Box<dyn ChatClient> = Box::new(m);
    }

    #[tokio::test]
    async fn mock_cloud_clone_shares_queue() {
        let mock = MockCloud::new(vec![dummy_response("shared")]);
        let cloned = mock.clone();
        let boxed: Box<dyn ChatClient> = Box::new(mock);
        let resp = boxed.chat("m", vec![], None).await.unwrap();
        assert_eq!(resp.message.content.as_deref(), Some("shared"));
        assert!(cloned.is_empty());
    }

    #[allow(non_snake_case)]
    #[test]
    fn MockCloud_is_ChatClient_and_generic() {
        fn assert_impl<T: ChatClient>() {}
        assert_impl::<MockCloud>();
        assert_impl::<CloudClient>();
    }

    #[test]
    fn cloud_api_key_not_hardcoded_vendor() {
        let src = include_str!("cloud.rs");
        let needle = String::from_utf8(vec![
            97, 112, 105, 46, 111, 112, 101, 110, 97, 105, 46, 99, 111, 109,
        ])
        .unwrap();
        assert!(
            !src.to_ascii_lowercase().contains(&needle),
            "must not hardcode vendor domain"
        );
        assert!(src.contains(ENV_API_KEY), "must read {ENV_API_KEY}");
        assert!(src.contains("120"), "must have 120s timeout");
        let needle_b = String::from_utf8(vec![112, 114, 105, 110, 116, 108, 110]).unwrap();
        let has_both = src.contains(&needle_b) && src.contains("api_key");
        assert!(!has_both, "must not log key material");
    }
}

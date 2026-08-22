//! Cloud fallback — generic OpenAI-compatible `ChatClient`.
//!
//! `CloudClient` is vendor-agnostic: it speaks the OpenAI `/v1/chat/completions`
//! wire format against any base URL (OpenAI, Azure, OpenRouter, etc.).
//! No vendor is hardcoded. API key is read from `KITTY_CLOUD_API_KEY` and
//! never logged. Timeout defaults to 120s and can be raised via
//! `KITTY_OLLAMA_TIMEOUT_SECS` (shared with [`crate::client::OllamaClient`],
//! whose own default is 300s); invalid/missing values fall back to 120s.
//! Cost is metered via `tracing::info!`.
//!
//! The key is held in [`SecretString`] (pattern stolen from Catalyst
//! `apohara-secrets`): zeroized on drop, redacted in `Debug`, and exposed
//! as cleartext only at the HTTP transport boundary via
//! [`SecretString::expose`]. Env vars forwarded to child processes should be
//! filtered through [`is_secret_env_name`] / [`sanitize_env`].

use std::collections::VecDeque;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};
use zeroize::Zeroize;

use crate::client::parse_timeout_secs;
use crate::client::{ChatResponse, ENV_TIMEOUT_SECS, Message, ToolCall, ToolCallFunction};
use crate::error::{Result, RuntimeError};
use crate::tools::ToolDefinition;

const ENV_API_KEY: &str = "KITTY_CLOUD_API_KEY";
const ENV_BASE_URL: &str = "KITTY_CLOUD_BASE_URL";

/// Default request timeout for [`CloudClient`] — 120s.
pub const DEFAULT_CLOUD_TIMEOUT_SECS: u64 = 120;

// ---------------------------------------------------------------------------
// SecretString — zeroize-on-drop, Debug-redacted (Catalyst apohara-secrets)
// ---------------------------------------------------------------------------

/// Wrapper around a secret string that:
///   - zeroes its memory on drop (defeats heap re-read attacks),
///   - hides its value in `Debug` so `tracing::info!(?secret, ...)` cannot
///     accidentally log the cleartext token.
///
/// Use [`SecretString::expose`] only at the moment you have to hand the bytes
/// to a transport (HTTP header, env var, file write) and never `.clone()`
/// the inner String — that would extend the lifetime of plaintext into
/// callers that may forget the zeroize discipline.
pub struct SecretString(String);

impl SecretString {
    pub fn new(value: String) -> Self {
        Self(value)
    }

    /// Borrow the cleartext. Caller is responsible for not copying the
    /// returned slice into long-lived plaintext storage.
    pub fn expose(&self) -> &str {
        &self.0
    }

    pub fn into_inner(self) -> String {
        // Bypasses the Drop scrubber — only use when the caller has its
        // own zeroize discipline (rare).
        let mut s = self;
        std::mem::take(&mut s.0)
    }
}

impl Drop for SecretString {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SecretString(***)")
    }
}

// ---------------------------------------------------------------------------
// Secret env-name predicate — ported from Catalyst apohara-sandbox runner
// ---------------------------------------------------------------------------

/// Patterns (substring match on uppercased name) that ALWAYS strip the
/// variable: `*_API_KEY`, `*_TOKEN`, `AWS_*`, provider prefixes, etc.
/// Mirrors Catalyst `apohara-sandbox/src/runner/imp.rs:is_secret_env_name`.
#[must_use]
pub fn is_secret_env_name(name: &str) -> bool {
    let up = name.to_ascii_uppercase();
    const SUFFIXES: &[&str] = &[
        "_API_KEY",
        "_KEY",
        "_TOKEN",
        "_SECRET",
        "_PASSWORD",
        "_PASSWD",
    ];
    const PREFIXES: &[&str] = &[
        "ANTHROPIC_",
        "OPENAI_",
        "GROQ_",
        "TOGETHER_",
        "MISTRAL_",
        "OPENROUTER_",
        "GEMINI_",
        "GOOGLE_",
        "COHERE_",
        "XAI_",
        "DEEPSEEK_",
        "PERPLEXITY_",
        "FIREWORKS_",
        "PINECONE_",
        "VOYAGE_",
        "REPLICATE_",
        "HUGGINGFACE_",
        "AWS_",
        "GCP_",
        "GCLOUD_",
        "AZURE_",
        "CLOUDFLARE_",
        "DIGITALOCEAN_",
        "VERCEL_",
        "NETLIFY_",
        "HEROKU_",
        "RENDER_",
        "SUPABASE_",
        "STRIPE_",
        "SENTRY_",
        "CIRCLE_",
        "GITLAB_",
        "BITBUCKET_",
    ];
    if PREFIXES.iter().any(|p| up.starts_with(p)) {
        return true;
    }
    if SUFFIXES.iter().any(|s| up.ends_with(s)) {
        return true;
    }
    matches!(
        up.as_str(),
        "HF_TOKEN"
            | "GITHUB_TOKEN"
            | "GH_TOKEN"
            | "FLY_API_TOKEN"
            | "RAILWAY_TOKEN"
            | "NPM_TOKEN"
            | "DATABASE_URL"
            | "MONGODB_URI"
            | "MYSQL_URL"
            | "POSTGRES_URL"
            | "REDIS_URL"
            | "NOTION_TOKEN"
            | "SLACK_TOKEN"
            | "SLACK_BOT_TOKEN"
            | "DISCORD_TOKEN"
            | "TELEGRAM_TOKEN"
            | "LINEAR_API_KEY"
    )
}

/// Strip every secret-looking variable from an env pair collection before it
/// reaches a child process or log line. Especially strips
/// `KITTY_CLOUD_API_KEY`, so sandboxed children can never enable paid cloud
/// fallback with host credentials.
#[must_use]
pub fn sanitize_env<I, K, V>(vars: I) -> Vec<(String, String)>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: AsRef<str>,
{
    vars.into_iter()
        .filter(|(k, _)| !is_secret_env_name(k.as_ref()))
        .map(|(k, v)| (k.as_ref().to_string(), v.as_ref().to_string()))
        .collect()
}

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
/// - `api_key`: bearer token held as [`SecretString`] (zeroized on drop,
///   redacted in `Debug`), read from `KITTY_CLOUD_API_KEY` via
///   [`CloudClient::from_env`]. Cleartext exists only at the transport
///   boundary ([`CloudClient::auth_header`]).
/// - Timeout defaults to 120s per request (rustls), overridable via
///   `KITTY_OLLAMA_TIMEOUT_SECS` (see [`parse_timeout_secs`]).
/// - Cost is metered via `tracing::info!` on every successful chat (tokens + cost_usd).
pub struct CloudClient {
    base_url: String,
    api_key: SecretString,
    http: reqwest::Client,
    timeout: Duration,
}

impl std::fmt::Debug for CloudClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CloudClient")
            .field("base_url", &self.base_url)
            // SecretString's Debug renders "SecretString(***)" — the key can
            // never surface through `{client:?}` logging.
            .field("api_key", &self.api_key)
            .finish()
    }
}

impl CloudClient {
    /// Create with explicit base URL and API key.
    ///
    /// Timeout comes from `KITTY_OLLAMA_TIMEOUT_SECS` when set to a valid
    /// `u64`, else [`DEFAULT_CLOUD_TIMEOUT_SECS`] (120s).
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        let api_key = SecretString::new(api_key.into());
        let raw = std::env::var(ENV_TIMEOUT_SECS).ok();
        let timeout = Duration::from_secs(parse_timeout_secs(raw.as_deref(), DEFAULT_CLOUD_TIMEOUT_SECS));
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .expect("reqwest::Client builder should not fail");
        Self {
            base_url,
            api_key,
            http,
            timeout,
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

    /// Expose timeout for verification (default 120s, env-overridable).
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    fn auth_header(&self) -> String {
        // Only place the cleartext leaves the SecretString: the HTTP
        // transport boundary. Never logged, never stored elsewhere.
        format!("Bearer {}", self.api_key.expose())
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

    #[test]
    fn cloud_timeout_default_flows_through_parse_fn() {
        // The constructor's env read reduces to this pure call; asserting it
        // here pins CloudClient's 120s default without process-env mutation.
        assert_eq!(
            parse_timeout_secs(None, DEFAULT_CLOUD_TIMEOUT_SECS),
            120
        );
        assert_eq!(
            parse_timeout_secs(Some("900"), DEFAULT_CLOUD_TIMEOUT_SECS),
            900
        );
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

    #[allow(non_snake_case)]
    #[test]
    fn CloudClient_debug_redacts() {
        let client = CloudClient::new("https://example.com/v1", "sk-test-1234567890");
        let rendered = format!("{client:?}");
        assert!(
            !rendered.contains("sk-"),
            "Debug must never leak key material, got: {rendered}"
        );
        assert!(
            rendered.contains("SecretString(***)"),
            "Debug must show the redacted SecretString marker, got: {rendered}"
        );
    }

    #[test]
    fn cloud_secret_string_debug_redacts_verbatim() {
        let s = SecretString::new("sk-123".to_string());
        assert_eq!(format!("{s:?}"), "SecretString(***)");
        assert_eq!(s.expose(), "sk-123");
    }

    #[test]
    fn cloud_secret_string_zeroize_scrubs_buffer() {
        // Validates the exact scrubbing call `SecretString::drop` performs
        // (`self.0.zeroize()`), inspected while the buffer is still alive.
        // Reading the buffer *after* drop is unsound: the allocator reuses
        // freed-chunk headers for its own metadata.
        const SECRET: &[u8] = b"top-secret-key-value";
        let s = SecretString::new(String::from_utf8(SECRET.to_vec()).unwrap());
        let mut inner = s.into_inner(); // bypasses the Drop scrubber on purpose
        assert_eq!(inner.as_bytes(), SECRET);
        inner.zeroize();
        assert!(
            inner.as_bytes().iter().all(|&b| b == 0),
            "buffer must be fully scrubbed by zeroize"
        );
    }

    #[test]
    fn cloud_is_secret_env_name_strips_patterns() {
        for name in [
            "KITTY_CLOUD_API_KEY",
            "OPENAI_API_KEY",
            "MY_SERVICE_CLIENT_KEY",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "GITHUB_TOKEN",
            "MY_DB_PASSWORD",
            "SERVICE_PASSWD",
            "DATABASE_URL",
            "LINEAR_API_KEY",
            "huggingface_token",
        ] {
            assert!(
                is_secret_env_name(name),
                "{name} must be classified as secret"
            );
        }
        for name in [
            "PATH",
            "HOME",
            "KITTY_BENCH_MODEL",
            "KITTY_CLOUD_BASE_URL",
            "OLLAMA_HOST",
            "TERM",
        ] {
            assert!(
                !is_secret_env_name(name),
                "{name} must NOT be classified as secret"
            );
        }
    }

    #[test]
    fn cloud_sanitize_env_strips_secrets_keeps_rest() {
        let env = vec![
            ("KITTY_CLOUD_API_KEY", "sk-host-secret"),
            ("AWS_SESSION_TOKEN", "tok"),
            ("PATH", "/usr/bin"),
            ("KITTY_BENCH_MODEL", "ornith:9b"),
        ];
        let sanitized = sanitize_env(env);
        assert!(!sanitized.iter().any(|(k, _)| k == "KITTY_CLOUD_API_KEY"));
        assert!(!sanitized.iter().any(|(k, _)| k == "AWS_SESSION_TOKEN"));
        assert_eq!(
            sanitized,
            vec![
                ("PATH".to_string(), "/usr/bin".to_string()),
                ("KITTY_BENCH_MODEL".to_string(), "ornith:9b".to_string()),
            ]
        );
    }
}

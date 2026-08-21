use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::error::{Result, RuntimeError};
use crate::tools::ToolDefinition;

// ---------------------------------------------------------------------------
// ChatClient trait — abstraction for Agent (Lane A)
// ---------------------------------------------------------------------------

/// Abstraction over an LLM chat backend (Ollama, llama.cpp, mock).
///
/// `Agent` should depend on this trait, not on `OllamaClient` concretely,
/// so that tests can inject a [`MockClient`] and future backends can be
/// added without touching the harness.
pub trait ChatClient: Send + Sync {
    fn chat<'a>(
        &'a self,
        model: &'a str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>>;

    fn count_tokens<'a>(
        &'a self,
        model: &'a str,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<usize>> + Send + 'a>>;

    fn health_check<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}

// ---------------------------------------------------------------------------
// Wire types — Ollama chat API
// ---------------------------------------------------------------------------

/// Tool call returned by Ollama.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: Option<String>,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl Message {
    /// Convenience constructor for a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: Some(content.into()),
            thinking: None,
            tool_calls: None,
        }
    }

    /// Convenience constructor for an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: Some(content.into()),
            thinking: None,
            tool_calls: None,
        }
    }

    /// Convenience constructor for a tool-result message.
    pub fn tool_result(content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: Some(content.into()),
            thinking: None,
            tool_calls: None,
        }
    }

    /// Build an assistant message that carries tool_calls (no text content).
    pub fn tool_calls(calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".into(),
            content: None,
            thinking: None,
            tool_calls: Some(calls),
        }
    }
}

/// Chat request to Ollama.
#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<RequestOptions>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RequestOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<i32>,
}

/// Chat response from Ollama.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatResponse {
    pub model: String,
    pub message: Message,
    pub done: bool,
    #[serde(default)]
    pub eval_count: Option<i64>,
    #[serde(default)]
    pub eval_duration: Option<i64>,
}

/// Response from Ollama's `/api/tokenize` endpoint.
#[derive(Debug, Deserialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<i64>,
}

// ---------------------------------------------------------------------------
// Model listing
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub size: Option<i64>,
    pub digest: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ModelListResponse {
    pub models: Vec<ModelInfo>,
}

// ---------------------------------------------------------------------------
// OllamaClient
// ---------------------------------------------------------------------------

/// Typed client for the Ollama HTTP API.
///
/// All network errors are surfaced as [`RuntimeError::Http`]. Non-2xx HTTP
/// status codes are mapped to [`RuntimeError::Api`].
#[derive(Clone)]
pub struct OllamaClient {
    base_url: String,
    http: reqwest::Client,
}

impl OllamaClient {
    /// Create a client pointing at the given base URL.
    pub fn new(base_url: &str) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300)) // 5 min for long generations
            .build()
            .expect("reqwest::Client::builder should not fail");
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            http,
        }
    }

    /// Convenience: client pointing at `http://localhost:11434`.
    pub fn default_url() -> Self {
        Self::new("http://localhost:11434")
    }

    /// Send a chat completion request (non-streaming).
    pub async fn chat(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        options: Option<RequestOptions>,
    ) -> Result<ChatResponse> {
        let request = ChatRequest {
            model: model.to_string(),
            messages,
            tools,
            stream: Some(false),
            options,
        };

        let url = format!("{}/api/chat", self.base_url);
        debug!(url = %url, model = %model, "sending chat request");

        let resp = self.http.post(&url).json(&request).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            warn!(status = %status, body = %body, "ollama chat returned error");
            return Err(RuntimeError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let response: ChatResponse = resp.json().await?;
        Ok(response)
    }

    /// Health check — returns `true` if the Ollama server is reachable and
    /// responds with 200 on `/api/tags`.
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/api/tags", self.base_url);
        match self.http.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(e) => {
                debug!(error = %e, "health check failed");
                Ok(false)
            }
        }
    }

    /// List locally available models.
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = format!("{}/api/tags", self.base_url);
        let resp = self.http.get(&url).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(RuntimeError::Api {
                status: status.as_u16(),
                message: body,
            });
        }
        let list: ModelListResponse = resp.json().await?;
        Ok(list.models)
    }

    /// Show model metadata.
    pub async fn show_model(&self, model: &str) -> Result<serde_json::Value> {
        let url = format!("{}/api/show", self.base_url);
        let body = serde_json::json!({ "name": model });
        let resp = self.http.post(&url).json(&body).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(RuntimeError::Api {
                status: status.as_u16(),
                message: text,
            });
        }
        let info: serde_json::Value = resp.json().await?;
        Ok(info)
    }

    /// Count tokens of `prompt` using the model's own tokenizer via Ollama's
    /// `/api/tokenize` endpoint. This is the *real* token count for the model,
    /// far more accurate than a character/word heuristic and required for
    /// correct context budgeting on the actual model being run.
    pub async fn count_tokens(&self, model: &str, prompt: &str) -> Result<usize> {
        let url = format!("{}/api/tokenize", self.base_url);
        let body = serde_json::json!({ "model": model, "prompt": prompt });
        let resp = self.http.post(&url).json(&body).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(RuntimeError::Api {
                status: status.as_u16(),
                message: text,
            });
        }
        let parsed: TokenizeResponse = resp.json().await?;
        Ok(parsed.tokens.len())
    }
}

impl ChatClient for OllamaClient {
    fn chat<'a>(
        &'a self,
        model: &'a str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>> {
        Box::pin(async move { OllamaClient::chat(self, model, messages, tools, None).await })
    }

    fn count_tokens<'a>(
        &'a self,
        model: &'a str,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<usize>> + Send + 'a>> {
        Box::pin(async move { OllamaClient::count_tokens(self, model, text).await })
    }

    fn health_check<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let ok = OllamaClient::health_check(self).await?;
            if ok {
                Ok(())
            } else {
                Err(RuntimeError::ConnectionRefused(
                    "ollama server not reachable or returned non-2xx".into(),
                ))
            }
        })
    }
}

impl<T: ChatClient + ?Sized> ChatClient for Box<T> {
    fn chat<'a>(
        &'a self,
        model: &'a str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>> {
        (**self).chat(model, messages, tools)
    }

    fn count_tokens<'a>(
        &'a self,
        model: &'a str,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<usize>> + Send + 'a>> {
        (**self).count_tokens(model, text)
    }

    fn health_check<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        (**self).health_check()
    }
}

// ---------------------------------------------------------------------------
// MockClient — test double for ChatClient (Lane A)
// ---------------------------------------------------------------------------

pub struct MockClient {
    responses: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<ChatResponse>>>,
    health_ok: bool,
}

impl MockClient {
    /// Create a mock with scripted responses in order.
    pub fn new(responses: Vec<ChatResponse>) -> Self {
        Self {
            responses: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::VecDeque::from(responses),
            )),
            health_ok: true,
        }
    }

    /// Create a mock with a single response.
    pub fn with_response(resp: ChatResponse) -> Self {
        Self::new(vec![resp])
    }

    /// Push a response to the back of the queue (thread-safe via Arc).
    pub fn push_response(&self, resp: ChatResponse) {
        self.responses.lock().unwrap().push_back(resp);
    }

    /// Returns true if no more scripted responses remain.
    pub fn is_empty(&self) -> bool {
        self.responses.lock().unwrap().is_empty()
    }

    /// Override health check result.
    pub fn with_health(mut self, ok: bool) -> Self {
        self.health_ok = ok;
        self
    }
}

impl Clone for MockClient {
    fn clone(&self) -> Self {
        Self {
            responses: std::sync::Arc::clone(&self.responses),
            health_ok: self.health_ok,
        }
    }
}

impl ChatClient for MockClient {
    fn chat<'a>(
        &'a self,
        _model: &'a str,
        _messages: Vec<Message>,
        _tools: Option<Vec<ToolDefinition>>,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>> {
        Box::pin(async move {
            let mut guard = self.responses.lock().unwrap();
            guard.pop_front().ok_or_else(|| RuntimeError::Api {
                status: 500,
                message: "MockClient: no more scripted responses".into(),
            })
        })
    }

    fn count_tokens<'a>(
        &'a self,
        _model: &'a str,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<usize>> + Send + 'a>> {
        // Heuristic fallback similar to context.rs estimation.
        let len = text.len();
        Box::pin(async move { Ok(len.div_ceil(4)) })
    }

    fn health_check<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            if self.health_ok {
                Ok(())
            } else {
                Err(RuntimeError::ConnectionRefused(
                    "mock health check failed".into(),
                ))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::error::RuntimeError;

    use super::*;

    #[test]
    fn message_user_constructor() {
        let m = Message::user("hello");
        assert_eq!(m.role, "user");
        assert_eq!(m.content.as_deref(), Some("hello"));
        assert!(m.tool_calls.is_none());
    }

    #[test]
    fn chat_request_serialization_omits_none_fields() {
        let req = ChatRequest {
            model: "test".into(),
            messages: vec![Message::user("hi")],
            tools: None,
            stream: Some(false),
            options: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert!(!json.as_object().unwrap().contains_key("tools"));
        assert!(!json.as_object().unwrap().contains_key("options"));
    }

    #[test]
    fn tokenize_response_parses_token_count() {
        let json = r#"{"tokens":[1,2,3,4]}"#;
        let parsed: TokenizeResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.tokens.len(), 4);
    }

    fn dummy_response(content: &str) -> ChatResponse {
        ChatResponse {
            model: "test-model".into(),
            message: Message::assistant(content),
            done: true,
            eval_count: None,
            eval_duration: None,
        }
    }

    #[tokio::test]
    async fn mock_client_returns_scripted_responses_in_order() {
        let r1 = dummy_response("first");
        let r2 = dummy_response("second");
        let mock = MockClient::new(vec![r1.clone(), r2.clone()]);
        let client: Box<dyn ChatClient> = Box::new(mock);

        let resp1 = client
            .chat("test-model", vec![Message::user("hi")], None)
            .await
            .unwrap();
        assert_eq!(resp1.message.content.as_deref(), Some("first"));

        let resp2 = client
            .chat("test-model", vec![Message::user("hi again")], None)
            .await
            .unwrap();
        assert_eq!(resp2.message.content.as_deref(), Some("second"));

        // Exhausted queue should error
        let err = client
            .chat("test-model", vec![Message::user("hi")], None)
            .await
            .unwrap_err();
        assert!(matches!(err, RuntimeError::Api { status: 500, .. }));
    }

    #[tokio::test]
    async fn mock_client_count_tokens_and_health_check() {
        let mock = MockClient::new(vec![dummy_response("ok")]);
        let count = mock
            .count_tokens("test-model", "hello world")
            .await
            .unwrap();
        // "hello world" len 11 -> ceil(11/4)=3
        assert_eq!(count, 3);
        assert!(mock.health_check().await.is_ok());

        let failing = MockClient::new(vec![]).with_health(false);
        assert!(failing.health_check().await.is_err());
    }

    #[tokio::test]
    async fn mock_client_is_object_safe_and_clone_shares_queue() {
        let mock = MockClient::new(vec![dummy_response("shared")]);
        let cloned = mock.clone();
        let boxed: Box<dyn ChatClient> = Box::new(mock);
        // cloned shares Arc, so consuming via boxed should affect clone
        let resp = boxed.chat("m", vec![], None).await.unwrap();
        assert_eq!(resp.message.content.as_deref(), Some("shared"));
        assert!(cloned.is_empty());
    }

    #[tokio::test]
    async fn ollama_client_implements_chat_client_trait() {
        // Compile-time check that OllamaClient satisfies ChatClient.
        fn assert_impl<T: ChatClient>() {}
        assert_impl::<OllamaClient>();
        // Also check MockClient
        assert_impl::<MockClient>();

        let client = OllamaClient::new("http://localhost:11434");
        // Ensure we can coerce to Box<dyn ChatClient>
        let _: Box<dyn ChatClient> = Box::new(client);
    }
}

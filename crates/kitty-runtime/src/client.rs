use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::error::{Result, RuntimeError};
use crate::tools::ToolDefinition;

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
#[derive(Debug, Deserialize)]
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

#[cfg(test)]
mod tests {
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
}

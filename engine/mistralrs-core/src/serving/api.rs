//! OpenAI-compatible API shapes for kitty-infer.
//!
//! Pure request/response types + small handler helpers. A thin binary wires
//! these to HTTP; this module owns the contract: chat (stream/non-stream
//! with tool_calls), models, admin load/status/cancel, gpu status/reclaim,
//! health+hint, metrics, logs.
//!
//! Auth: none enabled by default. If a deploy sets `KITTY_API_TOKEN`,
//! [`check_auth`] enforces it fail-closed (missing/wrong bearer => deny).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use super::errors::KittyError;

// ---------- chat ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ChatFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
}

impl ChatRequest {
    pub fn validate(&self) -> Result<(), KittyError> {
        if self.model.trim().is_empty() {
            return Err(KittyError::Validation { detail: "field `model` is required".to_string() });
        }
        if self.messages.is_empty() {
            return Err(KittyError::Validation { detail: "field `messages` must not be empty".to_string() });
        }
        for m in &self.messages {
            if m.role.trim().is_empty() {
                return Err(KittyError::Validation { detail: "message role must not be empty".to_string() });
            }
        }
        Ok(())
    }
}

// ---------- models ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub object: String,
    pub resident: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelEntry>,
}

// ---------- admin ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminLoadRequest {
    /// `hf:<repo>/<file>` | `ollama:<name>` | `local:<path.gguf>` | bare `.gguf` path.
    pub spec: String,
    #[serde(default)]
    pub gpu_fraction: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminStatusResponse {
    pub resident: Option<String>,
    pub gpu_fraction: Option<f32>,
    pub queue_depth: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminCancelResponse {
    pub cancelled: bool,
    pub task_id: String,
}

// ---------- gpu ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStatusResponse {
    pub backend: String,
    pub devices: Vec<super::health::GpuDeviceStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuReclaimResponse {
    pub reclaimed: bool,
    pub detail: String,
}

// ---------- health / metrics / logs ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub ok: bool,
    pub model: Option<String>,
    pub gpu_backend: String,
    pub hint: String,
}

pub fn health(model: Option<&str>, gpu_backend: &str) -> HealthResponse {
    let (ok, hint) = match model {
        Some(_) => (true, "ready".to_string()),
        None => (
            false,
            "no model resident: load one with POST /admin/load (`kitty load <spec>`)".to_string(),
        ),
    };
    HealthResponse { ok, model: model.map(|s| s.to_string()), gpu_backend: gpu_backend.to_string(), hint }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricsResponse {
    pub requests_total: u64,
    pub errors_total: u64,
    pub resident: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogsResponse {
    pub entries: Vec<super::errors::LogEntry>,
}

// ---------- admin task registry with real cancellation ----------

/// Tracks in-flight admin jobs (loads). `cancel` aborts the Tokio task —
/// real `JoinHandle::abort` (`Task.cancel` semantics), not a boolean flag
/// the job would have to poll.
pub struct AdminTasks {
    inner: Mutex<HashMap<String, JoinHandle<()>>>,
}

impl AdminTasks {
    pub fn new() -> Self {
        Self { inner: Mutex::new(HashMap::new()) }
    }

    pub async fn spawn<F>(&self, task_id: String, fut: F)
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        let handle = tokio::spawn(fut);
        self.inner.lock().await.insert(task_id, handle);
    }

    pub async fn finish(&self, task_id: &str) {
        self.inner.lock().await.remove(task_id);
    }

    /// Abort the task. Returns true if a task was aborted.
    pub async fn cancel(&self, task_id: &str) -> bool {
        let handle = self.inner.lock().await.remove(task_id);
        match handle {
            Some(h) => {
                h.abort();
                true
            }
            None => false,
        }
    }

    pub async fn len(&self) -> usize {
        self.inner.lock().await.len()
    }
}

impl Default for AdminTasks {
    fn default() -> Self {
        Self::new()
    }
}

// ---------- auth (fail-closed when enabled) ----------

/// If `expected` (deploy token) is set, `provided` bearer must match exactly
/// or the request is denied. If no token is configured, requests are allowed
/// (documented for loopback-only deploys). No `change-me` defaults anywhere.
pub fn check_auth(expected: Option<&str>, provided: Option<&str>) -> Result<(), KittyError> {
    match expected {
        Some(token) if !token.is_empty() => match provided {
            Some(given) if given == token => Ok(()),
            _ => Err(KittyError::Validation { detail: "unauthorized: valid bearer token required".to_string() }),
        },
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_validate() {
        let bad = ChatRequest {
            model: "".to_string(),
            messages: vec![],
            stream: false,
            temperature: None,
            max_tokens: None,
            tools: None,
            tool_choice: None,
        };
        assert!(bad.validate().is_err());
        let ok = ChatRequest {
            model: "m".to_string(),
            messages: vec![ChatMessage { role: "user".to_string(), content: "hi".to_string() }],
            stream: false,
            temperature: None,
            max_tokens: None,
            tools: None,
            tool_choice: None,
        };
        assert!(ok.validate().is_ok());
    }

    #[test]
    fn auth_fail_closed() {
        assert!(check_auth(Some("secret"), Some("secret")).is_ok());
        assert!(check_auth(Some("secret"), Some("wrong")).is_err());
        assert!(check_auth(Some("secret"), None).is_err());
        assert!(check_auth(None, None).is_ok());
    }

    #[tokio::test]
    async fn cancel_aborts_real_task() {
        let tasks = AdminTasks::new();
        tasks
            .spawn("t1".to_string(), async {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            })
            .await;
        assert!(tasks.cancel("t1").await);
        assert!(!tasks.cancel("t1").await);
    }
}

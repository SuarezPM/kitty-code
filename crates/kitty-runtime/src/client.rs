//! Chat backend abstraction: [`ChatClient`] trait, wire types, and
//! [`OllamaClient`] (typed Ollama HTTP API client).
//!
//! Request timeout defaults to 300s (long reasoning generations) and can be
//! raised for heavy bench workloads via `KITTY_OLLAMA_TIMEOUT_SECS`
//! (invalid/missing value falls back to the default). The same variable also
//! overrides [`crate::cloud::CloudClient`], whose own default is 120s.
//!
//! # Explicit context window (F2)
//!
//! Ollama's server-side default for `num_ctx` is TIERED BY SERVER VRAM
//! (<24 GB VRAM → 4096) and prompts are TRUNCATED SILENTLY when they exceed
//! it. To make the window explicit and audible:
//!
//! - Every `/api/chat` payload carries `"options": {"num_ctx": N}` INSIDE the
//!   options object (never top-level). N resolves once per client from
//!   `KITTY_NUM_CTX` (parsed as `u32`) or [`DEFAULT_NUM_CTX`] (32768), and
//!   stays FIXED for the whole session — changing `num_ctx` between requests
//!   forces an expensive model reload server-side.
//! - Every payload carries `"truncate": false`, so overflow produces HTTP 400
//!   (`exceed_context_size_error`) mapped to
//!   [`RuntimeError::ContextOverflow`] instead of silent degradation.
//! - Responses expose `prompt_eval_count` (actual prompt tokens processed);
//!   when it reaches 95% of the window a warning is logged (truncation tell).
//!
//! Note: KV-cache quantization (`OLLAMA_KV_CACHE_TYPE=q8_0`) is a SERVER env
//! var applied at model load — it CANNOT be set via API/Modelfile/request.
//! [`kv_cache_preflight_warning`] implements the pragmatic client-side check.
//!
//! # Qwen thinking/reasoning normalization (E3)
//!
//! Qwen-family "think" models emit reasoning either server-side (Ollama's
//! `message.thinking`) or as an OpenAI-compat `reasoning_content` field.
//! Handling is deterministic and requires zero model intelligence:
//!
//! - Request side: when `KITTY_NO_THINK=1` **or** the model id contains
//!   "think", [`OllamaClient`] sends `"think": false` in `/api/chat`
//!   ([`should_disable_thinking`]), asking the server not to reason.
//! - Response side: reasoning is parsed into [`Message::thinking`] — stored
//!   separately and NEVER merged back into assistant `content`. On the
//!   OpenAI-compat path ([`crate::cloud::CloudClient`]) the provider-specific
//!   `reasoning_content` field is normalized into `thinking`, so it never
//!   leaks back into the conversation as if it were user-visible text
//!   (opencode `openaiCompatible[field]` transform pattern).
//!
//! # Dynamic max-output cap (E3)
//!
//! Requesting a fixed large output budget on a nearly-full context window
//! fails server-side ("requested 8192 output but prompt 24577"). The harness
//! caps requested output via [`dynamic_max_output`] =
//! `context_limit − estimated_input − 1024` (floor 256), estimating input
//! conservatively at chars/4 via [`estimate_input_tokens`], and passes it
//! through [`ChatClient::chat_with_options`] (honored by [`OllamaClient`]).

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::error::{Result, RuntimeError};
use crate::tools::ToolDefinition;

/// Env var overriding the HTTP request timeout, in seconds, for both
/// [`OllamaClient`] (default 300s) and [`crate::cloud::CloudClient`]
/// (default 120s). Invalid or missing values fall back to each default.
pub const ENV_TIMEOUT_SECS: &str = "KITTY_OLLAMA_TIMEOUT_SECS";

/// Env flag (E3): when set to exactly `1`, thinking mode is disabled for
/// qwen-family models regardless of the model id.
pub const ENV_NO_THINK: &str = "KITTY_NO_THINK";

/// Env var (F2): explicit context window in tokens for `/api/chat`
/// `options.num_ctx`. Parsed as `u32`; missing/invalid values fall back to
/// [`DEFAULT_NUM_CTX`]. Resolved ONCE per [`OllamaClient`] — the value must
/// stay fixed for a whole session (changing it reloads the model).
pub const ENV_NUM_CTX: &str = "KITTY_NUM_CTX";

/// Default context window requested from the server when `KITTY_NUM_CTX` is
/// unset. Chosen to defeat the tiered-by-VRAM server default (4096 on <24 GB
/// machines) that silently truncates prompts.
pub const DEFAULT_NUM_CTX: u32 = 32768;

/// Env var (F2 KV preflight): set to `q8_0` or `q4_0` to acknowledge that the
/// Ollama SERVER runs with `OLLAMA_KV_CACHE_TYPE` quantized. Large windows
/// (`num_ctx > 16384`) hold f16 KV caches that can OOM an 8 GB GPU
/// (~4.8 GB at 32k); q8_0 halves it. The client cannot introspect the
/// server's env, so this acknowledgement silences the warning.
pub const ENV_KV_QUANT_ACK: &str = "KITTY_KV_QUANT_ACK";

/// Pure resolver behind [`ENV_NUM_CTX`] — testable without process-env
/// mutation. Missing, empty, unparsable, or zero values fall back to
/// [`DEFAULT_NUM_CTX`].
#[must_use]
pub fn parse_num_ctx(raw: Option<&str>) -> u32 {
    match raw.map(str::trim) {
        Some(s) => s.parse::<u32>().ok().filter(|v| *v > 0),
        None => None,
    }
    .unwrap_or(DEFAULT_NUM_CTX)
}

/// F2 KV preflight: returns a human-readable warning when a large context
/// window is requested without acknowledging server-side KV quantization.
///
/// Pure predicate — the CLI prints the warning at agentic-session startup;
/// this function decides IF one is due. Returns `None` when num_ctx fits a
/// small GPU comfortably (≤16384) or the ack var is set to a known quant
/// type (`q8_0`, `q4_0`). No hard failure by design.
#[must_use]
pub fn kv_cache_preflight_warning(num_ctx: u32, kv_ack: Option<&str>) -> Option<String> {
    let acked = matches!(kv_ack.map(str::trim), Some("q8_0") | Some("q4_0"));
    if num_ctx <= 16384 || acked {
        return None;
    }
    Some(format!(
        "num_ctx={num_ctx} on a typical 8GB GPU holds a ~{:.1}GB f16 KV cache (OOM risk). \
         Start the Ollama server with OLLAMA_KV_CACHE_TYPE=q8_0 (halves KV memory), \
         or set {ENV_KV_QUANT_ACK}=q8_0 to acknowledge.",
        f64::from(num_ctx) * 150_000.0 / 1_000_000_000.0
    ))
}

/// True when `prompt_eval_count` reached 95% of the resolved window — the
/// truncation tell exposed by the response field.
#[must_use]
pub fn prompt_near_capacity(prompt_eval_count: i64, num_ctx: u32) -> bool {
    if prompt_eval_count < 0 || num_ctx == 0 {
        return false;
    }
    u64::try_from(prompt_eval_count)
        .map(|c| c >= u64::from(num_ctx) * 95 / 100)
        .unwrap_or(false)
}

/// True when an HTTP error body identifies a context-window overflow
/// (`truncate:false` turns silent truncation into this audible failure).
/// Matches Ollama's `exceed_context_size_error` type as well as plain-text
/// variants like "input length exceeds context size".
#[must_use]
pub fn is_context_overflow_body(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("exceed_context_size_error")
        || (lower.contains("exceed") && lower.contains("context"))
}

/// Map a non-2xx chat response to a [`RuntimeError`]: HTTP 400 bodies that
/// indicate context overflow become [`RuntimeError::ContextOverflow`] with an
/// actionable message; everything else stays [`RuntimeError::Api`]. Pure —
/// unit-testable without HTTP.
#[must_use]
pub fn classify_api_error(status: u16, body: &str, requested_num_ctx: u32) -> RuntimeError {
    if status == 400 && is_context_overflow_body(body) {
        return RuntimeError::ContextOverflow {
            requested: requested_num_ctx,
            message: body.to_string(),
        };
    }
    RuntimeError::Api {
        status,
        message: body.to_string(),
    }
}

/// Pure predicate behind [`should_disable_thinking`] — testable without
/// process-env mutation.
///
/// Thinking is disabled when `no_think_env == Some("1")` or the model id
/// contains "think" (case-insensitive; matches qwen think variants).
#[must_use]
pub fn should_disable_thinking_with(no_think_env: Option<&str>, model: &str) -> bool {
    no_think_env == Some("1") || model.to_ascii_lowercase().contains("think")
}

/// Whether thinking mode should be disabled for `model`: true when
/// `KITTY_NO_THINK=1` or the model id contains "think".
#[must_use]
pub fn should_disable_thinking(model: &str) -> bool {
    should_disable_thinking_with(std::env::var(ENV_NO_THINK).ok().as_deref(), model)
}

/// Default request timeout for [`OllamaClient`] — 5 min, sized for long
/// reasoning generations on consumer GPUs.
pub const DEFAULT_OLLAMA_TIMEOUT_SECS: u64 = 300;

/// Parse an optional raw env value into a timeout in seconds.
///
/// Missing, empty, or unparsable input yields `default`; otherwise the parsed
/// `u64`. Pure function — never touches process env — so callers and tests
/// can exercise override semantics without parallel-unsafe env mutation.
#[must_use]
pub fn parse_timeout_secs(raw: Option<&str>, default: u64) -> u64 {
    raw.and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Dynamic max-output cap (E3)
// ---------------------------------------------------------------------------

/// Token headroom reserved between prompt and generated output so the sum
/// always fits the context window.
pub const OUTPUT_RESERVE_TOKENS: usize = 1024;
/// Floor for the capped output size — small but still usable.
pub const MIN_OUTPUT_TOKENS: i32 = 256;

/// Conservative input-token estimate over a conversation: chars/4 across
/// message content, thinking, and serialized tool-call arguments.
#[must_use]
pub fn estimate_input_tokens(messages: &[Message]) -> usize {
    let mut chars = 0usize;
    for m in messages {
        if let Some(c) = m.content.as_deref() {
            chars += c.len();
        }
        if let Some(t) = m.thinking.as_deref() {
            chars += t.len();
        }
        if let Some(calls) = &m.tool_calls {
            for tc in calls {
                chars += tc.function.name.len();
                chars += serde_json::to_string(&tc.function.arguments)
                    .map(|s| s.len())
                    .unwrap_or(0);
            }
        }
    }
    chars.div_ceil(4)
}

/// Cap the requested output tokens so prompt + output fit the context window:
/// `context_limit − estimated_input_tokens − 1024`, floored at 256.
///
/// Prevents the "requested 8192 output but prompt 24577" failure class on
/// small local models.
#[must_use]
pub fn dynamic_max_output(context_limit: usize, estimated_input_tokens: usize) -> i32 {
    context_limit
        .saturating_sub(estimated_input_tokens)
        .saturating_sub(OUTPUT_RESERVE_TOKENS)
        .max(MIN_OUTPUT_TOKENS as usize) as i32
}

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

    /// Chat with explicit request options (e.g., a capped `num_predict` from
    /// [`dynamic_max_output`]).
    ///
    /// Default implementation IGNORES the options and delegates to
    /// [`ChatClient::chat`], keeping existing backends source-compatible;
    /// [`OllamaClient`] overrides it and honors them.
    fn chat_with_options<'a>(
        &'a self,
        model: &'a str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        _options: Option<RequestOptions>,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>> {
        Box::pin(async move { self.chat(model, messages, tools).await })
    }

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
    /// E3 qwen thinking normalization: `Some(false)` serializes as
    /// `"think": false` (Ollama) to disable thinking server-side; `None`
    /// leaves the field out entirely (server default).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<bool>,
    /// F2: `Some(false)` serializes as `"truncate": false`, turning context
    /// overflow into an audible HTTP 400 instead of silent prompt truncation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RequestOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,
    /// Context window override. HARNESS-OWNED: callers must leave this unset —
    /// [`OllamaClient`] injects the session-fixed window resolved once from
    /// `KITTY_NUM_CTX` / [`DEFAULT_NUM_CTX`]. Setting it here would alternate
    /// values across requests and reload the model server-side (guarded by a
    /// `debug_assert` in [`OllamaClient::build_chat_request`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<i32>,
    /// Caller-side thinking override (F2): `Some(false)` forces `"think":
    /// false` on this request regardless of model-id/env heuristics. Used by
    /// the harness to express a per-model profile's `ThinkingMode::Off`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<bool>,
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
    /// F2 truncation tell: actual prompt tokens processed server-side. When
    /// this nears the requested window, earlier turns were likely truncated.
    #[serde(default)]
    pub prompt_eval_count: Option<i64>,
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

/// Capacity of the per-client token-count LRU cache (f8 T4.3).
const TOKEN_CACHE_CAPACITY: usize = 256;

/// Mutex-protected LRU map for `/api/tokenize` results, keyed by
/// (model id, hashed prompt). Insertion-order eviction: the oldest entry is
/// dropped once [`TOKEN_CACHE_CAPACITY`] is exceeded. Interior mutability on
/// [`OllamaClient`]; `Arc` keeps clones sharing one cache.
#[derive(Default)]
struct TokenCache {
    map: std::collections::HashMap<(String, u64), usize>,
    order: std::collections::VecDeque<(String, u64)>,
}

impl TokenCache {
    /// Cached count for `key`, marking it most-recently-used.
    fn get(&mut self, key: &(String, u64)) -> Option<usize> {
        let hit = self.map.get(key).copied()?;
        // Refresh recency.
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        self.order.push_back(key.clone());
        Some(hit)
    }

    /// Insert `key → value`, evicting the oldest entry at capacity.
    fn insert(&mut self, key: (String, u64), value: usize) {
        if self.map.contains_key(&key) {
            // Refresh + overwrite (counts can't change, but stay correct).
            self.insert_touch(&key);
            self.map.insert(key, value);
            return;
        }
        if self.order.len() >= TOKEN_CACHE_CAPACITY {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            }
        }
        self.map.insert(key.clone(), value);
        self.order.push_back(key);
    }

    fn insert_touch(&mut self, key: &(String, u64)) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        self.order.push_back(key.clone());
    }
}

/// Stable cache key for a tokenize request: model id plus a hash of the
/// prompt text (DefaultHasher is process-stable enough for an in-memory
/// cache; collisions across 256 entries are negligible and only cost an
/// inaccurate cached count — never correctness of the request path).
fn token_cache_key(model: &str, prompt: &str) -> (String, u64) {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    prompt.hash(&mut hasher);
    (model.to_string(), hasher.finish())
}

/// Typed client for the Ollama HTTP API.
///
/// All network errors are surfaced as [`RuntimeError::Http`]. Non-2xx HTTP
/// status codes are mapped to [`RuntimeError::Api`], except context-window
/// overflows on 400 which become [`RuntimeError::ContextOverflow`] (F2).
///
/// The context window (`num_ctx`) is resolved ONCE at construction from
/// `KITTY_NUM_CTX` (or [`DEFAULT_NUM_CTX`]) and stays fixed for the whole
/// session — alternating values across requests would reload the model.
#[derive(Clone)]
pub struct OllamaClient {
    base_url: String,
    http: reqwest::Client,
    timeout: Duration,
    num_ctx: u32,
    /// f8 T4.3: LRU cache over `/api/tokenize` counts; clones share it.
    token_cache: std::sync::Arc<std::sync::Mutex<TokenCache>>,
}

impl OllamaClient {
    /// Create a client pointing at the given base URL.
    ///
    /// Timeout comes from `KITTY_OLLAMA_TIMEOUT_SECS` when set to a valid
    /// `u64`, else [`DEFAULT_OLLAMA_TIMEOUT_SECS`] (300s). Context window
    /// comes from `KITTY_NUM_CTX` when set to a valid `u32`, else
    /// [`DEFAULT_NUM_CTX`] (32768).
    pub fn new(base_url: &str) -> Self {
        let raw = std::env::var(ENV_TIMEOUT_SECS).ok();
        let timeout_secs = parse_timeout_secs(raw.as_deref(), DEFAULT_OLLAMA_TIMEOUT_SECS);
        let num_ctx = parse_num_ctx(std::env::var(ENV_NUM_CTX).ok().as_deref());
        let timeout = Duration::from_secs(timeout_secs);
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .expect("reqwest::Client::builder should not fail");
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            http,
            timeout,
            num_ctx,
            token_cache: std::sync::Arc::new(std::sync::Mutex::new(TokenCache::default())),
        }
    }

    /// Override the resolved context window explicitly (F2). Useful for tests
    /// and for callers that resolve configuration themselves; keeps the
    /// once-per-client guarantee without touching process env.
    #[must_use]
    pub fn with_num_ctx(mut self, num_ctx: u32) -> Self {
        self.num_ctx = if num_ctx == 0 {
            DEFAULT_NUM_CTX
        } else {
            num_ctx
        };
        self
    }

    /// The fixed per-session context window this client requests.
    pub fn num_ctx(&self) -> u32 {
        self.num_ctx
    }

    /// Request timeout actually configured (default 300s, env-overridable).
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Convenience: client pointing at `http://localhost:11434`.
    pub fn default_url() -> Self {
        Self::new("http://localhost:11434")
    }

    /// Send a chat completion request (non-streaming).
    ///
    /// E3: when [`should_disable_thinking`] is true for `model`, the request
    /// carries `"think": false` so qwen-family models skip reasoning.
    ///
    /// F2: the payload always carries `options.num_ctx` (fixed per-session
    /// value) and `"truncate": false`; a 400 context-overflow response maps
    /// to [`RuntimeError::ContextOverflow`]. When `prompt_eval_count`
    /// reaches 95% of the window a warning is logged.
    pub async fn chat(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        options: Option<RequestOptions>,
    ) -> Result<ChatResponse> {
        let num_ctx = self.num_ctx;
        let request = self.build_chat_request(model, messages, tools, options);

        let url = format!("{}/api/chat", self.base_url);
        debug!(url = %url, model = %model, num_ctx, "sending chat request");

        let resp = self.http.post(&url).json(&request).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            warn!(status = %status, body = %body, "ollama chat returned error");
            return Err(classify_api_error(status.as_u16(), &body, num_ctx));
        }

        let response: ChatResponse = resp.json().await?;
        if let Some(count) = response.prompt_eval_count {
            if prompt_near_capacity(count, num_ctx) {
                // G2 finding 5: with truncate:false, ≥95% means the NEXT
                // oversized request will fail audibly — recommend action now.
                warn!(
                    prompt_eval_count = count,
                    num_ctx,
                    "prompt nearly filled the context window (>=95%); with truncate:false the \
                     NEXT oversized request will fail with ContextOverflow — compact now or \
                     raise KITTY_NUM_CTX"
                );
            }
        }
        Ok(response)
    }

    /// Pure payload builder shared by [`OllamaClient::chat`] and tests:
    /// merges the fixed per-session `num_ctx` into the caller's options
    /// (caller-explicit values win), forces `"truncate": false`, and applies
    /// the thinking-disable heuristics (E3 env/model-id + F2 caller override).
    #[must_use]
    pub fn build_chat_request(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        options: Option<RequestOptions>,
    ) -> ChatRequest {
        let mut opts = options.unwrap_or_default();
        // G2-FIX4: guard session-fixing against caller injection — a caller
        // that set its own num_ctx would alternate values across requests and
        // reload the model server-side. Debug builds trap this; release keeps
        // the caller-explicit value (documented on RequestOptions::num_ctx).
        let caller_num_ctx = opts.num_ctx;
        // Fixed per-session window: never alternate values across requests.
        opts.num_ctx
            .get_or_insert(i32::try_from(self.num_ctx).unwrap_or(i32::MAX));
        debug_assert!(
            caller_num_ctx.is_none() || caller_num_ctx == opts.num_ctx,
            "callers must not inject their own num_ctx: the session-fixed \
             context window is harness-owned (set KITTY_NUM_CTX instead)"
        );
        let think = match opts.think {
            Some(false) => Some(false),
            _ => {
                if should_disable_thinking(model) {
                    Some(false)
                } else {
                    None
                }
            }
        };
        ChatRequest {
            model: model.to_string(),
            messages,
            tools,
            stream: Some(false),
            options: Some(opts),
            think,
            truncate: Some(false),
        }
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

    /// Extract the model's native context length from an `/api/show` payload.
    ///
    /// Pure helper behind [`OllamaClient::show_model`] (f8 T2.1).
    /// `model_info` keys are family-prefixed (`llama.context_length`,
    /// `qwen2.context_length`, `qwen3.context_length`, …), so ANY key ending
    /// in `.context_length` is accepted — a literal `llama.` lookup would be
    /// dead for qwen-family models. When several family keys appear, the
    /// smallest value wins. Falls back to a `num_ctx <N>` entry in the
    /// `parameters` string. Returns `None` when neither source yields a
    /// positive number.
    pub fn parse_native_context_length(show_payload: &serde_json::Value) -> Option<u32> {
        if let Some(info) = show_payload.get("model_info").and_then(|v| v.as_object()) {
            let mut best: Option<u32> = None;
            for (key, val) in info {
                if !key.ends_with(".context_length") {
                    continue;
                }
                // as_u64 filters negatives, floats and non-numeric types.
                if let Some(n) = val.as_u64() {
                    if n == 0 || n > u64::from(u32::MAX) {
                        continue;
                    }
                    let n = n as u32;
                    best = Some(best.map_or(n, |b| b.min(n)));
                }
            }
            if best.is_some() {
                return best;
            }
        }
        // Fallback: `parameters` is a plain "key value\n" string.
        let params = show_payload.get("parameters")?.as_str()?;
        let mut tokens = params.split_whitespace();
        while let Some(tok) = tokens.next() {
            if tok.eq_ignore_ascii_case("num_ctx") {
                return tokens.next()?.parse::<u32>().ok().filter(|n| *n > 0);
            }
        }
        None
    }

    /// Count tokens of `prompt` using the model's own tokenizer via Ollama's
    /// `/api/tokenize` endpoint. This is the *real* token count for the model,
    /// far more accurate than a character/word heuristic and required for
    /// correct context budgeting on the actual model being run.
    pub async fn count_tokens(&self, model: &str, prompt: &str) -> Result<usize> {
        // f8 T4.3: identical (model, prompt) pairs repeat heavily across
        // turns (system prompt, knowledge/episode injections); serve them
        // from the LRU without touching the network. Errors are NOT cached:
        // miss → compute → insert.
        let key = token_cache_key(model, prompt);
        if let Some(hit) = self.token_cache.lock().expect("token cache").get(&key) {
            debug!(model = %model, cached = true, "token count served from LRU");
            return Ok(hit);
        }
        let tokens = self.count_tokens_remote(model, prompt).await?;
        self.token_cache
            .lock()
            .expect("token cache")
            .insert(key, tokens);
        Ok(tokens)
    }

    /// Uncached `/api/tokenize` round-trip behind [`OllamaClient::count_tokens`].
    async fn count_tokens_remote(&self, model: &str, prompt: &str) -> Result<usize> {
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

    fn chat_with_options<'a>(
        &'a self,
        model: &'a str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        options: Option<RequestOptions>,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>> {
        Box::pin(async move { OllamaClient::chat(self, model, messages, tools, options).await })
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

    /// f8 AD4 rider: forward options on the boxed path too. Without this the
    /// blanket impl inherited the DEFAULT chat_with_options (delegate to
    /// plain `chat`), silently dropping RequestOptions — num_predict caps,
    /// sampling overrides and thinking flags never reached ANY boxed client.
    fn chat_with_options<'a>(
        &'a self,
        model: &'a str,
        messages: Vec<Message>,
        tools: Option<Vec<ToolDefinition>>,
        options: Option<RequestOptions>,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>> {
        (**self).chat_with_options(model, messages, tools, options)
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
            think: None,
            truncate: Some(false),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert!(!json.as_object().unwrap().contains_key("tools"));
        assert!(!json.as_object().unwrap().contains_key("options"));
        assert!(!json.as_object().unwrap().contains_key("think"));
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
            prompt_eval_count: None,
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

    /// Serve `n` canned /api/tokenize responses on an ephemeral port while
    /// counting accepted connections. Returns (client, port, counter).
    fn spawn_tokenize_server(
        n: usize,
    ) -> (
        OllamaClient,
        u16,
        std::sync::Arc<std::sync::atomic::AtomicUsize>,
    ) {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
        let port = listener.local_addr().unwrap().port();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_thread = Arc::clone(&counter);
        std::thread::spawn(move || {
            for stream in listener.incoming().take(n) {
                let Ok(mut stream) = stream else { break };
                counter_thread.fetch_add(1, Ordering::SeqCst);
                // Drain the full request (headers + Content-Length body).
                let mut data = Vec::new();
                let mut buf = [0u8; 4096];
                loop {
                    let read = match stream.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(read) => read,
                    };
                    data.extend_from_slice(&buf[..read]);
                    let Ok(text) = std::str::from_utf8(&data) else {
                        continue;
                    };
                    let Some(header_end) = text.find("\r\n\r\n") else {
                        continue;
                    };
                    let content_length = text
                        .split("\r\n")
                        .find_map(|line| {
                            line.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .and_then(|v| v.trim().parse::<usize>().ok())
                        })
                        .unwrap_or(0);
                    if data.len() >= header_end + 4 + content_length {
                        break;
                    }
                }
                let body = "{\"tokens\":[1,2,3,4,5]}";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        let client = OllamaClient::new(&format!("http://127.0.0.1:{port}"));
        (client, port, counter)
    }

    #[tokio::test]
    async fn token_cache_serves_repeat_without_second_http_call() {
        // f8 T4.3 AC: second identical count_tokens performs ZERO additional
        // /api/tokenize round-trips; distinct content still computes.
        let (client, _port, counter) = spawn_tokenize_server(4);
        let first = client.count_tokens("m", "hello world").await.unwrap();
        let second = client.count_tokens("m", "hello world").await.unwrap();
        assert_eq!(first, 5);
        assert_eq!(second, 5);
        assert_eq!(
            counter.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "repeat must be served from the LRU cache"
        );

        // Distinct prompt → one more real call.
        let _ = client.count_tokens("m", "different").await.unwrap();
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 2);

        // Same prompt, DIFFERENT model → separate cache entry, real call.
        let _ = client.count_tokens("other", "hello world").await.unwrap();
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 3);

        // Warm again across all three — no further calls.
        let _ = client.count_tokens("m", "hello world").await.unwrap();
        let _ = client.count_tokens("m", "different").await.unwrap();
        let _ = client.count_tokens("other", "hello world").await.unwrap();
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[test]
    fn token_cache_lru_eviction_bounded() {
        // f8 T4.3 AC: >capacity inserts evict oldest without panicking.
        let mut cache = TokenCache::default();
        for i in 0..400usize {
            cache.insert(("m".to_string(), i as u64), i);
        }
        assert_eq!(cache.order.len(), TOKEN_CACHE_CAPACITY);
        assert_eq!(cache.map.len(), TOKEN_CACHE_CAPACITY);
        // Oldest entries are gone; newest are present.
        assert_eq!(cache.get(&("m".to_string(), 0)), None);
        assert_eq!(cache.get(&("m".to_string(), 399)), Some(399));
    }

    #[test]
    fn native_context_length_llama_style_payload() {
        // f8 T2.1: classic unprefixed-family payload.
        let payload = serde_json::json!({
            "model_info": {
                "llama.context_length": 131072,
                "general.architecture": "llama"
            },
            "parameters": "stop <|im_end|>\nnum_ctx 4096"
        });
        assert_eq!(
            OllamaClient::parse_native_context_length(&payload),
            Some(131072),
            "model_info wins over the parameters fallback"
        );
    }

    #[test]
    fn native_context_length_qwen_style_prefixed_key() {
        // f8 T2.1 (Oracle-corrected): family-prefixed keys are the norm for
        // qwen models — a literal `llama.` lookup would be dead here.
        let payload = serde_json::json!({
            "model_info": {
                "qwen3.context_length": 40960,
                "qwen3.embedding_length": 4096,
                "general.architecture": "qwen3"
            }
        });
        assert_eq!(
            OllamaClient::parse_native_context_length(&payload),
            Some(40960)
        );

        // Several family keys present → smallest numeric value wins.
        let multi = serde_json::json!({
            "model_info": {
                "qwen2.context_length": 32768,
                "qwen3.context_length": 131072
            }
        });
        assert_eq!(
            OllamaClient::parse_native_context_length(&multi),
            Some(32768)
        );
    }

    #[test]
    fn native_context_length_empty_or_weird_types_yield_none() {
        // Empty object / missing model_info.
        assert_eq!(
            OllamaClient::parse_native_context_length(&serde_json::json!({})),
            None
        );
        // Wrong value types under a matching key.
        let weird = serde_json::json!({
            "model_info": {
                "llama.context_length": "very long",
                "qwen3.context_length": null,
                "qwen2.context_length": -5,
                "qwen2.context_length_f": 1.5
            }
        });
        assert_eq!(OllamaClient::parse_native_context_length(&weird), None);
        // Zero and absurdly large values are rejected too.
        let zero = serde_json::json!({"model_info": {"llama.context_length": 0}});
        assert_eq!(OllamaClient::parse_native_context_length(&zero), None);

        // Fallback path: `parameters` carries num_ctx when model_info doesn't.
        let params_only = serde_json::json!({
            "parameters": "temperature 0.7\nnum_ctx 8192\nstop <s>"
        });
        assert_eq!(
            OllamaClient::parse_native_context_length(&params_only),
            Some(8192)
        );
        // Neither source usable → None.
        let junk = serde_json::json!({"parameters": "temperature 0.7", "model_info": {}});
        assert_eq!(OllamaClient::parse_native_context_length(&junk), None);
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

    #[test]
    fn ollama_client_default_timeout_is_300s() {
        let client = OllamaClient::new("http://localhost:11434");
        assert_eq!(client.timeout(), Duration::from_secs(300));
    }

    #[test]
    fn parse_timeout_secs_missing_or_invalid_uses_default() {
        assert_eq!(parse_timeout_secs(None, 300), 300);
        assert_eq!(parse_timeout_secs(None, 120), 120);
        assert_eq!(parse_timeout_secs(Some(""), 300), 300);
        assert_eq!(parse_timeout_secs(Some("   "), 120), 120);
        assert_eq!(parse_timeout_secs(Some("abc"), 300), 300);
        assert_eq!(parse_timeout_secs(Some("-5"), 300), 300);
        assert_eq!(parse_timeout_secs(Some("12.5"), 120), 120);
    }

    #[test]
    fn parse_timeout_secs_parses_valid_override() {
        assert_eq!(parse_timeout_secs(Some("900"), 300), 900);
        assert_eq!(parse_timeout_secs(Some("42"), 120), 42);
        assert_eq!(parse_timeout_secs(Some(" 600 "), 300), 600);
    }

    #[test]
    fn should_disable_thinking_pure_predicate() {
        // Default: thinking stays on
        assert!(!should_disable_thinking_with(None, "qwen3:8b"));
        assert!(!should_disable_thinking_with(Some("0"), "qwen3:8b"));
        // KITTY_NO_THINK=1 disables regardless of model id
        assert!(should_disable_thinking_with(Some("1"), "ornith:9b"));
        assert!(should_disable_thinking_with(Some("1"), ""));
        // Model id containing "think" (case-insensitive) disables
        assert!(should_disable_thinking_with(None, "Qwen3-Thinking"));
        assert!(should_disable_thinking_with(None, "qwen3:8b-think"));
        // Other values than "1" don't count as enabled flag
        assert!(!should_disable_thinking_with(Some("yes"), "llama3"));
    }

    #[test]
    fn chat_request_serializes_think_false_when_set() {
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![],
            tools: None,
            stream: None,
            options: None,
            think: Some(false),
            truncate: Some(false),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["think"], serde_json::json!(false));
    }

    #[test]
    fn dynamic_max_output_caps_and_floors() {
        // 16K ctx with ~12K input → 16384 − 12288 − 1024 = 3072
        assert_eq!(dynamic_max_output(16384, 12288), 3072);
        // Nearly-full context floors at 256 instead of requesting an
        // impossible output budget
        assert_eq!(dynamic_max_output(16384, 16000), MIN_OUTPUT_TOKENS);
        // Input larger than the context saturates to the floor too
        assert_eq!(dynamic_max_output(4096, 9000), MIN_OUTPUT_TOKENS);
        // Zero-input small context: 4096 − 0 − 1024 = 3072
        assert_eq!(dynamic_max_output(4096, 0), 3072);
    }

    #[test]
    fn estimate_input_tokens_conservative_chars_per_four() {
        // 8 chars → 2 tokens
        assert_eq!(estimate_input_tokens(&[Message::user("12345678")]), 2);
        // Empty conversation → 0
        let empty: Vec<Message> = vec![];
        assert_eq!(estimate_input_tokens(&empty), 0);
        // Tool-call arguments are counted (name + serialized args)
        let m = Message {
            role: "assistant".into(),
            content: None,
            thinking: None,
            tool_calls: Some(vec![ToolCall {
                id: None,
                function: ToolCallFunction {
                    name: "Read".into(),
                    arguments: serde_json::json!({"path": "abcd"}),
                },
            }]),
        };
        // args JSON is {"path":"abcd"} = 15 chars + name 4 = 19 → ceil(19/4)=5
        assert_eq!(estimate_input_tokens(&[m]), 5);
    }

    // ── F2: explicit context window + truncate:false ──────────────────

    #[test]
    fn parse_num_ctx_good_and_bad_inputs() {
        // Good inputs parse.
        assert_eq!(parse_num_ctx(Some("16384")), 16384);
        assert_eq!(parse_num_ctx(Some("4096")), 4096);
        assert_eq!(parse_num_ctx(Some(" 8192 ")), 8192);
        // Bad inputs fall back to the default (never zero, never panic).
        assert_eq!(parse_num_ctx(None), DEFAULT_NUM_CTX);
        assert_eq!(parse_num_ctx(Some("")), DEFAULT_NUM_CTX);
        assert_eq!(parse_num_ctx(Some("   ")), DEFAULT_NUM_CTX);
        assert_eq!(parse_num_ctx(Some("abc")), DEFAULT_NUM_CTX);
        assert_eq!(parse_num_ctx(Some("-5")), DEFAULT_NUM_CTX);
        assert_eq!(parse_num_ctx(Some("12.5")), DEFAULT_NUM_CTX);
        assert_eq!(parse_num_ctx(Some("0")), DEFAULT_NUM_CTX);
    }

    #[test]
    fn chat_payload_carries_fixed_num_ctx_inside_options_and_truncate_false() {
        let client = OllamaClient::new("http://localhost:11434").with_num_ctx(8192);
        let messages = vec![Message::user("hi")];

        let req1 = client.build_chat_request("m", messages.clone(), None, None);
        let json1 = serde_json::to_value(&req1).unwrap();
        // num_ctx must live INSIDE options, never top-level.
        let top = json1.as_object().unwrap();
        assert!(
            !top.contains_key("num_ctx"),
            "num_ctx must not be top-level"
        );
        assert_eq!(json1["options"]["num_ctx"], serde_json::json!(8192));
        assert_eq!(json1["truncate"], serde_json::json!(false));

        // Second consecutive build must carry the IDENTICAL fixed value
        // (alternating num_ctx across requests reloads the model).
        let req2 = client.build_chat_request("m", messages, None, None);
        let json2 = serde_json::to_value(&req2).unwrap();
        assert_eq!(json1["options"]["num_ctx"], json2["options"]["num_ctx"]);
        assert_eq!(json1["truncate"], json2["truncate"]);
    }

    #[test]
    fn chat_payload_merges_into_existing_caller_options_without_clobbering() {
        let client = OllamaClient::new("http://localhost:11434").with_num_ctx(4096);
        let opts = RequestOptions {
            temperature: Some(0.3),
            ..Default::default()
        };
        let json =
            serde_json::to_value(client.build_chat_request("m", vec![], None, Some(opts))).unwrap();
        assert_eq!(json["options"]["num_ctx"], serde_json::json!(4096));
        let temp = json["options"]["temperature"].as_f64().unwrap();
        assert!((temp - f64::from(0.3_f32)).abs() < 1e-9);
        assert_eq!(json["truncate"], serde_json::json!(false));
    }

    #[test]
    fn is_context_overflow_body_matches_overflow_only() {
        // Good: recognized overflow shapes.
        assert!(is_context_overflow_body(
            r#"{"error":{"message":"input length exceeds context size","type":"exceed_context_size_error"}}"#
        ));
        assert!(is_context_overflow_body(
            r#"{"error":"prompt exceeds context size (25000 > 4096)"}"#
        ));
        // Bad: unrelated 400/500 bodies must NOT match.
        assert!(!is_context_overflow_body(
            r#"{"error":"model 'x' not found"}"#
        ));
        assert!(!is_context_overflow_body(
            r#"{"error":"grammar and tools are not supported together"}"#
        ));
        assert!(!is_context_overflow_body(""));
    }

    #[test]
    fn classify_api_error_routes_overflow_vs_plain_api_errors() {
        let overflow_body =
            r#"{"error":"exceed_context_size_error: input length 33000 > context 32768"}"#;
        let err = classify_api_error(400, overflow_body, 32768);
        match &err {
            RuntimeError::ContextOverflow { requested, message } => {
                assert_eq!(*requested, 32768);
                assert!(message.contains("33000"), "server body preserved verbatim");
            }
            other => panic!("expected ContextOverflow, got {other:?}"),
        }
        // Actionable message: states requested vs available + suggests KITTY_NUM_CTX.
        let display = err.to_string();
        assert!(
            display.contains("KITTY_NUM_CTX"),
            "actionable hint missing: {display}"
        );
        assert!(display.contains("32768"));

        // Bad: non-400 or unrelated bodies stay plain Api errors.
        assert!(matches!(
            classify_api_error(500, overflow_body, 32768),
            RuntimeError::Api { status: 500, .. }
        ));
        assert!(matches!(
            classify_api_error(400, r#"{"error":"model not found"}"#, 32768),
            RuntimeError::Api { status: 400, .. }
        ));
    }

    #[test]
    fn prompt_near_capacity_threshold_is_95_percent() {
        // Good: at/above 95% warns; below does not.
        assert!(prompt_near_capacity(31_200, 32_768)); // 95.2%
        assert!(prompt_near_capacity(32_768, 32_768));
        assert!(!prompt_near_capacity(30_000, 32_768));
        assert!(!prompt_near_capacity(100, 4096));
        // Bad inputs never warn.
        assert!(!prompt_near_capacity(-1, 32_768));
        assert!(!prompt_near_capacity(10_000, 0));
    }

    #[test]
    fn kv_cache_preflight_warning_good_and_bad_inputs() {
        let msg = kv_cache_preflight_warning(32768, None).expect("large ctx unacked must warn");
        assert!(msg.contains("OLLAMA_KV_CACHE_TYPE=q8_0"));
        assert!(msg.contains(ENV_KV_QUANT_ACK));
        // Acknowledged quant types silence the warning.
        assert!(kv_cache_preflight_warning(32768, Some("q8_0")).is_none());
        assert!(kv_cache_preflight_warning(32768, Some(" q4_0 ")).is_none());
        // Small windows never warn.
        assert!(kv_cache_preflight_warning(16384, None).is_none());
        assert!(kv_cache_preflight_warning(4096, None).is_none());
        // Unknown ack values do NOT count as acknowledgement.
        assert!(kv_cache_preflight_warning(65536, Some("yes")).is_some());
        assert!(kv_cache_preflight_warning(65536, Some("")).is_some());
    }

    #[test]
    fn with_num_ctx_rejects_zero_and_reports_resolved_value() {
        let client = OllamaClient::new("http://localhost:11434");
        assert_eq!(client.num_ctx(), DEFAULT_NUM_CTX);
        let custom = OllamaClient::new("http://localhost:11434")
            .with_num_ctx(16384)
            .with_num_ctx(0);
        assert_eq!(custom.num_ctx(), DEFAULT_NUM_CTX);
        assert_eq!(
            OllamaClient::new("http://localhost:11434")
                .with_num_ctx(2048)
                .num_ctx(),
            2048
        );
    }

    #[test]
    fn caller_think_false_override_forces_think_field_off() {
        let client = OllamaClient::new("http://localhost:11434");
        // F2 profile Off: caller-side think override reaches the wire even
        // for models whose id/env would keep thinking on.
        let opts = RequestOptions {
            think: Some(false),
            ..Default::default()
        };
        let json =
            serde_json::to_value(client.build_chat_request("ornith:9b", vec![], None, Some(opts)))
                .unwrap();
        assert_eq!(json["think"], serde_json::json!(false));
        // Without the override, ornith keeps thinking on (no think field).
        let json = serde_json::to_value(client.build_chat_request("ornith:9b", vec![], None, None))
            .unwrap();
        assert!(json.as_object().unwrap().get("think").is_none());
    }
}

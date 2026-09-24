// kitty-infer serving surface: OpenAI-compatible shapes over the core engine.
// Single-resident registry, SSE framing, friendly errors, cached GPU snapshot.
// No auth by default; see api::check_auth for the fail-closed rule when enabled.

pub mod api;
pub mod config;
pub mod errors;
pub mod health;
pub mod infer_policy;
pub mod model_manager;
pub mod sse;

pub use api::{check_auth, AdminTasks};
pub use config::{
    CorsConfig, RateLimitConfig, RateLimiter, ServingConfig, DEFAULT_BIND, DEFAULT_PORT, ENV_BIND,
    ENV_BURST, ENV_CORS, ENV_PORT, ENV_REQUIRE_TOKEN, ENV_RPS, ENV_TOKEN,
};
pub use errors::{ErrorBody, KittyError, RequestLog};
pub use health::{GpuDeviceStatus, GpuSnapshot};
pub use infer_policy::{
    extra_f32_prefixes, needs_external_tokenizer, resolve_dtype, resolve_tokenizer_json,
    sidecar_candidate, F32_OVERRIDE_PREFIXES,
};
pub use model_manager::{ModelRegistry, ModelSource, ModelSpec, Resident, global_registry};
pub use sse::{chunk_to_frame, done_frame, final_frame};

/// VRAM pre-flight for embedders (re-exported so out-of-tree clients share
/// the same gate as the daemon path).
pub use crate::engine::gguf_loader::{file_bytes, resolve_ctx_len, vram_precheck};

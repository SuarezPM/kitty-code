# engine/mistralrs-core/src/serving/

## Responsibility
Serving surface for `kitty run`: OpenAI-compatible request/response shapes over the core engine, single-resident GGUF registry, SSE framing, friendly `{ error, hint }` errors, cached GPU snapshot, and embedded-inference policy. Pure types + small handler helpers; a thin binary wires these to HTTP.

## Design
- Single-resident registry (`model_manager.rs`): `ModelRegistry` behind one `tokio::sync::Mutex`, process-wide via `OnceLock` (`global_registry`). Exactly one GGUF resident; `load_with_retry` uses evict-before-load + OOM retry (gpu_fraction 0.9 → 0.6, shared `is_oom_message` predicate) with single-flight reuse for same-spec waiters. `preload_with_retry` (Fase 2 warm-resident) loads then runs a caller-supplied liveness probe lock-free; probe failure evicts and bails loudly — never measure over a dead CUDA context. `ModelSpec::auto` parses `hf:<repo>/<file>` | `ollama:<name>` | `local:<path.gguf>` | bare `.gguf`.
- API shapes (`api.rs`): `ChatRequest` (+`validate`), `ChatMessage`/`ChatToolCall`, `ModelsResponse` (`resident` flag), admin `AdminLoadRequest`/`AdminStatusResponse`/`AdminCancelResponse`, `GpuStatusResponse`/`GpuReclaimResponse`, `HealthResponse` (+`health()` helper), `MetricsResponse`, `LogsResponse`. `AdminTasks` tracks in-flight loads with real `JoinHandle::abort` cancellation. `check_auth` is fail-closed when `KITTY_API_TOKEN` is set.
- Errors (`errors.rs`): `KittyError` (`Oom` | `KvExhausted` | `EngineCore` | `NotFound` | `Validation`) → `ErrorBody { error, hint }`; `RequestLog` bounded ring.
- Health (`health.rs`): `GpuSnapshot`/`GpuDeviceStatus` cached 1s (`OnceLock<Mutex>`); `nvidia-smi` → `rocm-smi` → `cpu` fallback; argv-based bounded commands; `first_device_free_bytes`.
- SSE (`sse.rs`): `data: <json>\n\n` events, `done_frame` = `data: [DONE]`, `chunk_to_frame`, `final_frame` (usage + `finish_reason: stop`).
- Policy (`infer_policy.rs`, F2): dtype default F16 (`Auto`) + `F32_OVERRIDE_PREFIXES` (`qwen2.5-coder*`, isolated F16-CUDA NaN on sm_75) + env extras; MiniCPM5/K2-Horizon external `tokenizer.json` sidecar-or-honest-error (`needs_external_tokenizer`, `resolve_tokenizer_json`, `sidecar_candidate`).

## Flow
1. Admin `POST /admin/load` → `ModelSpec::auto(spec)` → `global_registry().load_with_retry` (fast path if resident; else evict → build → store, OOM retry) → `Resident { spec, gpu_fraction, loaded_at_secs }`.
2. Startup/bench warm: `preload_with_retry` once → first turn hits the fast path (no rebuild).
3. Chat turn → `ChatRequest::validate` → engine inference → frames via `chunk_to_frame` … `final_frame` … `done_frame`.
4. Failure → `KittyError::classify` → `{ error, hint }` body.

## Integration
- Consumed by: `crates/kitty-runtime` embedded client (`KittyInferClient::ensure_loaded`), future thin serving binary, `mistralrs::serving` re-export (`engine/mistralrs/src/lib.rs:301`).
- Depends on: `crate::engine::gguf_loader` (`file_bytes`, `resolve_ctx_len`, `vram_precheck`, re-exported in `mod.rs`), engine `Model`, `tracing`, `tokio`, `serde`.

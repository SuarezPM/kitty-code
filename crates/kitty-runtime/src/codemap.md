# crates/kitty-runtime/src/

## Responsibility

One file per backend or shared piece behind the `ChatClient` trait. Each
client owns exactly one transport (Ollama HTTP, cloud OpenAI-compat, llama.cpp
in-process, embedded GGUF) plus file-local pure helpers; cross-backend
concerns (wire types, errors, grammar filter) live in their own modules so no
client imports another.

## Design

- `client.rs`: `ChatClient` trait (`chat` / `chat_with_options` /
  `count_tokens` / `health_check`, object-safe via `Pin<Box<dyn Future>>`) +
  `OllamaClient` (typed `reqwest`, session-fixed `num_ctx`,
  `KV preflight`, `TokenCache`, `MockClient`) + wire types (`ChatRequest`,
  `ChatResponse`, `Message` with `tool_call_id`, `ToolCall`,
  `ToolCallFunction`, `ToolDefinition`, `RequestOptions`).
- `cloud.rs`: `CloudClient` (vendor OpenAI-compat, `SecretString` key,
  cost metering) + `MockCloud`.
- `llama.rs`: `LlamaCppClient` (in-process GGUF vocab probe + Ollama
  fallback, `grammar_lazy` sampler); `llama`/`llama-vulkan` features.
- `kitty_infer.rs`: `KittyInferConfig::from_env` (GGUF path, ctx, tokenizer
  sidecar, GPU-layer spill knob, F32 prefixes) + `KittyInferClient`
  (single-resident registry, `ensure_loaded`, `to_request` with
  `tool_call_id` threading, env-mapped `SamplerOpts`, heuristic
  `count_tokens`, file-level `health_check`) + pure mapping helpers
  (`backend_selected`, `parse_gpu_layers`, `validate_gpu_layers`,
  `spill_estimate_bytes`, GGUF header readers, `tool_format`,
  `sampler_env_override`, `tool_fallback_id`, `thinking_enabled`) +
  `unavailable_error()` stub when the feature is off. Dual-declared in
  `lib.rs` so both feature configurations compile the same module.
- `openai_compat.rs`: `OpenAiCompatClient` (keyless OpenAI-compat, `tool_call_id`
  positional pairing with `kitty_{n}` fallback, `_raw` args salvage).
- `grammar.rs`: `GrammarClient` post-hoc strict filter + `GrammarConfig::to_gbnf`.
- `tools.rs`: shared tool wire types (`ToolDefinition`, `Message` ctors).
- `error.rs`: `RuntimeError` taxonomy + `ErrorClass` + retry predicates.
- `lib.rs`: re-exports + feature-gated client exports.
- `bin/`: `gbnf_probe` measurement binary (`llama`-gated).

## Flow

`ChatClient::chat_with_options` per backend: Ollama builds the fixed-`num_ctx`
payload and POSTs; cloud adds Bearer; llama probes vocab then delegates;
kitty-infer ensures the resident, translates history (roles, tool calls with
ids, tools) into an engine `RequestBuilder`, and maps usage counters back.
Errors classify into `RuntimeError` with actionable hints at the client
boundary.

## Integration

- Consumed by: `kitty-harness` (via `ChatClient` trait objects only) and
  `kitty-cli` (`OllamaClient` construction, `KittyInferClient::from_env`,
  `unavailable_error`, `ENV_*` constants).
- External deps: `reqwest`, `tokio`, `serde`/`serde_json`, `thiserror`,
  `tracing`, `zeroize`, optional `llama-cpp-2`; optional internal
  `engine/mistralrs` (feature `kitty-infer[-cuda]` only).

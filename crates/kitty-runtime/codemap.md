# crates/kitty-runtime/

## Responsibility
LLM client crate: typed backends behind one trait plus shared tool wire-types.
Owns `ChatClient` abstraction, Ollama / Cloud / llama.cpp clients, GBNF grammar
spike, `RuntimeError` taxonomy, and the `gbnf-probe` measurement binary.

## Design
- `ChatClient` (client.rs): `chat` + `chat_with_options` + `count_tokens` +
  `health_check` via `Pin<Box<dyn Future>>` object-safe. `Box<T: ChatClient>`
  blanket impl forwards `chat_with_options` (f8 AD4 rider). Sole seam for harness.
- `OllamaClient` (client.rs): typed `reqwest` POST `/api/chat`, `/api/tokenize`,
  `/api/tags`, `/api/show`. `build_chat_request` merges session-fixed `num_ctx`
  into `options` (never top-level) + `truncate: false` + `think: false` heuristic
  (`should_disable_thinking`, `KITTY_NO_THINK`). `TokenCache` LRU-256 over
  (model, hashed prompt); `parse_num_ctx` / `parse_native_context_length` /
  `classify_api_error` / `dynamic_max_output` / `estimate_input_tokens` pure helpers.
  `MockClient` (`Arc<Mutex<VecDeque<ChatResponse>>>`, clone shares queue) for tests.
- `CloudClient` (cloud.rs): vendor-agnostic OpenAI-compat POST `{base}/chat/completions`
  (`KITTY_CLOUD_API_KEY` + `KITTY_CLOUD_BASE_URL`, 120s default via shared
  `KITTY_OLLAMA_TIMEOUT_SECS`). Key held in `SecretString` (zeroize-on-drop,
  `Debug` redacted, `auth_header` Bearer at transport only). Cost metering
  `tracing::info!`; `CloudMessage::reasoning_content` normalized into
  `Message::thinking`. `is_secret_env_name` / `sanitize_env` strip secrets from
  child env. `MockCloud` mirrors `MockClient` semantics.
- `OpenAiCompatClient` (openai_compat.rs): keyless OpenAI-compat POST
  `{base}/v1/chat/completions` for models Ollama can't serve (K2-Horizon-3.7B
  via SGLang; `KITTY_OPENAI_BASE_URL`, default `:30000`). `reasoning_content`
  ↔ `thinking` both directions, `tool_call_id` positional pairing
  (`kitty_{n}` fallback), server tool names validated identifier-shaped
  (anti prompt-injection), malformed args salvaged as `_raw`, options
  always forwarded, `extra_body.reasoning_effort` (K2 `high`;
  `think:false` drops it), chars/4 `count_tokens`, `/v1/models` health.
  11 wiremock tests. K2 serving frozen (no SGLang on box): `docs/SERVING-K2-SGLANG.md`.
- `LlamaCppClient` (llama.rs): in-process GGUF adapter (`model_path: PathBuf` +
  `fallback: OllamaClient` + `state: Arc<Mutex<Option<Inner>>>>` + `load_error`,
  `unsafe impl Send/Sync`). `try_load` vocab-only probe, missing GGUF delegates
  immediately to fallback (never blocks). `tool_schemas_to_gbnf` (`oneOf` per-tool
  `json_schema_to_grammar` + fallbacks), `build_grammar_lazy_sampler` /
  `build_lazy_sampler_with` (`grammar_lazy` + word trigger `{"name"` +
  tokenizer-derived `trigger_tokens_from_lookup` over `TRIGGER_PHRASES`).
  Behind optional `llama` feature (`llama-cpp-2`, `llama-vulkan` for GPU offload).
- `GrammarClient` spike (grammar.rs): `GrammarConfig::to_gbnf()` (GBNF for
  `{"name": enum, "arguments": object}` + JSON value rules) + `validate()` /
  `validate_value()` (`serde_json`, flat or nested `function` shape, `required`
  check). `GrammarClient<C: ChatClient>::apply_strict_filter` drops invalid
  `tool_calls` on BOTH `chat` and `chat_with_options` paths. No decode-time
  enforcement — post-hoc filter only.
- Tool wire-types (tools.rs + client.rs): `ToolDefinition` / `ToolFunction`
  (`type: function`, `parameters` JSON Schema), `Message` (`role/content/thinking/
  tool_calls`, `user/assistant/tool_result/tool_calls` ctors), `ToolCall` /
  `ToolCallFunction`, `ChatRequest` (`stream: false`, `options: RequestOptions`,
  `think`, `truncate`), `ChatResponse` (`prompt_eval_count` truncation tell,
  `eval_count/eval_duration`), `RequestOptions` (`temperature/top_p/top_k/
  num_predict`, harness-owned `num_ctx`, caller `think` override). Helpers
  `required_string` / `optional_string` / `optional_u64` / `json_params`.
- `error.rs`: `RuntimeError` (`Http/Json/Io/Api/ContextOverflow/ModelNotFound/
  ConnectionRefused/ToolExecution/InvalidArgument/UnexpectedResponse`) +
  `ErrorClass` (`Transient/RateLimit/Permanent/Overflow/Quota`) +
  `is_transient/error_class/is_retryable/is_overflow` for harness retry/compaction.
- `bin/gbnf_probe.rs` (`llama`-gated): empirical A/B harness decoding each prompt
  per arm (`Lazy` production `grammar_lazy_sampler_for` / `LazyTok` / `CanaryWord` /
  `CanaryTok` / `Full` / `Off` baseline) via `generate()` + `run_diag_tokens()`
  (`--diag-tokens`), `bench_tools()` Read/Write/Edit/Bash. No double-`accept`
  (`sample` already accepts). `lib.rs` re-exports all clients + wire types.
- `KittyInferClient` (kitty_infer.rs, feature `kitty-infer[-cuda]`): embedded
  GGUF backend behind `ChatClient`, selected by `KITTY_BACKEND=kitty-infer`
  (`backend_selected`, pure). `KittyInferConfig::from_env` reads
  `KITTY_INFER_MODEL` (.gguf path) + `KITTY_INFER_CTX` (default 4096) +
  `KITTY_INFER_TOKENIZER_DIR` (`<dir>/MiniCPM5-2B/tokenizer.json` sidecar,
  honest error when missing) + `KITTY_INFER_GPU_LAYERS` (spill knob, internal
  ≤12) + `KITTY_INFER_F32_PREFIXES`. Single-resident registry
  (`global_registry` evict-before-load + OOM retry, `loaded_map` reuse,
  `vram_precheck` fail-closed with bytes, spill estimate vs
  `SPILL_WARN_HEADROOM`): concurrent same-spec loads compile once.
  `to_request` threads harness `tool_call_id` through `add_tool_message`
  (positional `tool-{idx}` fallback, warn past the first); assistant history
  with tool calls re-emits engine `ToolCallResponse` (missing ids →
  `"call-0"`); engine ids surface as `Some(id)` on `ToolCall`.
  `chat_with_options` honors `RequestOptions` (`num_predict` clamped to 8192)
  plus env-mapped sampler (`SamplerOpts` from `KITTY_INFER_TEMP/TOP_P/TOP_K`,
  `KITTY_INFER_NO_THINK`, `KITTY_INFER_TOOL_FORMAT`,
  `KITTY_INFER_REASONING_EFFORT`). `count_tokens` is a len/4 heuristic on
  purpose (tokenizing must not trigger a 5GB load); usage counters flow into
  `eval_count`/`prompt_eval_count`. `health_check` is file-level only (GGUF
  exists + tokenizer resolvable), never loads. Without the feature the module
  still compiles with `unavailable_error()` so the CLI names the missing
  feature instead of `cfg`-branching call sites.
- Features: `kitty-infer = ["dep:mistralrs", "dep:anyhow"]`,
  `kitty-infer-cuda = ["kitty-infer", "mistralrs/cuda"]` (internal
  `engine/mistralrs` path-dep post-mono; default builds stay CPU-only).

## Flow
Harness → `ChatClient::chat_with_options(model, messages, tools, RequestOptions)`
→ `OllamaClient::build_chat_request` (inject fixed `options.num_ctx` from
`KITTY_NUM_CTX`/`DEFAULT_NUM_CTX=32768`, force `truncate: false`, apply
`think: false`) → POST `/api/chat` (`CloudClient`: `/chat/completions` + Bearer;
`LlamaCppClient`: vocab probe + `grammar_lazy` sampler exercise, then fallback
generation) → status check (`classify_api_error`: 400 + `exceed_context_size_error`
→ `ContextOverflow`; `prompt_near_capacity` 95% warn) → `ChatResponse`
(`GrammarClient::apply_strict_filter` drops schema-invalid `tool_calls`) →
harness parser normalizes downstream (`thinking` kept separate from `content`,
never fed back as visible text; `dynamic_max_output` = `ctx − estimate − 1024`,
floor 256). `KittyInferClient` arm: `ensure_loaded` (registry reuse or
evict→vram_precheck→build→`resident ready` log) → `to_request` (env sampler +
tool_format + reasoning effort) → engine `send_chat_request` → classified
`RuntimeError::KittyInfer` with hints on failure, never a silent fallback.

## Integration
- Consumed by: `kitty-harness` (agent loop injects `Box<dyn ChatClient>`,
  owns tool defs + options + compaction on `Overflow`); `gbnf-probe` binary
  exercises `llama::grammar_lazy_sampler_for/build_lazy_sampler_with/
  derive_trigger_tokens_pub/tool_schemas_to_gbnf_pub` + `GrammarConfig::validate`.
- Depends on: `reqwest` (rustls HTTP), `serde`/`serde_json`, `thiserror`,
  `tokio`, `tracing`, `zeroize`; optional `llama-cpp-2` (`llama`/`llama-vulkan`
  features); optional internal `engine/mistralrs` + `anyhow`
  (`kitty-infer[-cuda]` features, never default).

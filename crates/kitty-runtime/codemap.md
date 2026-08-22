# crates/kitty-runtime/

## Responsibility
Inference runtime layer. Typed HTTP clients for the Ollama chat API plus the
serializable tool wire-types (`ToolDefinition`/`ToolFunction`) shared with the harness.
Pluggable via `ChatClient` trait (Ollama / Cloud / llama.cpp + Grammar spike).

## Design
- `ChatClient` (client.rs: ~555L trait): `trait ChatClient: Send+Sync { chat, count_tokens, health_check }` via `Pin<Box<dyn Future>>` object-safe. Sole abstraction for Agent loop.
- `OllamaClient` (client.rs: ~555L): thin `reqwest` (rustls only) wrapper. One typed `chat()` (non-streaming POST `/api/chat`), plus `health_check`, `list_models`, `show_model`, and `count_tokens` (calls Ollama's `/api/tokenize` to return the model's real token count, heuristic fallback). 5min timeout. `MockClient` (`Arc<Mutex<VecDeque<ChatResponse>>>`, clone shares queue) for tests.
- `CloudClient` (cloud.rs: 578L, F6 W2): generic OpenAI-compatible `/v1/chat/completions` (no vendor hardcoded, `KITTY_CLOUD_API_KEY` + `KITTY_CLOUD_BASE_URL` env), timeout 120s rustls, cost metering `tracing::info!(prompt_tokens, completion_tokens, total_tokens, cost_usd)` ($2/1M in + $6/1M out placeholder), `auth_header Bearer` never logged, `Debug` redacted `<redacted>`, `MockCloud` clone shares queue. 9 tests (MockCloud, timeout, env, ChatClient bound, no hardcoded domain).
- `GrammarClient` (grammar.rs: 398L): **spike defer** (F6.4 ADR-005). `GrammarConfig::to_gbnf()` generates GBNF string for `{"name": enum, "arguments": object}` with JSON value rules; `validate()` = `serde_json` + `required`-field check. No decode-time enforcement — `ChatClient` has no grammar field; `OllamaClient::chat` does not send GBNF. `GrammarClient<C: ChatClient>` strict filtering drops invalid `tool_calls`.
- `LlamaCppClient` (llama.rs: 392L, F6 W4): in-process `llama.cpp` adapter (F6.4 GO path). `model_path: PathBuf` GGUF + `fallback: OllamaClient` + `state: Arc<Mutex<Option<Inner>>>` (backend/model `Send` via `unsafe impl`) + `load_error`. Vocab-only probe `LlamaModelParams::vocab_only(true)`, `tool_schemas_to_gbnf` per-tool `json_schema_to_grammar` + combined, `LlamaSampler::grammar_lazy` (triggers `["{\"name\""]`, avoids `grammar+tools` HTTP 400 `server-common.cpp:1289`). `try_load` non-blocking: if GGUF absent, immediately delegates to fallback (MUST NOT block). 4 tests: Send/Sync + fallback paths + no grammar+tools together. Feature `llama` optional (no dep on defer).
- Wire types (client.rs / tools.rs: ~101L): plain `serde` structs mirroring Ollama's `/api/chat` schema (`Message`, `ToolCall`, `ToolCallFunction`, `ChatRequest`, `ChatResponse`). `Message` carries optional `thinking` and `tool_calls`; `skip_serializing_if` minimal. `ToolDefinition` is the only exported artifact used by the harness.
- `error.rs` (34L): `RuntimeError` (`Http` / `Api` / `InvalidArgument` / `ConnectionRefused` / `UnexpectedResponse`).

## Flow
CLI → `ChatClient::chat(model, messages, tools)` → (OllamaClient POST `/api/chat` | CloudClient POST `/v1/chat/completions` | LlamaCppClient vocab probe + `grammar_lazy` sampler then fallback) → status check → `ChatResponse` (+ `GrammarClient` filter) → `ChatResponse`. Tool definitions flow harness → client → Ollama.

## Integration
- Consumed by: `kitty-harness` (agent loop, tool defs), `kitty-cli` (entry). `kitty-harness` owns `ChatClient` injection (`Agent::new` + `set_cloud_client` + Llama tiny path).
- Depends on: `reqwest` (rustls), `serde_json`, `tokio`. No internal crate deps.
- Note: `tree-sitter` is declared in the workspace but **unused** (dead dependency, noted in codemap).

## Tests
18 lib tests (ChatClient, MockClient, grammar validate, tool_definition, cloud/llama when wired) + cloud 9 + llama 4 = up to 31 when F6 modules `pub mod` wired. `cargo test -p kitty-runtime` green, clippy `-D warnings` clean.

## Honest status (F6.4 ADR-005 final)
- Grammar: spike `to_gbnf` + `validate` post-hoc only; real decode-time enforcement requires `llama-cpp-2 = "0.1.154"` `LlamaSampler::grammar` / `grammar_lazy` (GO when tiny validity <95% on `benches/ab_gbnf.py` — measured 60% spike vs 98% grammar_lazy, verdict GO). `llama.rs` implements the GO contract without blocking without GGUF. **AMEND 2026-08-21 (post-audit): veredicto GO REVERTIDO** — medición empírica real (probe `gbnf-probe`, GGUF qwen2.5-coder-7b-q4_k_m, 50 prompts): 94.0% baseline, delta 0.0pp; el trigger lazy `{"name"` nunca se activa bajo Qwen2.5 tokenization (no-op silencioso); decoding grammar-constrained sigue SIN MEDIR. Ver docs/ADR-GBNF-GO.md.
- Cloud: vendor-agnostic, no hardcoded domain, key never logged, `Debug` redacted.
- See `docs/STRATEGY.md` F6.4 for ADR, alternatives (llama-cpp-2 in-process preferred vs llama-server daemon fallback) and pitfall `custom grammar + tools` exclusive.

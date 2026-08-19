# crates/kitty-runtime/

## Responsibility
Inference runtime layer. Typed HTTP client for the Ollama chat API plus the
serializable tool wire-types (`ToolDefinition`/`ToolFunction`) shared with the harness.

## Design
- `OllamaClient` (client.rs): thin `reqwest` wrapper. One typed `chat()` (non-streaming),
  plus `health_check`, `list_models`, `show_model`, and `count_tokens` (calls
  Ollama's `/api/tokenize` to return the model's real token count). All errors
  normalized to `RuntimeError` (`Http` / `Api` / `InvalidArgument`).
- Wire types (client.rs / tools.rs): plain `serde` structs mirroring Ollama's
  `/api/chat` schema (`Message`, `ToolCall`, `ChatRequest`, `ChatResponse`).
  `Message` carries optional `thinking` and `tool_calls` fields; uses
  `skip_serializing_if = "Option::is_none"` to keep payloads minimal.
- `ToolDefinition` is the only exported artifact used by the harness. The generic
  `trait Tool` was removed (2026-08-18) — it was declared but never implemented.

## Flow
CLI → `OllamaClient::chat(model, messages, tools, options)` → POST `/api/chat`
→ status check → `ChatResponse`. Tool definitions flow harness → client → Ollama.

## Integration
- Consumed by: `kitty-harness` (agent loop, tool defs), `kitty-cli` (entry).
- Depends on: `reqwest`, `serde_json`, `tokio`. No internal crate deps.
- Note: `tree-sitter = "0.23"` is declared in the workspace but **unused** here
  (and everywhere) — dead dependency.

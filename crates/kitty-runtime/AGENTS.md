# kitty-runtime AGENTS.md

## OVERVIEW
Leaf crate. Runtime abstraction for LLM backends. Trait plus Ollama HTTP plus Mock plus deferred Grammar.

## WHERE TO LOOK
| Symbol | Location | Notes |
|---|---|---|
| `ChatClient` | `src/client.rs:19` | trait Send+Sync, `chat` / `count_tokens` / `health_check` returns `Pin<Box<dyn Future + Send>>` |
| `OllamaClient` | `src/client.rs:177` | reqwest with rustls, POST `/api/chat` and POST `/api/tokenize`, 5min timeout |
| `MockClient` | `src/client.rs:362` | `Arc<Mutex<VecDeque<ChatResponse>>>`, clone shares queue, pop_front per call |
| `GrammarClient` | `src/grammar.rs:6` | spike. Generates GBNF string, validates post hoc via serde_json only |
| `ToolDefinition` | `src/client.rs` | wire type for tool calling, serialized to Ollama tools array |
| `ChatMessage` / `ChatResponse` | `src/client.rs` | wire types, role, content, tool_calls, optional fields skipped |
| `OllamaConfig` | `src/client.rs` | base_url, model, timeout wiring for OllamaClient::new |

## CONVENTIONS
- Trait object safe via `Pin<Box<dyn Future>>`. All methods require Send+Sync bounds.
- OllamaClient uses `reqwest` with `rustls` only, no native tls. Timeout fixed at 300s for long generations.
- `count_tokens` calls `/api/tokenize`, falls back to heuristic estimate if endpoint absent.
- `health_check` uses cheap GET probe, no model load, called by Agent preflight.
- MockClient holds `Arc<Mutex<VecDeque>>`. Lock briefly, clone response, drop guard before await.
- GrammarClient stays isolated. Pure GBNF string generation, no ChatClient integration yet.
- Wire types use `serde` with `skip_serializing_if` for Option fields, minimal derives.
- Clone on MockClient is intentional sharing. Test scripts push VecDeque in order of expected calls.

## ANTI-PATTERNS
- NEVER send `grammar` and `tools` together in one `/api/chat` request. Server returns HTTP 400 per `server-common.cpp:1289`.
- NEVER add `llama-cpp-2` dependency now. GrammarClient stays post hoc serde_json strict filter until GO gate.
- GO gate: wire llama binding only when tiny model validity drops below 95 percent on Harbor subset.
- NEVER hold MockClient mutex across await. Lock, clone, drop, then await.
- NEVER assume `/api/tokenize` exists. Always handle fallback path.
- NEVER block OllamaClient on short timeout. Long reasoning needs full 5min window.
- NEVER mix GBNF generation with HTTP logic. Keep grammar.rs pure and dependency free.

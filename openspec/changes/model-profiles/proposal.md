## Why

Kitty Code está acoplado a defaults de ornith:9b (contexto 16K, thinking 2048, set fijo de 6 tools, canal de tool-calling nativo). Para servir el rango 8B–35B y el tier sub-2B, el comportamiento por modelo debe ser **datos, no código**. Además, la abstracción de runtime (`ChatClient`) es prerrequisito de la gramática (grammar-enforcement) y de los tests del loop con mock — hoy `Agent` tiene un `OllamaClient` concreto (agent.rs:67).

## What Changes

- Nuevo `ModelProfile` (struct): `context_window`, `max_thinking_tokens`, `max_iterations`, `tool_subset`, `use_native_tools`, `parser_strictness`, caps de Read/Bash/Grep, `keep_count` de compactación, system prompt.
- Tres presets hardcodeados: `tiny` (≤2B: 4K ctx, sin native tools, 3-4 tools, sin thinking), `medium` (8–13B: 16K, native tools, 6 tools), `large` (27–35B: 32K, todos).
- `AgentConfig` consume un perfil; todos los defaults actuales (agent.rs:47-63, context.rs:111, thinking_budget.rs:106, knowledge.rs:139-140, tools.rs:83/413/514) pasan a ser datos del perfil.
- Trait `ChatClient` (`chat`, `count_tokens`, `health_check`) en kitty-runtime; `OllamaClient` lo implementa; `Agent` recibe `Box<dyn ChatClient>`.
- Flag CLI `--profile tiny|medium|large` (default `medium`).

## Capabilities

### New Capabilities
- `model-profiles`: perfiles de modelo que parametrizan el comportamiento del harness (contexto, thinking, tools, caps, canal de tool-calling).

## Impact

- `crates/kitty-harness/src/agent.rs` (AgentConfig, inyección de perfil), `context.rs`, `thinking_budget.rs`, `knowledge.rs`, `tools.rs` (caps por perfil).
- `crates/kitty-runtime/src/` (nuevo trait `ChatClient`, `OllamaClient` lo implementa).
- `crates/kitty-cli/src/main.rs` (flag `--profile`).
- Prerrequisito de: grammar-enforcement, tests del loop con mock, bench matrix.

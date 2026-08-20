## Context

See proposal.md Why: acoplado a ornith:9b (16K ctx, thinking 2048, 6 tools fijas, native tool-calling); para 8B–35B + tier sub-2B el comportamiento debe ser datos, no código. Constraints: defaults dispersos (agent.rs:47-63, context.rs:111, thinking_budget.rs:106, knowledge.rs:139-140, tools.rs:83/413/514); prerrequisito de grammar-enforcement y bench matrix.

## Goals / Non-Goals

**Goals:**
- Struct `ModelProfile` con contexto, thinking, tool_subset, use_native_tools, caps y keep_count.
- Tres presets `tiny`/`medium`/`large` y `AgentConfig` que consume perfil.
- Trait `ChatClient` en kitty-runtime (`OllamaClient` lo implementa); `--profile` CLI.

**Non-Goals:**
- Auto-detección de modelo (selección explícita).
- Perfiles por modelo concreto (solo 3 buckets por tamaño).
- Migración de sesiones existentes (no hay estado persistido).

## Decisions

- **Struct + 3 presets hardcodeados vs config file** — presets cubren 80% sin complejidad de parsing; alternativa: TOML externo (overhead, validación extra).
- **Trait `ChatClient` en kitty-runtime** — desacopla Agent de Ollama, habilita mock y llama.cpp; alternativa: enum con match (no extensible).
- **Caps como datos del perfil** — Read/Bash/Grep/knowledge derivan del perfil; alternativa: constantes globales + overrides (inconsistente).

## Risks / Trade-offs

- [Perfil tiny demasiado restrictivo rompe tareas medianas] → Mitigación: presets conservadores + override futuro, bench matrix por perfil.
- [Cambio de Agent a `Box<dyn ChatClient>` rompe tests] → Mitigación: impl por defecto con OllamaClient, migración de tests a MockClient.
- [Drift de presets si defaults cambian] → Mitigación: single source en ModelProfile, tests que asertan valores esperados.

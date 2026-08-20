## Context

See proposal.md Why: modelos sub-2B emiten tool-calls malformadas; GBNF las hace imposibles. Requiere `ChatClient` de model-profiles (hoy `Agent` con `OllamaClient` concreto, agent.rs:67) y el set filtrado del 2-stage routing (small-model-tool-loop). Spike A/B previo validó 88%→98%.

## Goals / Non-Goals

**Goals:**
- Adapter llama.cpp que implementa `ChatClient` con gramática GBNF por schemas activos.
- Gramática generada dinámicamente desde el set filtrado (perfil + categoría).
- Paridad de loop con OllamaClient.

**Non-Goals:**
- Servidor HTTP/daemon (llama.cpp embebido, sin Node).
- Soporte de gramáticas arbitrarias fuera de tool-calls.
- Entrenamiento/fine-tuning del modelo.

## Decisions

- **llama.cpp directo (sin daemon HTTP)** — elimina hop de red y Node; alternativa: Ollama con GBNF (no expone control de gramática).
- **Generación GBNF desde schemas activos** — garantiza coherencia con routing; alternativa: gramática estática (desfasada al filtrar).
- **Implementa `ChatClient`** — reutiliza trait de model-profiles; alternativa: módulo separado con duplicación de loop.

## Risks / Trade-offs

- [GBNF demasiado estricta bloquea tool válida nueva] → Mitigación: regeneración por turno desde schemas activos, A/B validity.
- [Dependencia nativa llama.cpp complica build] → Mitigación: feature flag `grammar`, fallback sin ella.
- [Colisión con routing si gramática desactualizada] → Mitigación: single source of truth: set filtrado → gramática.

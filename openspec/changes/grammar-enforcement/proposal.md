## Why

Los modelos sub-2B producen tool-calls malformadas; la gramática (GBNF) las hace **físicamente imposibles** — es el make-or-break del tier sub-2B (spike de de-risking inmediatamente después del benchmark del piso). Requiere la abstracción `ChatClient` (model-profiles): hoy `Agent` tiene un `OllamaClient` concreto. Además, la gramática debe conocer el set de schemas filtrado por el 2-stage routing (small-model-tool-loop) o colisionan.

## What Changes

- Adapter **llama.cpp** que implementa `ChatClient` con gramática GBNF para tool-calls (sin daemon, sin Node).
- La gramática se genera a partir del set de schemas activo (perfil + categoría del routing).
- A/B de tool-call validity con/sin gramática (target: 88% → 98% según el plan original).

## Capabilities

### New Capabilities
- `grammar-enforcement`: adapter llama.cpp con gramática GBNF que restringe las tool-calls al schema activo.

## Impact

- Nuevo adapter en `kitty-runtime` (o crate nuevo) implementando `ChatClient`.
- Depende de: model-profiles (trait `ChatClient`), small-model-tool-loop (set de schemas filtrado).
- Objetivo: >95% tool-call validity en tier sub-2B.

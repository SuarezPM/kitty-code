## Why

Los modelos pequeños desperdician contexto cuando se envían los schemas completos de todas las tools en cada turno (smallcode ahorra ~800 tokens/turno con 2-stage routing), y entran en bucles de tool-calls repetidas (el dedup actual de Kitty Code es solo por nombre, quality_monitor.rs:125-152). Además, sin un done-guard, el agente puede declarar "terminado" sin satisfacer aserciones verificables.

## What Changes

- **2-stage tool routing**: un clasificador determinista (regex/keywords, zero-token) selecciona la categoría (read/write/search/run/plan) y solo se inyectan los schemas de esa categoría. **Affirmation guard**: "yes"/"ok" no re-clasifica fuera de las write tools.
- **Args-aware tool dedup**: la ventana de detección de loops considera nombre + hash de argumentos, no solo nombre; persiste entre turnos.
- **Contract/DoD done-guard**: aserciones declarativas (p.ej. "el archivo X existe y contiene Y") que el agente debe satisfacer antes de declarar completado; el harness verifica y rechaza la finalización si no se cumplen.

## Capabilities

### New Capabilities
- `tool-loop`: routing de tools en dos etapas, dedup por argumentos y done-guard de contrato para el loop de agentes.

## Impact

- `crates/kitty-harness/src/agent.rs` (routing, done-guard), `tools.rs` (schemas por categoría), `quality_monitor.rs` (dedup por args), nuevo módulo de router.
- Interactúa con grammar-enforcement: la gramática debe conocer el set de schemas filtrado.
- Interactúa con model-profiles: el routing se activa en perfiles con contexto pequeño.

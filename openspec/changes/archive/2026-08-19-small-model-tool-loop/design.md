## Context

See proposal.md Why: modelos pequeños desperdician ~800 tokens/turno enviando todos los schemas y entran en loops (dedup actual solo por nombre, quality_monitor.rs:125-152); sin done-guard declaran "terminado" sin verificación. Constraints: interactúa con model-profiles (routing por perfil) y grammar-enforcement (gramática conoce set filtrado).

## Goals / Non-Goals

**Goals:**
- 2-stage routing: clasificador determinista (regex/keywords, zero-token) → solo schemas de categoría ganadora; affirmation guard.
- Dedup por `name + hash(args)` persistente entre turnos.
- Contract/DoD done-guard con aserciones verificables y parser robusto sin drops silenciosos.

**Non-Goals:**
- Clasificador LLM (costo token).
- Replanificación automática del agente tras dedup (solo mensaje correctivo).
- Lingua franca de aserciones Turing-completa (existencia de archivo + contenido).

## Decisions

- **Clasificador regex/keywords determinista** — zero-token, ~200 tokens por turno ahorrados, estable; alternativa: LLM classifier (costo, latencia, no determinista).
- **Dedup por args hash con ventana persistente** — distingue `Read(a)` vs `Read(b)`; alternativa: solo por nombre (falsos positivos actuales).
- **Contract/DoD como aserciones declarativas verificadas por harness** — rechaza finalización si no se cumplen; alternativa: validación solo post-hoc (no guía al modelo).

## Risks / Trade-offs

- [Clasificador elige categoría errónea y oculta tool necesaria] → Mitigación: affirmation guard + fallback a todas las tools si ninguna categoría matchea.
- [Dedup demasiado sensible bloquea reintentos legítimos] → Mitigación: umbral 3 repeticiones con mismos args, no penaliza args distintos.
- [Parser correctivo entra en loop si modelo no corrige] → Mitigación: mensaje con formato esperado + límite de reintentos antes de escalar.

# Vision & Roadmap — Kitty Code

**Fecha:** 2026-08-21 · **Plan:** f6-p0-quickwins-nightbench todo 11 · **Alcance:** own-inference worth,
posición competitiva, y los tres hechos narrativos del día registrados como permanente.

---

## 1. Own inference: ¿vale la pena? — Veredicto: KEEP Ollama para v0.1

| Criterio | Ollama (hoy) | Motor propio (Dekanus Phase-0 skeleton) |
|---|---|---|
| Mantenimiento | ~0 (daemon del sistema, updates del upstream) | Alto: candle 0.11 + cudarc 0.19, kernels vendor, build ~5 GB |
| Throughput medida | ~13 s/tarea ornith:9b (bench real) | **Ninguna** — sin tok/s real medido |
| Riesgo | Bajo (fallback CloudClient ya cableado) | Alto: skeleton sin bench, API inestable entre candles |
| Diferencial | Ninguno (todos lo usan) | Potencial — solo si se ejecuta la tesis 397B |

**Decisión:** se conserva la abstracción `ChatClient` (`crates/kitty-runtime/src/client.rs`) + Ollama
como backend default para v0.1. El motor propio NO entra en v0.1.

**Condición de re-apertura:** un motor in-process solo se justifica cuando la tesis grande lo EXIJA:
Ornith-397B MoE sparsity (saltar expertos muertos), FWHT 3-bit (después del paper), SelectivePolicy
y `layer_streaming` con `pinned_buffer`. Mientras el target sea 9B/8GB vía HTTP, Ollama gana por
coste de mantenimiento y cero regresión. "Phase 0 skeleton sin tok/s real" no es un candidato a
backend productivo — es una apuesta de investigación separada.

## 2. Visión director: la celda vacía — NO pivot

Kitty ocupa la celda **Rust single-binary para modelos 9B en GPUs de 8 GB** — que hoy está vacía:

- **little-coder** (`Proyectos/Repos/little-coder/package.json:40`): Node, scaffold sobre
  `@earendil-works/pi-coding-agent ^0.83.0` — no es single-binary Rust.
- **smallcode** (`Proyectos/Repos/smallcode/bin/rag-index.js:77`): Node, índice RAG en tmpdir
  (`os.tmpdir()` mkdtemp) — mismo stack runtime que kitty explícitamente no requiere.

**Estrategia: competir cerrando P0s, no añadiendo features.** La ventaja reclamable es el harness
honesto (sandbox kernel-real, guard interpreter-aware, memoria episódica, presupuesto de contexto)
empaquetado en un binario estático. Cada feature nueva que no cierre un P0 diluye esa ventaja.

## 3. Hechos narrativos del día (registro permanente, dirigido por Oracle)

1. **Sandbox cableado con modelo de dos fronteras y opt-out ruidoso.** Los hijos de Bash corren
   contenidos por Landlock+seccomp+userns por defecto (`crates/kitty-harness/src/sandbox.rs`);
   las corridas de bench harbor ponen `KITTY_SANDBOX=off` porque la frontera Docker exterior ya
   está activa; el opt-out es por-run y ruidoso (`tracing::warn!` en cada activación). Decisión
   registrada en `docs/AUDIT.md` §6 y `bench/harbor/adapters/kitty/README.md`.
2. **GO de GBNF REVERTIDO por evidencia empírica.** La medición real con qwen2.5-coder-7b-q4_k_m
   dio 94.0% vs 94.0% (con/sin gramática): el trigger lazy `{"name"` nunca activó bajo la
   tokenización de Qwen2.5, así que el 94% es capacidad baseline del modelo — la decodificación
   restringida por gramática sigue SIN MEDIRSE. Nadie puede citar "GBNF funciona al 94%".
   Detalle y regla de citación: `docs/ADR-GBNF-GO.md` (amendment + citation rule).
3. **Cola nocturna muestreada first-N pre-registrada.** Las corridas kitty usan `-l N` = PRIMERAS N
   tareas del orden del dataset (cero cherry-pick post-hoc), con nombres honestos `_l50`/`_l15`,
   gate de trials erróneos ≤10%, y fallback `_partialN` con nota `not empirically closed`.
   Pre-registro completo: `docs/AUDIT.md` §6 "Approved night strategy".
4. **Prove-harness Fases 0–4 cerradas, HOLD antes de la batería pesada (2026-09-09).**
   7 commits, Gates A–E (2 re-reviews); primer par A/B citable N=15 casero:
   empate 14/14 (`ornith:9b` vs `ornith-1.5:9b`), string-match-homemade,
   nunca tasa oficial; oracle smoke TB2.1 25/25 (infra validada, no señal
   de modelo); mutación full descoped por OOM del box (quedan killer test
   + evidencia parcial); ClawEval descoped (mismatch de upstream);
   GPQA bloqueado (approval+token HF). Estimaciones calibradas y HOLD con
   GO pendiente en `handoff.md`; estado vivo en
   `.slim/deepwork/prove-harness-2026-09.md`.

## 4. Estado de métricas — honestidad ante todo

- `SMOKE.md`: números de smoke etiquetados como smoke; sin extrapolación a full-suite.
- `CNAS.md`: N/A hasta que los números del harbor adapter aterricen (no publicar tasas consumer
  oficiales antes).
- Gate de avance: cerrar P0s del audit (hecho: rename honesto, zeroize SecretString, sandbox real;
  pendiente: trigger fix de GBNF antes de re-decidir el GO).

## 5. Qué NO hacemos (guardrails de visión)

- No pivotar a producto generalista ni homepage/docs-site.
- No sustituir Ollama por motor propio en v0.1 (ver §1).
- No añadir wasmtime/plugins WASM como gate (plugins estáticos bastan).
- No crecer el quiz casero con ejercicios train-on-test.
- No fabricar porcentajes: ni 98% simulado, ni 94% atribuido a la gramática.

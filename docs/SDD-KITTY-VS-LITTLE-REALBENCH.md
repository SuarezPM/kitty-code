# SDD — Kitty vence a Little: kitty-infer operativo + Context-Forge + benchmark REAL

Fecha: 2026-10-24 · Fases previas: Pass 1-3 (kittycode, integridad, serie demo-12 paridad). Objetivo: **kitty code full (con kitty-infer dentro) > little-coder y opencode, medido en benchmarks REALES con MiniCPM5 y Ornith-1.5-9B**. Obsesión declarada: no parar hasta conseguirlo.

## 0. Diagnóstico (por qué hoy no ganamos, verificado)

1. **Mismo cerebro**: ornith:9b vía Ollama en ambos; 98% del wall es `wall_model`. Little hace ~4-5 turnos/ejercicio, kitty ~7.2.
2. **Pagamos el seguro**: guard+sandbox+verify+quality+doom-loop = +2.8s que compran robustez (0 ceros vs flakeables).
3. **El loop no termina**: texto final consume iteración hasta `max_iterations`; thinking sin cap; retries encadenados.
4. **kitty-infer fuera del ring**: Ornith 32/32 en CPU por presupuestación KV (no por tamaño): `required_whole_capacity = pesos + activations + KV(ctx=16K) + reservation > cap` → `gpu-fits-zero-spill-to-cpu`. Solo 8/32 capas pagan KV (híbrido GDN).
5. **Juguetes**: demo-12 n=36 no discrimina; falta benchmark real.

## 1. kitty-infer operativo — CORREGIDO por Oracle (F1a bloqueante; P0-3 FUERA)

Causa raíz real (no era `ContextSize`): el estimador marca TODAS las capas como paged cuando solo 8/32 pagan KV (híbrido GDN) → KV sobreestimada ~4x → `gpu-fits-zero-spill-to-cpu`. Cambiar a `Utilization` sin arreglar el conteo solo mueve el OOM a `KvExhausted`.
- **F1a-1**: cablear la máscara híbrida (índices paged desde el arch GGUF) en el `model_cfg` que usan `auto_device_map` y `calculate_cache_config`; mantener `ContextSize(16384)`; exponer `available_context_tokens` + assert `>= ctx_len` + registrar en manifiesto. `Utilization` solo como fallback con el mismo assert.
- **F1a-2 (verificar y cerrar)**: P0-2 ya existe (`synchronize_cuda_context` + `trim_cuda_mempool` en `calculate_cache_config`); solo falta trim/query fresco en el precheck de `ensure_loaded` si el fallo es ese. P0-4 ya existe (`decision_reason` a stderr+tracing); falta política (prohibir `KITTY_INFER_GPU_LAYERS`) + plumbarlo al manifiesto.
- **P0-3 ELIMINADO**: `cuda_supports_fp8` exige major ≥ 8; la 2060S es sm_75 → hard-fail. Ni en F1 ni después en esta GPU.
- Verificación: Ornith full-GPU 16K + assert `available_context_tokens >= 16384` en manifiesto + `ollama ps`/`nvidia-smi`.

## 2. Context-Forge portado a Rust (P0 primero, resto P1/P2; de R4/R5/R7)

- **P0-a `SegmentKind` + presupuestos** en `ContextBudget`/`push_message` (UserTurn y SystemPreflight intocables, min 512 tok, emergencia ×0.8). Toca 3 ficheros, cero riesgo.
- **P0-b `tier_adjust` pre-fusión** en `fuse_rankings_weighted` + test qrels con corpus DURO nuevo (el actual está saturado 1.000).
- **P0-c safety-salt en `response_cache::cache_key`** + test `hit_rate(juez reordenado)==0`.
- **P1**: `evict_to_budget` 5 modos (CRITICAL=offload, no borra) + `assemble_system_prompt` con invariantes (SEP `\n\n`, hash prefijo, reemplazo en sitio).
- **P2**: `kv_gate.rs` (JCR risk + INV-15 + salt shared/iso) + `CompactionStrategy`.
- **Vetos**: plugin ROMY vLLM, LLMLingua-2, TokenDance/AnchorPool, WORKFLOW_AWARE, Z3 en runtime (solo test). Calibración solo con datos reales, nunca manifiestos.

## 3. Loop que termina (ataque directo a los 7.2 turnos)

- **E1 text-terminal (enum, no break)**: `process_response` devuelve `Result<()>` y el loop corre hasta `max_iterations`; hace falta enum `Terminal{text,reason}` + `finalize` único que ejecuta verify + `save_episode` post-verify (hoy se guarda éxito pre-verify). El done-guard/nudge se excluye del predicado terminal. `iters_used` sigue siendo turnos LLM + `terminal_reason` obligatorio en manifiesto.
- **E2 caps**: Ornith resuelve thinking ON sin cap y `num_predict` default 8192 → floor ≥4096 (no 2K: trunca thinking/JSON y genera más turnos); medir `invalid_tool_calls` en el A/B. Solo tras E1.
- **E3 response-cache opt-in** con umbral 64 tok (no reusar prefijos minúsculos). **P0-c Forge CUTADO**: el salt pertenece a la clave KV del engine (F2c), no al response cache (ya hashea mensajes ordenados).
- **E4 retención KV (F2c, lane separada post-F1)**: session id fresco por run/ejercicio + salt por run + TTL/eviction + test 0 hits cruzados; flag off en bench hasta pasar el test. No es "no resetear SingleCache" (kitty-infer usa paged).
- **E5 dieta de trayectoria** última; su gate es calidad, no solo p50.
- Gate A/B: McNemar exacto pareado pre-registrado ("sin caída >1 + mejora clara p50"); n=12, k≥3 no aumenta n; veredicto real en F3.

## 4. Benchmark REAL (de R6)

- **Primario: Terminal-Bench 2.0 (89 tareas) vía Harbor.** Vendored + oracle 15/15 + techo 81/89 ya verificados. Adapters: kitty `KittyAgent` (1 archivo, binario release headless + Ollama), little `LittleCoderAgent` (`-p`/`--mode rpc` verificado), opencode **0 código** (PR harbor#2940 + `OPENAI_BASE_URL` + provider `openai-compatible`). Modelos: MiniCPM5 y Ornith-1.5-9B (2 celdas). k=3, serie exclusiva, 1 concurrente.
- **Secundario: Aider Polyglot 225** (paridad little 45.56% vs Aider vanilla 19.11%; kitty ~11.9% baseline propio). Mismo harness adapters sobre dataset aider-harbor.
- **Regla de victoria pre-registrada**: parity-only por defecto; superioridad exige +tareas con McNemar p<0.05 pareado por tarea Y p50≤rival. p99 descriptivo/censurado.

## 5. Fases (corregidas por Oracle)

- **F1a (bloqueante)**: máscara híbrida + `ContextSize(16K)` + assert + manifiesto; P0-2/P0-4 verificar-y-cerrar; P0-3 fuera.
- **F1b**: suite `inferperf` steady-state tok/s kitty-infer vs Ollama (warmup + ventana limpia + N/A si contaminada; logging off en ventana; `load_duration` separado), NO dentro de demo-12. Fingerprint con `backend_features` (`kv_fp8`, `response_cache`, `kv_retention`, `paged_mem_config`) + `available_context_tokens` + `terminal_reason` + `invalid_tool_calls`. demo-12 declara `fairness.backend=ollama`: kitty-infer corre en arm/manifiesto NUEVO, nunca bajo ese manifiesto.
- **F2a (loop, Ollama primero)**: E1 enum+finalize, E3 opt-in, E2 ≥4K tras E1. A/B k≥3 demo-12 pareado.
- **F2b (Forge, paralelo)**: P0-a presupuestos primero; P0-b tier+qrels duro como lane independiente (no comparte gate de turnos); P0-c cut.
- **F2c (engine, post-F1)**: E4 session/prefix cache con test de aislamiento; lane separada con reversión propia.
- **E5 último**; **F3 bench real** y **F4 iterar** como estaban. Crash/partial: el ledger reserva antes del primer LLM; celda crash ⇒ marcar fail, no re-ejecutar en silencio, reanudar solo con `rep` nuevo y pre-registro.

## 6. Vetos y honestidad

Serie exclusiva; k≥3 + manifiesto; pre-reg committed antes de la primera celda; runmark+ledger; tripwire answer-key; gates_off donde aplique; sin inventar celdas/timeouts; per-exercise margin 1/12; p99 censurado. Cifras del SDD anterior (35/36 vs 32/36, Δp50 2.8s) quedan como baseline, no como victoria.

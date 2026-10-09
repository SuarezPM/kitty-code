# Modelos soportados 1B–35B en 8GB (W8, SDD estado-arte)

Tabla curada para hardware de consumo. Fuente de tamaños: serie medida
`bench/gpu-ventana-2026-10-09/manifest-w6-vram.json` (2026-10-09T19:45Z,
Ollama 0.40.1, kitty 0.1.0, KV q8_0, ctx justo 16384 salvo nota).

> **Caveat CPU-fallback (honesto):** en esa serie Ollama NO veía la GPU
> (`gpu_count=0`; `vram_pico_gpu_mib=0` en las 10 celdas). Los tamaños de
> abajo son **residente medido** (RAM/CPU), no pico GPU. Columna GPU =
> PENDIENTE en todas las filas hasta la ventana serial con CUDA real.

## Medidos (residente real, sin OOM en ninguna celda)

| Modelo (id Ollama) | Params | Residente | Cuant | ctx celda | OOM | Tier harness | Nota |
|---|---|---|---|---|---|---|---|
| `smollm2:1.7b` | 1.7B | 2.8 GB | Q8_0 ⚠️ | 8192 | no | Ejecución | Único no-Q4 del lote (desviación del default Q4, declarada en manifiesto) |
| `minicpm5:2b` | 2.5B | 2.5 GB | Q4_K_M | 32768 | no | Ejecución | ctx_actual 32768 vino de fit/env previo (ver nota ctx del manifiesto) |
| `alibayram/smollm3:latest` | 3.1B | 2.7 GB | Q4_K_M | 16384 | no | Ejecución | — |
| `granite4:3b` | 3.4B | 3.0 GB | Q4_K_M | 16384 | no | **Triaje** | Token `:3b` = nano-shape: el harness lo auto-detecta triaje (`Tier::for_model`, fijado por test) |
| `qwen3:4b` | 4.0B | 4.0 GB | Q4_K_M | 16384 | no | Ejecución | — |
| `qwen3.5:4b` | 4.7B | 3.4 GB | Q4_K_M | 16384 | no | Ejecución | — |
| `ornith:9b` | 9.0B | 6.2 GB | Q4_K_M | 16384 | no | Ejecución | Representante 9B medido (variantes `-opt`/`-1.5` también residentes sin OOM) |
| `qwen38-27b:latest` | 27.3B | 18 GB RAM | Q4_K_S | 8192 | no | Ejecución | ctx 8192 del Modelfile propio; +17GB RAM durante residencia (46GB totales: en 8GB reales NO cabe — fila informativa, no recomendación) |

GPU pico: **pendiente de medición** en las 8 filas. Backend: Ollama.
En GPU real los 9B/27B probablemente OOM en 8GB (sin datos: no se afirma).

## Pendientes de celda VRAM (sin números: no medidos, no se inventan)

| Clase (plan) | Estado |
|---|---|
| Granite-8B | Pendiente de celda (el 3B medido NO lo representa) |
| Gemma E4B | Pendiente de celda |
| Phi-4-mini | Pendiente de celda |
| Qwen3.5-9B (id canónico) | Pendiente de celda (`ornith:9b` medido como representante 9B) |
| Clase 32–35B | Pendiente de celda |

## Regla de tiers (enforcement)

`Tier::for_model(id)` en `crates/kitty-harness/src/model_profile.rs`
replica el orden auto de `from_cli` (small-check, luego nano-shape;
resto fail-open a Ejecución). `--profile` explícito sigue mandando en
runtime (`profile.tier()` es la autoridad). El router aplica el subset
del tier (`TRIAGE_TOOL_SUBSET` 4 tools / `EXECUTION_TOOL_SUBSET` 6) en
`tools_for_category_tiered`, idéntico a perfiles estándar (test
`tier_routing_matches_profile_routing`). Modelos fuera de esta tabla
corren con tier Ejecución fail-open (sin degradación silenciosa).

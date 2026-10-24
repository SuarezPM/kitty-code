# Modelos soportados 1B–35B en 8GB (W8, SDD estado-arte)

Tabla curada para hardware de consumo. Fuente de tamaños CPU: serie medida
`bench/gpu-ventana-2026-10-09/manifest-w6-vram.json` (2026-10-09T19:45Z,
Ollama 0.40.1, kitty 0.1.0, KV q8_0, ctx justo 16384 salvo nota;
campos `matriz[].size_residente`, `vram_pico_gpu_mib`, `oom`).
Fuente GPU: serie 1-celda
`bench/gpu-ventana-2026-10-09/manifest-w6-vram-gpu.json`
(2026-10-09T23:14:51Z, `timestamp_utc`; Ollama 0.40.1 + ollama-cuda,
`backend`; RTX 2060 SUPER 8192 MiB, `hardware.gpu`; HEAD 5d8090d,
`git_head`; `n_celdas=4`, una celda por modelo — serie 1-celda,
**nunca k≥3**, no publicable como tasa ni como p50/p99).

> **Caveat CPU-fallback (honesto):** en la serie CPU Ollama NO veía la GPU
> (`hardware.ollama_gpu_count=0`, `gpu_visible_a_ollama: false`;
> `vram_pico_gpu_mib=0` en las 10 celdas de
> `manifest-w6-vram.json`). Los tamaños de la columna Residente son
> **residente medido** (RAM/CPU), no pico GPU. La columna GPU solo cubre
> los 4 modelos re-seriados con CUDA real; el resto sigue PENDIENTE.

## Medidos (residente real, sin OOM en ninguna celda CPU)

| Modelo (id Ollama) | Params | Residente CPU (`size_residente`) | GPU pico 1-celda (`peak_mib` / `pico_delta_mib`, serie `w6-vram-gpu`, NO k≥3) | Cuant | ctx celda | OOM | Tier harness | Nota |
|---|---|---|---|---|---|---|---|---|
| `smollm2:1.7b` | 1.7B | 2.8 GB | PENDIENTE (sin celda GPU) | Q8_0 ⚠️ | 8192 | no | Ejecución | Único no-Q4 del lote (desviación del default Q4, declarada en manifiesto) |
| `minicpm5:2b` | 2.5B | 2.5 GB | peak 3393 MiB / delta 2386 MiB · 43/43 capas GPU · runner 2.2 GiB / ps 2.3 GB (`matriz[0]`) | Q4_K_M | 32768 | no | Ejecución | ctx_actual 32768 vino de fit/env previo (ver nota ctx del manifiesto) |
| `alibayram/smollm3:latest` | 3.1B | 2.7 GB | PENDIENTE (sin celda GPU) | Q4_K_M | 16384 | no | Ejecución | — |
| `granite4:3b` | 3.4B | 3.0 GB | PENDIENTE (sin celda GPU) | Q4_K_M | 16384 | no | **Triaje** | Token `:3b` = nano-shape: el harness lo auto-detecta triaje (`Tier::for_model`, fijado por test) |
| `qwen3:4b` | 4.0B | 4.0 GB | peak 4976 MiB / delta 3931 MiB · 37/37 capas GPU · runner 3.7 GiB / ps 4.0 GB (`matriz[1]`) | Q4_K_M | 16384 | no | Ejecución | — |
| `qwen3.5:4b` | 4.7B | 3.4 GB | peak 5094 MiB / delta 4014 MiB · 34/34 capas GPU · runner 3.2 GiB / ps 3.4 GB (`matriz[2]`) | Q4_K_M | 16384 | no | Ejecución | — |
| `ornith:9b` | 9.0B | 6.2 GB | peak 6466 MiB / delta 5391 MiB · 33/33 capas GPU (+545.62 MiB CPU-mapped) · runner 5.1 GiB / ps 5.5 GB (`matriz[3]`, `oom=false`) | Q4_K_M | 16384 | no | Ejecución | Representante 9B medido (variantes `-opt`/`-1.5` también residentes sin OOM **solo en serie CPU**; en GPU solo `ornith:9b` tiene celda) |
| `qwen38-27b:latest` | 27.3B | 18 GB RAM | SIN CELDA GPU (no re-seriado; `oom_si_no_global`: 18GB > 8GB, OOM seguro por diseño; veredicto: no cabe, sin celda) | Q4_K_S | 8192 | no (CPU) | Ejecución | ctx 8192 del Modelfile propio; +17GB RAM durante residencia (46GB totales: en 8GB reales NO cabe — fila informativa, no recomendación) |

GPU pico: medido **solo** en las 4 filas con celda (`manifest-w6-vram-gpu.json`,
`metodologia_pico`: nvidia-smi tras 1 inferencia corta, delta = peak −
baseline desktop; el pico real de prefill puede exceder ligeramente el
residente; `KV q8_0` servidor). Serie 1-celda citada como serie,
nunca como k≥3. Resto de filas: **pendiente de medición**.
Backend: Ollama.
En GPU real los 9B no re-seriados / 27B probablemente OOM en 8GB
(sin datos: no se afirma; para el 27B el manifiesto GPU lo declara
no-cabe sin celda).

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

# Borrador de anuncio a comunidad — DO NOT PUBLISH (W9-prep)

> Estructura + resultados parity-only con letra pequeña. CERO claims
> de victoria o latencia: cada `[MEDIR: …]` se rellena solo con
> manifiestos citados de la ventana serial (GPU real, HEAD congelado,
> serie exclusiva). Publicar sin los números medidos viola el AC de
> W9 (anuncio solo con claims reproducidos).

## Qué es Kitty Code

Agente de código local para SLMs de consumo: un harness diseñado para
modelos pequeños locales + motor embebido + sandbox real (Landlock /
seccomp / namespaces), sin nube obligatoria.

## Tesis (una frase)

[Sin cambios.]

## Qué incluye esta versión

- Un solo binario `kitty` con `run`, `bench`, TUI e instalación en minutos
  (`kitty doctor` propone el `pull` exacto para tu VRAM medida).
- Sandbox real fail-closed para cada comando (denegación con control
  negativo, nunca teatro de seguridad).
- `kitty undo`: cada edición del agente queda en journal revertible.
- Tiers triaje/ejecución con routing y resúmenes compact-QR.
- Suite comparativa propia `bench/demo-12` (12 ejercicios fijos,
  protocolo de justicia embebido, manifiestos por corrida).

## Cómo reproducir (comandos, sin resultados)

```bash
curl ... | bash
kitty doctor
kitty run "..."
kitty bench demo-12 --k 3 -o manifiesto.json
```

## Resultados (parity-only, k≥3 con letra pequeña; serie Pass 3 medida)

> Los tres titulares citan manifiestos k=3 existentes (tb2-gpu = 5d8090d,
> demo12-fasef = c6bd16b, demo12-pass3 = 73a1542). Doc actual: tag v0.2.0.
> La serie Pass 3 ya corrió
> (`bench/demo-12/manifest-demo12-pass3-2026-10-24.json`, HEAD 73a1542,
> 6 celdas K1/L1/K2/L2/K3/L3 con paridad num_ctx 16384).

- **Titular 1 — tb2-8 casera 24/24 vs 18/24 (k=3):** kitty 24/24 (100.0%)
  vs little-coder 18/24 (75.0%), mismo modelo `ornith:9b`, misma GPU real,
  mismos timeouts 60s/celda, serie 1-residente con `ollama stop` entre reps.
  Fuente: `bench/gpu-ventana-2026-10-09/manifest-tb2-justicia-gpu.json`
  (`comparativa.kitty.pass_rate`, `comparativa.little.pass_rate`,
  `comparativa.kitty.reps=[8,8,8]`, `comparativa.little.reps=[6,6,6]`,
  `k=3`, `git_head=5d8090d`).
  Letra pequeña: suite **casera, no oficial** (8 ejercicios locales, no el
  Terminal-Bench oficial); little fue **más rápido por ejercicio**
  (p50 10.9s vs 27.2s: `comparativa.little.p50_ms=10875`,
  `comparativa.kitty.p50_ms=27152`); los 2 fallos little son sistemáticos
  (`fails_por_celda`: `tb2-sh-backup 0/3`, `tb2-sh-find-modified 0/3`);
  asimetría declarada en el manifiesto: kitty guard+sandbox on vs little
  accept-all, kitty num_ctx 32768 parcial CPU/GPU vs little 16384 100% GPU.
  Re-medición en curso: [MEDIR-PASS3: tb2-8 no entra en Pass 3 (solo
  demo-12); citar solo tras nueva serie k≥3 con HEAD congelado].
- **Titular 2 — demo-12 empate técnico 35/36 vs 34/36, Δp50 190ms (k=3):**
  kitty 35/36 (97.2%, reps 12/12/11) vs little 34/36 (94.4%, reps
  12/11/11), Δp50 +190ms, Δpass +2.8pp = 1 ejercicio.
  Fuente: `bench/demo-12/manifest-demo12-fasef-2026-10-10.json`
  (`agent_kitty.pass_rate`, `agent_little.pass_rate`,
  `comparativa.delta_p50_ms=190`, `comparativa.delta_pass_pp=2.8`,
  `k=3`, `git_head=c6bd16b`, paridad VRAM `KITTY_NUM_CTX=16384`).
  Letra pequeña: **paridad, no victoria** (`interpretation`: "EMPATE
  TÉCNICO en velocidad … EMPATE en pass-rate, sin ventaja legible";
  regla del manifiesto: no leer deltas <5s como señal); **HU-LATENCIA
  abierta** (gates_off en bench, p50/p99 brutos no cumplen `≤ little`);
  el +13.9pp de ambos brazos vs baseline es varianza en celdas flakeables,
  no atribuible a num_ctx (little sin cambios ganó lo mismo).
   Re-medición Pass 3 (HEAD 73a1542, paridad ctx 16384, k=3 interleaved):
   demo-12 35/36 vs 32/36, Δp50 +2807ms (< 5s = ruido).
   Fuente: `bench/demo-12/manifest-demo12-pass3-2026-10-24.json`
   (`kitty.passed=35`, `little.passed=32`, `delta_p50_ms=2806.8`,
   `desglose_por_ejercicio`, sin ningún 0/3 en ningún lado;
   kitty reps 12/12/11, little reps 10/11/11). HU-LATENCIA sigue
   ABIERTA (p50 nominal kitty > little); tb2-8 no entra en Pass 3.]
- **Titular 3 — cero ceros en tres series k=3 con sandbox on:** ningún
  ejercicio 0/3 en comparativo (30/36 vs 29/36) ni en fasef (35/36 vs 34/36).
  Fuentes: `bench/demo-12/manifest-demo12-comparativo.json`
  (`comparativa.kitty.fails_por_ejercicio` sin entradas 0/3;
  `comparativa.little.fails_por_ejercicio` con 2/3 en backup/find-modified;
  `k=3`) + `bench/demo-12/manifest-demo12-fasef-2026-10-10.json`
  (`agent_kitty.fails_por_ejercicio={demo-js-fetch:1/3}`,
  `agent_little.fails_por_ejercicio={demo-sh-find-modified:1/3,
  demo-sh-words:1/3}`).
  Letra pequeña: robustez observada en suites **caseras** (demo-12 v1.0.0,
  12 ejercicios fijos), no en benchmark oficial; con sandbox+guard activos
  en el brazo kitty (= producto); k=3 no autoriza p99 ni claims de
   producción. Pass 3: cero ceros confirmado (kitty 12/12, 11/12,
   12/12; little 10/12, 11/12, 11/12; fails kitty {demo-rs-args:1/3},
   little {demo-bash-count:1/3, demo-sh-find-modified:2/3,
   demo-sh-words:1/3}).
   Fuente: `bench/demo-12/manifest-demo12-pass3-2026-10-24.json`.
- **Reserva (serie 1-celda, nunca k≥3):** `ornith:9b` Q4 cabe en 8GB:
  pico-delta 5391 MiB (peak 6466 MiB), 33/33 capas GPU (+545.62 MiB
  CPU-mapped), runner 5.1 GiB / ps 5.5 GB, sin OOM.
  Fuente: `bench/gpu-ventana-2026-10-09/manifest-w6-vram-gpu.json`
  (`matriz[3].pico_delta_mib/peak_mib/capas_gpu/runner_vram/size_ps/oom`,
  `n_celdas=4`, `metodologia_pico`, `oom_si_no_global`).
  Letra pequeña: **una celda por modelo, sin repetición**: se cita como
  serie, nunca como tasa k≥3 ni como p50/p99; el pico real de prefill puede
  exceder ligeramente el residente; `qwen38-27b` sin celda GPU (no cabe).
- VRAM por modelo Q4 + KV-quant: verificado en
  `docs/SUPPORTED-MODELS-8GB.md` (CPU `manifest-w6-vram.json` + GPU
   1-celda `manifest-w6-vram-gpu.json`, columna GPU citada como serie).
   Pass 3: warmup 100% GPU + CONTEXT 16384 en ambos brazos
   (`ollama ps`: 5.5 GB, llamacpp, ctx 16384; little vía pi RPC 0.83.0
   también negocia 16384 sin reiniciar el servidor).

## Qué falta / honest scope

- HU-LATENCIA abierta: Pass 3 k=3 no la cierra (p50 nominal kitty >
  little: 13.3s vs 10.5s); falta una serie que cumpla p50/p99 ≤ little.
- Plataformas fuera de ubuntu x86_64 hasta CI por objetivo.
- Auditoría externa, desktop/IDE, i18n: fuera de alcance declarado.

## Llamada a la comunidad

Probar `kitty doctor` + `run` en su hardware y reportar VRAM real por
modelo (formato del manifiesto W6). Sin promesas de rendimiento hasta
los números medidos de arriba.

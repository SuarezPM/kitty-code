# Borrador de anuncio a comunidad — DO NOT PUBLISH (W9-prep)

> Estructura solamente. CERO claims numéricos: cada `[MEDIR: …]`
> se rellena solo con manifiestos citados de la ventana serial (GPU real,
> HEAD congelado, serie exclusiva). Publicar sin los números medidos
> viola el AC de W9 (anuncio solo con claims reproducidos).

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

## Resultados (BLOQUEADO tras medición GPU)

- Latencia p50/p99 vs little-coder con pass-rate: `[MEDIR: manifiesto tb2-8, k≥3]`
- VRAM por modelo Q4 + KV-quant: `[MEDIR: serie VRAM exclusiva]`
- Tasa de colapso vs baseline: `[MEDIR: goldens + E2E]`
- Tabla de modelos 1B–35B con residentes medidos:
  `docs/SUPPORTED-MODELS-8GB.md` (columna GPU pendiente hasta la serie).

## Qué falta / honest scope

- Serie GPU real (Ollama no veía la GPU en la ventana de medición;
  `bench/gpu-ventana-2026-10-09/manifest-w6-vram.json`, caveat CPU-fallback).
- Plataformas fuera de ubuntu x86_64 hasta CI por objetivo.
- Auditoría externa, desktop/IDE, i18n: fuera de alcance declarado.

## Llamada a la comunidad

Probar `kitty doctor` + `run` en su hardware y reportar VRAM real por
modelo (formato del manifiesto W6). Sin promesas de rendimiento hasta
los números medidos de arriba.

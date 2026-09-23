# EMBEDDED-BENCH-SPEC — bench sobre el motor nuevo (spec, Fase 3)

Estado: el bench (`bench`, `bench-ab`) exige `OllamaClient` concreto y hace
bail honesto con `KITTY_BACKEND=kitty-infer`. Generalizarlo es un refactor
de `BenchConfig` + drivers + assertions de servidor; NO se hace a ciegas.
Este documento fija el diseño mínimo + el spike que lo valida.

## Qué mide el bench hoy (contrato a preservar)

Por ejercicio: wall latency del turno agente, veredicto pass/fail/error/timeout
(30s por defecto), tasa de éxito agregada + `bench_exit_code` (gate 50% +
delta vs baseline), persistencia como `Episode`. Todo eso es agnóstico al
backend **excepto**: construcción del cliente, `num_ctx` probe (`/api/show`),
ventana de contexto y `is_small_model` por id.

Nota D1 (gate k>=3): el `latency_ms` de un veredicto `timeout` es SENTINELA
(`30000.0` hard-coded en `run.rs`, no elapsed real); el cap efectivo vive en
el manifiesto Fase 1 (`crates/kitty-cli/src/bench/timeouts.json`, celda por
modelo/backend). No promediar el sentinela como latencia.

## Diseño mínimo (cuando se apruebe)

1. `BenchConfig.client: OllamaClient` → `Box<dyn ChatClient>`; la sonda
   `num_ctx` solo corre en el brazo Ollama (infer usa `KITTY_INFER_CTX`).
2. `BenchCliCtx` suma `backend: BackendKind` (`ollama` default; `infer`
   opt-in por env, nunca default silencioso).
3. Métricas por ejercicio idénticas (latencia, veredicto, salida); el reporte
   añade columna `backend` para comparar Ollama vs infer en la misma suite.
4. Prohibido: cambiar thresholds, semillas o `interleave` al generalizar
   (rompe reproducibilidad contra baselines existentes).

## Spike instrumentado (disponible hoy)

`crates/kitty-harness/examples/infer_spike.rs` — mide por prompt fijo lo que
el núcleo del bench necesita (latencia, ok, tamaño de respuesta) a través de
`Box<dyn ChatClient>`, con el MISMO binario para ambos backends:

```bash
cargo run -p kitty-harness --example infer_spike
KITTY_BACKEND=kitty-infer KITTY_INFER_MODEL=<m.gguf> \
  cargo run -p kitty-harness --example infer_spike --features kitty-infer
```

Criterio de salida del spike antes del bench completo: 3/3 ok en ambos
backends con latencias del mismo orden (tolerancia GPU vs CPU documentada
por caso). Sin bench completo silencioso: si el spike diverge, se investiga,
no se esconde bajo un promedio.

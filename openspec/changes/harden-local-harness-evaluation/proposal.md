## Why

Kitty Code ya puede medir ejercicios de código, pero aún no puede evaluar de
forma reproducible razonamiento científico ni tareas agentic con las restricciones
reales de sus datasets. Además, los fallos de proceso, contexto, secretos y
llamadas de herramientas siguen siendo poco observables en los perfiles de 1B–7B.

## What Changes

- Corrige los dos warnings preexistentes de Clippy sin cambiar la serialización de
  los tests que modifican el entorno.
- Añade adquisición offline-first, con revisión y hash fijados, para GPQA Diamond;
  ejecuta una aproximación local etiquetada, sin publicar prompts ni presentarla
  como resultado oficial.
- Añade integración de ClawEval como wrapper del harness oficial, con comprobación
  de Docker, fixtures y credenciales; no implementa un grader reducido que se haga
  pasar por `Pass^3`.
- Endurece la ejecución de herramientas con cancelación, terminación del grupo de
  procesos, redacción de secretos, clasificación de errores y reintentos acotados.
- Añade telemetría local rotada por turno y benchmark, sin prompts ni secretos.
- Reduce el coste de perfiles pequeños mediante superficie de herramientas mínima,
  breaker de llamadas inválidas y masking/snapshot de contexto.
- Ejecuta los benchmarks pesados solo tras pre-registro, comprobación de disco y
  la batería final de verificación.

## Capabilities

### New Capabilities
- `local-evaluation-datasets`: adquisición, integridad y ejecución honesta de GPQA Diamond y el wrapper de ClawEval.
- `local-run-telemetry`: registros locales, rotados y sin secretos para diagnosticar turnos y benchmarks.

### Modified Capabilities
- `benchmarking`: añade suites etiquetadas con método de grading, procedencia y precondiciones verificables.
- `tool-loop`: limita llamadas de herramientas inválidas y conserva contexto operativo mínimo en perfiles pequeños.
- `safety`: cancela árboles de procesos y redacciona secretos antes de persistir salidas.
- `model-profiles`: hace que los caps de output, masking y herramientas sean explícitos por perfil.

## Impact

- Código: `crates/kitty-harness`, `crates/kitty-runtime`, `crates/kitty-cli` y `crates/kitty-memory` si se necesita persistencia de métricas.
- Datos y documentación: `benches/`, `docs/AUDIT.md`, `docs/STRATEGY.md`, `codemap.md` y documentación de operación.
- Dependencias y entorno: posible `tokio-util` para cancelación; GPQA requiere aceptación/acceso a Hugging Face; ClawEval requiere su CLI, Docker, fixtures y, para la evaluación oficial, credenciales de sus servicios.

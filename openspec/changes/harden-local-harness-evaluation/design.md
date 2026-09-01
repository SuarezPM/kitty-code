## Context

El loop actual ya tiene perfiles, routing, guardas, sandbox, compaction,
truncación y resultados JSON de benchmark. La propuesta y sus specs amplían esas
capacidades sin convertir el harness en un orquestador de proyectos.

## Goals / Non-Goals

**Goals:**
- Añadir límites y telemetría que funcionen con ventanas de 4K–32K.
- Integrar datasets solo con trazabilidad, almacenamiento offline y etiquetas honestas.
- Garantizar que timeout/cancelación no deja procesos hijos vivos.

**Non-Goals:**
- No crear un grader local alternativo para `ClawEval Pass^3`.
- No guardar prompts GPQA ni publicar sus ejemplos.
- No añadir worktrees, una base de datos de workflow ni delegación de subagentes al runtime.

## Decisions

### Datos externos: caché materializada y metadatos separados

GPQA se materializa fuera del árbol versionado con revisión, hash, permutación y
plantilla fijos. El runtime solo lee una representación ya validada. Como GPQA es
gated, la revisión y el hash esperados se fijan en un manifiesto de pre-registro
creado antes de la medición y se cotejan con el sidecar y el archivo materializado;
sin ese manifiesto el comando queda bloqueado. ClawEval se invoca mediante su
propio CLI para cualquier claim de score oficial. Esto evita reimplementar graders
opacos y evita filtrar GPQA.

Alternativa descartada: vendorizar datasets en `benches/`. GPQA es gated y su
licencia/condición anti-fuga hacen esta opción inadecuada.

### Robustez: una frontera de ejecución de herramientas

La ejecución Bash concentra cancelación, grupo de procesos, timeout y redacción
antes de devolver `ToolResult`. Las salidas completas se redactan antes de spill;
el loop solo recibe el extracto que permita su perfil. El retry se limita a errores
transitorios de chat, nunca a comandos ni guardas.

Alternativa descartada: reintentar desde cada caller. Duplicaría presupuestos y
haría difícil diagnosticar storms.

### Eficiencia: estado pequeño y reversible en el loop

El breaker se basa en llamadas inválidas consecutivas y reduce la herramienta
visible durante un turno de recuperación. Antes de compactar se construye un
snapshot pequeño y los resultados antiguos se enmascaran, no se eliminan ni se
resumen de nuevo sin control.

Alternativa descartada: añadir un segundo selector LLM para tools. Añade latencia
y complejidad antes de medir que el routing existente sea insuficiente.

### Telemetría: JSONL local rotado

La telemetría se escribe como JSONL local con campos allowlisted y objetivo
hasheado. Mantiene los datos necesarios para comparar perfiles sin coste de base
de datos ni dependencia externa.

## Risks / Trade-offs

- [GPQA gated] → el fetch queda bloqueado hasta que el usuario acepte términos y aporte `HF_TOKEN`.
- [ClawEval requiere servicios externos] → el wrapper preflight bloquea sin score cuando faltan Docker, fixtures o credenciales.
- [Terminación de proceso en Unix] → se valida con un proceso que crea descendiente y se conserva el kill actual como fallback.
- [Regex de redacción] → se prueban falsos negativos comunes y se evita registrar texto no allowlisted.
- [Tests finales] → se escriben junto al código, pero por instrucción se ejecuta una única batería completa en el gate final; antes solo se emplean revisión estática y checks focalizados.

## Migration Plan

1. Mantener valores actuales de perfiles y comandos como defaults compatibles.
2. Activar telemetría y caches como estado local ignorado por Git.
3. Publicar loaders y wrappers sin resultados hasta completar preflight y manifiesto.
4. Ejecutar la batería final; si falla, revertir por fase los commits independientes y conservar los manifiestos/resultados como evidencia local.

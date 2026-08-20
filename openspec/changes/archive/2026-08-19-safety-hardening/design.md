## Context

See proposal.md Why: guard bloquea sustitución de comandos pero no redirects de shell (`cat > file`, clase #70), sin invariante write-vs-edit y sin remedio accionable; bypasses conocidos en competidores (SSRF, semicolon tras /dev/null). Constraints: extender `guard.rs` y `tools.rs` sin romper casos benignos (sh-backup, `tar -czf`).

## Goals / Non-Goals

**Goals:**
- Invariante write-vs-edit: Write rechaza existente, Edit exige Read previo (scoped a sesión).
- Detección shell-redirect (`>`, `>>`, `tee`) como writes.
- Permission-gate segment-wise que nombra el remedio.

**Non-Goals:**
- Sandbox de FS (solo guarda a nivel de comando).
- Bloqueo de lectura de secretos (fuera de scope de esta guarda).
- Policy engine configurable por proyecto.

## Decisions

- **Detección de redirects en guard.rs (parseo de `>`/`>>`/`tee`)** — captura clase #70 sin tocar cada tool; alternativa: hook en cada Write (incompleto, bypass por shell).
- **Invariante en tools.rs con estado de sesión** — Read marca archivo como editable; alternativa: check solo en guard (no tiene contexto de Read).
- **Permission-gate segment-wise** — evalúa cada `;`/`&&`/`||`/`|` por separado; alternativa: todo-o-nada (pierde granularidad y remedio por segmento).

## Risks / Trade-offs

- [Falso positivo en `> /dev/null` o `backup > file.bak`] → Mitigación: allowlist de destinos benignos (/dev/null, backups), tests de casos benignos.
- [Bypass `tee` sin `>` literal] → Mitigación: matching de `tee` + heurística de args, tests de regresión por caso F1.2.
- [Estado de sesión crece sin límite] → Mitigación: set de paths leídos es trivial en memoria, scoped a sesión.

## Why

Ambos competidores tienen bypasses de seguridad abiertos (little-coder: #70 ShellSession saltó write-guard, #107 semicolon tras `/dev/null`, #94 SAFE_PREFIXES bypasseable; smallcode: SSRF #39). La guarda de Kitty Code bloquea sustitución de comandos pero **no los redirects de shell** (`cat > file` — la clase de bypass #70), no hay invariante write-vs-edit, y no hay permission-gate que nombre el remedio.

## What Changes

- **Write-vs-edit invariant**: `Write` rechaza archivos existentes (fuerza `Edit`); `Edit` requiere `Read` previo del archivo en el turno.
- **Shell-redirect write detection**: la guarda detecta escrituras vía redirects (`cat > file`, `>>`, `tee`) y las trata como writes (mismas reglas que Write/Edit).
- **Permission-gate segment-wise**: juzga cada segmento de la cadena de shell, rechaza cualquier write-redirect no permitido, y **nombra el remedio** (no solo "refused").

## Capabilities

### New Capabilities
- `safety`: invariante write-vs-edit, detección de writes vía redirects de shell y permission-gate que nombra el remedio.

## Impact

- `crates/kitty-harness/src/guard.rs` (detección de redirects, permission-gate), `tools.rs` (invariante write-vs-edit, read-before-write).
- Extiende la guarda existente (sustitución de comandos, eval hard-block) sin romper los casos benignos (sh-backup, `tar -czf`).
- Tests de regresión por cada caso (F1.2 del plan de ejecución + nuevos casos de redirect).

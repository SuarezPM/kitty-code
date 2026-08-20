## Context

See proposal.md Why: table stakes 2026, Kitty Code sin MCP (solo backlog F6.5); competidores ya consumibles como infraestructura. Constraints: reutilizar `kitty-harness` (tools + guarda) sin duplicar; heredar safety-hardening; stdio como transporte.

## Goals / Non-Goals

**Goals:**
- Servidor MCP sobre stdio que expone Read/Edit/Write/Bash/Glob/Grep con schemas MCP.
- Cada llamada pasa por guarda, invariante write-vs-edit y permission-gate.
- Subcomando `kitty mcp` inicia el servidor.

**Non-Goals:**
- Transporte SSE/HTTP (solo stdio).
- Auth/billing del servidor MCP.
- Tools nuevas exclusivas de MCP (reexposición de las existentes).

## Decisions

- **Stdio transport** — estándar MCP más interoperable; alternativa: HTTP/SSE (más ops, no necesario).
- **Reuso de `kitty-harness` sin duplicar** — nuevo crate/módulo que llama al stack existente; alternativa: reimplementar tools (divergencia).
- **Subcomando `kitty mcp`** — ergonomía CLI existente; alternativa: binario separado (fricción de instalación).

## Risks / Trade-offs

- [Bypass de guarda vía MCP si ruta distinta] → Mitigación: mismo código de guarda, tests `rm -rf` y `> file` vía MCP.
- [Cliente MCP espera streaming/progress] → Mitigación: documentar sync-only inicial, no prometer streaming.
- [Exposición de Bash amplía superficie] → Mitigación: permission-gate segment-wise heredado, mensajes con remedio.

## Why

MCP es table stakes para un harness en 2026: smallcode envía un servidor MCP (`mcp.json`, `src/tools/mcp_client.js`) y little-coder cabalga el ecosistema de extensiones de pi. Kitty Code no tiene ninguno (solo backlog F6.5). Un MCP server hace a Kitty Code consumible por otros agentes (el canal que lo convierte en infraestructura, no solo en CLI).

## What Changes

- **MCP server** sobre stdio que expone las tools de Kitty Code (Read, Edit, Write, Bash, Glob, Grep) como tools MCP.
- Las llamadas MCP pasan por el mismo stack de seguridad (guarda, invariante write-vs-edit, permission-gate) que el loop normal.
- Nuevo subcomando CLI (`kitty mcp`) que inicia el servidor.

## Capabilities

### New Capabilities
- `mcp-server`: servidor MCP sobre stdio que expone las tools de Kitty Code con el mismo stack de seguridad.

## Impact

- Nuevo módulo/crate en `kitty-cli` (o crate `kitty-mcp`).
- Reutiliza `kitty-harness` (tools + guarda) sin duplicar lógica.
- Interactúa con safety-hardening (las llamadas MCP heredan las reglas).

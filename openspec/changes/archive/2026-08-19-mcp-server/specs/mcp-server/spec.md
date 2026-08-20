## Purpose

Servidor MCP sobre stdio que expone las tools de Kitty Code a otros agentes, con el mismo stack de seguridad del loop normal.

## ADDED Requirements

### Requirement: Servidor MCP sobre stdio

El subcomando `kitty mcp` inicia un servidor MCP sobre stdio que anuncia las tools de Kitty Code (Read, Edit, Write, Bash, Glob, Grep) y las ejecuta.

#### Scenario: Listar tools
- **GIVEN** un cliente MCP conectado al servidor
- **WHEN** el cliente solicita la lista de tools
- **THEN** el servidor responde con las tools de Kitty Code y sus schemas

#### Scenario: Llamar una tool
- **GIVEN** un cliente MCP conectado al servidor
- **WHEN** el cliente invoca una tool (p.ej. Grep con un patrón)
- **THEN** el servidor ejecuta la tool y devuelve el resultado en el formato MCP

### Requirement: Las llamadas MCP heredan el stack de seguridad

Toda llamada MCP pasa por la guarda, el invariante write-vs-edit y el permission-gate; un comando peligroso se rechaza igual que en el loop normal.

#### Scenario: Comando peligroso vía MCP
- **GIVEN** un cliente MCP que invoca Bash con un comando peligroso (p.ej. `rm -rf /` o un write-redirect a un archivo existente)
- **WHEN** el servidor ejecuta la tool
- **THEN** la llamada se rechaza con el mismo mensaje de remedio que en el loop normal

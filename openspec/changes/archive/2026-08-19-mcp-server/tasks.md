## 1. Servidor MCP stdio

- [ ] 1.1 Implementar servidor MCP sobre stdio que anuncie tools (Read/Edit/Write/Bash/Glob/Grep) con schemas
- [ ] 1.2 Implementar dispatch de tool-calls al stack de `kitty-harness` y retorno en formato MCP

## 2. Seguridad heredada

- [ ] 2.1 Enrutar cada llamada MCP por guarda, invariante write-vs-edit y permission-gate (mismo remedio que loop normal)
- [ ] 2.2 Añadir subcomando CLI `kitty mcp` que inicia el servidor

## 3. Verificación

- [ ] 3.1 Tests de listar tools y de rechazo de `rm -rf /` / `cat > file` vía MCP con mensaje de remedio

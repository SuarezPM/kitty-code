## Purpose

Perfiles de modelo que parametrizan el comportamiento del harness (contexto, thinking, tools, caps, canal de tool-calling) para servir LLMs de 1B a 35B.

## ADDED Requirements

### Requirement: Los perfiles de modelo definen el comportamiento por tamaño

El harness expone tres presets (`tiny`, `medium`, `large`) que determinan contexto, thinking, subset de tools, canal de tool-calling y caps de salida. El usuario selecciona el perfil explícitamente; el default es `medium`.

#### Scenario: Perfil tiny para modelos sub-2B
- **GIVEN** un modelo sub-2B (p.ej. qwen2.5:1.5b) y el perfil `tiny`
- **WHEN** el agente inicia una sesión
- **THEN** usa contexto ≤4K, sin thinking budget, expone solo Read/Edit/Write/Bash, y no asume tool-calling nativo

#### Scenario: Perfil medium para modelos 8–13B
- **GIVEN** un modelo 8–13B y el perfil `medium` (default)
- **WHEN** el agente inicia una sesión
- **THEN** usa contexto 16K, thinking budget 2048, las 6 tools, y tool-calling nativo vía Ollama

#### Scenario: Perfil large para modelos 27–35B
- **GIVEN** un modelo 27–35B y el perfil `large`
- **WHEN** el agente inicia una sesión
- **THEN** usa contexto 32K y el set completo de tools

### Requirement: El runtime está abstraído tras un trait

El agente depende de un trait `ChatClient` (`chat`, `count_tokens`, `health_check`), no de una implementación concreta. Esto permite múltiples backends (Ollama, llama.cpp) y tests del loop con un mock.

#### Scenario: El agente acepta cualquier implementación de ChatClient
- **GIVEN** una implementación de `ChatClient` distinta de `OllamaClient`
- **WHEN** se construye el agente con ella
- **THEN** el loop de agentes funciona sin cambios en el harness

#### Scenario: Tests del loop con mock
- **GIVEN** un `MockClient` con respuestas guionizadas (tool_call → error de tool → texto final)
- **WHEN** se ejecuta el loop de agentes en un test
- **THEN** el test verifica el flujo completo sin servidor real

### Requirement: Los caps de contexto son datos del perfil

Los límites de Read (líneas), salida de Bash (chars), Grep (resultados), knowledge (cards/tokens) y `keep_count` de compactación se derivan del perfil activo, no de constantes globales.

#### Scenario: Caps reducidos en perfil tiny
- **GIVEN** el perfil `tiny` con contexto 4K
- **WHEN** el agente lee un archivo grande o recibe salida de Bash extensa
- **THEN** los caps de Read/Bash/Grep son menores que en `medium` y se sugiere Grep para archivos grandes

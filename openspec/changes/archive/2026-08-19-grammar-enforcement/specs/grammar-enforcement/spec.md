## Purpose

Adapter llama.cpp con gramática GBNF que restringe las tool-calls al schema activo, haciendo imposibles las llamadas malformadas de modelos sub-2B.

## ADDED Requirements

### Requirement: Tool-calls restringidas por gramática

El adapter llama.cpp genera una gramática GBNF a partir del set de schemas activo y la aplica a cada llamada; una tool-call que no cumpla el schema es imposible de emitir.

#### Scenario: Llamada malformada imposible
- **GIVEN** el adapter llama.cpp con gramática activa para el set de schemas
- **WHEN** el modelo intenta emitir una tool-call fuera del schema (nombre erróneo, args inválidos)
- **THEN** la salida no contiene tool-calls malformadas (la gramática las excluye)

#### Scenario: Gramática refleja el set filtrado
- **GIVEN** el 2-stage routing seleccionó la categoría `write`
- **WHEN** se genera la gramática para el turno
- **THEN** la gramática solo admite los schemas de la categoría ganadora

### Requirement: Paridad de comportamiento con el cliente Ollama

El loop de agentes funciona de forma idéntica con el adapter llama.cpp y con `OllamaClient` (misma interfaz `ChatClient`, mismo flujo de tools).

#### Scenario: Loop completo vía llama.cpp
- **GIVEN** el agente construido con el adapter llama.cpp
- **WHEN** se ejecuta una tarea que requiere Read → Edit → verificación
- **THEN** el loop completa la tarea con el mismo comportamiento observable que con Ollama

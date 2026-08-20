# tool-loop Specification

## Purpose
Mejoras del loop de tools para modelos pequeños: routing en dos etapas, dedup por argumentos y done-guard de contrato.

## Requirements

### Requirement: Routing de tools en dos etapas

En perfiles con contexto pequeño, el harness envía primero un selector de categoría (~200 tokens) y solo después los schemas de la categoría ganadora, en vez de todos los schemas en cada turno.

#### Scenario: Selector de categoría primero
- **GIVEN** el perfil `tiny` o `medium` y un turno del agente
- **WHEN** el agente va a llamar a una tool
- **THEN** el harness envía el selector de categoría y solo los schemas de la categoría ganadora, ahorrando tokens por turno

#### Scenario: Affirmation guard conserva las write tools
- **GIVEN** el agente respondió "yes"/"ok" a una sugerencia de edición
- **WHEN** el clasificador evalúa la categoría
- **THEN** la categoría de write tools no se descarta por la afirmación

### Requirement: Dedup de tool-calls por argumentos

La detección de bucles considera nombre + hash de argumentos y persiste entre turnos; una misma tool con argumentos distintos no se marca como bucle.

#### Scenario: Misma tool con mismos args en turnos consecutivos
- **GIVEN** el agente llamó a la misma tool con los mismos argumentos en 3 turnos seguidos
- **WHEN** el quality monitor evalúa la ventana
- **THEN** se detecta un tool-loop y se emite el mensaje correctivo con los args reales

#### Scenario: Misma tool con args distintos
- **GIVEN** el agente llamó a la misma tool con argumentos distintos en turnos consecutivos
- **WHEN** el quality monitor evalúa la ventana
- **THEN** no se marca como bucle

### Requirement: Done-guard de contrato (Definition of Done)

El agente puede declarar un contrato con aserciones verificables; el harness rechaza la finalización si las aserciones no se cumplen.

#### Scenario: Contrato satisfecho
- **GIVEN** un contrato con la aserción "el archivo X existe y contiene Y"
- **WHEN** el agente declara completado y el harness verifica
- **THEN** la finalización se acepta

#### Scenario: Contrato insatisfecho
- **GIVEN** un contrato con la aserción "el archivo X existe y contiene Y"
- **WHEN** el agente declara completado pero el archivo no contiene Y
- **THEN** la finalización se rechaza y se devuelve el fallo de verificación al agente

### Requirement: Parser de tool-calls robusto (sin drops silenciosos)

El parser de tool-calls no descarta silenciosamente llamadas malformadas: ante una llamada que no puede parsear, emite un mensaje correctivo con el formato esperado para que el modelo reintente. Es el canal de tool-calling del perfil `tiny` (que no asume tool-calling nativo).

#### Scenario: Llamada malformada con mensaje correctivo
- **GIVEN** una respuesta del modelo con una tool-call malformada (JSON roto, tags sin cerrar, formato desconocido)
- **WHEN** el parser no puede recuperarla
- **THEN** el harness emite un mensaje correctivo con el formato esperado y el loop continúa (sin drop silencioso ni round-trip huérfano)

#### Scenario: Recuperación multi-formato
- **GIVEN** una respuesta con tool-calls en JSON, YAML, XML o bloques fenced
- **WHEN** el parser procesa la respuesta
- **THEN** las llamadas válidas se recuperan y ejecutan

## ADDED Requirements

### Requirement: Breaker de llamadas inválidas
El loop MUST limitar las llamadas de herramientas inválidas consecutivas y dar
una corrección breve antes de terminar con una causa diagnóstica.

#### Scenario: Tres llamadas inválidas consecutivas
- **WHEN** el agente supera el límite configurado de llamadas inválidas
- **THEN** el loop restringe la superficie o fuerza una respuesta final sin repetir indefinidamente

### Requirement: Contexto operativo recuperable
Antes de compactar, el loop MUST conservar un resumen acotado del objetivo,
archivos activos, último resultado de herramienta y error pendiente.

#### Scenario: Compaction bajo presión de contexto
- **WHEN** el contexto alcanza su umbral de compactación
- **THEN** el siguiente turno conserva el snapshot operativo dentro del presupuesto del perfil

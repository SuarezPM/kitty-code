## ADDED Requirements

### Requirement: Cancelación de árbol de procesos
Al vencer un timeout o solicitarse cancelación, el harness MUST terminar el
grupo de procesos de la herramienta y confirmar su recolección antes de continuar.

#### Scenario: Herramienta que excede el timeout
- **WHEN** una herramienta supera su presupuesto
- **THEN** el proceso y sus descendientes reciben terminación y el resultado indica timeout

### Requirement: Redacción antes de persistencia
El harness MUST redactar secretos reconocidos antes de guardar o exponer salidas
de herramientas, archivos de truncación, telemetría o mensajes de error.

#### Scenario: Token en stdout
- **WHEN** stdout contiene un token reconocido
- **THEN** los artefactos persistidos contienen un marcador redactado y no el token

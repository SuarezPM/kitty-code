## ADDED Requirements

### Requirement: Provenencia explícita de suites externas
Todo resultado de una suite externa MUST incluir versión, revisión, hash,
método de grading, nivel de compatibilidad y precondiciones de ejecución.

#### Scenario: Resultado local de una suite externa
- **WHEN** el harness escribe el resultado de una suite descargada
- **THEN** el JSON identifica inequívocamente dataset, método y limitaciones de comparabilidad

### Requirement: Pre-registro para baterías pesadas
Las baterías pesadas MUST tener un manifiesto creado antes de la ejecución que
fije entorno, selección, modelo, límites y artefactos esperados.

#### Scenario: Batería sin manifiesto
- **WHEN** se solicita una batería pesada sin manifiesto válido
- **THEN** el harness no inicia la medición

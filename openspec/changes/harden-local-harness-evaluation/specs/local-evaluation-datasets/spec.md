## Purpose

Permite preparar y ejecutar evaluaciones locales reproducibles sin confundirlas
con los resultados producidos por los harnesses oficiales de cada benchmark.

## ADDED Requirements

### Requirement: Dataset versionado y verificable
El sistema MUST requerir una revisión y un hash de contenido esperados antes de
cargar un dataset externo, y MUST fallar sin ejecutar si su integridad no coincide.

#### Scenario: Dataset íntegro en caché
- **WHEN** existe una copia local con revisión y hash esperados
- **THEN** el sistema la carga sin requerir red

#### Scenario: Dataset no verificado
- **WHEN** falta la revisión, el hash o su verificación falla
- **THEN** el sistema rechaza la ejecución y explica cómo preparar el dataset

### Requirement: Evaluación local compatible de GPQA Diamond
El sistema MUST calificar las 198 preguntas GPQA Diamond como alternativas de
cuatro opciones con una permutación y plantilla declaradas en el resultado.

#### Scenario: Respuesta correcta normalizada
- **WHEN** el agente entrega una alternativa única A–D
- **THEN** el sistema la compara con la respuesta permutada y reporta accuracy

#### Scenario: Resultado de GPQA publicado
- **WHEN** finaliza una ejecución GPQA
- **THEN** el resultado se etiqueta como aproximación local y no como leaderboard oficial

### Requirement: Integración fiel de ClawEval
El sistema MUST delegar las ejecuciones que declaren `Pass^3` al harness publicado
de ClawEval con Docker, fixtures, tres trials y sus graders.

#### Scenario: Precondiciones de ClawEval ausentes
- **WHEN** Docker, fixtures o credenciales requeridas no están disponibles
- **THEN** el sistema marca la ejecución como bloqueada sin producir score

#### Scenario: Ejecución oficial delegada
- **WHEN** todas las precondiciones están disponibles y se solicitan tres trials
- **THEN** el sistema ejecuta el harness publicado y conserva sus artefactos de score

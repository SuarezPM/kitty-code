## ADDED Requirements

### Requirement: Presupuesto de resultados y masking por perfil
Cada perfil MUST definir el máximo de resultado visible y la política de masking
de resultados antiguos para preservar contexto útil.

#### Scenario: Perfil tiny con salida extensa
- **WHEN** una herramienta devuelve más contenido que permite el perfil tiny
- **THEN** el agente recibe un extracto acotado y una referencia local al contenido completo redactado

### Requirement: Superficie mínima durante recuperación
Los perfiles pequeños MUST poder reducir temporalmente las herramientas
anunciadas tras errores de schema repetidos.

#### Scenario: Breaker abierto en perfil tiny
- **WHEN** se abre el breaker de schema
- **THEN** el siguiente intento expone solo la superficie de lectura permitida por el perfil

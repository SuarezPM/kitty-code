## Purpose

Endurecimiento de seguridad del harness: invariante write-vs-edit, detección de writes vía redirects de shell y permission-gate que nombra el remedio.

## ADDED Requirements

### Requirement: Invariante write-vs-edit

`Write` rechaza archivos existentes y fuerza `Edit`; `Edit` requiere que el archivo haya sido leído en la sesión actual (coincide con el guard read-before-edit existente, scoped a sesión).

#### Scenario: Write sobre archivo existente
- **GIVEN** un archivo que ya existe en disco
- **WHEN** el agente llama a `Write` sobre ese path
- **THEN** la tool se rechaza con un mensaje que indica usar `Edit` (tras `Read`)

#### Scenario: Edit sin Read previo
- **GIVEN** un archivo no leído en la sesión actual
- **WHEN** el agente llama a `Edit` sobre ese path
- **THEN** la tool se rechaza con un mensaje que indica leer el archivo primero

#### Scenario: Edit tras Read en un turno anterior
- **GIVEN** un archivo leído en un turno anterior de la misma sesión
- **WHEN** el agente llama a `Edit` sobre ese path
- **THEN** la tool se permite (el guard es scoped a sesión, no a turno)

### Requirement: Detección de writes vía redirects de shell

La guarda trata las escrituras por redirect (`>`, `>>`, `tee`) como writes y les aplica las mismas reglas que a Write/Edit.

#### Scenario: Redirect de escritura a archivo del proyecto
- **GIVEN** un comando como `cat > src/main.rs` o `echo x >> file`
- **WHEN** la guarda evalúa el comando
- **THEN** se aplican las reglas de write (archivo existente → rechazo con remedio)

#### Scenario: Redirect benigno a /dev/null
- **GIVEN** un comando como `cmd > /dev/null 2>&1`
- **WHEN** la guarda evalúa el comando
- **THEN** se permite (no es una escritura a un archivo del proyecto)

### Requirement: Permission-gate que nombra el remedio

Cuando un comando se rechaza, el mensaje nombra el segmento ofensor y el remedio accionable, no solo "refused".

#### Scenario: Comando rechazado con remedio
- **GIVEN** un comando con un segmento peligroso (p.ej. write-redirect a un archivo existente)
- **WHEN** la guarda lo rechaza
- **THEN** el mensaje identifica el segmento y sugiere la acción correcta (p.ej. "usa Edit tras Read")

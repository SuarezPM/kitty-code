## Purpose

Registra evidencia diagnóstica local de los turnos y benchmarks sin persistir
prompts, secretos o datos sensibles de los usuarios.

## ADDED Requirements

### Requirement: Métricas estructuradas por ejecución
El sistema MUST registrar de forma local el modelo, perfil, latencias, tokens,
uso de herramientas, compactaciones, reintentos y veredicto de cada ejecución.

#### Scenario: Turno terminado
- **WHEN** un turno termina por éxito, error, timeout o cancelación
- **THEN** se añade un evento estructurado con su causa y métricas disponibles

### Requirement: Persistencia acotada y segura
El sistema MUST rotar sus registros y MUST excluir prompts y valores secretos.

#### Scenario: Registro supera el límite configurado
- **WHEN** el archivo de telemetría supera su presupuesto local
- **THEN** el sistema rota el archivo conservando solo el número configurado de rotaciones

#### Scenario: Salida con secreto
- **WHEN** una salida contiene un valor que coincide con un secreto reconocido
- **THEN** la telemetría guarda únicamente su versión redactada

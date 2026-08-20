# benchmarking Specification

## Purpose
Harness de bench reproducible con veredicto exit-code y gate go/no-go del piso sub-2B.

## Requirements

### Requirement: Veredicto de bench con exit-code

El harness de bench compara el run actual contra un baseline guardado y emite un código de salida CI: 0 = mejoró, 1 = regresó, 2 = ruido (sin cambio significativo).

#### Scenario: Mejora detectada
- **GIVEN** un baseline guardado y un run nuevo con mejor pass rate
- **WHEN** el harness compara ambos
- **THEN** el proceso sale con código 0 (mejoró)

#### Scenario: Regresión detectada
- **GIVEN** un baseline guardado y un run nuevo con peor pass rate
- **WHEN** el harness compara ambos
- **THEN** el proceso sale con código 1 (regresó)

### Requirement: Gate go/no-go del piso sub-2B

Un benchmark dedicado corre la suite homemade con un modelo sub-2B y reporta si el single-file success supera el umbral (50%). Si no lo supera, el reporte marca el tier como "re-encuadrar" y el proceso sale con un código distinto de éxito.

#### Scenario: Piso superado
- **GIVEN** un modelo sub-2B y la suite homemade
- **WHEN** se ejecuta el benchmark del piso
- **THEN** si el single-file success ≥50%, el reporte valida el tier sub-2B

#### Scenario: Piso no superado
- **GIVEN** un modelo sub-2B y la suite homemade
- **WHEN** se ejecuta el benchmark del piso
- **THEN** si el single-file success <50%, el reporte marca "re-encuadrar el tier" y el proceso sale con código de fallo

### Requirement: Método de grading explícito en cada métrica

Toda métrica de éxito reportada fija el método de grading (string-match homemade vs. unit-test oficial) para evitar números vanity.

#### Scenario: Métrica con grading etiquetado
- **GIVEN** un reporte de bench con métricas
- **WHEN** se publica el reporte
- **THEN** cada métrica indica su método de grading y si es comparable a un benchmark oficial

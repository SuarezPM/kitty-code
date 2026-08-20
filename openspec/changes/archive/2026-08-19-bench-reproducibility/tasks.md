## 1. Bench veredicto exit-code

- [ ] 1.1 Implementar comparación contra baseline guardado en `benches/` con códigos 0/1/2
- [ ] 1.2 Etiquetar cada métrica con método de grading (homemade vs unit-test) en el reporte
- [ ] 1.3 Integrar veredicto en `crates/kitty-cli/src/main.rs` (subcomando bench)

## 2. Gate go/no-go piso sub-2B

- [ ] 2.1 Implementar bench dedicado que corre KittyQuiz-225 / KittyShell-89 con modelo sub-2B
- [ ] 2.2 Reportar single-file success y marcar tier como "re-encuadrar" si <50% (exit != 0)
- [ ] 2.3 Persistir baseline del gate y documentar umbral en CI

## 3. Verificación

- [ ] 3.1 Validar `openspec validate --all` y escenarios de spec (mejora/regresión/ruido, piso superado/no superado)

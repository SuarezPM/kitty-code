## 1. 2-stage tool routing

- [ ] 1.1 Implementar clasificador determinista (regex/keywords, zero-token) que seleccione categoría (read/write/search/run/plan) y filtrar schemas a solo la ganadora
- [ ] 1.2 Implementar affirmation guard ("yes"/"ok" no descarta write tools) y fallback a todas si ninguna categoría matchea
- [ ] 1.3 Cablear routing por perfil (activo en `tiny`/`medium`, interactúa con grammar-enforcement)

## 2. Dedup y parser robusto

- [ ] 2.1 Cambiar dedup en `quality_monitor.rs` a `name + hash(args)` con ventana persistente entre turnos
- [ ] 2.2 Hacer parser de tool-calls robusto (multi-formato JSON/YAML/XML/fenced) sin drops silenciosos; emitir correctivo con formato esperado

## 3. Contract/DoD done-guard

- [ ] 3.1 Implementar contrato declarativo con aserciones (archivo existe y contiene Y) y verificación en el harness
- [ ] 3.2 Rechazar finalización si aserciones no se cumplen, devolviendo fallo al agente; aceptar si se cumplen

## 4. Verificación

- [ ] 4.1 Tests de routing, affirmation guard, dedup por args, contrato satisfecho/insatisfecho y parser correctivo

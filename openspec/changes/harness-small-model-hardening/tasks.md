## 1. Aider Polyglot Loader (P5)

- [ ] 1.1 Crear módulo `crates/kitty-cli/src/polyglot.rs` con parser de ejercicios Exercism
- [ ] 1.2 Implementar extracción de prompts desde `.meta/.docs/problem.md` o `.meta/description.md`
- [ ] 1.3 Implementar detección de test runner por lenguaje (pytest, go test, cargo test, node, javac)
- [ ] 1.4 Implementar ejecución de tests y determinación pass/fail por exit code
- [ ] 1.5 Integrar con `cargo run -- bench polyglot` en main.rs
- [ ] 1.6 Generar resultados en formato comparable al leaderboard (pass_rate_1, pass_rate_2)
- [ ] 1.7 Escribir pruebas de carga de ejercicios, extracción de prompts y selección de test runner

## 2. JSON Repair Pipeline (P1)

- [ ] 2.1 Crear módulo `crates/kitty-harness/src/json_repair.rs`
- [ ] 2.2 Implementar `repair_json()` — fix control chars, invalid escapes
- [ ] 2.3 Implementar `coerce_with_schema()` — type coercion string→int/bool, union types, nested objects
- [ ] 2.4 Implementar normalización de null para optional properties
- [ ] 2.5 Integrar en pipeline de parsing de tool calls en agent.rs
- [ ] 2.6 Escribir pruebas de repair: control chars, escapes, coercion, unions, nested, optional null

## 3. Structured Compaction (P2)

- [ ] 3.1 Crear módulo `crates/kitty-harness/src/compaction.rs`
- [ ] 3.2 Implementar formato estructurado: Goal/Constraints/Progress/Decisions/Next Steps/Critical Context
- [ ] 3.3 Implementar file operation tracking (read-files, modified-files)
- [ ] 3.4 Implementar iterative summary updates (preserve existing context)
- [ ] 3.5 Integrar con ContextBudget existente y CompactionSnapshot de F4.2
- [ ] 3.6 Escribir pruebas de formato, file tracking, iterative updates

## 4. Progressive Skills (P3)

- [ ] 4.1 Crear módulo `crates/kitty-harness/src/skills.rs`
- [ ] 4.2 Implementar discovery de skills desde `.kitty/skills/` y `~/.kitty/skills/`
- [ ] 4.3 Implementar formateo XML para system prompt (name+description+location)
- [ ] 4.4 Implementar deduplicación por nombre (first wins, warning on collision)
- [ ] 4.5 Integrar con system prompt builder
- [ ] 4.6 Escribir pruebas de discovery, formateo, deduplicación

## 5. Tool-Gated Prompt Sections (P4)

- [ ] 5.1 Implementar filtrado de secciones de prompt por tools activos
- [ ] 5.2 Integrar con ToolRouter existente en router.rs
- [ ] 5.3 Mover guidelines de herramientas a secciones condicionales
- [ ] 5.4 Escribir pruebas de inclusión/exclusión por tool activo

## 6. Multi-Tier Retry (P6)

- [ ] 6.1 Crear módulo `crates/kitty-harness/src/retry.rs` (o extender error.rs)
- [ ] 6.2 Implementar clasificación de errores: transient, permanent, overflow, rate-limit, auth, quota
- [ ] 6.3 Implementar Tier 1: provider HTTP retry con exponential backoff + jitter
- [ ] 6.4 Implementar Tier 2: application retry con 60+ patrones de error
- [ ] 6.5 Integrar con retry existente de F3.3
- [ ] 6.6 Escribir pruebas de clasificación para cada tier

## 7. Overflow Detection (P7)

- [ ] 7.1 Crear módulo `crates/kitty-harness/src/overflow.rs`
- [ ] 7.2 Implementar 25+ regex patterns para overflow de providers (Ollama, llama.cpp, OpenAI, etc.)
- [ ] 7.3 Implementar detección de overflow silencioso (zero output tokens)
- [ ] 7.4 Implementar auto-compact + retry once con flag de prevención de loops
- [ ] 7.5 Implementar filtrado de overflow stale (timestamp comparison)
- [ ] 7.6 Escribir pruebas de cada patrón, overflow silencioso, stale filtering

## 8. Harness Comparison (P8)

- [ ] 8.1 Crear módulo `crates/kitty-harness/src/comparison.rs`
- [ ] 8.2 Implementar comparación pareada: correctness lift, wins/ties/losses
- [ ] 8.3 Implementar métricas: token delta, latency delta, cost delta
- [ ] 8.4 Implementar formato de salida y persistencia de artefactos
- [ ] 8.5 Escribir pruebas de cálculo, formato, persistencia

## 9. Integración y Verificación

- [ ] 9.1 Integrar todos los módulos en agent.rs, tools.rs, main.rs
- [ ] 9.2 Ejecutar `cargo check --workspace`
- [ ] 9.3 Ejecutar `cargo clippy --workspace -- -D warnings`
- [ ] 9.4 Ejecutar `cargo test --workspace`
- [ ] 9.5 Ejecutar polyglot l15 smoke test con el loader real
- [ ] 9.6 Actualizar tasks.md del change `harden-local-harness-evaluation`

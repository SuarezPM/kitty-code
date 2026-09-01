## 1. Fundamentos de evaluación

- [x] 1.1 Corregir los dos avisos de Clippy que protegen la serialización de tests de entorno.
- [x] 1.2 Implementar metadatos y preflight offline-first para datasets con revisión, hash y licencia.
- [x] 1.3 Añadir el loader GPQA Diamond con permutación determinista, respuesta A–D y resultado etiquetado como aproximación local.
- [x] 1.4 Escribir pruebas unitarias de parsing, hash, permutación y grading GPQA sin ejecutar la batería completa.

## 2. Integración de ClawEval

- [x] 2.1 Añadir un preflight que compruebe CLI, Docker, fixtures, configuración y credenciales de ClawEval.
- [x] 2.2 Implementar el wrapper de ejecución y la extracción de artefactos del harness publicado.
- [x] 2.3 Distinguir estados bloqueado, ejecutado y resultado oficial en la salida y documentación.
- [x] 2.4 Escribir pruebas de preflight y parsing de resultados sin ejecutar la batería completa.

## 3. Robustez y telemetría del harness

- [x] 3.1 Añadir cancelación y terminación/recolección de grupos de procesos para herramientas Bash.
- [x] 3.2 Redactar secretos antes de spills, errores, telemetría y resultados de herramientas.
- [x] 3.3 Clasificar errores de runtime y aplicar reintentos acotados solo a fallos transitorios de chat.
- [x] 3.4 Registrar telemetría JSONL local rotada y sin prompts, con métricas por turno y benchmark.
- [x] 3.5 Escribir pruebas de cancelación, redacción, clasificación y rotación sin ejecutar la batería completa.

## 4. Eficiencia para perfiles pequeños

- [x] 4.1 Añadir caps de output y masking de resultados antiguos configurados por perfil.
- [x] 4.2 Crear snapshot operativo acotado antes de compactar contexto.
- [x] 4.3 Añadir breaker de schema y reducción temporal de superficie de herramientas.
- [x] 4.4 Escribir pruebas del breaker, masking y snapshots sin ejecutar la batería completa.

## 5. Verificación final y medición

- [x] 5.1 Ejecutar una única batería final de formato, Clippy, tests del workspace y diagnósticos Rust.
- [x] 5.2 Crear manifiestos pre-registrados y ejecutar GPQA cuando el dataset gated esté disponible.
- [x] 5.3 Ejecutar ClawEval solo si el preflight oficial queda satisfecho; de otro modo registrar bloqueo honesto.
- [ ] 5.4 Comprobar disco antes de cada descarga/benchmark pesado y ejecutar las baterías aprobadas con telemetría.
- [ ] 5.5 Actualizar AUDIT, STRATEGY, codemaps y el estado de OpenSpec con evidencia y limitaciones.

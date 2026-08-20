## 1. Invariante write-vs-edit

- [ ] 1.1 Hacer que `Write` rechace archivos existentes (forzar `Edit`) y que `Edit` exija `Read` previo en la sesión
- [ ] 1.2 Añadir tracking de archivos leídos por sesión en `tools.rs`

## 2. Detección shell-redirect y permission-gate

- [ ] 2.1 Detectar writes vía redirects (`>`, `>>`, `tee`) en `guard.rs` y tratarlas como writes (misma regla Write/Edit)
- [ ] 2.2 Evaluar permission-gate segment-wise (cada `;`/`&&`/`|`/`||`) y nombrar segmento ofensor + remedio accionable
- [ ] 2.3 Permitir redirects benignos (`> /dev/null 2>&1`, `tar -czf`) sin falso positivo

## 3. Verificación

- [ ] 3.1 Tests de regresión por cada caso (F1.2 + redirects) incluyendo Write/Edit/redirect benigno y mensajes con remedio

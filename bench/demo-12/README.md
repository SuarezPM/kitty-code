# demo-12 — suite fija versionada (W8, SDD estado-arte)

Comparativa propia kitty vs PI/opencode/little-coder: 12 ejercicios fijos
con protocolo de justicia embebido en `demo-12.json` (objeto `fairness`).

## Contenido

- `demo-12.json` v1.0.0: 12 ejercicios + `fairness` (modelo pineado al
  correr, mismo backend/turnos/timeouts, k≥3, `gates_off` declarado,
  exclusividad de serie, asimetrías declaradas, veredictos, estadística).
- `validate.py`: validador stdlib (`python3 bench/demo-12/validate.py`).
- 8 ejercicios portados de `tb2-homemade` (mismos ids/prompts, checks
  adaptados al esquema `JsonExercise` de un solo predicado) + 4 nuevos
  (javascript, rust, json, bash-words).

## Protocolo de justicia (resumen; el JSON manda)

Mismo modelo+digest/backend/turnos/timeouts ambos lados; k≥3 reps;
1 modelo residente con `ollama stop` entre reps; `gates_off`
(`autorun_hops: 0`, sin knowledge/memory, router en pass-through con
`effort_off_in_bench` por celda); p50/p99 sobre wall real; manifiesto por
corrida. Medición GPU/serie en ventana serial (HEAD congelado, árbol
limpio, 1 celda a la vez).

## TRAMPA CONOCIDA: `--exercises` no ejecuta demo-12

`kitty bench --exercises <json>` **no filtra ni carga la suite polyglot**
(`crates/kitty-cli/src/bench/mod.rs`: `suite == "polyglot"` retorna al
runner oficial ANTES del manejo de `--exercises`), y además espera un
array pelado (`Vec<JsonExercise>`), no el envelope versionado con
`fairness` de este fichero. demo-12 necesita runner propio (lane cli):
parsear el envelope, fijar modelo+digest+versiones en el manifiesto,
correr los 12 ejercicios k≥3 veces por agente en serie exclusiva.
No usar `--exercises` como atajo: pierde versión, justicia y grado.

## Checks de un predicado

El esquema `JsonExercise` admite UN check por ejercicio (polyglot
`check_str`+`check_file` XOR tb2 `check_type`+`check_path`). Respecto a
`tb2-homemade` hay debilitamientos documentados: `demo-bash-count`,
`demo-sh-find-modified`, `demo-sh-backup`, `demo-sh-words` solo exigen el
bit ejecutable (el `&& contains` original no cabe en el esquema);
`demo-gh-actions` exige `cargo test` (el original aceptaba test OR build).
La lane cli puede extender el esquema; hasta entonces estos son los
veredictos oficiales de la suite.

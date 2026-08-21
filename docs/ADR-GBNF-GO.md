# ADR-GBNF-GO — Verdict GO para Grammar Enforcement GBNF

**Status:** Accepted — GO
**Date:** 2026-08-21
**Deciders:** kitty-harness / kitty-runtime
**Scope:** `crates/kitty-runtime/src/grammar.rs`, `crates/kitty-runtime/src/client.rs`, `Cargo.toml`
**Relacionado:** `docs/STRATEGY.md` F6.4 ADR-005, `crates/kitty-runtime/src/grammar.rs:6` SPIKE defer, `grammar::tests::test_gbnf_spike_documented`

---

## Verdict

**GO.** Se aprueba pasar de spike post-hoc a enforcement real con gramatica GBNF.

La condicion GO de `docs/STRATEGY.md` F6.4 ADR-005 dice: implementar adapter real solo cuando el tier sub-2B requiera mas de 95% de tool-call validity y el filtrado post-hoc no alcance. El probe tiny demuestra que esa condicion ya se cumple.

## Evidencia

### Probe tiny — /tmp/tiny_probe_validity.json

Fuente primaria: `/tmp/tiny_probe_validity.json` (derivado de `/tmp/tiny_probe.json`).

```
model: smollm2:1.7b
profile: tiny
total_tool_calls: 100
valid_tool_calls: 60
invalid_tool_calls: 40
valid_pct: 60.0
invalid_pct: 40.0
gate_threshold: 95.0
gate_status: BELOW_GATE_NEEDS_GRAMMAR
below_gate: true
method: GrammarConfig::validate (serde_json + required fields) replication in Python
bench_summary: 50 tasks, 18 passed, 36.0%
```

El metodo replica `GrammarConfig::validate` de `crates/kitty-runtime/src/grammar.rs:79`: `serde_json` parse mas chequeo de `required` fields por tool. No es un estimado manual. Es la misma logica que usa `GrammarClient` en modo strict.

**Lectura:** 60.0% es menor que 95.0%. El gate falla por 35 puntos. Ejemplos invalidos observados incluyen `{"name":"Bash","arguments":{}}` sin `command`, `Edit` sin `path`/`hash`, `Write` con `arguments` no objeto, y forma `{"function":{"name":"Write"...}}` truncada. Todos son tool calls que el spike post-hoc solo filtra despues de generar, no evita. En tiny esto paga un turno completo perdido por generacion invalida.

Cita STRATEGY F6.4: target mayor que 95% para sub-2B, plan T21 SC-GBNF-VALIDITY, roadmap P1 Grammar enforcement 3 a 5 dias. `docs/STRATEGY.md:104` dice Accepted spike defer. `docs/STRATEGY.md:117` dice condicion GO es `bench polyglot` con `ModelProfile::tiny` validity menor que 95% tras tuning de prompt y router. El probe cumple esa condicion con smollm2:1.7b en tiny.

### Evidencia de spike documentado

`crates/kitty-runtime/src/grammar.rs:6` header documenta `SPIKE defer: no llama.cpp bind — validated post-hoc via serde_json + required-field check. Real enforcement requires llama-cpp-2 crate`. El test `grammar::tests::test_gbnf_spike_documented` verifica que ese header existe:

```
cargo test -p kitty-runtime --lib -- test_gbnf_spike_documented
test grammar::tests::test_gbnf_spike_documented ... ok
```

El spike genera `GrammarConfig::to_gbnf()` para `{"name": enum, "arguments": object}` con reglas `object`, `string`, `value`, `array`, `number`, `ws`, pero solo valida con `validate()` post-hoc. `ChatClient` no tiene campo grammar. `OllamaClient::chat` no envia GBNF.

### Cargo test verde

Workspace completo verde al momento del veredicto:

```
cargo test --workspace
kitty-runtime 27 passed
kitty-harness 243 passed
kitty-memory 18 passed
kitty-cli 17 passed
total 305 tests, 0 failed
```

Incluye `grammar::tests::*` 7 tests: `to_gbnf_contains_tool_names`, `to_gbnf_empty_returns_null`, `validate_accepts_valid_tool_call`, `validate_rejects_invalid_tool`, `validate_accepts_nested_function_shape`, `grammar_client_strict_filters_invalid`, `grammar_client_non_strict_passes_through`, mas `test_gbnf_spike_documented`.

---

## Razonamiento: Option A vs Option B

F6.4 ya compara las alternativas. Esta ADR elige entre ellas.

### Option A — llama-cpp-2 in-process

Via `llama-cpp-2` crate `0.1.154`, `LlamaSampler::grammar` o `LlamaSampler::grammar_lazy` con triggers, `json_schema_to_grammar` para convertir schema a GBNF, integrado via `ChatClient` como `LlamaCppClient: ChatClient`.

- Esfuerzo: 3 a 5 dias (F6.4 est.)
- Validity: mayor que 95% decode-time, tokens restringidos durante muestreo
- Pro: control maximo, sin daemon, no fragmenta runtime, GBNF generado por `to_gbnf()` se usa directo como sampler input, interfaz `GrammarConfig` + `GrammarClient<C: ChatClient>` ya es el seam
- Contra: build FFI, carga GGUF Q4, ciclo de vida sampler, cuidado `Send/Sync` para `ChatClient`
- Recomendado cuando GO por STRATEGY F6.4

### Option B — llama-server HTTP daemon

Modo daemon HTTP donde `tools` auto genera lazy grammar nativo (PR ggml-org/llama.cpp #9639), tool schemas convertidos a grammar perezosa disparada solo al emitir prefijo tool-call.

- Esfuerzo: 1 a 2 dias
- Validity: mayor que 95% server-side
- Pro: ops simple, sin bindgen Rust, sin FFI
- Contra: requiere ciclo de vida daemon, fragmenta runtime que hoy estandariza en Ollama HTTP, y tiene trampa critica: grammar custom mas `tools` son exclusivos, `server-common.cpp:1289` retorna HTTP 400 si se envian juntos. `grammar_lazy` con triggers resuelve para tool calling nativo, pero exige no enviar `grammar` string y `tools` array juntos en path HTTP
- STRATEGY F6.4 dice rechazar B como default porque ya se usa Ollama HTTP. Viable como fallback si el build in-process es muy pesado

### Decision entre A y B

**Se elige A como via principal.** La tesis sub-2B necesita control decode-time sin segundo daemon. El tier tiny ya corre con ventana 4K y 5 iteraciones, cualquier roundtrip HTTP extra o daemon extra suma latencia y ops. A mantiene un solo proceso Rust con sampler embebido. B queda documentado como fallback si A bloquea por build o GGUF, pero no es la via default.

La trampa `grammar` + `tools` juntos igual HTTP 400 aplica a ambos si se usa path HTTP. En A no aplica porque el sampler es in-process, no HTTP. Si se mantiene Ollama HTTP para modelos grandes y se anade LlamaCppClient solo para tiny, la interfaz `ChatClient` abstrae el runtime sin mezclar `grammar` y `tools` en la misma request.

---

## Consequences

- Sin bind ahora: esta ADR no anade dependencia. `Cargo.toml` no suma `llama-cpp-2 = "0.1.154"` aun. El spike sigue como unico enforcement. Esta ADR es el doc GO. El bind se hace en task 21/22.
- Cuando se ejecute task 21, agregar `llama-cpp-2` a `[workspace.dependencies]` y `kitty-runtime`, implementar `LlamaCppClient: ChatClient` con `LlamaSampler::grammar` o `grammar_lazy` + triggers, y pasar `to_gbnf()` como input sampler.
- Coste build: `llama.cpp` build mas carga GGUF in-process, manejo FFI y sampler lifecycle a mantener. Mitigacion: feature gate `grammar` opt-in para no penalizar CI sin build.
- No cambia comportamiento tiny hoy: `GrammarConfig::validate` sigue siendo gate post-hoc hasta que el adapter real aterrice. El riesgo de cola es que modelos que emiten JSON invalido siguen costando generacion completa hasta entonces.
- Interfaz preservada: `GrammarConfig` + `GrammarClient` es el seam. `to_gbnf()` pasa a ser input sampler cuando GO se materializa.

---

## Verificacion

1. Probe existe y es menor que gate: `cat /tmp/tiny_probe_validity.json` muestra `valid_pct 60.0` y `gate_status BELOW_GATE_NEEDS_GRAMMAR`.
2. Spike documentado: `cargo test -p kitty-runtime --lib -- test_gbnf_spike_documented` verde.
3. Workspace verde: `cargo test --workspace` 305 passed, 0 failed.
4. Evidencia guardada: `.omo/evidence/f6-remaining-tracks-opt-order/task-20.log` con outputs de probe y tests.

---

## Referencias

- `docs/STRATEGY.md` F6.4 ADR-005 — defer spike, condicion GO, alternativas A/B/C
- `crates/kitty-runtime/src/grammar.rs:6-8` — header SPIKE defer
- `crates/kitty-runtime/src/grammar.rs:44,79,146` — `to_gbnf`, `validate`, `GrammarClient`
- `crates/kitty-runtime/src/grammar.rs:354` — `test_gbnf_spike_documented`
- `ggml-org/llama.cpp` `grammars/README.md` — spec GBNF, `json_schema_to_grammar`
- `llama-cpp-2` crate `0.1.154` — `LlamaSampler::grammar`, `grammar_lazy`, `json_schema_to_grammar`
- PR `ggml-org/llama.cpp#9639` — lazy grammars
- `server-common.cpp:1289` — custom grammar mas tools exclusivo HTTP 400
- `/tmp/tiny_probe_validity.json` y `/tmp/tiny_probe.json` — probe tiny 60%

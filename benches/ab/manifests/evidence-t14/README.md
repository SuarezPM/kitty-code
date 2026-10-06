# T1.4 evidence — tokio-console vs kitty-cli MCP workload (SDD-82-10 Q2)

Captura directa (no reconstruida de memoria): `tokio-console 0.1.14` CONNECTED a
`console-subscriber 0.5.0` (console-api 0.9.0) en `127.0.0.1:6669`, binario
compilado con `RUSTFLAGS="--cfg tokio_unstable"` (solo sesión). Workload:
Read/Grep/Edit/Glob vía `kitty mcp` stdio → `ToolRegistry::execute`.

## Frame reconstruido (grilla 40×160, sort Total▿)

- `Tasks (82) ▶ Running (1) ⏸ Idle (0)`, Warn vacío en filas visibles.
- Tool-path (`tokio::fs::read_to_string`, Read/Grep/Glob): 10+ filas,
  Total máx **676µs**, Busy máx **39µs**, Sched ≤8µs, Polls=2.
- Parked internals (`tokio::io::blocking::Blocking`, blocking.rs:76:46):
  **14× Total=Busy=500ms** (2 polls) + **1× 449ms Running** (1 poll),
  Sched ~30µs, Idle ~60-90µs. Son workers del blocking-pool aparcados, NO
  trabajo de tools — se declaran, no se cuentan como violación.

## Churn

60/60 tools ok (Read/Grep/Glob rotados ~0.5s, rt 1-3ms) durante la ventana de
captura; puerto 6669 abierto antes, durante y después (`t14-churn.log`).

## Conclusión

**0 tasks del tool-path con poll>100ms** (esperado 0 tras W1) ✓.

## Huecos declarados

- Parked internals 500ms: artefacto del pool, fuera del AC (AC = tool-path).
- `id=4 old_string not found`: workload no-idempotente entre pasadas
  (la 1ª ya aplicó "beta two"→"beta TWO"); error funcional esperado, no flake.
- `workload_ok=False`: solo refleja el wart MCP-tracing-stdout (abajo);
  7/8 y 60/60 confirman que las tools ejecutaron.

## Wart MCP-tracing-stdout (causa, preexistente, fuera de T1.4)

`tracing_subscriber::fmt` escribe a **stdout** por defecto y `kitty mcp`
también emite JSON-RPC por stdout → 1 línea INFO (`kitty_harness::sandbox:
sandbox armed…`) contaminó la respuesta de `id=8 Bash` →
`INVALID RESP` en el driver (`t14-workload3.log`). Console-subscriber va por
gRPC (:6669), ajeno al problema. Fix propuesto (no aplicado, fuera de
ownership): `fmt::layer().with_writer(std::io::stderr)` o `.json()` a stderr
en la vía `mcp`.

## Ficheros y SHA256

- `t14-console-raw2.bin.gz` (102912B raw → gzip lossless; raw
  `aa8deb6c30094f3655e6d683e440e3737c1719ac8c0718e8890b1e93d5542cc1`)
- `t14-churn.log` (`a2788c7a…c44d`), `t14-workload3.log` (`fa0a8055…eb138`)
- Reconstrucción grilla: `/tmp/t14_frame.py` (fuera del repo, efímero).

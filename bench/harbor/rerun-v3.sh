#!/bin/bash
# rerun-v3.sh — validación l50 post-palancas Tier 3+4+7 (rama levers/tier-3-4-7)
#
# Secuencia autónoma:
#   1. Espera a que la cola nocturna v2 termine (QUEUE_V2_COMPLETE) — nunca toca
#      jobs/ ni reinicia Ollama mientras S2-S4 corren.
#   2. Merge --ff-only de levers/tier-3-4-7 → main.
#   3. Rebuild del binario del adapter (build-binary.sh).
#   4. Corrida polyglot l50 ornith:9b — protocolo first-N pre-registrado
#      (mismos flags que rerun-v2.sh para comparabilidad con S1-v2 17.39%).
#   5. Gate v3 (fixture-validado) + copia del result.json a evidencia.
#
# Estado en /tmp/harbor-night/status-v3 (NO toca status de la cola v2).
set -u -o pipefail
cd /home/thelinconx/Proyectos/Kitty-Code

# G2-F3: knobs KITTY_* del shell lanzador no deben filtrarse a la corrida medida
unset KITTY_MODEL_TURN_CAPS

EV=.omo/evidence/tier-3-4-7
LOG=$EV/rerun-v3.log
HB=/home/thelinconx/.local/bin/harbor
PP=/home/thelinconx/Proyectos/Kitty-Code/bench/harbor
ST=/tmp/harbor-night

source tests/fixtures/gate-v3.sh
mkdir -p "$ST" "$EV"

log(){ echo "=== [$(date -Is)] $*" >> "$LOG"; }
newest_result(){ local d; d=$(ls -t jobs/ 2>/dev/null | head -1); if [ -n "$d" ]; then find "jobs/$d" -name 'result.json' 2>/dev/null | head -1; fi; }

log "########## RERUN-V3 INICIO (post Tier 3+4+7) ##########"
env | grep '^KITTY_' >> "$LOG" || true   # G2-F3: provenancia de knobs efectivos
echo "WAITING_QUEUE_V2" > "$ST/status-v3"

# G2-F1: árbol sucio compilaría WIP ajeno al binario validado — abortar
if [ -n "$(git status --porcelain crates Cargo.toml Cargo.lock rust-toolchain.toml)" ]; then
  echo "DIRTY_TREE" > "$ST/status-v3"
  log "FATAL: arbol sucio (build compilaria WIP): $(git status --porcelain crates Cargo.toml Cargo.lock rust-toolchain.toml | tr '\n' '; ')"
  exit 1
fi

# 1. Espera cola nocturna v2 (G2-N2: watchdog 12h máx)
log "esperando QUEUE_V2_COMPLETE (poll 60s, max 12h)"
WAITED=0
while [ -f "$ST/status" ] && [ "$(cat "$ST/status")" != "QUEUE_V2_COMPLETE" ]; do
  sleep 60
  WAITED=$((WAITED+1))
  if [ "$WAITED" -ge 720 ]; then
    echo "WAIT_TIMEOUT" > "$ST/status-v3"
    log "FATAL: cola v2 no termino en 12h — abortando"
    exit 1
  fi
done
log "cola v2 completa — procediendo"

# G2-F4: re-validar tras la espera — el árbol puede ensuciarse durante la
# ventana de hasta 12h (pistas paralelas mutan este mismo working tree)
if [ -n "$(git status --porcelain crates Cargo.toml Cargo.lock rust-toolchain.toml)" ]; then
  echo "DIRTY_TREE" > "$ST/status-v3"
  log "FATAL: arbol se ensucio durante la espera: $(git status --porcelain crates Cargo.toml Cargo.lock rust-toolchain.toml | tr '\n' '; ')"
  exit 1
fi
# G2-F4: pin de contenido — lo merged debe CONTENER el estado revisado
# (777d21f: impl Tier 3+4+7 + guards G2) y solo puede diferir de él en el
# propio script de entrega (evolución del delivery no toca lo validado).
# NOTA: igualdad exacta de tip es imposible (el pin viviría dentro del
# commit que pinea — auto-referencia SHA); esto preserva la intención.
REVIEWED_SHA="777d21fe054fd45b3ad8ae3b0d534b57abe357dd"
if ! git merge-base --is-ancestor "$REVIEWED_SHA" levers/tier-3-4-7; then
  echo "BRANCH_MOVED" > "$ST/status-v3"
  log "FATAL: $REVIEWED_SHA ya no es ancestro de la rama (rebase/revert durante la espera)"
  exit 1
fi
STRAY=$(git diff --name-only "$REVIEWED_SHA" levers/tier-3-4-7 | grep -v '^bench/harbor/rerun-v3\.sh$' || true)
if [ -n "$STRAY" ]; then
  echo "BRANCH_MOVED" > "$ST/status-v3"
  log "FATAL: la rama difiere del estado revisado fuera del script de entrega: $(echo "$STRAY" | tr '\n' '; ')"
  exit 1
fi

# 2. Merge rama de palancas a main
if ! git checkout main >> "$LOG" 2>&1; then
  echo "MERGE_FAILED" > "$ST/status-v3"; log "FATAL: checkout main falló (¿working tree sucio?)"; exit 1
fi
if ! git merge --ff-only levers/tier-3-4-7 >> "$LOG" 2>&1; then
  echo "MERGE_FAILED" > "$ST/status-v3"; log "FATAL: merge --ff-only falló"; exit 1
fi
log "merge levers/tier-3-4-7 → main OK (full SHA: $(git rev-parse HEAD))"

# 3. Rebuild binario adapter
if ! sudo -n bash bench/harbor/adapters/kitty/build-binary.sh >> "$LOG" 2>&1; then
  echo "REBUILD_FAILED" > "$ST/status-v3"; log "FATAL: rebuild binario falló"; exit 1
fi
log "binario adapter rebuilt con palancas Tier 3+4+7"
sha256sum bench/harbor/adapters/kitty/bin/* >> "$LOG" 2>/dev/null || true  # G2-N3

# 4. Corrida de validación l50 ornith:9b (protocolo idéntico a S1-v2)
echo "RUNNING_V3_L50" > "$ST/status-v3"
PRE=$(newest_result)   # G2-F2: snapshot previo para detectar resultado stale
log "V3-L50 START model=ornith:9b dataset=polyglot limit=50 baseline_cmp=partial42:11.9,S1v2:17.39 pre_run_result='$PRE'"
sudo -n -E env PYTHONPATH="$PP" KITTY_BENCH_MODEL="ornith:9b" "$HB" run \
  -p bench/official/aider-harbor/aider-polyglot \
  -a adapters.kitty.agent:KittyAgent -l 50 -k 1 \
  --ae KITTY_SANDBOX=off --ae KITTY_OLLAMA_TIMEOUT_SECS=900 \
  --agent-timeout-multiplier 3 >> "$LOG" 2>&1
RC=$?
log "V3-L50 END rc=$RC"

# 5. Preservar resultado — SOLO si rc==0 Y el result.json es de ESTA corrida
#    (G2-F2: harbor elimina jobs/ previos; un run muerto antes de crear job dir
#    dejaría el result.json viejo de la cola v2 → falso V3_PASS)
RES=$(newest_result)
if [ "$RC" -eq 0 ] && [ -n "$RES" ] && [ "$RES" != "$PRE" ] && cp "$RES" "$EV/result-v3-l50-ornith9b.json" 2>>"$LOG"; then
  log "result.json preservado en $EV/result-v3-l50-ornith9b.json"
else
  RES=""
  log "WARN: resultado de esta corrida no disponible (rc=$RC RES='$RES' PRE='$PRE') — no se gatea ni preserva nada stale"
fi

# 6. Gate v3 (solo sobre resultado fresco de esta corrida)
if [ -n "$RES" ] && gate_v3 "$RES" "v3-l50-ornith9b-tier347"; then
  echo "V3_PASS" > "$ST/status-v3"
  log "GATE v3 PASS"
else
  echo "V3_FAIL" > "$ST/status-v3"
  log "GATE v3 FAIL"
fi

log "########## RERUN-V3 COMPLETO ##########"

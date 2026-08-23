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
echo "WAITING_QUEUE_V2" > "$ST/status-v3"

# 1. Espera cola nocturna v2
log "esperando QUEUE_V2_COMPLETE (poll 60s)"
while [ -f "$ST/status" ] && [ "$(cat "$ST/status")" != "QUEUE_V2_COMPLETE" ]; do
  sleep 60
done
log "cola v2 completa — procediendo"

# 2. Merge rama de palancas a main
if ! git checkout main >> "$LOG" 2>&1; then
  echo "MERGE_FAILED" > "$ST/status-v3"; log "FATAL: checkout main falló (¿working tree sucio?)"; exit 1
fi
if ! git merge --ff-only levers/tier-3-4-7 >> "$LOG" 2>&1; then
  echo "MERGE_FAILED" > "$ST/status-v3"; log "FATAL: merge --ff-only falló"; exit 1
fi
log "merge levers/tier-3-4-7 → main OK ($(git rev-parse --short HEAD))"

# 3. Rebuild binario adapter
if ! sudo -n bash bench/harbor/adapters/kitty/build-binary.sh >> "$LOG" 2>&1; then
  echo "REBUILD_FAILED" > "$ST/status-v3"; log "FATAL: rebuild binario falló"; exit 1
fi
log "binario adapter rebuilt con palancas Tier 3+4+7"

# 4. Corrida de validación l50 ornith:9b (protocolo idéntico a S1-v2)
echo "RUNNING_V3_L50" > "$ST/status-v3"
log "V3-L50 START model=ornith:9b dataset=polyglot limit=50 baseline_cmp=partial42:11.9,S1v2:17.39"
sudo -n -E env PYTHONPATH="$PP" KITTY_BENCH_MODEL="ornith:9b" "$HB" run \
  -p bench/official/aider-harbor/aider-polyglot \
  -a adapters.kitty.agent:KittyAgent -l 50 -k 1 \
  --ae KITTY_SANDBOX=off --ae KITTY_OLLAMA_TIMEOUT_SECS=900 \
  --agent-timeout-multiplier 3 >> "$LOG" 2>&1
RC=$?
log "V3-L50 END rc=$RC"

# 5. Preservar resultado (harbor elimina jobs/ previos al iniciar otra corrida)
RES=$(newest_result)
if [ -n "$RES" ] && cp "$RES" "$EV/result-v3-l50-ornith9b.json" 2>>"$LOG"; then
  log "result.json preservado en $EV/result-v3-l50-ornith9b.json"
else
  log "WARN: no se pudo preservar result.json (RES='$RES')"
fi

# 6. Gate v3
if [ -n "$RES" ] && gate_v3 "$RES" "v3-l50-ornith9b-tier347"; then
  echo "V3_PASS" > "$ST/status-v3"
  log "GATE v3 PASS"
else
  echo "V3_FAIL" > "$ST/status-v3"
  log "GATE v3 FAIL"
fi

log "########## RERUN-V3 COMPLETO ##########"

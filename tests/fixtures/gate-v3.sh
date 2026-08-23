#!/bin/bash
# gate-v3.sh — harbor result.json errored-trials gate (N-G1 finding 1 fix)
#
# Incident N1: gate v1 used [..|objects|select(has("n_total_trials"))][0] which grabbed
# a nested eval-level object instead of top-level .stats, producing vacuous gates.
# gate v2 (night-rerun.sh) read .stats correctly but trusted .stats.n_total_trials
# as the denominator, which is 0/unreliable in real result.json files.
#
# gate v3 contract (verified against real harbor 0.21.0 result.json schema):
#   root keys: finished_at,id,n_total_trials,started_at,stats,updated_at
#   .stats.n_completed_trials / .stats.n_errored_trials are the ONLY trustworthy counters.
#   - Parse completed/errored directly from top-level .stats
#   - Derive total outcomes = completed + errored (NEVER trust a stats scalar for totals)
#   - HARD FAIL schema-mismatch detector: n_total_trials exists but == 0 while outcomes > 0
#   - Must be validated against known-good AND known-bad fixtures before use
#     (run tests/fixtures/gate-selftest.sh)
#
# Usage: gate_v3 <result.json> <name>   ; exit 0 = PASS, 1 = FAIL
# Env:   GATE_ERROR_THRESHOLD_PCT (default 10)
set -u

GATE_ERROR_THRESHOLD_PCT="${GATE_ERROR_THRESHOLD_PCT:-10}"

gate_v3(){
  local f="$1" name="$2"
  if [ -z "$f" ] || [ ! -f "$f" ]; then
    echo "GATE $name: FAIL sin result.json ('$f')"
    return 1
  fi
  local row
  row=$(jq -c '{c:(.stats.n_completed_trials//0), e:(.stats.n_errored_trials//0), declared:(.n_total_trials//-1)}' "$f" 2>/dev/null)
  if [ -z "$row" ]; then
    echo "GATE $name: FAIL result.json ilegible/vacuo"
    return 1
  fi
  local c e declared outcomes
  c=$(printf '%s' "$row" | jq '.c')
  e=$(printf '%s' "$row" | jq '.e')
  declared=$(printf '%s' "$row" | jq '.declared')
  case "$c" in ''|*[!0-9]*) c=0;; esac
  case "$e" in ''|*[!0-9]*) e=0;; esac
  case "$declared" in ''|*[!0-9]*) declared=-1;; esac
  outcomes=$((c + e))
  # Schema mismatch detector: declared total present but zero while real outcomes exist.
  # (declared == -1 means the key is absent, which is NOT a mismatch)
  if [ "$declared" -eq 0 ] && [ "$outcomes" -gt 0 ]; then
    echo "GATE $name: FAIL schema-mismatch (n_total_trials=0 pero completed+errored=$outcomes)"
    return 1
  fi
  if [ "$outcomes" -le 0 ]; then
    echo "GATE $name: FAIL vacuo (completed=$c errored=$e outcomes=$outcomes)"
    return 1
  fi
  if awk -v e="$e" -v t="$outcomes" -v th="$GATE_ERROR_THRESHOLD_PCT" 'BEGIN{exit !(e*100 <= t*th)}'; then
    echo "GATE $name: PASS ($e/$outcomes errored <= ${GATE_ERROR_THRESHOLD_PCT}%)"
    return 0
  fi
  echo "GATE $name: FAIL env-comprometido ($e/$outcomes > ${GATE_ERROR_THRESHOLD_PCT}%)"
  return 1
}

# Allow sourcing without execution side-effects:
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  gate_v3 "${1:-}" "${2:-gate-v3}"
fi

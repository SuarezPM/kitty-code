#!/usr/bin/env bash
# Extrae métricas por request del log de ollama entre dos timestamps
# Uso: extract_ollama_speed.sh "2026-08-17 00:10" "2026-08-17 00:40" [etiqueta]
set -euo pipefail
SINCE="$1"
UNTIL="${2:-now}"
LABEL="${3:-run}"
OUT="/tmp/opencode/speed_${LABEL}.txt"
sudo journalctl -u ollama --since "$SINCE" --until "$UNTIL" 2>/dev/null \
  | grep -E "print_timing" \
  | grep -oE "(prompt eval time|eval time) =[^)]*" \
  | sed 's/^ *//' > "$OUT"
echo "requests capturados: $(grep -c "eval time" "$OUT" || true)"
echo "--- decode (tokens/s por request) ---"
grep "eval time" "$OUT" | grep -oE "[0-9.]+ tokens per second$" | sort -n | awk '{a[NR]=$1; s+=$1} END {printf "min=%.2f median=%.2f max=%.2f mean=%.2f (n=%d)\n", a[1], a[int(NR/2)+1], a[NR], s/NR, NR}'
echo "--- prefill (tokens/s) ---"
grep "prompt eval time" "$OUT" | grep -oE "[0-9.]+ tokens per second$" | sort -n | awk '{a[NR]=$1; s+=$1} END {printf "min=%.2f median=%.2f max=%.2f mean=%.2f (n=%d)\n", a[1], a[int(NR/2)+1], a[NR], s/NR, NR}'
echo "--- total tokens generados ---"
grep "eval time" "$OUT" | grep -oE "/ [0-9]+ tokens" | grep -oE "[0-9]+" | awk '{s+=$1} END {print s}'
echo "(log: $OUT)"
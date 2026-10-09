#!/usr/bin/env bash
# Serie VRAM W6 — UNA celda cada vez, 1 modelo residente, ollama stop entre series.
# Uso: bash bench/gpu-ventana-2026-10-09/measure-vram.sh [modelo...]
# Si no se pasan modelos, corre la matriz completa en orden ascendente de tamaño.
set -euo pipefail

MODELS_DEFAULT=(
  "smollm2:1.7b"
  "minicpm5:2b"
  "alibayram/smollm3:latest"
  "granite4:3b"
  "qwen3:4b"
  "qwen3.5:4b"
  "ornith:9b"
  "ornith:9b-opt"
  "ornith-1.5:9b"
  "qwen38-27b:latest"
)
MODELS=("${@:-${MODELS_DEFAULT[@]}}")

OUT_DIR="bench/gpu-ventana-2026-10-09"
mkdir -p "$OUT_DIR"
RAW="$OUT_DIR/vram-raw-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "== Serie VRAM W6 $(date -u +%FT%TZ) ==" | tee "$RAW"
echo "HEAD $(git rev-parse HEAD)" | tee -a "$RAW"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv | tee -a "$RAW"

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '
}

for m in "${MODELS[@]}"; do
  echo "" | tee -a "$RAW"
  echo "===== CELDA $m =====" | tee -a "$RAW"
  # Disciplina serial: nada residente antes de cargar
  echo "--- ollama ps (pre, debe estar vacío salvo esta celda) ---" | tee -a "$RAW"
  ollama ps 2>&1 | tee -a "$RAW"
  base=$(gpu_used_mib)
  echo "BASELINE_GPU_MiB=$base" | tee -a "$RAW"
  ollama show "$m" 2>&1 | head -12 | tee -a "$RAW"
  echo "--- carga + 1 inferencia corta ---" | tee -a "$RAW"
  if timeout 300 ollama run "$m" "Responde solo con la palabra: OK" 2>&1 | head -20 | tee -a "$RAW"; then
    echo "INFER_EXIT=0" | tee -a "$RAW"
  else
    code=$?
    echo "INFER_EXIT=$code (posible OOM/timeout — ver abajo)" | tee -a "$RAW"
  fi
  echo "--- ollama ps (residente) ---" | tee -a "$RAW"
  ollama ps 2>&1 | tee -a "$RAW"
  peak=$(gpu_used_mib)
  echo "PEAK_GPU_MiB=$peak DELTA_MiB=$((peak - base))" | tee -a "$RAW"
  nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv 2>&1 | tee -a "$RAW"
  echo "--- ollama stop $m (libera para la siguiente celda) ---" | tee -a "$RAW"
  ollama stop "$m" 2>&1 | tee -a "$RAW" || true
  sleep 3
  ollama ps 2>&1 | tee -a "$RAW"
  after=$(gpu_used_mib)
  echo "AFTER_STOP_GPU_MiB=$after" | tee -a "$RAW"
done

echo "" | tee -a "$RAW"
echo "== fin serie ==" | tee -a "$RAW"
echo "RAW=$RAW"

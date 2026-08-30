#!/usr/bin/env bash
# Pre-registered A/B launcher for `kitty bench-ab` (f7 lever batch).
#
# Usage: ./run_ab.sh <MODEL_A> <MODEL_B> [N]
#   MODEL_A : baseline model id (arm A, no new-lever env vars)
#   MODEL_B : candidate model id  (arm B, new-lever env vars on)
#   N       : exercises per arm, FIRST-N in dataset order (default 15)
#
# Safety properties:
#   - refuses to start on a dirty working tree (staged or unstaged)
#   - writes a pre-registration manifest BEFORE any launch
#   - no secrets hardcoded; all credentials stay in the operator env
#
# This script is tooling only. It has NOT been executed as part of the f7
# phase; see README.md for the protocol it enforces.
set -euo pipefail

MODEL_A="${1:?usage: run_ab.sh <MODEL_A> <MODEL_B> [N]}"
MODEL_B="${2:?usage: run_ab.sh <MODEL_A> <MODEL_B> [N]}"
N="${3:-15}"
SUITE="${SUITE:-polyglot}"
THINKING="${THINKING:-2048}"
PROFILE="${PROFILE:-medium}"
# f9 calibration: real 9B-model polyglot tasks on split-GPU hardware run
# 60-1200s; the 30s binary default voids most runs. Measurement default: 900.
TIMEOUT_SECS="${TIMEOUT_SECS:-900}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# ── Guard 1: clean tree (results must be attributable to a committed sha) ──
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: working tree is dirty — commit or stash before an A/B run." >&2
    exit 1
fi

HEAD_SHA="$(git rev-parse HEAD)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

# ── Binary: prefer release build; build once if missing (no benchmark here) ──
BIN="${KITTY_BIN:-$REPO_ROOT/target/release/kitty-cli}"
if [[ ! -x "$BIN" ]]; then
    echo "release binary missing — building kitty-cli..."
    cargo build --release -p kitty-cli
fi

mkdir -p benches/ab/manifests benches/ab/results
MANIFEST="benches/ab/manifests/manifest-${TS}.json"

# ── Exercises fingerprint (F1 MATERIAL, G2 2026-08-26) ───────────────────────
# Built-in polyglot suite has only 15 items; N>15 requires
# `--exercises benches/exercises_polyglot.json` (228 exercises). Record the
# fingerprint so manifests are reproducible and comparable (l15-builtin mix of
# 6 langs vs l50-json first-50 = 50 python-only by dataset order).
EXERCISES_PATH="benches/exercises_polyglot.json"
EXERCISES_SHA="$(sha256sum "$EXERCISES_PATH" 2>/dev/null | cut -d' ' -f1 || echo "unknown")"
EXERCISES_LOADED="$(jq -r 'length' "$EXERCISES_PATH" 2>/dev/null || echo 228)"

# ── Pre-registration manifest: written BEFORE any launch ────────────────────
cat > "$MANIFEST" <<EOF
{
  "timestamp_utc": "$TS",
  "git_head_sha": "$HEAD_SHA",
  "tree": "clean",
  "suite": "$SUITE",
  "n_per_arm": $N,
  "exercises": {
    "path": "$EXERCISES_PATH",
    "sha256": "$EXERCISES_SHA",
    "loaded": $EXERCISES_LOADED
  },
  "selection_rule": "first-N exercises in dataset order (-l N); chosen before launch, no post-hoc re-selection",
  "thinking_budget": $THINKING,
  "timeout_secs": $TIMEOUT_SECS,
  "profile": "$PROFILE",
  "interleave": true,
  "valid_run_gate": "errored trials <= 10% of N per arm; no early cuts",
  "harness_config": {
    "max_iterations": 5,
    "note": "kitty bench polyglot uses bench override (5), not the 16 default"
  },
  "env_baseline": "both arms launched from one normalized prefix; stripped: KITTY_NUM_CTX, KITTY_KV_QUANT_ACK, KITTY_MODEL_TURN_CAPS, KITTY_NO_THINK, KITTY_LINT_GATE, KITTY_EDIT_CHECK, KITTY_AUTORUN_HOPS, KITTY_BENCH_TIMEOUT_SECS",
  "arm_a": {
    "model": "$MODEL_A",
    "levers": "none — normalized baseline only"
  },
  "arm_b": {
    "model": "$MODEL_B",
    "levers": {
      "KITTY_NUM_CTX": "${ARM_B_NUM_CTX:-32768}",
      "KITTY_KV_QUANT_ACK": "${ARM_B_KV_ACK:-q8_0}",
      "note": "same normalized baseline + these declared levers layered on top"
    }
  }
}
EOF
echo "pre-registration manifest: $MANIFEST"

OUT_A="benches/ab/results/${TS}-armA.json"
OUT_B="benches/ab/results/${TS}-armB.json"

# ── Symmetric normalized env baseline (G4-FIX2 + f8 T1.3 + f9 timeout) ──────
# BOTH arms launch from this exact prefix so ambient operator env can never
# leak into one arm and not the other (the manifest's "levers" claim must be
# true). Stripped set: every harness lever env var, INCLUDING the interactive-
# only KITTY_AUTORUN_HOPS and the bench timeout fallback KITTY_BENCH_TIMEOUT_
# SECS (arms set their own timeout via --timeout-secs / TIMEOUT_SECS below).
CLEAN_ENV=(env -u KITTY_NUM_CTX -u KITTY_KV_QUANT_ACK -u KITTY_MODEL_TURN_CAPS -u KITTY_NO_THINK -u KITTY_LINT_GATE -u KITTY_EDIT_CHECK -u KITTY_AUTORUN_HOPS -u KITTY_BENCH_TIMEOUT_SECS)

# NOTE: `--model` is a TOP-LEVEL argument and must precede the subcommand.

echo "=== ARM A: $MODEL_A (normalized baseline, declared levers: none) ==="
"${CLEAN_ENV[@]}" \
    "$BIN" --model "$MODEL_A" bench-ab \
    --thinking "$THINKING" -l "$N" --suite "$SUITE" \
    --profile "$PROFILE" --interleave --timeout-secs "$TIMEOUT_SECS" \
    -o "$OUT_A"

echo "=== ARM B: $MODEL_B (same normalized baseline + declared levers on top) ==="
"${CLEAN_ENV[@]}" \
    KITTY_NUM_CTX="${ARM_B_NUM_CTX:-32768}" \
    KITTY_KV_QUANT_ACK="${ARM_B_KV_ACK:-q8_0}" \
    "$BIN" --model "$MODEL_B" bench-ab \
    --thinking "$THINKING" -l "$N" --suite "$SUITE" \
    --profile "$PROFILE" --interleave --timeout-secs "$TIMEOUT_SECS" \
    -o "$OUT_B"

echo "done."
echo "manifest: $MANIFEST"
echo "arm A:    $OUT_A"
echo "arm B:    $OUT_B"
echo "Next: apply the valid-run gate from benches/ab/README.md §2 before citing anything."

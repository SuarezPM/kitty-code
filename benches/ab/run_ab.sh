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

# ── Pre-registration manifest: written BEFORE any launch ────────────────────
cat > "$MANIFEST" <<EOF
{
  "timestamp_utc": "$TS",
  "git_head_sha": "$HEAD_SHA",
  "tree": "clean",
  "suite": "$SUITE",
  "n_per_arm": $N,
  "selection_rule": "first-N exercises in dataset order (-l N); chosen before launch, no post-hoc re-selection",
  "thinking_budget": $THINKING,
  "profile": "$PROFILE",
  "interleave": true,
  "valid_run_gate": "errored trials <= 10% of N per arm; no early cuts",
  "arm_a": {
    "model": "$MODEL_A",
    "levers": "none (no KITTY_NUM_CTX / no turn-cap overrides)"
  },
  "arm_b": {
    "model": "$MODEL_B",
    "levers": {
      "KITTY_NUM_CTX": "${ARM_B_NUM_CTX:-32768}",
      "KITTY_KV_QUANT_ACK": "${ARM_B_KV_ACK:-q8_0}",
      "note": "explicit context window + acknowledged KV quantization plan; sampling/thinking come from per-model profiles"
    }
  }
}
EOF
echo "pre-registration manifest: $MANIFEST"

OUT_A="benches/ab/results/${TS}-armA.json"
OUT_B="benches/ab/results/${TS}-armB.json"

# NOTE: `--model` is a TOP-LEVEL argument and must precede the subcommand.

echo "=== ARM A: $MODEL_A (baseline, levers off) ==="
env -u KITTY_NUM_CTX -u KITTY_KV_QUANT_ACK -u KITTY_MODEL_TURN_CAPS \
    "$BIN" --model "$MODEL_A" bench-ab \
    --thinking "$THINKING" -l "$N" --suite "$SUITE" \
    --profile "$PROFILE" --interleave \
    -o "$OUT_A"

echo "=== ARM B: $MODEL_B (new levers on) ==="
KITTY_NUM_CTX="${ARM_B_NUM_CTX:-32768}" \
KITTY_KV_QUANT_ACK="${ARM_B_KV_ACK:-q8_0}" \
    "$BIN" --model "$MODEL_B" bench-ab \
    --thinking "$THINKING" -l "$N" --suite "$SUITE" \
    --profile "$PROFILE" --interleave \
    -o "$OUT_B"

echo "done."
echo "manifest: $MANIFEST"
echo "arm A:    $OUT_A"
echo "arm B:    $OUT_B"
echo "Next: apply the valid-run gate from benches/ab/README.md §2 before citing anything."

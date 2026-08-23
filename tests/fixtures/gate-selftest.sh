#!/bin/bash
# gate-selftest.sh — validates gate v3 against known-good AND known-bad fixtures
# BEFORE it is trusted on real runs (project rule: gates validated against both).
#
# T1  gate-fixture-result.json (completed=10, errored=0)      -> expect PASS
# T2  gate-fixture-bad.json    (completed=5,  errored=28)     -> expect FAIL (>10% errors)
# T3  synthetic n_total_trials=0 with outcomes>0              -> expect FAIL (schema mismatch)
# T4  missing file                                            -> expect FAIL
#
# Usage: bash tests/fixtures/gate-selftest.sh ; exit 0 = all expectations met
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
GATE="$DIR/gate-v3.sh"
fail=0

# T1 known-good -> PASS
if out=$(bash "$GATE" "$DIR/gate-fixture-result.json" selftest-good); then
  echo "T1 PASS-as-expected: $out"
else
  echo "T1 UNEXPECTED-FAIL: $out"; fail=1
fi

# T2 known-bad -> FAIL
if out=$(bash "$GATE" "$DIR/gate-fixture-bad.json" selftest-bad); then
  echo "T2 UNEXPECTED-PASS: $out"; fail=1
else
  echo "T2 FAIL-as-expected: $out"
fi

# T3 schema mismatch (n_total_trials == 0 while outcomes > 0) -> FAIL
tmp=$(mktemp)
jq '.n_total_trials = 0' "$DIR/gate-fixture-result.json" > "$tmp"
if out=$(bash "$GATE" "$tmp" selftest-schema); then
  echo "T3 UNEXPECTED-PASS: $out"; fail=1
else
  echo "T3 FAIL-as-expected: $out"
fi
rm -f "$tmp"

# T4 missing file -> FAIL
if out=$(bash "$GATE" /nonexistent/result.json selftest-missing); then
  echo "T4 UNEXPECTED-PASS: $out"; fail=1
else
  echo "T4 FAIL-as-expected: $out"
fi

if [ "$fail" -eq 0 ]; then
  echo "SELFTEST: OK (4/4 expectations met)"
else
  echo "SELFTEST: FAILED — do not trust gate v3 on real runs"
fi
exit "$fail"

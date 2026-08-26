# Pre-registered A/B benchmark runbook (bench-ab)

Operational protocol for running `kitty bench-ab` comparisons with the
harness lever batch (f7). **This directory ships tooling only — nothing in
here has been executed.** No number produced by this protocol may be cited
anywhere until it follows the rules below.

## 1. Pre-registration protocol (mandatory, before launch)

1. **Pick first-N BEFORE seeing any result.** The deterministic selection
   rule is: *first N exercises in dataset order* (`-l N` takes the first N
   tasks of the suite dataset — no shuffling of the dataset itself, no
   post-hoc cherry-picking, no re-running with a different N after peeking).
2. **Record N in the manifest before launch.** `run_ab.sh` writes
   `manifests/manifest-<UTC-timestamp>.json` containing: UTC timestamp, git
   HEAD sha at launch time, both model ids, N, suite, selection rule, and
   which arm carries which levers. A results file without a matching
   pre-dated manifest is INVALID and must not be reported.
3. **Both arms share everything except the declared levers**: same suite,
   same N, same thinking budget (`--thinking 2048`, single budget per arm),
   same profile preset, same interleave setting. Any confound (different
   timeout multipliers, different server env beyond the declared levers)
   makes the pair non-comparable — record it or discard the run.

## 2. What counts as a valid run

* **Errored-trials gate:** errored trials must be ≤10% of N per arm,
  read from the serialized `per_budget[].errored` counter in the result
  JSON (bench-ab serializes verdict counts per budget; `errored` counts
  verdict `"error"`, and task-level timeouts are reported separately under
  `timed_out` — apply your own timeout policy, the gate itself reads
  `errored`). Parse the counters, not the driver's exit code.
* No early-cut: `finished_at` present / driver completed; aborted runs are
  `_partialN`, directional-only, and never comparable to completed arms.
* Manifest exists, predates the results file, and its HEAD sha matches a
  clean working tree (the launcher refuses dirty trees for exactly this
  reason).
* Both arms pass the gate independently; if only one passes, the pair is
  void (report as failed attempt, keep artifacts).

Provisional numbers NEVER go to validation, docs (`docs/AUDIT.md`,
`CNAS.md`), or any claim without this protocol having been followed
end-to-end.

### Sizing N against the gate

The gate makes N a design decision, not a convenience: with N=15 the ≤10%
gate allows at most ONE errored trial per arm, so against configs that
historically error around ~20% most attempts are void by construction.
Pick N so expected errors sit under the gate with margin. Note the
arithmetic honestly: at true error rate p against this 10% gate, expected
errors converge to p·N while the gate allows only 0.1·N — so **if pilot
p ≥ 0.10, NO N passes the gate**: record it as a FINDING and stop. Use a
larger N only when pilot p sits comfortably below 0.10, to tighten the CI.
And state it explicitly: **a model erroring >10% under this protocol is a
FINDING** (unmeasurable under current harness quality), not a failure to
retry — record it as such instead of re-rolling N until the gate passes.

## 3. Where things land

| Artifact | Path |
|----------|------|
| Pre-registration manifests | `benches/ab/manifests/manifest-*.json` |
| Arm A results (baseline) | `benches/ab/results/<ts>-armA.json` |
| Arm B results (levers on) | `benches/ab/results/<ts>-armB.json` |

Raw artifacts stay in place; aggregate numbers move into docs ONLY via the
audit process above.

## 4. Launcher

```bash
./benches/ab/run_ab.sh <MODEL_A> <MODEL_B> [N]   # N default 15
```

The script refuses to start on a dirty working tree (staged or unstaged),
writes the manifest first, then invokes the real CLI twice:

* Arm A — baseline model, NO new-lever env vars.
* Arm B — candidate model WITH the new-lever env vars (explicit context
  window etc., see inside the script).

Exact CLI syntax used (verified against this repo's clap definition):
`--model` is a TOP-LEVEL argument and must come BEFORE the subcommand;
`--profile`/`--knowledge` are global. Per-arm invocation shape:

```bash
kitty-cli --model "$MODEL" bench-ab \
  --thinking 2048 -l "$N" --suite polyglot \
  --profile medium -o <out.json>
```

(`bench-ab` flags: `--thinking`, `-l/--limit`, `--reps`, `--suite`,
`-o/--out`, `--interleave`, `--hops`, `--timeout-secs`; there is no
per-subcommand model flag.)

**Per-exercise timeout (`--timeout-secs` / `KITTY_BENCH_TIMEOUT_SECS`, f9
calibration):** every exercise runs under a wall-clock cap. Precedence:
explicit CLI flag > env fallback > default 30s; an invalid env value warns
once and falls back to the default; the launcher strips the env var from
BOTH arms and sets its own symmetric value (`TIMEOUT_SECS`, default 900).
Guidance from calibration: polyglot tasks on a ~9B hybrid model over a
split GPU take **60–1200 s per task** — the old hard-coded 30s cap voided
real measurement runs almost immediately. Use **≥900 s for measurement
runs** (and size N accordingly, see §2); shorter values are fine only for
smoke checks.

**`--hops` arm lever (f8):** autorun continuation hops grant each exercise
extra FULL tool-loop cycles on iteration exhaustion. Bench autorun is
FLAG-ONLY — the harness never reads `KITTY_AUTORUN_HOPS`, and the launcher
strips that var from BOTH arms, so an ambient export can never contaminate a
run; arms differ only when you explicitly pass different `--hops` values.
**Cost warning:** worst-case LLM calls per exercise ≈
max_iterations × 3 × (hops+1) — against the 30s per-exercise bench timeout,
hops > 0 on a slow model mostly converts timeouts into voided runs. Size N
and hops together, and prefer hops=1 with a raised timeout over large hop
counts.

**Honest attribution:** this pair is a MODEL-vs-MODEL comparison under the
new harness, NOT a per-lever ablation. Arm A also runs with the client's
unconditional defaults — `num_ctx=32768` and `"truncate": false` are
injected by OllamaClient on every request regardless of arm — the KV ack is
warning-only, and the ladder/parser levers have no env toggles at all. The
only declared deltas between arms are the model id and arm B's explicit
`KITTY_NUM_CTX`/ack layering on top of the shared normalized baseline.

## 5. Server-side prerequisites checklist (host, NOT settable via API)

- [ ] `OLLAMA_KV_CACHE_TYPE=q8_0` recommended whenever the resolved
      `num_ctx > 16384`: an f16 KV cache at 32k is ~4.8 GB — OOM risk on
      8 GB GPUs; q8_0 halves it. This is a SERVER env var applied at model
      load; it cannot be set via API, Modelfile, or request.
- [ ] `KITTY_NUM_CTX` fixed for the whole session (client default 32768).
      Changing num_ctx between requests reloads the model server-side —
      pick one value per run and keep it.
- [ ] If you cannot control the server env, set
      `KITTY_KV_QUANT_ACK=q8_0` to acknowledge the memory plan and silence
      the client-side KV preflight warning (warning only, never a failure).

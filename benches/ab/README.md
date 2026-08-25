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

* **Errored-trials gate:** errored trials must be ≤10% of N per arm
  (checked from the result JSON's top-level stats — harbor/bench drivers can
  exit 0 while trials error; parse the counters, not the exit code).
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
`-o/--out`, `--interleave`; there is no per-subcommand model flag.)

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

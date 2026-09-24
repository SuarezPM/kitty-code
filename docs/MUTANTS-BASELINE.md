# Mutants baseline — kitty-harness guard + sandbox

- Date: 2026-09-24 (America/Argentina/Buenos_Aires)
- Tool: `cargo mutants 27.1.0` (plan pin exact)
- Scope: `-f "crates/kitty-harness/src/guard/*.rs" -f "crates/kitty-harness/src/sandbox.rs"`
  (440 mutants listed). NOTE: `--glob` does not exist in mutants 27.1.0;
  `-f/--file` (repeatable) is the equivalent. The bare `-f src/guard/`
  form matches 0 (globs with slash match the entire path).
- Time-box: 25 min (`timeout -s KILL 1500`); run did NOT finish.

## Counts (partial, 87/440 evaluated = 20%)

| outcome  | count |
|----------|-------|
| caught   | 48    |
| missed   | 35    |
| timeout  | 0     |
| unviable | 4     |

- Score among decided (caught / (caught + missed)): **48/83 = 57.8%**
- Status: **HELD-timebox** — ~17–94 s per mutant (suite + coverage-slow
  parser test); full 440 needs ~3–4 h, belongs in the nocturnal job.

## Top-3 missed (file:line)

1. `crates/kitty-harness/src/guard/mod.rs:99` — `||` → `&&` in `Guard::scan`
2. `crates/kitty-harness/src/guard/mod.rs:98` — `||` → `&&` in `Guard::scan`
3. `crates/kitty-harness/src/guard/mod.rs:97` — `||` → `&&` in `Guard::scan`

Signal: boolean-composition mutants in `Guard::scan` survive (also :96);
sandbox `||`→`&&` / `Ok(())` replacements survive (`sandbox.rs:330,467,635`).
Remediation is another lane (this file is the ledger, not the fix).

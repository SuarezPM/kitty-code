# Delta Spec: testing/mutation-guard

## ADDED Requirements

### Requirement: Guard mutation results are collected and archived

Guard mutation runs SHALL persist their log inside the repo results
area (never only `/tmp`) so a later session can collect them.

#### Scenario: mutation log is collectible

- **GIVEN** a guard mutation run finishes
- **WHEN** anyone looks for its outcome
- **THEN** the full log exists at a committed-ignored results path with
  the mutant counts (caught/missed/timeout) readable without re-running

### Requirement: Heredoc line-arithmetic is covered by a killing test

`strip_heredoc_bodies` SHALL have a test whose body spans two or more
lines before the terminator, so the `consumed += line.len()+1`
accumulation executes at least twice and any `+=`→`*=` mutant diverges.

#### Scenario: multi-line heredoc kills the arithmetic mutant

- **GIVEN** `cat > f << EOF` followed by two body lines, the
  terminator, and a trailing chained command
- **WHEN** the body is stripped
- **THEN** the result is the exact prefix-plus-chain string with no
  body residue and no lost trailing command

#### Scenario: double heredoc re-scan is covered

- **GIVEN** one command containing two heredocs
- **WHEN** both bodies are stripped
- **THEN** neither body leaks and the re-scan position stays consistent

### Requirement: Mutation ratio above 70 percent is ratcheted

Deferred until a runner with headroom (see tasks 1.1/1.3); until
then the documented CONTRIBUTING target (>70% caught for guard
modules) SHOULD hold on every re-run; new MISSED mutants are either
killed with a test or recorded as accepted with a reason.

Status 2026-09-09 (non-normative): full ratchet descoped with tasks
1.1/1.3 — two full runs OOM-killed on this box (logs in `results/`).
Partial evidence stands: the killer test for the known `+=`→`*=`
MISSED (committed `aa424fc`) plus ~70 scored mutants showing a MISSED
cluster in `split_command_chain` arithmetic, recorded as future test
work.

#### Scenario: regression in mutation score fails the gate

- **GIVEN** a fresh guard mutation report below the archived ratio
- **WHEN** the change is reviewed
- **THEN** it is blocked until tests kill the new survivors or an
  accepted-with-reason entry covers each one

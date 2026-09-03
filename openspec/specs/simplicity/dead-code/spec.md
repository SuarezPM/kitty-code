# dead-code Specification

## Purpose
Cero supresores de dead_code en producción y raíz del repositorio limpia de scripts huérfanos.

## Requirements

### Requirement: No dead-code suppressors in production code

Production modules SHALL NOT contain `#[allow(dead_code)]`. Code that is
unused SHALL be deleted; code needed by tests SHALL be exercised through
the public API or moved into the test module.

- Applies to: `crates/kitty-cli/src/{gpqa,tui,polyglot,claweval}.rs`,
  `crates/kitty-harness/src/{guard,episodic}.rs`,
  `crates/kitty-runtime/src/{cloud,llama}.rs`.
- Inline `#[cfg(test)]` modules MAY suppress if justified with a comment.

#### Scenario: workspace builds without dead_code allows

- **WHEN** the workspace is searched for `#[allow(dead_code)]` outside
  `#[cfg(test)]` modules
- **THEN** zero occurrences remain

### Requirement: Repository root contains no orphaned bench scripts

The repository root SHALL NOT contain orphaned experiment scripts:
`extract_speed.py`, `extract_speed.sh`, `oc_bench.js`, `speed_proxy.js`,
`stress.js`, the empty `vendor/` directory, and the stale
`adapter-review-report.md`.

- Agent-run artifacts at the root (`affine_cipher.py`, `beer_song.py`,
  `bottle_song.py` pattern) SHALL be prevented by gitignore rules.

#### Scenario: root is clean

- **WHEN** the repository root is listed after applying this change
- **THEN** none of the listed orphan files exist and
  `git status` shows no untracked bench artifacts matching `/^[a-z_]+\.py$/`

#### Scenario: gitignore prevents recurrence

- **GIVEN** a future bench run writes `<word>.py` to the repository root
- **WHEN** `git status` runs
- **THEN** the file is ignored by the `/[a-z_]+\.py` root rule

## Why

Three benchmark runners have critical issues that produce invalid or incomparable results:

1. **Polyglot runner** (`polyglot.rs`): Test files visible to agent during coding, no retry, no `.meta/config.json` parsing, no subprocess timeouts. Violates the official Aider pattern.
2. **GPQA runner** (`gpqa.rs`): No chain-of-thought, prompt format not aligned with official (`ANSWER: $LETTER`), scoring not strict by format. Results not comparable to leaderboard.
3. **ClawEval runner** (`claweval.rs`): **Critical bug** — model hardcoded to `openrouter/default`, ignores `--model` flag. Dead code in `run_claweval_batch`. Results invalid.
4. **Terminal-Bench 2.1**: Dataset vendored but NO runner integrated. `bench tb2` is 8 trivial tasks, `run_tb2_89.py` is homemade with string-match grading. Not comparable to leaderboard.

## What Changes

- **Rewrite `polyglot.rs` + `run_polyglot_bench`**: Official pattern with config.json parsing, test hiding, retry loop, timeouts
- **Fix `gpqa.rs`**: Add CoT prompt, align with official `ANSWER: $LETTER` format, strict scoring
- **Fix `claweval.rs`**: Pass `--model` flag correctly, remove dead code, integrate `run_claweval_batch`
- **Add TB2 runner**: Integrate `harbor run` as subprocess wrapper for Terminal-Bench 2.1
- **Deprecate homemade runners**: Rename `run_polyglot_225.py` → `run_kittyquiz_225.py`

## Capabilities

### New Capabilities
- `polyglot-runner`: Official-pattern Polyglot runner with config.json parsing, test hiding, retry loop, and subprocess timeouts

### Modified Capabilities
- None — this is a complete rewrite of existing functionality

## Impact

- `crates/kitty-cli/src/polyglot.rs` — Complete rewrite (~400 lines)
- `crates/kitty-cli/src/main.rs` — Rewrite `run_polyglot_bench` function (~200 lines)
- `benches/run_polyglot_225.py` — Rename to `run_kittyquiz_225.py`

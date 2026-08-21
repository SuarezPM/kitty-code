# benches/ Homemade Quiz Harness

## OVERVIEW
Homemade Python harness that grades KittyQuiz-225 and KittyShell-89 by string-match in tempdir via Ollama tool calls, not official benchmarks.

## STRUCTURE
| File | Role |
|------|------|
| `run_polyglot_225.py` | KittyQuiz runner, 225 asserts 228 exercises, Write and Bash only |
| `run_tb2_89.py` | KittyShell runner, 89 tasks, Write Bash Read |
| `exercises_polyglot.json` | 228 exercises, 225 base plus 3 bonus py-list-comp, py-decorator-args, py-async-await |
| `exercises_tb2.json` | 89 TB2-style tasks |
| `baseline.json` | Reference 96.44 percent on polyglot, string-match-homemade, ornith:9b |
| `polyglot_225.py`, `week0_*.py`, `export_exercises.py` | Legacy runners and export helper, keep but prefer run_* |
| `bench_*.json`, `bench_qwen3_*.json` | Output artifacts, gitignored, leak to repo root |

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Polyglot language split | `run_polyglot_225.py:62` | 50 py, 50 js, 30 ts, 45 rust, 30 go, 10 cpp, 10 java equals 225 |
| TB2 categories | `run_tb2_89.py:98` | 20 file_ops, 20 text_proc, 20 shell, 15 data_transform, 14 sys_ops equals 89 |
| Tool definitions | `run_polyglot_225.py:26`, `run_tb2_89.py:26` | TOOLS array, Ollama native tool calling |
| Grading logic | `run_polyglot_225.py:395` `verify_exercise` | `check_str.lower() in content.lower()`, file existence only |
| TB2 checks | `run_tb2_89.py:98` | Lambdas like `os.path.isfile` or `os.access`, `safe_check` wrapper |
| Execution | Both `execute_tool_calls` | Tempdir per exercise, absolute paths coerced to basename, Bash 30s timeout |
| Ollama call | Both `call_ollama` | `http://localhost:11434/api/chat`, ornith:9b, temp 0.6, timeout 120s |

## CONVENTIONS
- Grading is string-match, not compilation or unit tests. `check_str` must appear in `check_file`.
- Each exercise runs isolated in `tempfile.TemporaryDirectory` with `kitty_bench_` or `kitty_tb2_` prefix.
- Outputs are JSON summaries with `passed`, `latency_ms`, `per_language` or `per_category`.
- `exercises_polyglot.json` is source of truth for prompts, keep in sync if editing `run_polyglot_225.py` EXERCISES.
- `bench_ab` in `kitty-cli` is interleaved shuffle seed 42 comparison, also homemade, not `bench/official`.

## ANTI-PATTERNS
- NEVER compare homemade pass rates to Aider Polyglot or Terminal-Bench 2.0. Label always as homemade string-match.
- NEVER edit `bench/official`. That is Harbor Docker per-task with `task.toml` and 1800s timeout, distinct infra.
- NEVER require Docker here. This dir runs bare Python plus Ollama, no containers.
- NEVER commit `bench_*.json` from root leakage. They are gitignored, keep outputs in `benches/` or `results/`.
- NEVER add heavy test harnesses. Grading stays `check_str.lower() in content.lower()`, file exists for TB2.

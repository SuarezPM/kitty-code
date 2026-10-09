#!/usr/bin/env python3
"""tb2-8 justicia little-coder — 8 prompts tb2-homemade, timeout 60s/ejercicio, k=3.
Protocolo espejo de kitty bench tb2 (mismo modelo/backend/turnos/timeouts).
Modelo: ollama/ornith:9b via pi RPC. Max turns 20 (= kitty max_iterations).
Grading: mismo string-match que kitty run.rs tb2_homemade_exercises.
Serie: 1 modelo residente, ollama stop entre reps (el llamador hace stop).
Uso: python3 bench/gpu-ventana-2026-10-09/little-tb2.py <rep> <out.json>
"""
import json
import os
import stat
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "/home/thelinconx/Proyectos/Repos/little-coder/benchmarks")
from rpc_client import PiRpc

MODEL = "ollama/ornith:9b"
TIMEOUT = 60
MAX_TURNS = 20

EXERCISES = [
    {"id": "tb2-bash-count", "lang": "bash",
     "prompt": "Create a bash script called count_lines.sh that counts the number of lines in all .rs files in the current directory recursively. Save it to count_lines.sh and make it executable.",
     "file": "count_lines.sh", "checks": [("exec", "count_lines.sh", None), ("contains", "count_lines.sh", "find")]},
    {"id": "tb2-py-csv", "lang": "python",
     "prompt": "Write a Python script called csv_to_json.py that reads a CSV file (input path as first argument) and outputs JSON to stdout. Save it to csv_to_json.py.",
     "file": "csv_to_json.py", "checks": [("contains", "csv_to_json.py", "csv"), ("contains", "csv_to_json.py", "json")]},
    {"id": "tb2-makefile", "lang": "makefile",
     "prompt": "Create a Makefile with three targets: test (runs cargo test), build (runs cargo build), and clean (removes target directory). Save to Makefile.",
     "file": "Makefile", "checks": [("contains", "Makefile", "test:"), ("contains", "Makefile", "build:"), ("contains", "Makefile", "clean:")]},
    {"id": "tb2-dockerfile", "lang": "docker",
     "prompt": "Write a Dockerfile for a Rust project that uses multi-stage builds: builder stage compiles, runtime stage copies binary. Save to Dockerfile.",
     "file": "Dockerfile", "checks": [("contains", "Dockerfile", "FROM"), ("contains", "Dockerfile", "COPY")]},
    {"id": "tb2-gh-actions", "lang": "ci",
     "prompt": "Create a GitHub Actions CI workflow at .github/workflows/ci.yml that runs cargo test and cargo build on push/PR. Save to .github/workflows/ci.yml.",
     "file": ".github/workflows/ci.yml", "checks": [("contains_any", ".github/workflows/ci.yml", ["cargo test", "cargo build"])]},
    {"id": "tb2-sh-find-modified", "lang": "bash",
     "prompt": "Write a bash script called find_recent.sh that finds all files modified in the last 24 hours in a given directory (first argument). Save to find_recent.sh.",
     "file": "find_recent.sh", "checks": [("exec", "find_recent.sh", None), ("contains", "find_recent.sh", "find")]},
    {"id": "tb2-py-dirsize", "lang": "python",
     "prompt": "Write a Python script called dirsize.py that calculates and prints the total size of each subdirectory in the current directory. Save to dirsize.py.",
     "file": "dirsize.py", "checks": [("contains", "dirsize.py", "os"), ("contains", "dirsize.py", "path")]},
    {"id": "tb2-sh-backup", "lang": "bash",
     "prompt": "Write a bash script called backup.sh that backs up files from a source directory (first argument) to a destination with a timestamp suffix (e.g. backup_20260817_120000). Save to backup.sh.",
     "file": "backup.sh", "checks": [("exec", "backup.sh", None), ("contains", "backup.sh", "cp")]},
]


def check(tmpdir, kinds):
    for kind, fname, needle in kinds:
        p = Path(tmpdir) / fname
        if not p.is_file():
            return False
        try:
            content = p.read_text(errors="ignore")
        except OSError:
            return False
        if kind == "exec":
            if not (p.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)):
                return False
        elif kind == "contains":
            if needle.lower() not in content.lower():
                return False
        elif kind == "contains_any":
            if not any(n.lower() in content.lower() for n in needle):
                return False
    return True


def main():
    rep = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    out = sys.argv[2] if len(sys.argv) > 2 else f"little-tb2-k{rep}.json"
    results = []
    for i, ex in enumerate(EXERCISES):
        print(f"[{i+1}/8] {ex['id']}...", flush=True)
        tmpdir = tempfile.mkdtemp(prefix="little_tb2_")
        prompt = (f"{ex['prompt']}\n\nWork in directory: {tmpdir} "
                  f"— Save to {tmpdir}/{ex['file']} (use absolute path)")
        start = time.time()
        verdict, passed = "error", False
        turns, tool_calls = 0, 0
        try:
            with PiRpc(model=MODEL, cwd=tmpdir, max_turns=MAX_TURNS) as rpc:
                r = rpc.prompt_and_collect(prompt, timeout=TIMEOUT)
                turns, tool_calls = r.turn_count, len(r.tool_calls)
            elapsed_ms = (time.time() - start) * 1000
            passed = check(tmpdir, ex["checks"])
            verdict = "pass" if passed else "fail"
            latency_ms = elapsed_ms
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            msg = str(e)
            if "timeout" in msg.lower() or latency_ms >= TIMEOUT * 1000 - 500:
                verdict = "timeout"
            print(f"  error: {e}", flush=True)
        # Si el timeout externo corta a 60s, la latencia real ≈ 60000
        if verdict == "timeout":
            latency_ms = 60000.0
        print(f"  {'PASS' if passed else 'FAIL'} ({verdict}, {latency_ms:.0f}ms, turns={turns}, tools={tool_calls})",
              flush=True)
        results.append({"id": ex["id"], "lang": ex["lang"], "passed": passed,
                        "verdict": verdict, "latency_ms": latency_ms,
                        "turns": turns, "tool_calls": tool_calls})
    passed_n = sum(1 for r in results if r["passed"])
    print(f"Passed: {passed_n}/8 ({100*passed_n/8:.1f}%)")
    with open(out, "w") as f:
        json.dump({"model": MODEL, "rep": rep, "timeout_secs": TIMEOUT,
                   "max_turns": MAX_TURNS, "results": results,
                   "passed": passed_n, "total": 8}, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

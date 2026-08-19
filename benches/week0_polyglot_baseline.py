#!/usr/bin/env python3
"""
Kitty Code Week 0 — Polyglot Baseline
Runs Aider Polyglot exercises with ornith:9B via Ollama API.
Measures raw baseline (no harness, no scaffold).
"""

import json
import os
import sys
import time
import tempfile
import subprocess
import urllib.request
import re
from dataclasses import dataclass, field
from typing import Optional

OLLAMA_URL = "http://localhost:11434/api/chat"

# Polyglot exercises (subset for baseline measurement)
EXERCISES = [
    # Python
    {
        "id": "py-fibonacci",
        "lang": "python",
        "prompt": "Write a function fibonacci(n: int) -> list[int] that returns the first n Fibonacci numbers. Use type hints.",
        "verify": lambda d: os.path.exists(os.path.join(d, "solution.py")) and "def fibonacci" in open(os.path.join(d, "solution.py")).read(),
    },
    {
        "id": "py-class",
        "lang": "python",
        "prompt": "Create a BankAccount class with deposit(), withdraw(), and balance property. Include type hints.",
        "verify": lambda d: os.path.exists(os.path.join(d, "solution.py")) and "class BankAccount" in open(os.path.join(d, "solution.py")).read(),
    },
    {
        "id": "py-sort",
        "lang": "python",
        "prompt": "Write a function merge_sort(arr: list[int]) -> list[int] that implements merge sort.",
        "verify": lambda d: os.path.exists(os.path.join(d, "solution.py")) and "def merge_sort" in open(os.path.join(d, "solution.py")).read(),
    },
    {
        "id": "py-decorator",
        "lang": "python",
        "prompt": "Write a retry decorator that retries a function up to 3 times on exception.",
        "verify": lambda d: os.path.exists(os.path.join(d, "solution.py")) and "def retry" in open(os.path.join(d, "solution.py")).read(),
    },
    {
        "id": "py-context",
        "lang": "python",
        "prompt": "Write a context manager that times code execution and prints the elapsed time.",
        "verify": lambda d: os.path.exists(os.path.join(d, "solution.py")) and "__enter__" in open(os.path.join(d, "solution.py")).read(),
    },
    # JavaScript
    {
        "id": "js-promise",
        "lang": "javascript",
        "prompt": "Write a function fetchWithTimeout(url, timeout) that fetches a URL with a timeout using AbortController.",
        "verify": lambda d: os.path.exists(os.path.join(d, "solution.js")) and "AbortController" in open(os.path.join(d, "solution.js")).read(),
    },
    {
        "id": "js-debounce",
        "lang": "javascript",
        "prompt": "Write a debounce function in JavaScript with proper TypeScript-style JSDoc.",
        "verify": lambda d: os.path.exists(os.path.join(d, "solution.js")) and "function debounce" in open(os.path.join(d, "solution.js")).read(),
    },
    {
        "id": "js-async",
        "lang": "javascript",
        "prompt": "Write an async function that reads a file and returns its contents as a string.",
        "verify": lambda d: os.path.exists(os.path.join(d, "solution.js")) and "async" in open(os.path.join(d, "solution.js")).read(),
    },
    # TypeScript
    {
        "id": "ts-interface",
        "lang": "typescript",
        "prompt": "Write a TypeScript interface User with id, name, email fields and a type guard function isUser.",
        "verify": lambda d: os.path.exists(os.path.join(d, "solution.ts")) and "interface User" in open(os.path.join(d, "solution.ts")).read(),
    },
    {
        "id": "ts-generic",
        "lang": "typescript",
        "prompt": "Write a generic function first<T>(arr: T[]): T | undefined that returns the first element.",
        "verify": lambda d: os.path.exists(os.path.join(d, "solution.ts")) and "function first" in open(os.path.join(d, "solution.ts")).read(),
    },
    # Shell
    {
        "id": "sh-backup",
        "lang": "bash",
        "prompt": "Write a bash script that backs up a directory to a timestamped tar.gz file.",
        "verify": lambda d: os.path.exists(os.path.join(d, "solution.sh")) and "tar" in open(os.path.join(d, "solution.sh")).read(),
    },
    {
        "id": "sh-replace",
        "lang": "bash",
        "prompt": "Write a bash script that finds and replaces text in all .txt files in a directory.",
        "verify": lambda d: os.path.exists(os.path.join(d, "solution.sh")) and "find" in open(os.path.join(d, "solution.sh")).read(),
    },
    # JSON
    {
        "id": "json-validate",
        "lang": "python",
        "prompt": "Write a Python function that validates a JSON string and returns the parsed data or None on error.",
        "verify": lambda d: os.path.exists(os.path.join(d, "solution.py")) and "json.loads" in open(os.path.join(d, "solution.py")).read(),
    },
    # Multi-file
    {
        "id": "multi-module",
        "lang": "python",
        "prompt": "Create a Python package with __init__.py, models.py (User dataclass), and utils.py (format_user function). Import models in utils.",
        "verify": lambda d: os.path.exists(os.path.join(d, "models.py")) and os.path.exists(os.path.join(d, "utils.py")),
    },
    # Rust
    {
        "id": "rust-struct",
        "lang": "rust",
        "prompt": "Write a Rust struct Point with x,y f64 fields and an impl block with a new() constructor and distance_to() method.",
        "verify": lambda d: os.path.exists(os.path.join(d, "solution.rs")) and "struct Point" in open(os.path.join(d, "solution.rs")).read(),
    },
    {
        "id": "rust-enum",
        "lang": "rust",
        "prompt": "Write a Rust enum Shape with Circle(f64), Rectangle(f64, f64), Triangle(f64, f64, f64) and an area() method.",
        "verify": lambda d: os.path.exists(os.path.join(d, "solution.rs")) and "enum Shape" in open(os.path.join(d, "solution.rs")).read(),
    },
    # Go
    {
        "id": "go-struct",
        "lang": "go",
        "prompt": "Write a Go struct Person with Name and Age fields, and a String() method that returns a formatted string.",
        "verify": lambda d: os.path.exists(os.path.join(d, "solution.go")) and "type Person struct" in open(os.path.join(d, "solution.go")).read(),
    },
    # C++
    {
        "id": "cpp-vector",
        "lang": "cpp",
        "prompt": "Write a C++ function that takes a vector<int> and returns a new vector with only even numbers.",
        "verify": lambda d: os.path.exists(os.path.join(d, "solution.cpp")) and "vector" in open(os.path.join(d, "solution.cpp")).read(),
    },
    # Fix bugs
    {
        "id": "fix-py-bug",
        "lang": "python",
        "prompt": "Fix the bug in this code: def divide(a, b): return a / b  # crashes on b=0",
        "verify": lambda d: os.path.exists(os.path.join(d, "solution.py")) and ("if" in open(os.path.join(d, "solution.py")).read() or "raise" in open(os.path.join(d, "solution.py")).read()),
    },
    {
        "id": "fix-js-bug",
        "lang": "javascript",
        "prompt": "Fix this code: function getLength(s) { return s.length } // crashes on null input",
        "verify": lambda d: os.path.exists(os.path.join(d, "solution.js")) and ("!" in open(os.path.join(d, "solution.js")).read() or "?." in open(os.path.join(d, "solution.js")).read()),
    },
]


@dataclass
class ExerciseResult:
    id: str
    lang: str
    passed: bool
    latency_ms: float
    files_created: int = 0
    error: Optional[str] = None


def call_ollama(prompt: str, model: str, tools: list, temperature: float = 0.6) -> tuple[dict, float]:
    """Call Ollama with tools."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "stream": False,
        "options": {"temperature": temperature, "top_p": 0.95, "top_k": 20, "num_predict": 4096},
    }).encode("utf-8")

    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data, (time.time() - start) * 1000
    except Exception as e:
        return {"error": str(e)}, (time.time() - start) * 1000


def run_exercise(exercise: dict, model: str, tools: list, temperature: float) -> ExerciseResult:
    """Run a single exercise."""
    tmpdir = tempfile.mkdtemp(prefix=f"kitty_bench_{exercise['id']}_")

    try:
        # Multi-turn: first get tool calls, then execute them
        response, latency = call_ollama(exercise["prompt"], model, tools, temperature)
        msg = response.get("message", {})
        tool_calls = msg.get("tool_calls", [])

        files_created = 0
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})

            if name == "Write":
                path = args.get("path", "")
                content = args.get("content", "")
                if path and content:
                    full_path = os.path.join(tmpdir, os.path.basename(path))
                    with open(full_path, "w") as f:
                        f.write(content)
                    files_created += 1
            elif name == "Bash":
                cmd = args.get("command", "")
                if cmd:
                    try:
                        result = subprocess.run(
                            ["bash", "-c", cmd],
                            cwd=tmpdir,
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        # Parse output for file creation
                        if "solution." in cmd or "create" in cmd.lower():
                            files_created += 1
                    except subprocess.TimeoutExpired:
                        pass

        # Verify
        passed = exercise["verify"](tmpdir)

        return ExerciseResult(
            id=exercise["id"],
            lang=exercise["lang"],
            passed=passed,
            latency_ms=latency,
            files_created=files_created,
        )
    except Exception as e:
        return ExerciseResult(
            id=exercise["id"],
            lang=exercise["lang"],
            passed=False,
            latency_ms=0,
            error=str(e),
        )
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ornith:9b")
    parser.add_argument("--count", type=int, default=len(EXERCISES))
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()

    tools = [
        {"type": "function", "function": {"name": "Write", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
        {"type": "function", "function": {"name": "Bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
        {"type": "function", "function": {"name": "Read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    ]

    exercises = EXERCISES[:args.count]
    results = []

    print(f"\n{'='*60}")
    print(f"Kitty Code Week 0 — Polyglot Baseline")
    print(f"Model: {args.model} | Exercises: {len(exercises)} | Temp: {args.temperature}")
    print(f"{'='*60}\n")

    for i, ex in enumerate(exercises, 1):
        sys.stdout.write(f"\r[{i:2d}/{len(exercises)}] {ex['id']:20s}...")
        sys.stdout.flush()

        result = run_exercise(ex, args.model, tools, args.temperature)
        results.append(result)

        status = "✅" if result.passed else "❌"
        sys.stdout.write(f"\r[{i:2d}/{len(exercises)}] {status} {result.id:20s} ({result.latency_ms:.0f}ms, {result.files_created} files)\n")
        sys.stdout.flush()

    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    rate = 100 * passed / total if total else 0
    avg_lat = sum(r.latency_ms for r in results) / total if total else 0

    print(f"\n{'='*60}")
    print(f"BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"Passed: {passed}/{total} ({rate:.1f}%)")
    print(f"Avg latency: {avg_lat:.0f}ms")

    # By language
    by_lang = {}
    for r in results:
        by_lang.setdefault(r.lang, []).append(r)
    for lang, res in sorted(by_lang.items()):
        p = sum(1 for r in res if r.passed)
        print(f"  {lang}: {p}/{len(res)} ({100*p/len(res):.0f}%)")

    # Gate check
    target = 45.0
    print(f"\nTarget: {target}% | Baseline: {rate:.1f}%")
    if rate >= target:
        print(f"✅ BASELINE MEETS TARGET")
    elif rate >= target * 0.8:
        print(f"⚠️ BASELINE CLOSE — harness may bridge gap")
    else:
        print(f"❌ BASELINE FAR FROM TARGET — 45% likely unreachable")

    # Save
    output = {
        "model": args.model,
        "temperature": args.temperature,
        "total": total,
        "passed": passed,
        "rate_pct": rate,
        "by_lang": {lang: {"passed": sum(1 for r in res if r.passed), "total": len(res)} for lang, res in by_lang.items()},
        "details": [{"id": r.id, "passed": r.passed, "latency_ms": r.latency_ms} for r in results],
    }
    with open("week0_polyglot_baseline.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to week0_polyglot_baseline.json")


if __name__ == "__main__":
    main()

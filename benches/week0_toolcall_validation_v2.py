#!/usr/bin/env python3
"""
Kitty Code Week 0 — Tool-Call Validation Suite (v2)
Uses Ollama's native tool-calling API (tools parameter).
Measures parse success rate. Gate: >=85%.
"""

import json
import sys
import time
import urllib.request
import urllib.error
from typing import Optional
from dataclasses import dataclass, field

OLLAMA_URL = "http://localhost:11434/api/chat"

# Tool definitions for Ollama API
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file and return its contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"},
                    "start_line": {"type": "integer", "description": "Optional start line (1-indexed)"},
                    "end_line": {"type": "integer", "description": "Optional end line (1-indexed)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Edit",
            "description": "Edit a file by replacing old text with new text",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit"},
                    "old_text": {"type": "string", "description": "Exact text to find and replace"},
                    "new_text": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a new file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write"},
                    "content": {"type": "string", "description": "Content to write to the file"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Glob",
            "description": "Find files matching a glob pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g. src/**/*.rs)"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Grep",
            "description": "Search file contents with regex",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory or file to search in"},
                    "include": {"type": "string", "description": "File pattern to include (e.g. *.rs)"},
                },
                "required": ["pattern"],
            },
        },
    },
]

VALID_TOOL_NAMES = {t["function"]["name"] for t in TOOLS}

# 100 prompts: mix of tool-requiring tasks
PROMPTS = [
    # Read (20)
    "Read the file Cargo.toml",
    "Read lines 10-25 of src/main.rs",
    "Show me the contents of README.md",
    "Read the config at .config/settings.json",
    "Display the first 50 lines of tests/test1.rs",
    "Read src/lib.rs and show the public API",
    "Show me the error handling in src/error.rs",
    "Read the CI config at .github/workflows/ci.yml",
    "Display the contents of deny.toml",
    "Read the LICENSE file",
    "Show me the dependencies in Cargo.toml",
    "Read the CHANGELOG.md",
    "Display src/parser/mod.rs",
    "Read tests/integration/test_parse.rs",
    "Show me the benchmark code in benches/main.rs",
    "Read the SECURITY.md file",
    "Display the workspace configuration",
    "Read src/harness/mod.rs",
    "Show me the CLI entry point at src/cli.rs",
    "Read the Makefile",
    # Edit (20)
    "In src/main.rs, change the default port from 8080 to 9090",
    "Replace 'TODO' with 'DONE' in src/lib.rs",
    "Update the version in Cargo.toml from 0.1.0 to 0.2.0",
    "Fix the typo in README.md: 'teh' should be 'the'",
    "Change the timeout in config.rs from 30 to 60 seconds",
    "Add a new function 'validate' to src/parser.rs",
    "Remove the unused import in src/main.rs",
    "Update the test assertion in tests/test1.rs to expect 42",
    "Rename 'old_name' to 'new_name' in src/utils.rs",
    "Add error handling for the unwrap() in src/network.rs",
    "Change the log level from Info to Debug in src/config.rs",
    "Update the comment in src/lib.rs to explain the algorithm",
    "Fix the off-by-one error in src/parser.rs line 42",
    "Add a doc comment to the public function in src/api.rs",
    "Change the return type in src/handler.rs from String to Result",
    "Update the constant MAX_RETRIES from 3 to 5 in src/config.rs",
    "Replace the println! with tracing::info! in src/main.rs",
    "Add a #[must_use] attribute to src/types.rs",
    "Fix the lifetime annotation in src/extractor.rs",
    "Update the test fixture in tests/fixtures/input.json",
    # Write (10)
    "Create a new file src/utils.rs with an empty module",
    "Write a new test file tests/test_parser.rs",
    "Create a configuration file config/default.toml",
    "Write a new function in src/math.rs that adds two numbers",
    "Create a new module src/auth/mod.rs",
    "Write a hello world in examples/hello.rs",
    "Create a .gitignore file for a Rust project",
    "Write a new integration test in tests/integration.rs",
    "Create documentation file docs/api.md",
    "Write a new error type in src/error.rs",
    # Bash (20)
    "Run cargo build and show the output",
    "Execute cargo test and report results",
    "Run cargo clippy and show warnings",
    "Execute ls -la to list files",
    "Run git status to check repository state",
    "Execute cargo fmt --check to verify formatting",
    "Run cargo audit to check for vulnerabilities",
    "Execute cat Cargo.toml to show dependencies",
    "Run cargo bench to run benchmarks",
    "Execute echo 'hello world'",
    "Run python3 -c 'print(1+1)'",
    "Execute curl -s http://localhost:11434/api/tags",
    "Run nvidia-smi to check GPU status",
    "Execute cargo clean to remove build artifacts",
    "Run git log --oneline -5 to show recent commits",
    "Execute wc -l src/main.rs to count lines",
    "Run grep 'fn ' src/main.rs to find functions",
    "Execute find . -name '*.rs' to find Rust files",
    "Run du -sh target/ to check build size",
    "Execute date to show current time",
    # Glob/Grep (15)
    "Find all Rust files in the src/ directory",
    "Search for 'unwrap()' in all .rs files",
    "Find all test functions in the codebase",
    "Search for TODO comments across all files",
    "Find all public functions in src/lib.rs",
    "Search for error handling patterns in src/",
    "Find all configuration files in the project",
    "Search for 'unsafe' blocks in the codebase",
    "Find all benchmark files",
    "Search for function signatures matching 'fn.*Result'",
    "Find all .toml files in the project",
    "Search for imports of 'anyhow' or 'thiserror'",
    "Find all files larger than 100KB",
    "Search for '#[cfg(test)]' modules",
    "Find all README files in subdirectories",
    # Mixed (15)
    "Read src/main.rs, then edit line 42 to fix the bug",
    "Run cargo test, and if it fails, read the error output",
    "Search for 'TODO' in src/, then create a plan to fix them",
    "Read Cargo.toml, check if serde is a dependency, add it if missing",
    "Run git diff, read the changes, and explain what was modified",
    "Find all unwrap() calls, read each file, and suggest safer alternatives",
    "Read the CI config, then run the same commands locally",
    "Check if the project builds, and if not, read the error and fix it",
    "Read src/parser.rs, understand the API, then write a test for it",
    "Run cargo clippy, read warnings, and fix the top 3 issues",
    "Read the README, check if the installation instructions are correct",
    "Search for 'panic!' in src/, read context, and suggest fixes",
    "Read src/error.rs, understand error types, and add a new variant",
    "Run benchmarks, read results, and identify the slowest test",
    "Read src/config.rs, then update the default values",
]


@dataclass
class ToolCallResult:
    prompt: str
    raw_response: str
    tool_calls: list = field(default_factory=list)
    has_tool_call: bool = False
    parse_success: bool = False
    latency_ms: float = 0.0
    error: Optional[str] = None


def call_ollama_with_tools(prompt: str, model: str = "ornith:9b", temperature: float = 0.6) -> tuple[dict, float]:
    """Call Ollama API with native tool definitions."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": TOOLS,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 20,
            "num_predict": 2048,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latency = (time.time() - start) * 1000
        return data, latency
    except urllib.error.URLError as e:
        latency = (time.time() - start) * 1000
        return {"error": str(e)}, latency


def extract_tool_calls(response: dict) -> list[dict]:
    """Extract tool calls from Ollama response."""
    msg = response.get("message", {})
    tool_calls = msg.get("tool_calls", [])
    return tool_calls if isinstance(tool_calls, list) else []


def run_validation(model: str, count: int, temperature: float = 0.6) -> list[ToolCallResult]:
    """Run validation suite."""
    results = []
    prompts = PROMPTS[:count]

    print(f"\n{'='*60}")
    print(f"Kitty Code Week 0 — Tool-Call Validation (v2)")
    print(f"Model: {model} | Prompts: {count} | Temp: {temperature}")
    print(f"API: Ollama native tool-calling (tools parameter)")
    print(f"{'='*60}\n")

    for i, prompt in enumerate(prompts, 1):
        sys.stdout.write(f"\r[{i:3d}/{count}] {prompt[:55]}...")
        sys.stdout.flush()

        response, latency = call_ollama_with_tools(prompt, model=model, temperature=temperature)
        tool_calls = extract_tool_calls(response)

        has_tc = len(tool_calls) > 0
        valid_tc = any(
            tc.get("function", {}).get("name") in VALID_TOOL_NAMES
            for tc in tool_calls
        )

        result = ToolCallResult(
            prompt=prompt,
            raw_response=json.dumps(response.get("message", {}))[:500],
            tool_calls=tool_calls,
            has_tool_call=has_tc,
            parse_success=valid_tc,
            latency_ms=latency,
        )
        results.append(result)

        status = "✅" if valid_tc else ("⚠️" if has_tc else "❌")
        tc_info = ""
        if tool_calls:
            names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
            tc_info = f" → {', '.join(names)}"
        sys.stdout.write(f"\r[{i:3d}/{count}] {status} {prompt[:50]}...{tc_info} ({latency:.0f}ms)\n")
        sys.stdout.flush()

    return results


def print_summary(results: list[ToolCallResult], temperature: float):
    """Print summary."""
    total = len(results)
    with_tc = sum(1 for r in results if r.has_tool_call)
    valid = sum(1 for r in results if r.parse_success)
    avg_lat = sum(r.latency_ms for r in results) / total if total else 0

    rate = 100 * valid / total if total else 0

    print(f"\n{'='*60}")
    print(f"RESULTS (temp={temperature})")
    print(f"{'='*60}")
    print(f"Total prompts:        {total}")
    print(f"With tool calls:      {with_tc}/{total} ({100*with_tc/total:.1f}%)")
    print(f"Valid tool calls:     {valid}/{total} ({rate:.1f}%)")
    print(f"Average latency:      {avg_lat:.0f}ms")
    print()

    gate = 85.0
    if rate >= gate:
        print(f"✅ GATE PASSED: {rate:.1f}% >= {gate}%")
    else:
        print(f"❌ GATE FAILED: {rate:.1f}% < {gate}%")

    # Per-tool breakdown
    tool_counts = {}
    for r in results:
        for tc in r.tool_calls:
            name = tc.get("function", {}).get("name", "unknown")
            tool_counts[name] = tool_counts.get(name, 0) + 1
    if tool_counts:
        print(f"\nTool call distribution:")
        for name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
            print(f"  {name}: {count}")

    return rate


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ornith:9b")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()

    results = run_validation(args.model, args.count, args.temperature)
    rate = print_summary(results, args.temperature)

    # Save
    output = {
        "model": args.model,
        "temperature": args.temperature,
        "total": len(results),
        "valid_tool_calls": sum(1 for r in results if r.parse_success),
        "rate_pct": rate,
        "details": [
            {
                "prompt": r.prompt,
                "has_tool_call": r.has_tool_call,
                "parse_success": r.parse_success,
                "latency_ms": r.latency_ms,
                "tool_calls": r.tool_calls,
            }
            for r in results
        ],
    }
    with open("week0_toolcall_results_v2.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to week0_toolcall_results_v2.json")

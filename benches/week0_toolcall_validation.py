#!/usr/bin/env python3
"""
Kitty Code Week 0 — Tool-Call Validation Suite
Sends 100 tool-call prompts to ornith:9b via Ollama API.
Measures parse success rate. Gate: ≥85%.

Usage: python3 week0_toolcall_validation.py [--model ornith:9b] [--count 100]
"""

import json
import sys
import time
import re
import urllib.request
import urllib.error
from typing import Optional
from dataclasses import dataclass, field

OLLAMA_URL = "http://localhost:11434/api/chat"

# 100 tool-call prompts: mix of simple, medium, complex
TOOL_CALL_PROMPTS = [
    # --- Read tools (20) ---
    "Read the file src/main.rs",
    "Read lines 10-25 of Cargo.toml",
    "Show me the contents of README.md",
    "Read the config file at .config/settings.json",
    "Display the first 50 lines of tests/test1.rs",
    "Read src/lib.rs and show the public API",
    "Show me the error handling in src/error.rs",
    "Read the CI configuration at .github/workflows/ci.yml",
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

    # --- Edit tools (20) ---
    "In src/main.rs, change the default port from 8080 to 9090",
    "Replace 'TODO' with 'DONE' in src/lib.rs line 15",
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
    "Change the return type in src/handler.rs from String to Result<String>",
    "Update the constant MAX_RETRIES from 3 to 5 in src/config.rs",
    "Replace the println! with tracing::info! in src/main.rs",
    "Add a #[must_use] attribute to src/types.rs",
    "Fix the lifetime annotation in src/extractor.rs",
    "Update the test fixture in tests/fixtures/input.json",

    # --- Write tools (10) ---
    "Create a new file src/utils.rs with an empty module",
    "Write a new test file tests/test_parser.rs",
    "Create a configuration file config/default.toml",
    "Write a new function in src/math.rs that adds two numbers",
    "Create a new module src/auth/mod.rs",
    "Write a simple hello world in a new file examples/hello.rs",
    "Create a .gitignore file for a Rust project",
    "Write a new integration test in tests/integration.rs",
    "Create a documentation file docs/api.md",
    "Write a new error type in src/error.rs",

    # --- Bash tools (20) ---
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

    # --- Glob/Grep tools (15) ---
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

    # --- Mixed/complex (15) ---
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
    "Read src/config.rs, then update the default values in config/default.toml",
]


@dataclass
class ToolCallResult:
    prompt: str
    raw_response: str
    parsed_tool_calls: list = field(default_factory=list)
    has_tool_call: bool = False
    parse_success: bool = False
    latency_ms: float = 0.0
    error: Optional[str] = None


def call_ollama(prompt: str, model: str = "ornith:9b", temperature: float = 0.6) -> tuple[str, float]:
    """Call Ollama API and return (response_text, latency_ms)."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 20,
            "num_predict": 2048,
        }
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
        content = data.get("message", {}).get("content", "")
        return content, latency
    except urllib.error.URLError as e:
        latency = (time.time() - start) * 1000
        return "", latency


def parse_tool_calls(response: str) -> list[dict]:
    """Extract tool calls from ornith response. Handles multiple formats."""
    tool_calls = []

    # Format 1: Qwen3 XML <tool_call>...</tool_call>
    xml_pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    for match in re.finditer(xml_pattern, response, re.DOTALL):
        try:
            tc = json.loads(match.group(1))
            tool_calls.append(tc)
        except json.JSONDecodeError:
            # Try to extract function name and args from XML
            name_match = re.search(r"<name>(.*?)</name>", match.group(1))
            args_match = re.search(r"<arguments>(.*?)</arguments>", match.group(1), re.DOTALL)
            if name_match:
                tool_calls.append({
                    "name": name_match.group(1).strip(),
                    "arguments": json.loads(args_match.group(1)) if args_match else {}
                })

    # Format 2: JSON tool_calls in content
    if not tool_calls:
        json_pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"[^{}]*\}'
        for match in re.finditer(json_pattern, response):
            try:
                tc = json.loads(match.group(0))
                if "name" in tc:
                    tool_calls.append(tc)
            except json.JSONDecodeError:
                pass

    # Format 3: Function call pattern like `function_name(args)`
    if not tool_calls:
        func_pattern = r'(\w+)\((.*?)\)'
        for match in re.finditer(func_pattern, response):
            name = match.group(1)
            if name in ("Read", "Edit", "Write", "Bash", "Glob", "Grep", "ShellStart", "Test",
                         "WebSearch", "WebFetch"):
                tool_calls.append({"name": name, "arguments": {"raw": match.group(2)}})

    return tool_calls


def run_validation(model: str, count: int) -> list[ToolCallResult]:
    """Run the full validation suite."""
    results = []
    prompts = TOOL_CALL_PROMPTS[:count]

    print(f"\n{'='*60}")
    print(f"Kitty Code Week 0 — Tool-Call Validation")
    print(f"Model: {model} | Prompts: {count}")
    print(f"{'='*60}\n")

    for i, prompt in enumerate(prompts, 1):
        sys.stdout.write(f"\r[{i:3d}/{count}] Testing: {prompt[:50]}...")
        sys.stdout.flush()

        raw_response, latency = call_ollama(prompt, model=model)
        tool_calls = parse_tool_calls(raw_response)

        result = ToolCallResult(
            prompt=prompt,
            raw_response=raw_response,
            parsed_tool_calls=tool_calls,
            has_tool_call=len(tool_calls) > 0,
            parse_success=len(tool_calls) > 0 and any(
                tc.get("name") in ("Read", "Edit", "Write", "Bash", "Glob", "Grep",
                                    "ShellStart", "Test", "WebSearch", "WebFetch")
                for tc in tool_calls
            ),
            latency_ms=latency,
        )
        results.append(result)

        status = "✅" if result.parse_success else "❌"
        sys.stdout.write(f"\r[{i:3d}/{count}] {status} {prompt[:50]}... ({latency:.0f}ms)\n")
        sys.stdout.flush()

    return results


def print_summary(results: list[ToolCallResult]):
    """Print validation summary."""
    total = len(results)
    with_tool_call = sum(1 for r in results if r.has_tool_call)
    parse_success = sum(1 for r in results if r.parse_success)
    avg_latency = sum(r.latency_ms for r in results) / total if total else 0

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total prompts:        {total}")
    print(f"With tool calls:      {with_tool_call}/{total} ({100*with_tool_call/total:.1f}%)")
    print(f"Parse success:        {parse_success}/{total} ({100*parse_success/total:.1f}%)")
    print(f"Average latency:      {avg_latency:.0f}ms")
    print()

    # Gate check
    rate = 100 * parse_success / total if total else 0
    gate = 85.0
    if rate >= gate:
        print(f"✅ GATE PASSED: {rate:.1f}% ≥ {gate}%")
    else:
        print(f"❌ GATE FAILED: {rate:.1f}% < {gate}%")
        print(f"   Model cannot reliably produce tool calls. Consider different model or fine-tuning.")

    print()

    # Show failures
    failures = [r for r in results if not r.parse_success]
    if failures:
        print(f"Failed prompts ({len(failures)}):")
        for r in failures[:10]:  # Show first 10
            print(f"  - {r.prompt[:60]}")
            if r.raw_response:
                print(f"    Response snippet: {r.raw_response[:100]}...")
        if len(failures) > 10:
            print(f"  ... and {len(failures)-10} more")

    # Save results
    output = {
        "model": results[0].prompt if results else "unknown",
        "total": total,
        "with_tool_call": with_tool_call,
        "parse_success": parse_success,
        "rate_pct": rate,
        "avg_latency_ms": avg_latency,
        "gate": gate,
        "passed": rate >= gate,
        "details": [
            {
                "prompt": r.prompt,
                "parse_success": r.parse_success,
                "has_tool_call": r.has_tool_call,
                "latency_ms": r.latency_ms,
                "tool_calls": r.parsed_tool_calls,
            }
            for r in results
        ]
    }

    with open("week0_toolcall_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to week0_toolcall_results.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kitty Code Week 0 Tool-Call Validation")
    parser.add_argument("--model", default="ornith:9b", help="Ollama model name")
    parser.add_argument("--count", type=int, default=100, help="Number of prompts to test")
    args = parser.parse_args()

    results = run_validation(args.model, args.count)
    print_summary(results)

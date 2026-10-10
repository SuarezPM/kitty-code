#!/usr/bin/env python3
"""demo-12 justicia little-coder — 12 prompts demo-12.json, timeout 60s/ejercicio, k=3.

Protocolo espejo de `kitty bench demo-12` (mismo modelo/backend/turnos/timeouts).
Modelo: ollama/ornith:9b via pi RPC (little-coder extensions). Max turns 20
(= kitty max_turns demo-12). Grading: mismo predicado único que kitty
`json_to_bench_item` (file_executable XOR check_str+check_file, case-sensitive).
Prompts+checks se cargan de bench/demo-12/demo-12.json (fuente única, sin drift).
Serie: 1 modelo residente, ollama stop entre reps (el llamador hace stop).
Uso: python3 bench/demo-12/rival-little-demo12.py <rep> <out.json>
"""

import json
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
SUITE_JSON = Path(__file__).resolve().parent / "demo-12.json"


def load_exercises():
    suite = json.loads(SUITE_JSON.read_text())
    assert suite["suite"] == "demo-12" and suite["version"] == "1.0.0"
    assert suite["fairness"]["k"] >= 3
    return suite["exercises"]


def check(tmpdir: str, ex: dict) -> bool:
    # Un estilo de check, espejo de kitty json_to_bench_item (case-sensitive).
    if ex.get("check_str") and ex.get("check_file"):
        p = Path(tmpdir) / ex["check_file"]
        try:
            content = p.read_text(errors="ignore")
        except OSError:
            return False
        return ex["check_str"] in content
    if ex.get("check_type") and (ex.get("check_path") or ex.get("check_type") == "always_pass"):
        ctype = ex["check_type"]
        if ctype == "always_pass":
            return True
        p = Path(tmpdir) / ex["check_path"]
        if ctype == "file_executable":
            try:
                return bool(p.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
            except OSError:
                return False
        if ctype == "file_exists":
            return p.is_file()
        if ctype == "file_not_exists":
            return not p.exists()
        if ctype == "file_content_exact":
            try:
                return p.read_text(errors="ignore").strip() == ex.get("expected", "").strip()
            except OSError:
                return False
    return False


def target_file(ex: dict) -> str:
    if ex.get("check_file"):
        return ex["check_file"]
    if ex.get("check_path"):
        return ex["check_path"]
    return "solution.py"


def main():
    rep = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    out = sys.argv[2] if len(sys.argv) > 2 else f"little-demo12-k{rep}.json"
    exercises = load_exercises()
    assert len(exercises) == 12, f"demo-12 fija 12, got {len(exercises)}"
    results = []
    for i, ex in enumerate(exercises):
        print(f"[{i+1}/12] {ex['id']}...", flush=True)
        tmpdir = tempfile.mkdtemp(prefix="little_demo12_")
        target = target_file(ex)
        prompt = (
            f"{ex['prompt']}\n\nWork in directory: {tmpdir} "
            f"— Save to {tmpdir}/{target} (use absolute path)"
        )
        start = time.time()
        verdict, passed = "error", False
        turns, tool_calls = 0, 0
        try:
            with PiRpc(model=MODEL, cwd=tmpdir, max_turns=MAX_TURNS) as rpc:
                r = rpc.prompt_and_collect(prompt, timeout=TIMEOUT)
                turns, tool_calls = r.turn_count, len(r.tool_calls)
            elapsed_ms = (time.time() - start) * 1000
            passed = check(tmpdir, ex)
            verdict = "pass" if passed else "fail"
            latency_ms = elapsed_ms
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            msg = str(e)
            if "timeout" in msg.lower() or latency_ms >= TIMEOUT * 1000 - 500:
                verdict = "timeout"
            print(f"  error: {e}", flush=True)
        if verdict == "timeout":
            latency_ms = 60000.0
        print(
            f"  {'PASS' if passed else 'FAIL'} ({verdict}, {latency_ms:.0f}ms, turns={turns}, tools={tool_calls})",
            flush=True,
        )
        results.append(
            {
                "id": ex["id"],
                "lang": ex.get("lang", ""),
                "passed": passed,
                "verdict": verdict,
                "latency_ms": latency_ms,
                "wall_ms": latency_ms,
                "turns": turns,
                "tool_calls": tool_calls,
            }
        )
    passed_n = sum(1 for r in results if r["passed"])
    print(f"Passed: {passed_n}/12 ({100*passed_n/12:.1f}%)")
    with open(out, "w") as f:
        json.dump(
            {
                "suite": "demo-12",
                "version": "1.0.0",
                "agent": "little-coder (pi RPC 0.83.0)",
                "model": MODEL,
                "rep": rep,
                "timeout_secs": TIMEOUT,
                "max_turns": MAX_TURNS,
                "results": results,
                "passed": passed_n,
                "total": 12,
            },
            f,
            indent=2,
        )
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

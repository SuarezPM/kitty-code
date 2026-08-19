#!/usr/bin/env python3
"""Export benchmark exercises from Python scripts to JSON for Rust harness."""

import ast
import inspect
import json
import os
import re
import sys

# Add benches/ to path to import the scripts
sys.path.insert(0, os.path.dirname(__file__))

# ── Polyglot export ────────────────────────────────────────────

def export_polyglot():
    """Import EXERCISES from run_polyglot_225.py and export to JSON."""
    from run_polyglot_225 import EXERCISES

    outpath = os.path.join(os.path.dirname(__file__), "exercises_polyglot.json")
    with open(outpath, "w") as f:
        json.dump(EXERCISES, f, indent=2)
    print(f"  polyglot: {len(EXERCISES)} exercises → {outpath}")
    return EXERCISES


# ── TB2 export ─────────────────────────────────────────────────

# Setup function → category name mapping
_SETUP_TO_CATEGORY = {
    "setup_file_ops": "file_ops",
    "setup_text_proc": "text_proc",
    "setup_shell": "shell",
    "setup_data_transform": "data_transform",
    "setup_sys_ops": "sys_ops",
}


def _extract_join_path(source, start_pattern):
    """Extract all string args from os.path.join(wd, "a", "b", ...) after a given start pattern."""
    # Find the os.path.join( opening
    m = re.search(r"os\.path\.join\(", source[start_pattern:])
    if not m:
        return None
    # Grab from join( and find the matching closing paren
    join_start = start_pattern + m.start() + len("os.path.join(")
    depth = 1
    i = join_start
    while i < len(source) and depth > 0:
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
        i += 1
    join_content = source[join_start:i - 1]
    # Extract all quoted strings inside the join args
    parts = re.findall(r"['\"]([^'\"]+)['\"]", join_content)
    return "/".join(parts) if parts else None


def _analyze_check_fn(check_fn):
    """Analyze a check lambda's source to determine check_type and params."""
    try:
        source = inspect.getsource(check_fn)
    except (OSError, TypeError):
        return {"check_type": "always_pass"}

    source = source.strip()

    # lambda wd: True
    if re.search(r"lambda\s+\w+\s*:\s*True\s*$", source):
        return {"check_type": "always_pass"}

    # not os.path.isfile(...)
    m = re.search(r"not\s+os\.path\.isfile\(", source)
    if m:
        path = _extract_join_path(source, m.start())
        if path:
            return {"check_type": "file_not_exists", "check_path": path}

    # os.path.isfile(...)
    m = re.search(r"os\.path\.isfile\(", source)
    if m:
        path = _extract_join_path(source, m.start())
        if path:
            return {"check_type": "file_exists", "check_path": path}

    # os.path.isdir(...)
    m = re.search(r"os\.path\.isdir\(", source)
    if m:
        path = _extract_join_path(source, m.start())
        if path:
            return {"check_type": "file_exists", "check_path": path}

    # os.path.islink(...)
    m = re.search(r"os\.path\.islink\(", source)
    if m:
        path = _extract_join_path(source, m.start())
        if path:
            return {"check_type": "file_exists", "check_path": path}

    # os.access(..., os.X_OK)
    m = re.search(r"os\.access\(", source)
    if m and "os.X_OK" in source:
        path = _extract_join_path(source, m.start())
        if path:
            return {"check_type": "file_executable", "check_path": path}

    # open(...).read().strip() == "value"
    m = re.search(r"open\(os\.path\.join\(", source)
    if m:
        path = _extract_join_path(source, m.start())
        m2 = re.search(r"\.read\(\)\.strip\(\)\s*==\s*['\"](.+?)['\"]", source)
        if path and m2:
            return {"check_type": "file_content_exact", "check_path": path, "expected": m2.group(1)}
        if path:
            # open(...).read().strip().isdigit() or similar
            return {"check_type": "file_exists", "check_path": path}

    # Fallback: try to find any file path in the source
    m = re.search(r"['\"]([\w./]+\.\w+)['\"]", source)
    if m:
        return {"check_type": "file_exists", "check_path": m.group(1)}

    return {"check_type": "always_pass"}


def export_tb2():
    """Import TASKS from run_tb2_89.py and export to JSON."""
    from run_tb2_89 import TASKS

    outpath = os.path.join(os.path.dirname(__file__), "exercises_tb2.json")

    exercises = []
    for name, setup_fn, prompt, check_fn in TASKS:
        # Determine category from setup function name
        setup_name = setup_fn.__name__ if hasattr(setup_fn, "__name__") else str(setup_fn)
        category = _SETUP_TO_CATEGORY.get(setup_name, setup_name)

        # Analyze check function
        check_info = _analyze_check_fn(check_fn)

        entry = {
            "id": name,
            "prompt": prompt,
            "setup": category,
            **check_info,
        }
        exercises.append(entry)

    with open(outpath, "w") as f:
        json.dump(exercises, f, indent=2)
    print(f"  tb2: {len(exercises)} tasks → {outpath}")
    return exercises


if __name__ == "__main__":
    export_polyglot()
    export_tb2()

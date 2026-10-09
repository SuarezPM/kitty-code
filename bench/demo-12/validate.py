#!/usr/bin/env python3
"""Validador de la suite fija demo-12 (solo stdlib, sin modelo).

Comprueba: envelope versionado, protocolo de justicia completo, 12
ejercicios con ids únicas y checks compatibles con el esquema
JsonExercise (polyglot-style check_str+check_file XOR tb2-style
check_type+check_path), prompts portables (sin paths absolutos).
Uso: python3 bench/demo-12/validate.py [--json PATH]
"""

import json
import re
import sys
from pathlib import Path

SUITE_DIR = Path(__file__).resolve().parent
DEFAULT_JSON = SUITE_DIR / "demo-12.json"

FAIRNESS_REQUIRED = [
    "protocol",
    "model",
    "backend",
    "pinned_versions",
    "max_turns",
    "timeout_secs_per_exercise",
    "k",
    "grade",
    "gates_off",
    "exclusivity",
    "declared_asymmetries",
    "verdicts",
    "stats",
]

CHECK_TYPES = {
    "file_exists",
    "file_not_exists",
    "file_executable",
    "file_content_exact",
    "always_pass",
}


from typing import NoReturn


def fail(msg: str) -> NoReturn:
    print(f"INVALID: {msg}")
    sys.exit(1)


def main() -> None:
    path = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--json" else DEFAULT_JSON
    try:
        suite = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        fail(f"{path} no es JSON válido: {e}")

    if suite.get("suite") != "demo-12":
        fail("suite != 'demo-12'")
    version = suite.get("version", "")
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        fail(f"version debe ser semver pineada, got {version!r}")

    fairness = suite.get("fairness")
    if not isinstance(fairness, dict):
        fail("falta objeto fairness")
    for key in FAIRNESS_REQUIRED:
        if key not in fairness:
            fail(f"fairness.{key} ausente")
    if not isinstance(fairness.get("k"), int) or fairness["k"] < 3:
        fail(f"fairness.k debe ser entero >= 3, got {fairness.get('k')!r}")
    for gate in ("autorun_hops", "knowledge", "memory", "effort_router"):
        if gate not in fairness.get("gates_off", {}):
            fail(f"fairness.gates_off.{gate} ausente (trampa conocida declarada)")

    exercises = suite.get("exercises")
    if not isinstance(exercises, list) or len(exercises) != 12:
        got = len(exercises) if isinstance(exercises, list) else type(exercises)
        fail(f"exercises debe tener 12 fijos, got {got}")

    ids = [ex.get("id") for ex in exercises]
    if len(set(ids)) != 12 or any(not i for i in ids):
        fail(f"ids deben ser 12 únicas no vacías: {ids}")

    for ex in exercises:
        tag = ex.get("id", "?")
        for field in ("lang", "prompt"):
            if not ex.get(field):
                fail(f"{tag}: campo {field} vacío")
        if "/tmp/" in ex["prompt"] or ex["prompt"].startswith("/"):
            fail(f"{tag}: prompt con path absoluto (usa 'current directory')")
        polyglot = bool(ex.get("check_str")) and bool(ex.get("check_file"))
        tb2 = bool(ex.get("check_type")) and bool(ex.get("check_path") or ex.get("check_type") == "always_pass")
        if polyglot == tb2:
            fail(
                f"{tag}: exactamente un estilo de check "
                "(polyglot check_str+check_file XOR tb2 check_type+check_path)"
            )
        if tb2 and ex["check_type"] not in CHECK_TYPES:
            fail(f"{tag}: check_type desconocido: {ex['check_type']!r}")
        if tb2 and ex["check_type"] == "file_content_exact" and not ex.get("expected"):
            fail(f"{tag}: file_content_exact exige expected")

    print(f"demo-12 v{version}: 12 ejercicios OK, justicia OK ({path})")


if __name__ == "__main__":
    main()

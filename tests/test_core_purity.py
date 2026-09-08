"""The architectural boundary, enforced by CI rather than by discipline.

SPEC 3.1: modules under `baobab_turn/core/` must never import a voice
framework. SPEC 13 lists this as a condition for v1 being done.

Uses `ast` rather than grep so that the word "livekit" appearing in a comment,
a docstring or a marker string does not fail the build -- only real imports do.
"""

from __future__ import annotations

import ast
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent / "baobab_turn" / "core"
PACKS = Path(__file__).resolve().parent.parent / "baobab_turn" / "packs"

# Top-level module names that must never be imported from core/.
FORBIDDEN_ROOTS = {
    "livekit",
    "pipecat",
    "vapi",
    "retell",
    "bland",
    "torch",
    "numpy",  # core is rules, not maths. If this ever needs relaxing, argue for it in review.
}


def _imported_roots(path: Path) -> set[str]:
    """Every top-level module name imported by a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import, which is always in-package.
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_core_imports_no_framework() -> None:
    offenders: list[str] = []
    for path in sorted(CORE.rglob("*.py")):
        bad = _imported_roots(path) & FORBIDDEN_ROOTS
        if bad:
            rel = path.relative_to(CORE.parent.parent)
            offenders.append(f"{rel} imports {sorted(bad)}")

    assert not offenders, (
        "core/ must stay framework-agnostic (SPEC 3.1). Move this into an "
        "adapter:\n  " + "\n  ".join(offenders)
    )


def test_packs_directory_contains_no_logic() -> None:
    """SPEC 4: packs/ is DATA ONLY, except for the registry that loads it.

    A language pack is a YAML file. If a .py file appears here, someone is
    about to make packs executable, which breaks the promise that a
    non-programmer can contribute a language.
    """
    if not PACKS.exists():
        return
    allowed = {"registry.py", "__init__.py"}
    stray = [p.name for p in sorted(PACKS.rglob("*.py")) if p.name not in allowed]
    assert not stray, f"packs/ must contain data, not code. Found: {stray}"

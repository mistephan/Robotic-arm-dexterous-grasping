"""Guard: every RELATIVE import in the wrc_demo package must resolve.

Root cause this pins (2026-09-17): apps/demo.py used ``from .voice.asr
import ...`` inside the real-arm branch. From ``wrc_demo/apps/`` a single
dot means ``wrc_demo.apps.voice`` (nonexistent) -- the correct spelling
is ``..voice``. The mistake survived every check because the import was
function-local and mock tests never execute the real-arm branch; it only
exploded at real-rig startup with ModuleNotFoundError.

This test AST-walks all package sources and importlib-resolves every
``from .x / from ..x`` -- including lazy imports inside rarely-taken
branches -- so that whole class of bug fails loudly in CI instead of on
the rig.
"""

import ast
import importlib
from pathlib import Path

import wrc_demo

SRC = Path(wrc_demo.__file__).parent


def _iter_relative_imports():
    for py in sorted(SRC.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        tree = ast.parse(py.read_text(), filename=str(py))
        rel = py.relative_to(SRC).with_suffix("")
        parts = list(rel.parts)
        # Dropping the last entry gives the containing package for a plain
        # module (agent/llm.py -> agent) AND the package itself for an
        # __init__.py (agent/__init__.py -> agent): both resolve level-1
        # relative imports against wrc_demo.agent.
        pkg_parts = parts[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                # level 1 -> this package, level 2 -> its parent, ...
                up = node.level - 1
                base = pkg_parts[: len(pkg_parts) - up] if up else pkg_parts
                target_parts = base + (node.module.split(".") if node.module else [])
                yield (py, node.lineno, node.level, node.module,
                       ".".join(target_parts))


def test_all_relative_imports_resolve():
    bad, checked = [], 0
    for py, lineno, level, module, target in _iter_relative_imports():
        fullname = f"wrc_demo.{target}" if target else "wrc_demo"
        try:
            importlib.import_module(fullname)
            checked += 1
        except ImportError as e:
            dots = "." * level
            bad.append(f"{py.relative_to(SRC)}:{lineno}  "
                       f"from {dots}{module or ''} -> {fullname}  ({e})")
    assert not bad, "unresolvable relative imports:\n  " + "\n  ".join(bad)
    assert checked > 50, "import scan unexpectedly small -- guard is broken"

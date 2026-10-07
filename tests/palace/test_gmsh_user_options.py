"""Tests that gsim does not read the user's Gmsh options file.

``gmsh.initialize()`` loads ``~/.gmsh-options`` (``gmsh-options`` in
``%GMSH_HOME%`` or ``%APPDATA%`` on Windows) unless told not to, which would
silently change every mesh gsim builds.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src" / "gsim"
_INIT_CALL = re.compile(r"gmsh\.initialize\(([^)]*)\)")
_SCRIPT = (
    "import gmsh\n"
    "gmsh.initialize({args})\n"
    'print(gmsh.option.getNumber("Mesh.MeshSizeFactor"))\n'
    "gmsh.finalize()\n"
)


def _size_factor(env: dict[str, str], args: str) -> float:
    """Return Mesh.MeshSizeFactor seen by a fresh Gmsh initialized with ``args``."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _SCRIPT.format(args=args)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip().splitlines()[-1])


def test_src_never_reads_user_gmsh_options() -> None:
    """Every gmsh.initialize call under src/gsim passes readConfigFiles=False."""
    offenders = []
    for path in sorted(_SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in _INIT_CALL.finditer(text):
            if not re.search(r"readConfigFiles\s*=\s*False", match.group(1)):
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{path.relative_to(_SRC.parents[1])}:{line}")
    assert not offenders, f"gmsh.initialize() reads user options at: {offenders}"


def test_read_config_files_false_ignores_user_options_file(tmp_path: Path) -> None:
    """readConfigFiles=False keeps a user's Mesh.MeshSizeFactor out of Gmsh."""
    for name in (".gmsh-options", "gmsh-options"):
        (tmp_path / name).write_text("Mesh.MeshSizeFactor = 0.5;\n")
    env = dict(os.environ)
    for key in ("GMSH_HOME", "HOME", "USERPROFILE", "APPDATA"):
        env[key] = str(tmp_path)

    if _size_factor(env, "") != 0.5:
        pytest.skip(
            "this Gmsh build does not read the options file from those locations"
        )
    assert _size_factor(env, "readConfigFiles=False") == 1.0

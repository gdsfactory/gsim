"""Check docs execution selection without GitHub access or cloud jobs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import nbformat
import pytest

from scripts.execute_paper_examples import EXAMPLES as REFERENCE_CASES
from scripts.execute_paper_examples import model_fingerprint

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "nbs/fdtd_lu_broadband_coupler.ipynb",
    "nbs/fdtd_sattari_suspended_coupler.ipynb",
    "nbs/fdtd_romero_visible_mmi.ipynb",
    "nbs/fdtd_sacher_visible_mmi.ipynb",
}


@pytest.mark.skipif(sys.platform == "win32", reason="Linux docs workflow")
@pytest.mark.parametrize(
    ("changed", "run_all", "notebooks"),
    [
        ("scripts/execute_paper_examples.py", "false", EXAMPLES),
        ("nbs/data/paper_examples/lu/field_xy.npz", "false", EXAMPLES),
        ("docs/fdtd-benchmark.md", "false", set()),
        ("src/gsim/fdtd/workflow.py", "true", set()),
    ],
)
def test_docs_execution_selection(tmp_path, changed, run_all, notebooks):
    """Reference changes render their examples; benchmark plots launch no notebooks."""
    workflow = (ROOT / ".github/workflows/docs.yml").read_text()
    step = workflow.split("    - name: Detect changed notebooks", 1)[1]
    step = step.split("\n  execute-notebooks:", 1)[0]
    script = textwrap.dedent(step.split("      run: |\n", 1)[1])
    script = script.replace("${{ github.event.pull_request.number }}", "123")
    binary = tmp_path / "bin"
    binary.mkdir()
    gh = binary / "gh"
    gh.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$DOCS_TEST_CHANGED_FILES"\n')
    gh.chmod(0o755)
    output = tmp_path / "output"
    bash = shutil.which("bash")
    assert bash is not None
    subprocess.run(  # noqa: S603 -- fixed workflow script with a local GitHub CLI stub.
        [bash, "-e", "-o", "pipefail", "-c", script],
        env={
            **os.environ,
            "PATH": f"{binary}{os.pathsep}{os.environ['PATH']}",
            "GITHUB_OUTPUT": str(output),
            "DOCS_TEST_CHANGED_FILES": changed,
        },
        capture_output=True,
        text=True,
        check=True,
    )
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values["run_all"] == run_all
    assert set(json.loads(values["notebooks"])) == notebooks


@pytest.mark.parametrize(("stem", "case"), REFERENCE_CASES.items())
def test_checked_in_models_match_reference_fingerprints(stem, case):
    """Verify the checked-in model matches its saved results."""
    notebook = nbformat.read(ROOT / "nbs" / f"{stem}.ipynb", as_version=4)
    provenance = json.loads(
        (ROOT / "nbs/data/paper_examples" / case / "provenance.json").read_text()
    )
    assert model_fingerprint(notebook) == provenance["model_code_sha256"]


@pytest.mark.parametrize(
    ("original", "changed"),
    [
        (
            "from shapely.geometry import LineString",
            "from another_geometry import LineString",
        ),
        ("import numpy as np", "import another_array_library as np"),
    ],
)
def test_reference_fingerprint_detects_import_changes(original, changed):
    """A preserved local symbol must not conceal a different imported model helper."""
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(original)])
    previous = model_fingerprint(notebook)
    notebook.cells[0].source = changed
    assert model_fingerprint(notebook) != previous


def test_reference_fingerprint_ignores_whitespace():
    """Formatting an unchanged model does not invalidate its saved results."""
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "from shapely.geometry import LineString\n"
                "shape = LineString([(0, 0), (1, 1)])"
            )
        ]
    )
    previous = model_fingerprint(notebook)
    notebook.cells[0].source = (
        "from shapely.geometry import (\n    LineString,\n)\n"
        "shape = LineString(\n    [(0, 0), (1, 1)],\n)"
    )
    assert model_fingerprint(notebook) == previous

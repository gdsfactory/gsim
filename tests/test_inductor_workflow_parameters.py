"""The manual inductor workflow must inject a real false boolean."""

from __future__ import annotations

import ast
import json
import shlex
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest


def test_fresh_inductor_cli_disables_cache(tmp_path):
    """Prepare a small notebook with the workflow's actual Papermill arguments."""
    pytest.importorskip(
        "papermill", reason="Papermill prepares docs notebook parameters"
    )
    workflow_path = Path(__file__).resolve().parents[1] / ".github/workflows/docs.yml"
    command = next(
        line.strip()
        for line in workflow_path.read_text(encoding="utf8").splitlines()
        if "papermill" in line and "check_cache" in line
    )
    command_args = shlex.split(command)
    notebook_path = tmp_path / "parameters.ipynb"
    notebook = nbformat.v4.new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        cells=[
            nbformat.v4.new_code_cell(
                "check_cache = True", metadata={"tags": ["parameters"]}
            )
        ],
    )
    nbformat.write(notebook, notebook_path)
    papermill_args = [
        str(notebook_path) if argument == "$NOTEBOOK" else argument
        for argument in command_args[command_args.index("papermill") + 1 :]
    ]
    subprocess.run(  # noqa: S603 -- Fixed runtime and repository-owned CLI options.
        [sys.executable, "-m", "papermill", *papermill_args, "--prepare-only"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    prepared = json.loads(notebook_path.read_text(encoding="utf8"))
    injected = next(
        cell
        for cell in prepared["cells"]
        if "injected-parameters" in cell["metadata"].get("tags", [])
    )
    assignment = next(
        node
        for node in ast.parse("".join(injected["source"])).body
        if isinstance(node, ast.Assign)
    )
    assert ast.literal_eval(assignment.value) is False

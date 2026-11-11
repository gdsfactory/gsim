"""Render paper examples using saved results, without submitting cloud jobs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
import numpy as np
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "fdtd_lu_broadband_coupler": "lu",
    "fdtd_sattari_suspended_coupler": "sattari",
    "fdtd_romero_visible_mmi": "romero",
    "fdtd_sacher_visible_mmi": "sacher",
}
REFERENCE_TAG = "docs_reference_result"


def model_fingerprint(notebook) -> str:
    """Identify model code and imports before the viewer cells, ignoring whitespace."""
    statements = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"plot_2d", "plot_3d"}
            for node in ast.walk(tree)
        ):
            break
        statements.extend(tree.body)
    model = ast.dump(ast.Module(body=statements, type_ignores=[]))
    return hashlib.sha256(model.encode()).hexdigest()


def prepare_reference(asset_directory: Path, output_directory: Path) -> Path:
    """Restore the compressed monitor for native FDTD result loading."""
    document = json.loads((asset_directory / "sparams.json").read_text())
    with np.load(asset_directory / "field_xy.npz") as saved:
        intensity = saved["intensity"]
        extent_nm = (saved["extent_um"] * 1000).tolist()
    np.save(output_directory / "field_xy.npy", intensity)
    wavelength_nm = document["config_resolved"]["center_wavelength"]
    core_midpoints_nm = {"lu": 110, "sattari": 107, "romero": 50, "sacher": 100}
    document["plane_monitors"] = {
        "overview_xy": {
            "normal_axis": "z",
            "normal_sign": 1,
            "u_axis": "x",
            "v_axis": "y",
            "plane_position_nm": core_midpoints_nm[asset_directory.name],
            "u_extent_nm": extent_nm[:2],
            "v_extent_nm": extent_nm[2:],
            "shape": [intensity.shape[1], intensity.shape[0]],
            "wavelength_nm": [wavelength_nm],
            "heatmaps": [
                {
                    "file": "field_xy.npy",
                    "quantity": "intensity",
                    "wavelength_nm": wavelength_nm,
                    "shape": list(intensity.shape),
                }
            ],
        }
    }
    output = output_directory / "sparams.json"
    output.write_text(json.dumps(document))
    return output


def reference_run(simulation, *args, result_path: str, **kwargs):
    """Return this example's completed result instead of making a cloud call."""
    from gsim import fdtd

    result = fdtd.FDTDResult.from_file(result_path)
    recorded = result.config_resolved
    if not np.isclose(
        simulation.solver.cell_size_nm, recorded["nanometers_per_cell"]
    ) or not np.isclose(
        simulation.source.wavelength_um * 1000, recorded["center_wavelength"]
    ):
        raise ValueError("The model no longer matches its reference resolution/band")
    return result


def execute(notebook_path: Path, *, kernel_name: str = "gsim") -> None:
    """Execute normal example code with reference loading injected only in memory."""
    notebook_path = notebook_path.resolve()
    if notebook_path.parent != ROOT / "nbs" or notebook_path.stem not in EXAMPLES:
        raise ValueError("Choose one of the four supported paper example notebooks")
    notebook = nbformat.read(notebook_path, as_version=4)
    asset_directory = ROOT / "nbs/data/paper_examples" / EXAMPLES[notebook_path.stem]
    provenance = json.loads((asset_directory / "provenance.json").read_text())
    if model_fingerprint(notebook) != provenance["model_code_sha256"]:
        raise ValueError("Update reference results before rendering a changed model")
    with TemporaryDirectory(prefix="gsim-docs-reference-") as directory:
        result_path = prepare_reference(asset_directory, Path(directory))
        injected = nbformat.v4.new_code_cell(
            "import sys\n"
            "from functools import partial\n"
            "from unittest.mock import patch\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from gsim import fdtd\n"
            "from scripts.execute_paper_examples import reference_run\n"
            "_reference_patch = patch.object(fdtd.Simulation, 'run', "
            "autospec=True, side_effect=partial(reference_run, "
            f"result_path={str(result_path)!r}))\n"
            "_reference_patch.start()"
        )
        injected.metadata.tags = [REFERENCE_TAG]
        notebook.cells.insert(0, injected)
        try:
            NotebookClient(
                notebook,
                kernel_name=kernel_name,
                timeout=300,
                resources={"metadata": {"path": str(ROOT / "nbs")}},
            ).execute()
        finally:
            notebook.metadata.get("language_info", {}).pop("version", None)
            notebook.cells = [
                cell
                for cell in notebook.cells
                if REFERENCE_TAG not in cell.metadata.get("tags", [])
            ]
            execution_count = 0
            for cell in notebook.cells:
                if cell.cell_type == "code" and cell.execution_count is not None:
                    execution_count += 1
                    cell.execution_count = execution_count
            nbformat.write(notebook, notebook_path)


def main() -> None:
    """Render the selected notebooks from a command line or the docs workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", type=Path, nargs="+")
    parser.add_argument("--kernel", default="gsim")
    arguments = parser.parse_args()
    for notebook in arguments.notebooks:
        execute(notebook, kernel_name=arguments.kernel)


if __name__ == "__main__":
    main()

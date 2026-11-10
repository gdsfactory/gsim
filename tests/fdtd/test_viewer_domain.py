"""Keep viewer geometry and grid estimates in sync with the domain."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from importlib.resources import files
from types import SimpleNamespace
from typing import Any, cast

import meshio
import pytest

from gsim import fdtd
from gsim.fdtd.metadata import _cell_counts
from gsim.fdtd.viz import plot_mesh


@pytest.mark.parametrize(("previous_y_extent", "new_y_extent"), [(1, 2), (2, 1)])
def test_viewer_refreshes_domain_without_overwriting_saved_run(
    tmp_path, fdtd_pdk_module, previous_y_extent, new_y_extent
):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module, default_port="o1")
    simulation.materials(background="SiO2", overrides={"Si": 3.5, "SiO2": 1.44})
    simulation.geometry("straight", settings={"length": 2.0})
    simulation.domain(
        y_bounds=(-previous_y_extent, previous_y_extent), z_bounds=(-0.5, 0.5)
    )
    artifacts = simulation.write(tmp_path / "saved")
    saved_mesh = artifacts.mesh_path.read_bytes()
    original_html = simulation.plot_3d().html

    simulation.domain(y_bounds=(-new_y_extent, new_y_extent))
    updated_html = simulation.plot_3d().html

    embedded_match = re.search(r"const meshText = (.*);", updated_html)
    assert embedded_match is not None
    embedded_json = embedded_match.group(1)
    preview_path = tmp_path / "preview.msh"
    preview_path.write_text(json.loads(embedded_json), encoding="utf8")
    preview_mesh = meshio.read(preview_path)
    assert preview_mesh.points[:, 1].min() == pytest.approx(-new_y_extent * 1000)
    assert preview_mesh.points[:, 1].max() == pytest.approx(new_y_extent * 1000)
    assert updated_html != original_html
    assert artifacts.mesh_path.read_bytes() == saved_mesh
    assert simulation._last_artifacts is artifacts


@pytest.mark.parametrize("method", ["plot_3d", "plot_2d_interactive"])
@pytest.mark.parametrize("edit", ["component", "settings", "mesh_size", "background"])
def test_viewer_refreshes_all_mesh_inputs(
    tmp_path, fdtd_pdk_module, monkeypatch, method, edit
):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module, default_port="o1")
    simulation.materials(background="SiO2", overrides={"Si": 3.5, "SiO2": 1.44})
    simulation.geometry("straight", settings={"length": 2.0})
    artifacts = simulation.write(tmp_path / "saved")
    saved_files = {path: path.read_bytes() for path in (tmp_path / "saved").iterdir()}
    original_mesh = meshio.read(artifacts.mesh_path)
    if edit == "component":
        simulation.geometry.component = fdtd_pdk_module.PDK.cells["straight"](length=4)
        simulation.geometry.settings.clear()
    elif edit == "settings":
        simulation.geometry.settings["length"] = 4
    elif edit == "mesh_size":
        simulation.geometry.mesh_size_nm = 300
    else:
        simulation.materials(background="Si", overrides={"Si": 3.5})
    written = []
    original_write = simulation.write

    def record_write(directory):
        preview = original_write(directory)
        written.append(
            (
                meshio.read(preview.mesh_path),
                json.loads(preview.config_path.read_text()),
            )
        )
        return preview

    monkeypatch.setattr(simulation, "write", record_write)
    viewer = getattr(simulation, method)(
        **({"axis": "x"} if method == "plot_2d_interactive" else {})
    )

    assert len(written) == 1
    preview_mesh, config = written[0]
    if edit in {"component", "settings"}:
        assert preview_mesh.points[:, 0].max() == pytest.approx(4000)
        if method == "plot_2d_interactive":
            assert '"positionUm":2.0' in viewer.html
    elif edit == "mesh_size":
        assert len(preview_mesh.points) > len(original_mesh.points)
    else:
        assert config["background_refractive_index"] == 3.5
    assert simulation._last_artifacts is artifacts
    assert all(path.read_bytes() == content for path, content in saved_files.items())


@pytest.mark.skipif(
    shutil.which("node") is None, reason="Node runs the viewer JavaScript"
)
@pytest.mark.parametrize(
    ("size_nm", "cell_size_nm", "pml_cells", "expected_shape"),
    [
        ((12000, 9000, 2500), 60, 32, [266, 216, 144]),
        ((12000, 6000, 2500), 60, 32, [266, 166, 144]),
        ((12000, 6000, 600), 60, 32, [266, 166, 112]),
        ((1000, 1000, 1000), 100, 0, [12, 12, 48]),
        ((1000, 1000, 1000), 100, 3, [18, 18, 80]),
        ((180, 180, 180), 60, 32, [69, 69, 112]),
        ((1260.00003, 1000, 1000), 60, 0, [23, 19, 64]),
        ((1260.00003, 1260.00003, 960.00003), 60, 0, [23, 23, 64]),
        ((1260.000029, 1260.000029, 960.000029), 60, 0, [23, 23, 48]),
        ((1260.000031, 1260.000031, 960.000031), 60, 0, [24, 24, 64]),
    ],
)
def test_viewer_grid_estimate_matches_metadata(
    tmp_path, size_nm, cell_size_nm, pml_cells, expected_shape
):
    mesh_path = tmp_path / "grid.msh"
    meshio.write(
        mesh_path,
        meshio.Mesh(
            points=[
                [0, 0, 0],
                [size_nm[0], 0, 0],
                [0, size_nm[1], 0],
                [0, 0, size_nm[2]],
            ],
            cells=[("tetra", [[0, 1, 2, 3]])],
            cell_data={"gmsh:physical": [[2]], "gmsh:geometrical": [[1]]},
            field_data={"core": [2, 3]},
        ),
        file_format="gmsh22",
        binary=False,
    )
    viewer = plot_mesh(mesh_path, cell_size_nm=cell_size_nm, pml_cells=pml_cells)
    options_match = re.search(r"const options = (.*);", viewer.html)
    assert options_match is not None
    options = json.loads(options_match.group(1))
    template = files("gsim.fdtd").joinpath("assets/mesh-viewer.html").read_text("utf8")
    stats_code = template[
        template.index("function formatNumber(") : template.index(
            "function buildModel("
        )
    ]
    # Exercise the actual statistics renderer with a minimal DOM and bounding box.
    script = """
const size = INPUT_SIZE;
const options = INPUT_OPTIONS;
const THREE = {
  Vector3: class {
    constructor(...values) { this.values = values; }
    toArray() { return this.values; }
  },
  Box3: class {
    constructor() {
      this.min = [Infinity, Infinity, Infinity];
      this.max = [-Infinity, -Infinity, -Infinity];
    }
    expandByPoint(point) {
      this.min = this.min.map((value, axis) => Math.min(value, point.values[axis]));
      this.max = this.max.map((value, axis) => Math.max(value, point.values[axis]));
    }
    getSize() {
      const values = this.max.map((value, axis) => value - this.min[axis]);
      return new THREE.Vector3(...values);
    }
  },
};
const document = {
  createElement(tag) {
    return {
      tag, children: [],
      append(...children) { this.children.push(...children); },
    };
  },
};
const statsPanel = {
  replaceChildren(...children) { this.children = children; },
};
STATS_CODE
updateStats({ nodes: new Map([[1, [0, 0, 0]], [2, size]]) });
const values = statsPanel.children[1].children.filter((item) => item.tag === 'dd');
const number = (value) => Number(value.replace(/[^0-9]/g, ''));
console.log(JSON.stringify({
  shape: values[1].textContent.split(String.fromCharCode(215)).map(number),
  total: number(values[2].textContent),
}));
"""
    script = (
        script.replace("INPUT_SIZE", json.dumps([value / 1000 for value in size_nm]))
        .replace("INPUT_OPTIONS", json.dumps(options))
        .replace("STATS_CODE", stats_code)
    )
    node_path = shutil.which("node")
    assert node_path is not None
    result = subprocess.run(  # noqa: S603 -- Fixed runtime and repository-owned JS.
        [node_path, "-e", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    shown = json.loads(result.stdout)
    reference = _cell_counts(
        cast(
            Any,
            SimpleNamespace(
                grid=SimpleNamespace(
                    non_pml_bounds=SimpleNamespace(min=[0, 0, 0], max=size_nm),
                    nanometers_per_cell=cell_size_nm,
                    pml_cells=pml_cells,
                )
            ),
        )
    )
    assert shown["shape"] == expected_shape == reference["total_cells_xyz"]
    assert shown["total"] == reference["total_cells"]

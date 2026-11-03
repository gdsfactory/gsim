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


@pytest.mark.parametrize(("previous_y_extent", "new_y_extent"), [(1, 2), (2, 1)])
def test_viewer_refreshes_domain_without_overwriting_saved_run(
    tmp_path, fdtd_pdk_module, previous_y_extent, new_y_extent
):
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module, default_port="o1")
    simulation.materials(overrides={"Si": 3.5, "SiO2": 1.44})
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
    assert simulation._last_artifacts_domain is not None
    assert simulation._last_artifacts_domain["y_bounds"] == (
        -previous_y_extent,
        previous_y_extent,
    )


@pytest.mark.skipif(
    shutil.which("node") is None, reason="Node runs the viewer JavaScript"
)
@pytest.mark.parametrize(
    ("size_um", "cell_size_nm", "pml_cells", "expected_shape"),
    [
        ((12, 9, 2.5), 60, 32, [266, 216, 144]),
        ((12, 6, 2.5), 60, 32, [266, 166, 144]),
        ((12, 6, 0.6), 60, 32, [266, 166, 112]),
        ((1, 1, 1), 100, 0, [12, 12, 48]),
        ((1, 1, 1), 100, 3, [18, 18, 80]),
        ((0.18, 0.18, 0.18), 60, 32, [69, 69, 112]),
    ],
)
def test_viewer_grid_estimate_matches_metadata(
    size_um, cell_size_nm, pml_cells, expected_shape
):
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
        script.replace("INPUT_SIZE", json.dumps(size_um))
        .replace(
            "INPUT_OPTIONS",
            json.dumps({"cellSizeNm": cell_size_nm, "pmlCells": pml_cells}),
        )
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
                    non_pml_bounds=SimpleNamespace(
                        min=[0, 0, 0], max=[value * 1000 for value in size_um]
                    ),
                    nanometers_per_cell=cell_size_nm,
                    pml_cells=pml_cells,
                )
            ),
        )
    )
    assert shown["shape"] == expected_shape == reference["total_cells_xyz"]
    assert shown["total"] == reference["total_cells"]

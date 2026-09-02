"""Tests for the solver-independent interactive Gmsh viewer."""

from __future__ import annotations

from pathlib import Path

import pytest

from gsim.common.viz import MeshViewer, plot_mesh_interactive
from gsim.common.viz.gmsh import generic_pdk_color_palette

_TRIANGLE_MESH = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
1
2 3 "surface"
$EndPhysicalNames
$Nodes
3
1 0 0 0
2 10 0 0
3 0 10 0
$EndNodes
$Elements
1
1 2 2 3 1 1 2 3
$EndElements
"""


def _write_mesh(path: Path) -> Path:
    path.write_text(_TRIANGLE_MESH, encoding="utf8")
    return path


def test_common_viewer_embeds_solver_configuration(tmp_path: Path) -> None:
    viewer = plot_mesh_interactive(
        _write_mesh(tmp_path / "mesh.msh"),
        mesh_unit_um=1.0,
        title="Custom solver",
        footer_title="Custom footer",
        cell_count_label="Custom cells",
        footer_tooltip="Custom explanation",
        port_group_prefixes=("P",),
    )

    assert isinstance(viewer, MeshViewer)
    assert '"meshUnitUm":1.0' in viewer.html
    assert '"title":"Custom solver"' in viewer.html
    assert '"footerTitle":"Custom footer"' in viewer.html
    assert '"cellCountLabel":"Custom cells"' in viewer.html
    assert '"footerTooltip":"Custom explanation"' in viewer.html
    assert '"portGroupPrefixes":["P"]' in viewer.html
    assert '"colorPalette":["#58c7a3","#ff7a7a"]' in viewer.html
    assert '"groupOpacity":null' in viewer.html
    assert '"portColor":"#f4f7fb"' in viewer.html
    assert '"largestGroupColor":"#b8b2b0"' in viewer.html
    assert '"includeInternalGroups":false' in viewer.html
    assert "function isInternalGroup" in viewer.html
    assert "function isOuterBoundaryGroup" in viewer.html
    assert "function groupType" in viewer.html
    assert 'type.className = "group-type"' in viewer.html
    assert "mesh.tetrahedra.length || mesh.triangles.length" in viewer.html


def test_common_viewer_can_include_internal_groups(tmp_path: Path) -> None:
    viewer = plot_mesh_interactive(
        _write_mesh(tmp_path / "mesh.msh"),
        include_internal_groups=True,
    )

    assert '"includeInternalGroups":true' in viewer.html


def test_common_viewer_embeds_custom_group_style(tmp_path: Path) -> None:
    viewer = plot_mesh_interactive(
        _write_mesh(tmp_path / "mesh.msh"),
        color_palette=("#123456", "gold"),
        group_opacity=0.4,
        port_color=None,
        largest_group_color=None,
    )

    assert '"colorPalette":["#123456","gold"]' in viewer.html
    assert '"groupOpacity":0.4' in viewer.html
    assert '"portColor":null' in viewer.html
    assert '"largestGroupColor":null' in viewer.html


def test_generic_pdk_palette_preserves_unique_layer_view_order() -> None:
    palette = generic_pdk_color_palette()

    assert palette[:8] == (
        "#ff9d9d",
        "#c0c0c0",
        "#0ff",
        "#00f",
        "#805000",
        "#c00",
        "#80a8ff",
        "#f00",
    )
    assert len(palette) == len(set(palette))


def test_common_viewer_rejects_invalid_unit_scale(tmp_path: Path) -> None:
    mesh_path = _write_mesh(tmp_path / "mesh.msh")

    with pytest.raises(ValueError, match="mesh_unit_um"):
        plot_mesh_interactive(mesh_path, mesh_unit_um=0)


@pytest.mark.parametrize("color_palette", [(), "#123456", ("",)])
def test_common_viewer_rejects_invalid_palette(
    tmp_path: Path, color_palette: object
) -> None:
    mesh_path = _write_mesh(tmp_path / "mesh.msh")

    with pytest.raises(ValueError, match="color_palette"):
        plot_mesh_interactive(mesh_path, color_palette=color_palette)  # type: ignore[arg-type]


@pytest.mark.parametrize("group_opacity", [-0.1, 1.1, float("nan")])
def test_common_viewer_rejects_invalid_opacity(
    tmp_path: Path, group_opacity: float
) -> None:
    mesh_path = _write_mesh(tmp_path / "mesh.msh")

    with pytest.raises(ValueError, match="group_opacity"):
        plot_mesh_interactive(mesh_path, group_opacity=group_opacity)

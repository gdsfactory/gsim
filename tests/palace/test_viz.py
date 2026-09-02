"""Tests for Palace's shared interactive Gmsh viewer integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from gsim.common.viz import MeshViewer
from gsim.palace import DrivenSim

_TETRAHEDRON_MESH = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
2
2 2 "P1"
3 3 "dielectric"
$EndPhysicalNames
$Nodes
4
1 0 0 0
2 10 0 0
3 0 10 0
4 0 0 10
$EndNodes
$Elements
2
1 2 2 2 1 1 2 3
2 4 2 3 1 1 2 3 4
$EndElements
"""


def test_plot_3d_uses_palace_units_and_footer(tmp_path: Path) -> None:
    simulation = DrivenSim()
    simulation.set_output_dir(tmp_path)
    (tmp_path / "palace.msh").write_text(_TETRAHEDRON_MESH, encoding="utf8")

    viewer = simulation.plot_3d(
        show_mesh=True,
        footer_title="Element statistics",
        cell_count_label="Palace cells",
        height=420,
    )

    assert isinstance(viewer, MeshViewer)
    assert '"mode":"mesh"' in viewer.html
    assert '"meshUnitUm":1.0' in viewer.html
    assert '"title":"GDSFactory Palace"' in viewer.html
    assert '"footerTitle":"Element statistics"' in viewer.html
    assert '"cellCountLabel":"Palace cells"' in viewer.html
    assert '"portGroupPrefixes":["P"]' in viewer.html
    assert '"includeInternalGroups":false' in viewer.html
    assert '"colorPalette":["#ff9d9d","#c0c0c0","#0ff","#00f"' in viewer.html
    assert '"groupOpacity":0.65' in viewer.html
    assert '"portColor":"#f4f7fb"' in viewer.html
    assert '"largestGroupColor":null' in viewer.html
    assert 'height="420"' in viewer._repr_html_()

    internal_viewer = simulation.plot_3d(include_internal_groups=True)
    assert '"includeInternalGroups":true' in internal_viewer.html

    custom_viewer = simulation.plot_3d(
        color_palette=("#123456", "gold"), group_opacity=0.3
    )
    assert '"colorPalette":["#123456","gold"]' in custom_viewer.html
    assert '"groupOpacity":0.3' in custom_viewer.html


def test_plot_3d_requires_generated_mesh(tmp_path: Path) -> None:
    simulation = DrivenSim()
    simulation.set_output_dir(tmp_path)

    with pytest.raises(ValueError, match=r"Call mesh\(\) first"):
        simulation.plot_3d()

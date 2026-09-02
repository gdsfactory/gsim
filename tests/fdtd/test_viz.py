"""Tests for the ZapFDTD-derived interactive Gmsh viewer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from gsim import fdtd
from gsim.fdtd.viz import MeshViewer, plot_mesh

_TETRAHEDRON_MESH = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
1
3 2 "core"
$EndPhysicalNames
$Nodes
4
1 0 0 0
2 1000 0 0
3 0 1000 0
4 0 0 1000
$EndNodes
$Elements
1
1 4 2 2 1 1 2 3 4
$EndElements
"""


def _write_mesh(path: Path) -> Path:
    path.write_text(_TETRAHEDRON_MESH, encoding="utf8")
    return path


def test_viewer_embeds_mesh_and_slice_options(tmp_path: Path) -> None:
    viewer = plot_mesh(
        _write_mesh(tmp_path / "mesh.msh"),
        mode="mesh_slice",
        axis="x",
        position_um=0.25,
        show_groups=["core"],
        cell_size_nm=50,
        pml_cells=8,
        footer_title="Grid estimate",
        cell_count_label="Estimated cells",
        height=420,
    )
    compact_html = "".join(viewer.html.split())

    assert isinstance(viewer, MeshViewer)
    assert '"mode":"mesh_slice"' in viewer.html
    assert '"positionUm":0.25' in viewer.html
    assert '"zoomToCursor":true' in viewer.html
    assert '"cellSizeNm":50' in viewer.html
    assert '"cellCountLabel":"Estimated cells"' in viewer.html
    assert '"footerTitle":"Grid estimate"' in viewer.html
    assert '"meshUnitUm":0.001' in viewer.html
    assert '"pmlCells":8' in viewer.html
    assert '"portGroupPrefixes":["port_"]' in viewer.html
    assert '"includeInternalGroups":false' in viewer.html
    assert '"title":"GDSFactory FDTD"' in viewer.html
    assert "$MeshFormat\\n2.2" in viewer.html
    assert "__GSIM_" not in viewer.html
    assert 'id="mode"' not in viewer.html
    assert 'id="stats" class="section" hidden' in viewer.html
    assert 'id="gizmo-center-target"' in viewer.html
    assert 'id="gizmo-x-target"' in viewer.html
    assert 'id="gizmo-y-target"' in viewer.html
    assert 'id="gizmo-z-target"' in viewer.html
    assert 'id="panel-toggle"' in viewer.html
    assert 'id="viewer-title"' in viewer.html
    assert 'aria-controls="panel-body"' in viewer.html
    assert 'role="radiogroup"' in viewer.html
    assert '<span id="zoom-focus-label" class="muted">Zoom toward</span>' in viewer.html
    assert 'id="zoom-pointer"' in viewer.html
    assert 'id="zoom-center"' in viewer.html
    physical_groups_index = viewer.html.index(
        '<div class="muted">Physical groups</div>'
    )
    zoom_controls_index = viewer.html.index('id="zoom-focus"')
    stats_index = viewer.html.index('id="stats" class="section" hidden')
    assert physical_groups_index < zoom_controls_index < stats_index
    assert 'd="m5 7.5 5 5 5-5"' in viewer.html
    assert 'panel.classList.toggle("collapsed")' in viewer.html
    assert "#panel-toggle{width:100%;" in compact_html
    assert ".gizmo-axis{stroke:#4f9cff;" in compact_html
    assert ".gizmo-axis-targetcircle{fill:#4f9cff;" in compact_html
    assert "camera.up.copy(zUp)" in viewer.html
    assert (
        "constdefaultViewDirection=newTHREE.Vector3(1.8,-1.5,1.8,).normalize();"
        in compact_html
    )
    assert "keyLight.position.set(3, -5, 4)" in viewer.html
    assert "zoomPointer.checked = options.zoomToCursor !== false" in viewer.html
    assert "controls.zoomToCursor = zoomPointer.checked" in viewer.html
    assert "const startTarget = controls.target.clone()" in viewer.html
    assert (
        "const modelBox = new THREE.Box3().setFromObject(currentModel)" in viewer.html
    )
    assert "controls.target.lerpVectors(" in viewer.html
    assert 'button.className = "group-button"' in viewer.html
    assert "options.portGroupPrefixes.some" in viewer.html
    assert "new THREE.Color(0xf4f7fb)" in viewer.html
    assert "constphysicalGroupPalette=[0x58c7a3,0xff7a7a];" in compact_html
    assert "constlargestGroupColor=0xb8b2b0;" in compact_html
    assert "if(group===currentLargestGroup)" in compact_html
    assert "footerTitle.textContent = options.footerTitle" in viewer.html
    assert "options.cellCountLabel" in viewer.html
    assert "\\u2248 ${totalCells.toLocaleString()}" in viewer.html
    assert '"Grid / PML",' not in viewer.html
    assert '"Transfer mesh",' not in viewer.html
    assert "setHSL" not in viewer.html
    assert "function tetrahedronVolume" in viewer.html
    assert "function largestMaterialGroup" in viewer.html
    assert "function orderedGroupEntries" in viewer.html
    assert "constopacity=isLargestMaterial?(isSlice?0.55:0.35):1;" in compact_html
    assert "transparent:isLargestMaterial" in compact_html
    assert "depthWrite:!isLargestMaterial" in compact_html
    assert 'height="420"' in viewer._repr_html_()
    assert viewer.save(tmp_path / "viewer.html").is_file()


def test_simulation_view_methods_reuse_last_mesh(tmp_path: Path) -> None:
    simulation = fdtd.Simulation()
    mesh_path = _write_mesh(tmp_path / "mesh.msh")
    simulation_state = cast(Any, simulation)
    simulation_state._last_artifacts = SimpleNamespace(mesh_path=mesh_path)
    simulation_state._resolved = SimpleNamespace(bounds=((0, -1, 0.1), (2, 1, 0.3)))

    assert '"mode":"surface"' in simulation.plot_3d().html
    assert '"hideGroups":[]' in simulation.plot_3d().html
    assert '"mode":"mesh"' in simulation.plot_3d(show_mesh=True).html
    assert '"cellSizeNm":60.0' in simulation.plot_3d().html
    assert '"pmlCells":32' in simulation.plot_3d().html
    custom_footer = simulation.plot_3d(
        footer_title="Cell estimate",
        cell_count_label="Approximate cells",
    ).html
    assert '"footerTitle":"Cell estimate"' in custom_footer
    assert '"cellCountLabel":"Approximate cells"' in custom_footer
    assert '"zoomToCursor":false' in simulation.plot_3d(zoom_to_cursor=False).html
    assert (
        '"includeInternalGroups":true'
        in simulation.plot_3d(include_internal_groups=True).html
    )
    slice_html = simulation.plot_2d().html
    assert '"mode":"slice"' in slice_html
    assert '"positionUm":0.2' in slice_html
    assert '"mode":"mesh_slice"' in simulation.plot_2d(show_mesh=True).html
    assert '"zoomToCursor":false' in simulation.plot_2d(zoom_to_cursor=False).html
    assert not hasattr(simulation, "plot_mesh")
    assert not hasattr(simulation, "plot_mesh_slice")


def test_simulation_view_can_generate_an_ephemeral_mesh(monkeypatch) -> None:
    simulation = fdtd.Simulation()
    generated_directories = []

    def write(directory: str | Path):
        generated_directories.append(Path(directory))
        mesh_path = _write_mesh(Path(directory) / "mesh.msh")
        artifacts = SimpleNamespace(mesh_path=mesh_path)
        cast(Any, simulation)._last_artifacts = artifacts
        return artifacts

    monkeypatch.setattr(simulation, "write", write)

    viewer = simulation.plot_3d()

    assert generated_directories
    assert '"mode":"surface"' in viewer.html
    assert simulation._last_artifacts is None

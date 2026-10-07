"""Gmsh-level tests for symmetry planes (no Palace run)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import gdsfactory as gf
import gmsh
import pytest

from gsim.common.stack import LayerStack
from gsim.common.stack.extractor import Layer
from gsim.palace import DrivenSim
from gsim.palace.mesh.generator import generate_mesh
from gsim.palace.mesh.geometry import (
    add_metals,
    add_pec_blocks,
    add_ports,
    extract_geometry,
    extract_pec_polygons,
)
from gsim.palace.mesh.validation import validate_mesh
from gsim.palace.models.pec import PECBlockConfig
from gsim.palace.models.symmetry import SymmetryPlaneConfig
from gsim.palace.ports.config import PalacePort, PortGeometry, PortType


def _m1():
    """Return the gpdk M1 layer as a ``(layer, datatype)`` tuple."""
    gf.gpdk.PDK.activate()
    layer = gf.gpdk.LAYER.M1
    return layer, (int(layer[0]), int(layer[1]))


def _stack() -> LayerStack:
    """One conductor layer on the gpdk M1 GDS layer."""
    _, gds = _m1()
    return LayerStack(
        layers={
            "metal1": Layer(
                name="metal1",
                gds_layer=gds,
                zmin=0.0,
                zmax=2.0,
                thickness=2.0,
                material="aluminum",
                layer_type="conductor",
            )
        }
    )


def _gsg(ground_shift: float = 0.0):
    """GSG strips, symmetric about y=0 unless the top ground is shifted."""
    layer, _ = _m1()
    c = gf.Component()
    for y0, y1 in [(-5.0, 5.0), (11.0, 61.0), (-61.0, -11.0)]:
        shift = ground_shift if y0 > 0 else 0.0
        c.add_polygon(
            [(0, y0 + shift), (100, y0 + shift), (100, y1 + shift), (0, y1 + shift)],
            layer=layer,
        )
    return c


def _plane(**kwargs) -> SymmetryPlaneConfig:
    """Plane at y=0 keeping the positive side."""
    return SymmetryPlaneConfig(**{"axis": "y", "position": 0.0, **kwargs})


def _is_flat(dim: int, tag: int, position: float = 0.0, tol: float = 1e-3) -> bool:
    """True if the entity's bbox is flat at y=position."""
    box = gmsh.model.occ.getBoundingBox(dim, tag)
    return abs(box[1] - position) < tol and abs(box[4] - position) < tol


@pytest.fixture
def gmsh_model():
    """Fresh gmsh model, finalized afterwards."""
    gmsh.initialize()
    gmsh.model.add("sym")
    yield gmsh.model.occ
    gmsh.finalize()


def test_extract_geometry_clips_to_kept_side():
    """Polygons are clipped, while bbox stays the full layout bbox."""
    geometry = extract_geometry(_gsg(), _stack(), symmetry_plane=_plane())
    assert geometry.bbox[1] == pytest.approx(-61.0)
    assert geometry.bbox[3] == pytest.approx(61.0)
    ys = [y for _, _, pts_y, _ in geometry.polygons for y in pts_y]
    assert min(ys) == pytest.approx(0.0)
    assert max(ys) == pytest.approx(61.0)
    layer_bbox = next(iter(geometry.layer_bboxes.values()))
    assert layer_bbox[1] == pytest.approx(0.0)


def test_extract_geometry_without_plane_is_unchanged():
    """No plane means the full polygons."""
    geometry = extract_geometry(_gsg(), _stack())
    ys = [y for _, _, pts_y, _ in geometry.polygons for y in pts_y]
    assert min(ys) == pytest.approx(-61.0)
    assert len(geometry.polygons) == 3


def test_extract_geometry_negative_keep():
    """keep='negative' keeps y <= position."""
    geometry = extract_geometry(
        _gsg(), _stack(), symmetry_plane=_plane(keep="negative")
    )
    ys = [y for _, _, pts_y, _ in geometry.polygons for y in pts_y]
    assert max(ys) == pytest.approx(0.0)


def test_extract_geometry_rejects_asymmetric_layout():
    """An asymmetric layout raises unless verification is off."""
    component = _gsg(ground_shift=3.0)
    with pytest.raises(ValueError, match="not mirror-symmetric"):
        extract_geometry(component, _stack(), symmetry_plane=_plane())
    extract_geometry(component, _stack(), symmetry_plane=_plane(verify_symmetry=False))


def test_extract_geometry_rejects_empty_kept_side():
    """Clipping that leaves nothing raises."""
    with pytest.raises(ValueError, match="no geometry"):
        extract_geometry(
            _gsg(),
            _stack(),
            symmetry_plane=_plane(position=100.0, verify_symmetry=False),
        )


def test_extract_pec_polygons_clips():
    """PEC block polygons are clipped to the kept side."""
    _, gds = _m1()
    polys = extract_pec_polygons(_gsg(), gds, symmetry_plane=_plane())
    ys = [y for _, pts_y, _ in polys for y in pts_y]
    assert min(ys) == pytest.approx(0.0)


def test_extract_pec_polygons_rejects_asymmetric_layout():
    """PEC block layers are verified too."""
    _, gds = _m1()
    with pytest.raises(ValueError, match="not mirror-symmetric"):
        extract_pec_polygons(_gsg(3.0), gds, symmetry_plane=_plane())


class _Kernel:
    """Kernel stub for ``add_ports`` with a patched rectangle factory."""

    def synchronize(self) -> None:
        """No-op."""


@pytest.fixture
def rects(monkeypatch):
    """Record the rectangles ``add_ports`` asks for."""
    calls: list[tuple[float, ...]] = []

    def _fake(_kernel, xmin, ymin, zmin, xmax, ymax, zmax):
        calls.append((xmin, ymin, zmin, xmax, ymax, zmax))
        return len(calls)

    monkeypatch.setattr(
        "gsim.palace.mesh.geometry.gmsh_utils.create_port_rectangle", _fake
    )
    return calls


def _ports(*ports, plane=None, bounds=None):
    """Run ``add_ports`` on the metal1 stack."""
    return add_ports(
        _Kernel(),
        list(ports),
        _stack(),
        domain_bounds=bounds,
        symmetry_plane=plane,
    )


def _lumped(center, name="o1"):
    """Lumped inplane port with its gap along y."""
    return PalacePort(
        name=name,
        port_type=PortType.LUMPED,
        geometry=PortGeometry.INPLANE,
        center=center,
        width=4.0,
        length=6.0,
        orientation=90.0,
        layer="metal1",
    )


def _wave(center, orientation=0.0, **kwargs):
    """Wave port on metal1."""
    return PalacePort(
        name="o1",
        port_type=PortType.WAVEPORT,
        geometry=PortGeometry.INPLANE,
        center=center,
        width=4.0,
        orientation=orientation,
        layer="metal1",
        **kwargs,
    )


def _cpw(centers):
    """CPW port with one element per center."""
    return PalacePort(
        name="o1",
        port_type=PortType.LUMPED,
        geometry=PortGeometry.CPW,
        multi_element=True,
        centers=centers,
        directions=["y"] * len(centers),
        width=4.0,
        length=2.0,
        orientation=0.0,
        layer="metal1",
    )


def test_lumped_port_on_kept_side_is_unchanged(rects):
    """A kept-side lumped port is built as without a plane."""
    _ports(_lumped((10.0, 8.0)), plane=_plane())
    assert rects[0][1] == pytest.approx(5.0)
    assert rects[0][4] == pytest.approx(11.0)


@pytest.mark.usefixtures("rects")
def test_lumped_port_straddling_plane_raises():
    """A lumped port across the plane is rejected by name."""
    with pytest.raises(ValueError, match="'o1' straddles the symmetry plane"):
        _ports(_lumped((10.0, 0.0)), plane=_plane())


@pytest.mark.usefixtures("rects")
def test_port_on_removed_side_raises():
    """A port in the removed half is rejected."""
    with pytest.raises(ValueError, match="removed half"):
        _ports(_lumped((10.0, -8.0)), plane=_plane())


def test_port_on_removed_side_ok_for_negative_keep(rects):
    """keep='negative' flips which side is allowed."""
    _ports(_lumped((10.0, -8.0)), plane=_plane(keep="negative"))
    assert len(rects) == 1


@pytest.mark.usefixtures("rects")
def test_cpw_port_with_gaps_on_both_sides_raises():
    """A CPW port whose element lies on the removed side is rejected."""
    with pytest.raises(ValueError, match="removed half"):
        _ports(_cpw([(0.0, 8.0), (0.0, -8.0)]), plane=_plane())


@pytest.mark.usefixtures("rects")
def test_cpw_element_straddling_plane_gives_hint():
    """A straddling CPW element suggests a single-element lumped port."""
    with pytest.raises(ValueError, match="single-element lumped"):
        _ports(_cpw([(0.0, 0.0)]), plane=_plane())


def test_waveport_max_size_uses_clamped_bounds(rects):
    """A max_size wave port spans plane to domain edge."""
    _, info = _ports(
        _wave((5.0, 0.0), max_size=True),
        plane=_plane(),
        bounds=(-100.0, 0.0, -20.0, 120.0, 80.0, 40.0),
    )
    assert rects[0][1] == pytest.approx(0.0)
    assert rects[0][4] == pytest.approx(80.0)
    assert not info[0].get("cut_by_symmetry_plane", False)


def test_waveport_centred_on_plane_is_halved(rects):
    """A wave port across the plane is clipped and flagged."""
    _, info = _ports(_wave((5.0, 0.0), lateral_margin=1.0), plane=_plane(), bounds=None)
    assert rects[0][1] == pytest.approx(0.0)
    assert rects[0][4] == pytest.approx(3.0)
    assert info[0]["width"] == pytest.approx(3.0)
    assert info[0]["cut_by_symmetry_plane"] is True


@pytest.mark.usefixtures("rects")
def test_waveport_lying_in_plane_raises():
    """A wave port whose face is on the plane is rejected."""
    with pytest.raises(ValueError, match="in the plane"):
        _ports(_wave((5.0, 0.0), orientation=90.0), plane=_plane())


def test_ports_without_plane_unchanged(rects):
    """No plane means no checks."""
    _ports(_lumped((10.0, 0.0)), _lumped((10.0, -8.0), "o2"))
    assert len(rects) == 2


def _flat_shells(tag_info) -> list[int]:
    """Shell faces of one layer that are flat at y=0."""
    return [
        tag
        for _vol, surfaces in tag_info["volumes"]
        for tag in surfaces
        if _is_flat(2, tag)
    ]


def test_add_metals_drops_shell_faces_on_plane(gmsh_model):
    """The cut face of a conductor is not a conductor shell face."""
    stack = _stack()
    geometry = extract_geometry(_gsg(), stack)
    full = add_metals(gmsh_model, geometry, stack)
    assert _flat_shells(full["metal1"]) == []  # nothing on y=0 in the full model

    gmsh.clear()
    gmsh.model.add("half")
    plane = _plane()
    geometry = extract_geometry(_gsg(), stack, symmetry_plane=plane)
    half = add_metals(gmsh.model.occ, geometry, stack, symmetry_plane=plane)
    assert _flat_shells(half["metal1"]) == []
    # The cut conductor still has its other faces.
    assert any(len(s) > 0 for _v, s in half["metal1"]["volumes"])


def test_add_metals_without_drop_would_keep_plane_face(gmsh_model):
    """Control: clipping alone leaves a face on the plane."""
    stack = _stack()
    plane = _plane()
    geometry = extract_geometry(_gsg(), stack, symmetry_plane=plane)
    tags = add_metals(gmsh_model, geometry, stack)
    assert len(_flat_shells(tags["metal1"])) == 1


def test_add_pec_blocks_drops_faces_on_plane(gmsh_model):
    """PEC block surfaces on the plane are not named block surfaces."""
    _, gds = _m1()
    plane = _plane()
    cfg = PECBlockConfig(from_layer="metal1", to_layer="metal1", gds_layer=gds)
    blocks = add_pec_blocks(gmsh_model, _gsg(), [cfg], _stack(), symmetry_plane=plane)
    tags = blocks["pec_block_0"]
    assert not [t for t in tags["surfaces_z"] + tags["surfaces_xy"] if _is_flat(2, t)]
    assert tags["surfaces_z"]


def _make_component(gssg: bool):
    """500 um line symmetric about y=0: GSG (signal on y=0) or GSSG (gap on y=0)."""
    layer, _ = _m1()
    length = 500.0
    if gssg:
        strips = [(-61.0, -11.0), (-8.0, -3.0), (3.0, 8.0), (11.0, 61.0)]
    else:
        strips = [(-61.0, -11.0), (-5.0, 5.0), (11.0, 61.0)]
    c = gf.Component()
    for y0, y1 in strips:
        c.add_polygon(
            [(-length / 2, y0), (length / 2, y0), (length / 2, y1), (-length / 2, y1)],
            layer=layer,
        )
    for name, x, orientation in [("o1", -length / 2, 0), ("o2", length / 2, 180)]:
        c.add_port(
            name=name,
            center=(x, 0),
            width=10,
            orientation=orientation,
            port_type="electrical",
            layer=layer,
        )
    return c


def _mesh_sim(tmp_path, *, gssg, kind, max_size):
    """Mesh a half model of the GSG or GSSG line with wave ports."""
    sim = DrivenSim()
    sim.set_output_dir(str(tmp_path / "palace-sim"))
    sim.set_geometry(_make_component(gssg))
    sim.set_stack(substrate_thickness=2.0, air_above=300.0)
    for name in ("o1", "o2"):
        if max_size:
            sim.add_wave_port(name, layer="metal1", max_size=True)
        else:
            sim.add_wave_port(name, layer="metal1", z_margin=20.0, lateral_margin=20.0)
    sim.set_driven(fmin=1e9, fmax=100e9, num_points=5)
    sim.add_symmetry_plane(axis="y", position=0.0, kind=kind)
    sim.mesh(preset="coarse")
    return sim


@pytest.fixture(scope="module")
def gsg_sim(tmp_path_factory):
    """GSG half model, PMC plane through the signal, max_size wave ports."""
    return _mesh_sim(
        tmp_path_factory.mktemp("sym_gsg"), gssg=False, kind="pmc", max_size=True
    )


@pytest.fixture(scope="module")
def gssg_sim(tmp_path_factory):
    """GSSG half model, PEC plane in the gap, small wave ports."""
    return _mesh_sim(
        tmp_path_factory.mktemp("sym_gssg"), gssg=True, kind="pec", max_size=False
    )


def _mesh_faces_by_group(msh_path, tol=1e-6):
    """Return ``{group name: True if some face is flat at y=0}`` and min node y."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.open(str(msh_path))
        flat_groups = {}
        for dim, pg in gmsh.model.getPhysicalGroups(2):
            name = gmsh.model.getPhysicalName(dim, pg)
            flat = False
            for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, pg):
                _, coords, _ = gmsh.model.mesh.getNodes(2, int(ent), True)
                ys = coords[1::3]
                # Entities with fewer than 3 nodes carry no triangle
                if len(ys) >= 3 and float(abs(ys).max()) < tol:
                    flat = True
            flat_groups[name] = flat
        _, coords, _ = gmsh.model.mesh.getNodes()
        return flat_groups, float(coords[1::3].min())
    finally:
        gmsh.finalize()


@pytest.mark.parametrize("fixture", ["gsg_sim", "gssg_sim"])
def test_symmetry_group_exists(fixture, request):
    """The plane has its own group with the right axis, kind and flat faces."""
    sim = request.getfixturevalue(fixture)
    result = sim._last_mesh_result
    info = result.groups["boundary_surfaces"]["symmetry"]
    kind = sim._symmetry_planes[0].kind
    assert (info["axis"], info["kind"], info["position"]) == ("y", kind, 0.0)
    assert info["tags"]
    assert result.symmetry_plane["kind"] == kind
    assert "absorbing" in result.groups["boundary_surfaces"]


@pytest.mark.parametrize("fixture", ["gsg_sim", "gssg_sim"])
def test_nothing_crosses_the_plane(fixture, request):
    """Only the plane group is flat at y=0, and no node lies below it."""
    sim = request.getfixturevalue(fixture)
    kind = sim._symmetry_planes[0].kind
    flat_groups, min_y = _mesh_faces_by_group(sim._last_mesh_result.mesh_path)
    assert min_y >= -1e-6
    assert [n for n, flat in flat_groups.items() if flat] == [f"symmetry_y_{kind}"]


def test_plane_group_is_not_absorbing(gsg_sim):
    """Plane faces are not in any ``*__None`` group."""
    groups = gsg_sim._last_mesh_result.groups["boundary_surfaces"]
    sym_pg = groups["symmetry"]["phys_group"]
    assert not set(sym_pg) & set(groups["absorbing"]["phys_group"])


def test_cut_conductor_has_no_z_face_on_plane(gsg_sim):
    """No ``metal1_z`` face lies on the plane (flat groups checked above)."""
    flat_groups, _ = _mesh_faces_by_group(gsg_sim._last_mesh_result.mesh_path)
    assert not flat_groups.get("metal1_z", False)


def test_max_size_waveport_spans_plane_to_domain_edge(gsg_sim):
    """P1 runs from y=0 to the far domain edge."""
    info = gsg_sim._last_mesh_result.port_info[0]
    assert info["ymin"] == pytest.approx(0.0)
    assert info["ymax"] > 50.0
    assert not info.get("cut_by_symmetry_plane", False)


def test_small_waveport_is_halved(gssg_sim):
    """A wave port centred on the plane keeps half its width and is flagged."""
    info = gssg_sim._last_mesh_result.port_info[0]
    assert info["ymin"] == pytest.approx(0.0)
    assert info["width"] == pytest.approx(info["ymax"])
    assert info["cut_by_symmetry_plane"] is True


@pytest.mark.parametrize(("fixture", "key"), [("gsg_sim", "PMC"), ("gssg_sim", "PEC")])
def test_write_config_and_validate_mesh(fixture, key, request):
    """The written config carries the plane and ``validate_mesh`` accepts it."""
    sim = request.getfixturevalue(fixture)
    sim.write_config()
    config = json.loads((sim._output_dir / "config.json").read_text())
    pg = sim._last_mesh_result.groups["boundary_surfaces"]["symmetry"]["phys_group"]
    boundaries = config["Boundaries"]
    assert set(pg) <= set(boundaries[key]["Attributes"])
    assert not set(pg) & set(boundaries["Absorbing"]["Attributes"])
    result = validate_mesh(sim)
    assert result.valid, result.errors
    assert any("Symmetry plane" in w for w in result.warnings)


def _fake_sim(tmp_path, config, *, kind="pmc"):
    """Minimal sim stand-in for ``validate_mesh`` with a hand-written config."""
    (tmp_path / "config.json").write_text(json.dumps(config))
    groups = {
        "volumes": {"air": {}},
        "conductor_surfaces": {"metal1_z": {"phys_group": 4}},
        "port_surfaces": {"P1": {"phys_group": 6}},
        "boundary_surfaces": {
            "absorbing": {"phys_group": [8]},
            "symmetry": {
                "phys_group": [30],
                "axis": "y",
                "position": 0.0,
                "kind": kind,
                "keep": "positive",
            },
        },
    }
    return SimpleNamespace(
        simulation_type="driven",
        _last_mesh_result=SimpleNamespace(
            groups=groups, mesh_path=tmp_path / "missing.msh"
        ),
        _output_dir=tmp_path,
        write_config=lambda **_kwargs: None,
    )


_GOOD_BOUNDARIES = {
    "Conductivity": [{"Attributes": [4]}],
    "WavePort": [{"Index": 1}],
    "Absorbing": {"Attributes": [8]},
}


def test_validate_mesh_flags_plane_missing_from_config(tmp_path):
    """A config without the plane attribute is an error."""
    sim = _fake_sim(tmp_path, {"Boundaries": _GOOD_BOUNDARIES})
    errors = validate_mesh(sim).errors
    assert any("PMC" in e and "30" in e for e in errors)


def test_validate_mesh_flags_plane_under_absorbing(tmp_path):
    """A plane attribute listed as absorbing is an error."""
    boundaries = {
        **_GOOD_BOUNDARIES,
        "PMC": {"Attributes": [30]},
        "Absorbing": {"Attributes": [8, 30]},
    }
    errors = validate_mesh(_fake_sim(tmp_path, {"Boundaries": boundaries})).errors
    assert any("Absorbing" in e for e in errors)


def test_validate_mesh_accepts_pec_plane(tmp_path):
    """A PEC plane is expected under PEC."""
    boundaries = {**_GOOD_BOUNDARIES, "PEC": {"Attributes": [30]}}
    sim = _fake_sim(tmp_path, {"Boundaries": boundaries}, kind="pec")
    assert validate_mesh(sim).valid


def test_periodic_axis_equal_to_plane_axis_raises(tmp_path):
    """periodic_axis on the plane axis is rejected before meshing."""
    with pytest.raises(ValueError, match="periodic_axis"):
        generate_mesh(
            _gsg(),
            _stack(),
            [],
            tmp_path,
            periodic_axis="y",
            symmetry_plane=_plane(),
        )


def test_boundarymode_with_plane_raises(tmp_path):
    """The native 2D boundarymode path rejects a plane."""
    with pytest.raises(ValueError, match="boundarymode"):
        generate_mesh(
            _gsg(),
            _stack(),
            [],
            tmp_path,
            simulation_type="boundarymode",
            symmetry_plane=_plane(),
        )


def test_plane_outside_domain_raises(tmp_path):
    """A plane beyond the domain raises a ValueError naming the domain."""
    sim = DrivenSim()
    sim.set_output_dir(str(tmp_path / "palace-sim"))
    sim.set_geometry(_make_component(gssg=False))
    sim.set_stack(substrate_thickness=2.0, air_above=300.0)
    sim.add_wave_port("o1", layer="metal1", max_size=True)
    sim.set_driven(fmin=1e9, fmax=100e9, num_points=5)
    sim.add_symmetry_plane(
        axis="y", position=5000.0, kind="pmc", keep="negative", verify_symmetry=False
    )
    with pytest.raises(ValueError, match="outside the simulation domain"):
        sim.mesh(preset="coarse")


def test_asymmetric_layout_raises_through_sim_mesh(tmp_path):
    """The mirror check runs through ``sim.mesh``."""
    sim = DrivenSim()
    sim.set_output_dir(str(tmp_path / "palace-sim"))
    sim.set_geometry(_gsg(ground_shift=3.0))
    sim.set_stack(substrate_thickness=2.0, air_above=300.0)
    sim.set_driven(fmin=1e9, fmax=100e9, num_points=5)
    sim.add_symmetry_plane(axis="y", position=0.0, kind="pmc")
    with pytest.raises(ValueError, match="not mirror-symmetric"):
        sim.mesh(preset="coarse")

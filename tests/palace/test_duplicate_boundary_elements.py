"""A mesh must not hold the same boundary element twice (#396).

gmsh writes an element once for every physical group it belongs to. Two groups on
one face give Palace a face with several boundary elements, and it stops with
"A non-periodic face cannot have multiple boundary elements".

* Native 2D BoundaryMode: two background dielectrics that overlap in z both
  claimed the overlap (fixed: the region listed first in the stack owns it).
  The claim is made per stack region: two regions of one material (an SiO2 box
  and an SiO2 cladding) are not merged before it, whatever is listed between.
* Every mesh is checked before it is written (``DuplicateBoundaryElementsError``).
* The 3D configurations with an air or gas slab at the airbox top never
  duplicated anything on main; they are kept here as a guard.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import gdsfactory as gf
import gmsh
import pytest

from gsim.common.stack import get_stack
from gsim.common.stack.extractor import Layer
from gsim.palace import BoundaryModeSim
from gsim.palace.mesh import generator, gmsh_utils
from tests._electrode_layouts import FULL, MESH, component, stack

GAS = {"type": "dielectric", "permittivity": 1.0001}
POLYMER = {"type": "dielectric", "permittivity": 2.5}
AIR = {"type": "dielectric", "permittivity": 1.0}
SIGNAL = ("m1", -20, 0, 20, 4)
GROUND = ("m1", -20, 8, 20, 12)
LINE, TRIANGLE = 1, 2


def _read_msh(path: Path):
    """Names, node coordinates and ``(type, group, nodes)`` of a text MSH 2.2."""
    text = Path(path).read_text()
    block = text.split("$PhysicalNames")[1].split("$EndPhysicalNames")[0]
    names = {}
    for line in block.strip().splitlines()[1:]:
        match = re.match(r'(\d+)\s+(\d+)\s+"(.*)"', line.strip())
        if match:
            names[(int(match.group(1)), int(match.group(2)))] = match.group(3)
    lines = text.splitlines()
    start = lines.index("$Nodes")
    xyz = {}
    for line in lines[start + 2 : start + 2 + int(lines[start + 1])]:
        node, x, y, z = line.split()
        xyz[int(node)] = (float(x), float(y), float(z))
    start = lines.index("$Elements")
    elements = []
    for line in lines[start + 2 : start + 2 + int(lines[start + 1])]:
        fields = [int(v) for v in line.split()]
        elements.append((fields[1], fields[3], fields[3 + fields[2] :]))
    return names, xyz, elements


def _duplicates(msh_path: Path, kind: int) -> dict[tuple[int, ...], list[str]]:
    """Elements of one type (1 = line, 2 = triangle) listed more than once.

    Maps the sorted node ids to the names of the groups listing them.
    """
    names, _, elements = _read_msh(msh_path)
    dim = {LINE: 1, TRIANGLE: 2}[kind]
    seen: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for etype, phys, nodes in elements:
        if etype == kind:
            seen[tuple(sorted(nodes))].append(names.get((dim, phys), str(phys)))
    return {nodes: groups for nodes, groups in seen.items() if len(groups) > 1}


def _vertical_ranges(msh_path: Path) -> dict[str, tuple[float, float]]:
    """``{group: (lowest, highest)}`` of the triangles of each 2D domain group."""
    names, xyz, elements = _read_msh(msh_path)
    out: dict[str, list[float]] = {}
    for etype, phys, nodes in elements:
        if etype == TRIANGLE:
            ys = [xyz[n][1] for n in nodes]
            lo, hi = out.setdefault(names[(2, phys)], [min(ys), max(ys)])
            out[names[(2, phys)]] = [min(lo, *ys), max(hi, *ys)]
    return {name: (round(lo, 6), round(hi, 6)) for name, (lo, hi) in out.items()}


# --------------------------------------------------------------- native 2D overlap
def _mesh_2d(tmp_path: Path, *, generic: bool, slab: tuple, first: bool = False):
    """Mesh a BoundaryMode cross-section whose stack has a slab over the oxide."""
    gf.gpdk.PDK.activate()
    name, z0, z1, z_above = slab
    if generic:
        st = stack()
        st.dielectrics = [{"name": "ox", "zmin": -2.0, "zmax": 5.0, "material": "sio2"}]
        st.materials["gas"] = GAS
        material = "gas"
        comp = component([SIGNAL, GROUND], ports=[])
        section, box, mesh = "x=0", {"margin_y": 5.0, "z_below": 2.0}, MESH
    else:
        st = get_stack(substrate_thickness=2.0)
        material = "air"
        comp = gf.Component()
        comp.add_polygon([(0, 0), (40, 0), (40, 4), (0, 4)], layer=(49, 0))
        comp.add_polygon([(0, 10), (40, 10), (40, 14), (0, 14)], layer=(49, 0))
        section, box = "x=20", {"margin_y": 0.0, "z_below": 0.0}
        mesh = {"refined_mesh_size": 0.5, "max_mesh_size": 3.0, "merge_via_distance": 0}
    slab_entry = {"name": name, "zmin": z0, "zmax": z1, "material": material}
    if first:
        st.dielectrics.insert(0, slab_entry)
    else:
        st.dielectrics.append(slab_entry)

    sim = BoundaryModeSim()
    sim.set_output_dir(str(tmp_path))
    sim.set_geometry(comp)
    sim.set_stack(st)
    sim.set_airbox(margin_x=0.0, z_above=z_above, **box)
    sim.set_cross_section(section)
    sim.set_boundary_mode(freq=10e9, num_modes=2, save=0)
    sim.metallic_boundaries = True
    sim.mesh(preset="coarse", verbose=False, **mesh)
    return sim.mesh_path


# (generic?, slab (name, zmin, zmax, z_above), slab listed first?, expected ranges)
OVERLAP_2D = {
    "generic gas 4-8 over the oxide, z_above=0": (
        True,
        ("cap", 4.0, 8.0, 0.0),
        False,
        {"sio2": (-2.0, 5.0), "gas": (5.0, 8.0)},
    ),
    "generic gas 4-8 listed before the oxide": (
        True,
        ("cap", 4.0, 8.0, 0.0),
        True,
        {"sio2": (-2.0, 4.0), "gas": (4.0, 8.0)},
    ),
    "gpdk air 3.0-8.2 over the oxide, no airbox": (
        False,
        ("air_top", 3.0, 8.2, 0.0),
        False,
        {"sio2": (-2.0, 5.2), "air": (5.2, 8.2)},
    ),
}


@pytest.mark.parametrize("case", list(OVERLAP_2D))
def test_native_2d_overlapping_dielectrics_have_one_owner(tmp_path, case):
    """Overlapping background dielectrics: no element twice, first listed wins.

    On main this wrote 150 / 252 triangles in two domain groups and 8 / 50 lines
    in several interface groups (the first two cases / the gpdk case).
    """
    generic, slab, first, expected = OVERLAP_2D[case]
    msh = _mesh_2d(tmp_path, generic=generic, slab=slab, first=first)

    assert _duplicates(msh, TRIANGLE) == {}
    assert _duplicates(msh, LINE) == {}
    ranges = _vertical_ranges(msh)
    for name, vertical in expected.items():
        assert ranges[name] == pytest.approx(vertical, abs=1e-6), ranges


# ------------------------------------------- native 2D: one material, two regions
def _mesh_regions(tmp_path: Path, regions: list[tuple]) -> Path:
    """Mesh a generic BoundaryMode cross-section with ``(name, z0, z1, material)``."""
    gf.gpdk.PDK.activate()
    st = stack()
    st.dielectrics = [
        {"name": name, "zmin": z0, "zmax": z1, "material": material}
        for name, z0, z1, material in regions
    ]
    st.materials["polymer"] = POLYMER
    sim = BoundaryModeSim()
    sim.set_output_dir(str(tmp_path))
    sim.set_geometry(component([SIGNAL, GROUND], ports=[]))
    sim.set_stack(st)
    sim.set_airbox(margin_x=0.0, margin_y=5.0, z_above=0.0, z_below=2.0)
    sim.set_cross_section("x=0")
    sim.set_boundary_mode(freq=10e9, num_modes=2, save=0)
    sim.metallic_boundaries = True
    sim.mesh(preset="coarse", verbose=False, **MESH)
    return sim.mesh_path


def _triangles_in_band(msh_path: Path, z0: float, z1: float) -> dict[str, int]:
    """Triangles per domain group whose centroid height lies between z0 and z1."""
    names, xyz, elements = _read_msh(msh_path)
    counts: dict[str, int] = defaultdict(int)
    for etype, phys, nodes in elements:
        if etype == TRIANGLE:
            height = sum(xyz[n][1] for n in nodes) / len(nodes)
            if z0 < height < z1:
                counts[names[(2, phys)]] += 1
    return dict(counts)


OXIDE = ("box", -2.0, 4.0, "sio2")
MID = ("mid", 4.0, 8.0, "polymer")
CLAD = ("clad", 6.0, 10.0, "sio2")
INNER = ("inner", 2.0, 3.0, "polymer")
WIDE_BOX = ("box", -2.0, 5.0, "sio2")

# (regions in stack order, band, group that must own the band)
SAME_MATERIAL_TWICE = {
    "polymer listed between the SiO2 box and the SiO2 cladding": (
        [OXIDE, MID, CLAD],
        (6.0, 8.0),
        "polymer",
    ),
    "polymer listed after the SiO2 cladding": (
        [OXIDE, CLAD, MID],
        (6.0, 8.0),
        "sio2",
    ),
    "polymer inside the oxide, listed first": (
        [INNER, WIDE_BOX],
        (2.0, 3.0),
        "polymer",
    ),
    "polymer inside the oxide, listed after it": (
        [WIDE_BOX, INNER],
        (2.0, 3.0),
        "sio2",
    ),
}


@pytest.mark.parametrize("case", list(SAME_MATERIAL_TWICE))
def test_native_2d_regions_of_one_material_are_claimed_one_by_one(tmp_path, case):
    """The first region listed owns an overlap, also against a merged material.

    On the branch before this fix the two SiO2 regions were merged into one entry
    before the claim, which then sat at the place of the box in the order: the
    cladding took the overlap from a polymer region listed before it (224
    triangles in the first case, polymer got none) with no element written twice,
    so the group check could not see it.
    """
    regions, (z0, z1), owner = SAME_MATERIAL_TWICE[case]
    msh = _mesh_regions(tmp_path, regions)

    assert _duplicates(msh, TRIANGLE) == {}
    assert _duplicates(msh, LINE) == {}
    in_band = _triangles_in_band(msh, z0, z1)
    in_band.pop("air", None)  # beside the stack, not in it
    assert set(in_band) == {owner}, in_band
    assert in_band[owner] > 0


def test_native_2d_region_fully_covered_by_an_earlier_one_has_no_group(tmp_path):
    """A region listed after one that covers it entirely gets no domain group."""
    msh = _mesh_regions(tmp_path, [WIDE_BOX, INNER])

    names, _, _ = _read_msh(msh)
    domain_groups = {name for (dim, _), name in names.items() if dim == 2}
    assert "sio2" in domain_groups
    assert "polymer" not in domain_groups
    assert _duplicates(msh, TRIANGLE) == {}


# --------------------------------------------------------------------- the guard
def _cubes(kernel) -> tuple[int, list[int]]:
    """A big box with a unit cube inside it, fragmented: 2 volumes."""
    big = kernel.addBox(-5, -5, -5, 10, 10, 10)
    cube = kernel.addBox(0, 0, 0, 1, 1, 1)
    kernel.fragment([(3, big)], [(3, cube)])
    kernel.synchronize()
    cube_face = next(
        int(t)
        for _, t in gmsh.model.getEntities(2)
        if len(gmsh.model.getAdjacencies(2, t)[0]) == 2
    )
    volumes = [int(t) for _, t in gmsh.model.getEntities(3)]
    return cube_face, volumes


def test_guard_rejects_a_face_in_two_groups(kernel):
    """A face in two groups is refused, naming both groups."""
    face, _volumes = _cubes(kernel)
    gmsh.model.addPhysicalGroup(2, [face], tag=1, name="air__sio2")
    gmsh.model.addPhysicalGroup(2, [face], tag=2, name="port_P1")

    with pytest.raises(gmsh_utils.DuplicateBoundaryElementsError) as info:
        gmsh_utils.check_unique_group_membership(dims=(2, 3))

    message = str(info.value)
    assert "air__sio2" in message
    assert "port_P1" in message
    assert f"surface {face}" in message
    assert "overlap in z" in message
    assert isinstance(info.value, ValueError)


def test_guard_accepts_a_clean_model(kernel):
    """One group per entity, volumes included, passes."""
    face, volumes = _cubes(kernel)
    gmsh.model.addPhysicalGroup(2, [face], tag=1, name="air__sio2")
    gmsh.model.addPhysicalGroup(3, volumes[:1], tag=1, name="air")
    gmsh.model.addPhysicalGroup(3, volumes[1:], tag=2, name="sio2")

    gmsh_utils.check_unique_group_membership(dims=(2, 3))


def test_guard_does_nothing_without_a_gmsh_session():
    """Unit tests that mock the mesh writer never start gmsh; the check is a no-op."""
    assert not gmsh.isInitialized()
    gmsh_utils.check_unique_group_membership(dims=(1, 2, 3))


def test_guard_checks_volumes_too(kernel):
    """A volume in two domain groups would be written twice as well."""
    _face, volumes = _cubes(kernel)
    gmsh.model.addPhysicalGroup(3, volumes, tag=1, name="air")
    gmsh.model.addPhysicalGroup(3, volumes[:1], tag=2, name="sio2")

    with pytest.raises(gmsh_utils.DuplicateBoundaryElementsError, match="volume"):
        gmsh_utils.check_unique_group_membership(dims=(2, 3))


def test_guard_only_looks_at_the_given_dimensions_and_names(kernel):
    """``dims`` and ``ignore_names`` leave groups out of the check."""
    face, _volumes = _cubes(kernel)
    gmsh.model.addPhysicalGroup(2, [face], tag=1, name="interface_a_b")
    gmsh.model.addPhysicalGroup(2, [face], tag=2, name="my_contact")

    gmsh_utils.check_unique_group_membership(dims=(3,))
    gmsh_utils.check_unique_group_membership(dims=(2,), ignore_names=("my_contact",))
    with pytest.raises(gmsh_utils.DuplicateBoundaryElementsError):
        gmsh_utils.check_unique_group_membership(dims=(2,))


def test_guard_lists_at_most_five_entities(kernel):
    """The message names five clashes and counts the rest."""
    faces = [kernel.addRectangle(i * 2.0, 0, 0, 1, 1) for i in range(7)]
    kernel.synchronize()
    gmsh.model.addPhysicalGroup(2, faces, tag=1, name="one")
    gmsh.model.addPhysicalGroup(2, faces, tag=2, name="two")

    with pytest.raises(
        gmsh_utils.DuplicateBoundaryElementsError, match=r"7 mesh entities"
    ) as info:
        gmsh_utils.check_unique_group_membership(dims=(2,))
    assert "(and 2 more)" in str(info.value)
    assert str(info.value).count("surface ") == 5


def test_guard_names_a_group_that_has_no_name(kernel):
    """A physical group without a name is reported by its tag."""
    face, _volumes = _cubes(kernel)
    gmsh.model.addPhysicalGroup(2, [face], tag=7)
    gmsh.model.addPhysicalGroup(2, [face], tag=8, name="port_P1")

    with pytest.raises(
        gmsh_utils.DuplicateBoundaryElementsError, match="<unnamed group 7>"
    ):
        gmsh_utils.check_unique_group_membership(dims=(2,))


def test_meshing_runs_the_guard_before_writing(tmp_path, monkeypatch):
    """Both mesh writers call the check, with the dimensions of their mesh."""
    calls = []
    real = gmsh_utils.check_unique_group_membership

    def spy(dims=(2, 3), ignore_names=()):
        calls.append(tuple(dims))
        return real(dims=dims, ignore_names=ignore_names)

    monkeypatch.setattr(gmsh_utils, "check_unique_group_membership", spy)

    _mesh_2d(tmp_path / "two", generic=True, slab=("cap", 5.0, 8.0, 0.0))
    assert calls == [(1, 2)]

    _generic_3d(tmp_path / "three")
    assert calls == [(1, 2), (2, 3)]


# ------------------------------------------------------------ 3D guard (passes on main)
def _generic_3d(out, *, slab=None, layer=None, z_above=3.0, ox_top=5.0, poly=None):
    """The generic stack in 3D with an optional air/gas slab or shaped layer."""
    gf.gpdk.PDK.activate()
    st = stack()
    st.dielectrics = [{"name": "ox", "zmin": -2.0, "zmax": ox_top, "material": "sio2"}]
    comp = component(FULL, ports=[])
    if slab:
        name, material, definition, z0, z1 = slab
        st.materials[material] = definition
        st.dielectrics.append(
            {"name": name, "zmin": z0, "zmax": z1, "material": material}
        )
    if layer:
        name, material, definition, z0, z1 = layer
        st.materials[material] = definition
        st.layers[name] = Layer(
            name=name,
            gds_layer=(4, 0),
            zmin=z0,
            zmax=z1,
            thickness=z1 - z0,
            material=material,
            layer_type="dielectric",
        )
        assert poly is not None
        x0, y0, x1, y1 = poly
        comp.add_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer=(4, 0))
    return generator.generate_mesh(
        comp,
        st,
        [],
        out,
        write_config=False,
        margin_x=0,
        margin_y=0,
        air_margin=0,
        airbox_margin_x=5.0,
        airbox_margin_y=5.0,
        airbox_z_above=z_above,
        airbox_z_below=2.0,
        **MESH,
    ).mesh_path


BOX = (0, 0, 10, 7)
SLAB_TOP_ON_AIRBOX_TOP = {
    "air slab 5-8, airbox top at 8": {"slab": ("air_top", "air", AIR, 5, 8)},
    "gas slab 5-8, airbox top at 8": {
        "slab": ("cap", "gas", GAS, 5, 8),
        "z_above": 0,
    },
    "shaped air layer 5-8, airbox top at 8": {
        "layer": ("air_layer", "air", AIR, 5, 8),
        "poly": BOX,
    },
    "shaped air layer 5-8 over the box top at 5": {
        "layer": ("air_layer", "air", AIR, 5, 8),
        "poly": BOX,
        "z_above": 0,
    },
    "conductor top on the airbox top": {"ox_top": 3.0, "z_above": 0},
    "gas slab overlapping the oxide": {
        "slab": ("cap", "gas", GAS, 4, 8),
        "z_above": 0,
    },
}


@pytest.mark.parametrize("case", list(SLAB_TOP_ON_AIRBOX_TOP))
def test_airbox_top_coincident_cases_have_no_duplicates(tmp_path, case):
    """Guard for #396, not a fail-before test: these 3D meshes were already clean.

    An air slab is dropped when an airbox is used, a gas slab and the airbox are
    separate volumes, and every piece of the one 3D fragment has a single owner.
    """
    msh = _generic_3d(tmp_path, **SLAB_TOP_ON_AIRBOX_TOP[case])

    assert _duplicates(msh, TRIANGLE) == {}
    assert _duplicates(msh, LINE) == {}

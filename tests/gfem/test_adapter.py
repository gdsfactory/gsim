"""gsim.gfem: mapping of the shared 2D mesh, readings and the missing-runtime path.

The first groups run without gfem. The solves at the end need it and skip
when it is missing.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import speed_of_light as C0  # noqa: N812

from gsim.common.modes import Conductor, LineReading
from gsim.common.stack.extractor import Layer, LayerStack
from gsim.common.stack.materials import MaterialProperties
from gsim.femwell.adapter import epsilon_by_region
from gsim.gfem import adapter
from gsim.gfem.adapter import (
    EdgeCondition,
    Medium,
    edge_conditions,
    impedance_paths_m,
    region_media,
    wall_mode_from_impedances,
)
from gsim.palace.line_impedance import MIN_VOLTAGE_POWER_RATIO, line_impedance_paths

pytest.importorskip("gmsh")

UM = 1e-6
SIGMA_AL = 3.77e7
EPS_OXIDE = 4.1
# Background regions are named after their stack material, as the native-2D
# mesher names them (a region that is no layer is its own material name).
OXIDE_REGION = "sio2_rf"
SILICON_REGION = "si_rf"

# --- fixtures -------------------------------------------------------------------


def _finish_mesh(gmsh, path: Path, *, fine: tuple[float, float, float, float]) -> None:
    """Size the mesh from a box field, then write msh v2.2 to ``path``."""
    x0, x1, y0, y1 = fine
    field = gmsh.model.mesh.field
    box = field.add("Box")
    for key, value in {
        "VIn": 1.0,
        "VOut": 12.0,
        "XMin": x0,
        "XMax": x1,
        "YMin": y0,
        "YMax": y1,
        "Thickness": 8.0,
    }.items():
        field.setNumber(box, key, value)
    field.setAsBackgroundMesh(box)
    # Pin the size sources so a developer ~/.gmsh-options cannot change the mesh.
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.model.mesh.generate(2)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(str(path))


def _write_hole_cpw_mesh(path: Path, *, suffix: str) -> Path:
    """A shielded CPW in one oxide region, the metal left out as holes.

    The 1D groups are named the way the native-2D mesher names them:
    ``m3_signal<suffix>``, ``m3_ground<suffix>`` and the outer ``absorbing`` wall.
    """
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        box = occ.addRectangle(-60.0, -40.0, 0.0, 120.0, 80.0)
        holes = [
            (2, occ.addRectangle(-5.0, 3.2, 0.0, 10.0, 2.0)),
            (2, occ.addRectangle(11.0, 3.2, 0.0, 49.0, 2.0)),
            (2, occ.addRectangle(-60.0, 3.2, 0.0, 49.0, 2.0)),
        ]
        occ.cut([(2, box)], holes)
        occ.synchronize()
        surfaces = [tag for _dim, tag in gmsh.model.getEntities(2)]
        pg = gmsh.model.addPhysicalGroup(2, surfaces)
        # The native-2D mesher names a background region after its material.
        gmsh.model.setPhysicalName(2, pg, OXIDE_REGION)
        curves: dict[str, list[int]] = {"signal": [], "ground": [], "wall": []}
        for _dim, tag in gmsh.model.getEntities(1):
            cx, cy, _cz = occ.getCenterOfMass(1, tag)
            in_metal_band = 3.2 - 1e-6 <= cy <= 5.2 + 1e-6
            if in_metal_band and abs(cx) <= 5.0 + 1e-6:
                curves["signal"].append(tag)
            elif in_metal_band and abs(cx) >= 11.0 - 1e-6:
                curves["ground"].append(tag)
            else:
                curves["wall"].append(tag)
        assert all(curves.values()), curves
        for key, name in (
            ("signal", f"m3_signal{suffix}"),
            ("ground", f"m3_ground{suffix}"),
            ("wall", "absorbing"),
        ):
            pg = gmsh.model.addPhysicalGroup(1, curves[key])
            gmsh.model.setPhysicalName(1, pg, name)
        _finish_mesh(gmsh, path, fine=(-20.0, 20.0, 1.0, 8.0))
    finally:
        gmsh.finalize()
    return path


def _write_layered_mesh(path: Path) -> Path:
    """Silicon, oxide and one volume-metal strip: three 2D regions, no 1D group."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        silicon = occ.addRectangle(-30.0, -20.0, 0.0, 60.0, 20.0)
        oxide = occ.addRectangle(-30.0, 0.0, 0.0, 60.0, 10.0)
        strip = occ.addRectangle(-5.0, 3.0, 0.0, 10.0, 2.0)
        cut, _ = occ.cut([(2, oxide)], [(2, strip)], removeTool=False)
        occ.synchronize()
        _out, mapping = occ.fragment([(2, silicon), *cut, (2, strip)], [])
        occ.synchronize()
        n_cut = len(cut)
        groups = {
            SILICON_REGION: sorted({t for _d, t in mapping[0]}),
            OXIDE_REGION: sorted({t for m in mapping[1 : 1 + n_cut] for _d, t in m}),
            "m3_signal": sorted({t for _d, t in mapping[1 + n_cut]}),
        }
        for name, tags in groups.items():
            pg = gmsh.model.addPhysicalGroup(2, tags)
            gmsh.model.setPhysicalName(2, pg, name)
        _finish_mesh(gmsh, path, fine=(-15.0, 15.0, 1.0, 7.0))
    finally:
        gmsh.finalize()
    return path


def _stack(*, conductor_layers: bool = False, sigma: float | None = SIGMA_AL):
    """A stack with one oxide dielectric, optionally the two metal layers."""
    layers = {}
    if conductor_layers:
        for name, gds in (("m3_signal", (49, 0)), ("m3_ground", (49, 1))):
            layers[name] = Layer(
                name=name,
                gds_layer=gds,
                zmin=3.2,
                zmax=5.2,
                thickness=2.0,
                material="aluminum",
                layer_type="conductor",
            )
    metal = (
        MaterialProperties(permittivity=1.0)
        if sigma is None
        else MaterialProperties(permittivity=1.0, conductivity=sigma)
    )
    return LayerStack(
        pdk_name="gfem_test",
        layers=layers,
        materials={
            "sio2_rf": MaterialProperties(permittivity=EPS_OXIDE).to_dict(),
            "aluminum": metal.to_dict(),
        },
        dielectrics=[
            {"name": "oxide", "zmin": -40.0, "zmax": 40.0, "material": "sio2_rf"}
        ],
    )


def _layered_stack() -> LayerStack:
    """Silicon (lossy), oxide and a volume-metal layer: every loss path at once."""
    layer = Layer(
        name="m3_signal",
        gds_layer=(49, 0),
        zmin=3.0,
        zmax=5.0,
        thickness=2.0,
        material="aluminum",
        layer_type="dielectric",
    )
    return LayerStack(
        pdk_name="gfem_layered",
        layers={"m3_signal": layer},
        materials={
            "sio2_rf": MaterialProperties(permittivity=4.1).to_dict(),
            "si_rf": MaterialProperties(
                permittivity=11.9, loss_tangent=0.01, conductivity=2.0
            ).to_dict(),
            "aluminum": MaterialProperties(
                permittivity=1.0, conductivity=SIGMA_AL
            ).to_dict(),
        },
        dielectrics=[
            {"name": "silicon", "zmin": -20.0, "zmax": 0.0, "material": "si_rf"},
            {"name": "oxide", "zmin": 0.0, "zmax": 10.0, "material": "sio2_rf"},
        ],
    )


@pytest.fixture(scope="module")
def layered_mesh(tmp_path_factory) -> Path:
    return _write_layered_mesh(tmp_path_factory.mktemp("gfem-layered") / "layered.msh")


@pytest.fixture(scope="module")
def pec_mesh(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("gfem-pec") / "cpw_pec.msh"
    return _write_hole_cpw_mesh(path, suffix="_pec")


@pytest.fixture(scope="module")
def conductivity_mesh(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("gfem-cond") / "cpw_conductivity.msh"
    return _write_hole_cpw_mesh(path, suffix="_conductivity")


# --- missing runtime ---------------------------------------------------------------


def test_package_imports_without_gfem(monkeypatch):
    monkeypatch.setitem(sys.modules, "gfem", None)
    import gsim.gfem  # noqa: F401


def test_require_gfem_names_the_extra(monkeypatch):
    from gsim.gfem.runtime import require_gfem

    monkeypatch.setitem(sys.modules, "gfem", None)
    with pytest.raises(ImportError, match=r"gsim\[gfem\]"):
        require_gfem()


def test_require_gfem_warns_outside_the_supported_series(monkeypatch):
    from gsim.gfem.runtime import require_gfem

    fake = types.ModuleType("gfem")
    fake.__dict__["__version__"] = "0.3.0"
    monkeypatch.setitem(sys.modules, "gfem", fake)
    with pytest.warns(UserWarning, match="0.2."):
        assert require_gfem() is fake


def test_solve_path_raises_without_gfem(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "gfem", None)
    with pytest.raises(ImportError, match=r"gsim\[gfem\]"):
        adapter.solve_line(tmp_path / "missing.msh", _stack(), frequency_hz=1e9)


def test_importing_the_package_does_not_import_gfem_or_jax():
    code = (
        "import sys\n"
        "import gsim\n"
        "watched = ('gfem', 'ngsolve', 'jax')\n"
        "before = {m for m in watched if m in sys.modules}\n"
        "import gsim.gfem\n"
        "after = {m for m in watched if m in sys.modules}\n"
        "print(sorted(after - before))\n"
    )
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "[]"


def test_gfem_extra_declared_in_packaging():
    from importlib import metadata

    try:
        requires = metadata.requires("gsim") or []
    except metadata.PackageNotFoundError:
        pytest.skip("gsim not installed as a distribution")
    gfem_requirements = [r for r in requires if "gfem" in r and "extra" in r]
    if not gfem_requirements:
        pytest.skip("installed gsim metadata predates the gfem extra: re-run uv sync")
    assert any("gfem>=0.2.0" in r.replace(" ", "") for r in gfem_requirements)


# --- region media ------------------------------------------------------------------


@pytest.mark.parametrize("frequency_hz", [1e9, 50e9])
def test_region_media_rebuild_the_femwell_permittivity(layered_mesh, frequency_hz):
    stack = _layered_stack()
    media = region_media(layered_mesh, stack, frequency_hz=frequency_hz)
    reference = epsilon_by_region(layered_mesh, stack, frequency_hz=frequency_hz)
    assert set(media) == set(reference) == {SILICON_REGION, OXIDE_REGION, "m3_signal"}
    for region, medium in media.items():
        assert medium.permittivity(frequency_hz) == pytest.approx(
            reference[region], rel=1e-12
        )
    assert media[OXIDE_REGION] == Medium(eps_r=4.1)
    assert media[SILICON_REGION] == Medium(eps_r=11.9, tan_delta=0.01, conductivity=2.0)
    assert media["m3_signal"].conductivity == SIGMA_AL


def test_region_media_honours_overrides(layered_mesh):
    stack = _layered_stack()
    override = {"sio2_rf": MaterialProperties(permittivity=3.9)}
    media = region_media(layered_mesh, stack, frequency_hz=1e9, overrides=override)
    assert media[OXIDE_REGION].eps_r == 3.9
    assert media[SILICON_REGION].eps_r == 11.9


def test_region_media_refuses_a_non_positive_permittivity(layered_mesh):
    stack = _layered_stack()
    override = {"sio2_rf": MaterialProperties(permittivity=-2.0)}
    with pytest.raises(ValueError, match=OXIDE_REGION):
        region_media(layered_mesh, stack, frequency_hz=1e9, overrides=override)


@pytest.mark.parametrize("frequency_hz", [0.0, -1e9, float("nan"), float("inf")])
def test_region_media_refuses_a_bad_frequency(layered_mesh, frequency_hz):
    with pytest.raises(ValueError, match="frequency_hz"):
        region_media(layered_mesh, _layered_stack(), frequency_hz=frequency_hz)


def test_medium_permittivity_signs():
    lossy = Medium(eps_r=4.0, tan_delta=0.1, conductivity=1.0).permittivity(1e9)
    assert lossy.real == pytest.approx(4.0)
    assert lossy.imag < 0.0


# --- edge conditions ---------------------------------------------------------------


def _write_group_mesh(path: Path) -> Path:
    """One oxide square whose four sides carry four differently named 1D groups."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        square = occ.addRectangle(0.0, 0.0, 0.0, 10.0, 10.0)
        occ.synchronize()
        pg = gmsh.model.addPhysicalGroup(2, [square])
        gmsh.model.setPhysicalName(2, pg, OXIDE_REGION)
        by_side: dict[str, int] = {}
        for _dim, tag in gmsh.model.getEntities(1):
            cx, cy, _cz = occ.getCenterOfMass(1, tag)
            if abs(cx) < 1e-9:
                by_side["left"] = tag
            elif abs(cx - 10.0) < 1e-9:
                by_side["right"] = tag
            elif abs(cy) < 1e-9:
                by_side["bottom"] = tag
            else:
                by_side["top"] = tag
        for side, name in (
            ("left", "m3_signal_conductivity"),
            ("right", "m3_ground_pec"),
            ("bottom", "absorbing"),
            ("top", "contact_a"),
        ):
            pg = gmsh.model.addPhysicalGroup(1, [by_side[side]])
            gmsh.model.setPhysicalName(1, pg, name)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


@pytest.fixture(scope="module")
def group_mesh(tmp_path_factory) -> Path:
    return _write_group_mesh(tmp_path_factory.mktemp("gfem-groups") / "groups.msh")


def _stack_with_signal_layer(*, sigma: float | None = SIGMA_AL) -> LayerStack:
    return _stack(conductor_layers=True, sigma=sigma)


def test_edge_conditions_default_leaves_the_wall_natural(group_mesh):
    conditions = edge_conditions(
        group_mesh, _stack_with_signal_layer(), frequency_hz=1e9
    )
    assert conditions == [
        EdgeCondition("conductor", "m3_signal_conductivity", SIGMA_AL, None),
        EdgeCondition("pec", "m3_ground_pec"),
    ]


def test_edge_conditions_metallic_wall_is_pec(group_mesh):
    conditions = edge_conditions(
        group_mesh,
        _stack_with_signal_layer(),
        frequency_hz=1e9,
        metallic_boundaries=True,
    )
    assert EdgeCondition("pec", "absorbing") in conditions
    assert len(conditions) == 3


def test_edge_conditions_absorbing_wall(group_mesh):
    conditions = edge_conditions(
        group_mesh,
        _stack_with_signal_layer(),
        frequency_hz=1e9,
        absorbing_boundary=True,
    )
    assert EdgeCondition("absorbing", "absorbing") in conditions


def test_edge_conditions_pec_wins_over_absorbing(group_mesh):
    conditions = edge_conditions(
        group_mesh,
        _stack_with_signal_layer(),
        frequency_hz=1e9,
        metallic_boundaries=True,
        absorbing_boundary=True,
    )
    assert [c.kind for c in conditions if c.group == "absorbing"] == ["pec"]


def test_edge_conditions_pec_conductor_boundary(group_mesh):
    conditions = edge_conditions(
        group_mesh,
        _stack_with_signal_layer(),
        frequency_hz=1e9,
        conductor_boundary="pec",
    )
    assert conditions == [
        EdgeCondition("pec", "m3_signal_conductivity"),
        EdgeCondition("pec", "m3_ground_pec"),
    ]


def test_edge_conditions_stack_thickness_is_the_layer_thickness_in_metres(group_mesh):
    conditions = edge_conditions(
        group_mesh,
        _stack_with_signal_layer(),
        frequency_hz=1e9,
        conductor_thickness="stack",
    )
    assert conditions[0].thickness_m == pytest.approx(2.0 * UM)


def test_edge_conditions_default_conductivity_when_the_material_has_none(group_mesh):
    stack = _stack_with_signal_layer(sigma=None)
    (conductor, _pec) = edge_conditions(group_mesh, stack, frequency_hz=1e9)
    assert conductor.conductivity == adapter.DEFAULT_CONDUCTIVITY_S_PER_M


def test_edge_conditions_name_a_missing_layer(group_mesh):
    with pytest.raises(ValueError, match="m3_signal"):
        edge_conditions(group_mesh, _stack(), frequency_hz=1e9)


def test_edge_conditions_refuse_an_unknown_thickness_choice(group_mesh):
    with pytest.raises(ValueError, match="conductor_thickness"):
        edge_conditions(
            group_mesh,
            _stack_with_signal_layer(),
            frequency_hz=1e9,
            conductor_thickness="thin",  # type: ignore[arg-type]
        )


# --- impedance paths and the wall rule ---------------------------------------------

SIGNAL = ((-5.0, 5.0), (3.2, 5.2))
GROUND = ((11.0, 150.0), (3.2, 5.2))
DOMAIN = ((-150.0, 150.0), (-203.0, 105.2))


def test_impedance_paths_are_metres_on_the_port_plane():
    voltage, loop = impedance_paths_m(SIGNAL, GROUND, DOMAIN)
    palace = line_impedance_paths(signal=SIGNAL, ground=GROUND, domain=DOMAIN)
    wide = line_impedance_paths(
        signal=SIGNAL,
        ground=GROUND,
        domain=DOMAIN,
        clearance_fraction=adapter.LOOP_CLEARANCE_FRACTION,
    )
    assert voltage.shape == (2, 3)
    assert loop.shape == (5, 3)
    # the voltage path is Palace's; the loop is the wider one
    np.testing.assert_allclose(voltage[:, :2], np.array(palace.voltage) * UM)
    np.testing.assert_allclose(loop[:4, :2], np.array(wide.current) * UM)
    np.testing.assert_array_equal(voltage[:, 2], 0.0)
    np.testing.assert_array_equal(loop[:, 2], 0.0)


def test_loop_clearance_fraction_moves_the_loop_only():
    voltage, loop = impedance_paths_m(SIGNAL, GROUND, DOMAIN)
    voltage_wide, loop_wide = impedance_paths_m(
        SIGNAL, GROUND, DOMAIN, loop_clearance_fraction=0.25
    )
    np.testing.assert_array_equal(voltage, voltage_wide)
    # half the smallest clearance (the 2 um thickness) is 1 um, a quarter 0.5 um
    np.testing.assert_allclose(loop[0, :2], [(-5.0 - 1.0) * UM, (3.2 - 1.0) * UM])
    np.testing.assert_allclose(loop_wide[0, :2], [(-5.0 - 0.5) * UM, (3.2 - 0.5) * UM])


def test_impedance_loop_is_closed_and_the_voltage_path_stays_off_the_metal():
    voltage, loop = impedance_paths_m(SIGNAL, GROUND, DOMAIN)
    np.testing.assert_array_equal(loop[0], loop[-1])
    # the smallest clearance is the 2 um thickness: 1e-3 of it is 0.002 um
    np.testing.assert_allclose(voltage[0, 0], (5.0 + 0.002) * UM)
    np.testing.assert_allclose(voltage[1, 0], (11.0 - 0.002) * UM)
    np.testing.assert_allclose(voltage[:, 1], 4.2 * UM)
    assert loop[:, 0].min() < -5.0 * UM
    assert loop[:, 0].max() > 5.0 * UM


def test_current_loop_without_a_return_conductor_encloses_the_signal():
    loop = adapter._current_loop_m(SIGNAL, DOMAIN)
    assert loop.shape == (5, 3)
    np.testing.assert_array_equal(loop[0], loop[-1])
    assert loop[:, 0].min() < -5.0 * UM < 5.0 * UM < loop[:, 0].max()
    assert loop[:, 1].min() < 3.2 * UM < 5.2 * UM < loop[:, 1].max()
    with pytest.raises(ValueError, match="not inside"):
        adapter._current_loop_m(SIGNAL, ((-150.0, 150.0), (-203.0, 5.2)))


def test_wall_mode_rule_on_both_sides_of_the_threshold():
    z_pi = 50.0 + 0.0j
    wall, clause = wall_mode_from_impedances(0.5 * MIN_VOLTAGE_POWER_RATIO * z_pi, z_pi)
    assert wall is True
    assert "outer wall" in clause
    wall, clause = wall_mode_from_impedances(2.0 * MIN_VOLTAGE_POWER_RATIO * z_pi, z_pi)
    assert wall is False
    assert "line mode" in clause
    wall, _ = wall_mode_from_impedances(60.0 - 20.0j, 66.0 - 19.0j)
    assert wall is False


def test_wall_mode_is_unknown_without_a_voltage_path_or_power():
    wall, clause = wall_mode_from_impedances(None, 50.0 + 0.0j)
    assert wall is None
    assert "voltage path" in clause
    wall, _ = wall_mode_from_impedances(1.0 + 0.0j, 0.0 + 0.0j)
    assert wall is None
    wall, _ = wall_mode_from_impedances(complex("nan"), 50.0 + 0.0j)
    assert wall is None


# --- solves (need gfem) --------------------------------------------------------------

FREQUENCY_HZ = 10e9
TEM_SIGNAL = Conductor("m3_signal", ((-5.0, 5.0), (3.2, 5.2)), "pec")
TEM_GROUND = Conductor("m3_ground", ((11.0, 60.0), (3.2, 5.2)), "pec")
TEM_DOMAIN = ((-60.0, 60.0), (-40.0, 40.0))


@pytest.fixture(scope="module")
def tem_solution(pec_mesh):
    """The homogeneous-oxide, perfect-conductor CPW: TEM, n_eff = sqrt(4.1)."""
    pytest.importorskip("gfem")
    voltage, _loop = impedance_paths_m(TEM_SIGNAL.extent, TEM_GROUND.extent, TEM_DOMAIN)
    return adapter.solve_line(
        pec_mesh,
        _stack(),
        frequency_hz=FREQUENCY_HZ,
        voltage_path_m=voltage,
        metallic_boundaries=True,
    )


def test_tem_line_has_the_dielectric_index(tem_solution):
    n_eff = np.asarray(tem_solution.n_eff)
    assert np.asarray(tem_solution.solved).all()
    assert n_eff.shape == (1,)
    # measured 5e-10 on this 944-triangle mesh: TEM is exact in the discretisation
    assert n_eff[0].real == pytest.approx(np.sqrt(EPS_OXIDE), abs=1e-6)
    assert abs(n_eff[0].imag) < 1e-6


def test_n_eff_follows_gamma(tem_solution):
    k0 = 2 * np.pi * FREQUENCY_HZ / C0
    np.testing.assert_allclose(
        np.asarray(tem_solution.n_eff),
        np.asarray(tem_solution.gamma) / (1j * k0),
        rtol=1e-12,
    )


def test_tem_impedance_definitions_agree(tem_solution):
    z = adapter.line_impedances(
        tem_solution,
        signal=TEM_SIGNAL.extent,
        ground=TEM_GROUND.extent,
        domain=TEM_DOMAIN,
    )
    assert set(z) == {"PV", "PI", "VI"}
    np.testing.assert_allclose(z["PV"], np.asarray(tem_solution.Z_PV), rtol=1e-6)
    # measured 4.5e-3 (PI) and 2.2e-3 (VI) on the 944-triangle mesh
    np.testing.assert_allclose(z["PI"], z["PV"], rtol=1e-2)
    np.testing.assert_allclose(z["VI"], z["PV"], rtol=1e-2)


def test_tem_line_reading(tem_solution):
    reading = adapter.line_reading(
        tem_solution, signal=TEM_SIGNAL, return_=TEM_GROUND, domain=TEM_DOMAIN
    )
    z = adapter.line_impedances(
        tem_solution,
        signal=TEM_SIGNAL.extent,
        ground=TEM_GROUND.extent,
        domain=TEM_DOMAIN,
    )
    assert isinstance(reading, LineReading)
    assert reading.z0_ohm == pytest.approx(z["PI"][0])
    assert reading.n_eff == pytest.approx(complex(np.asarray(tem_solution.n_eff)[0]))
    assert reading.wall_mode is False
    assert "Z_PI" in reading.diagnostic
    assert "Z_PV" in reading.diagnostic
    assert "Z_VI" in reading.diagnostic


def test_line_reading_without_a_return_conductor_has_no_wall_verdict(tem_solution):
    full = adapter.line_reading(
        tem_solution, signal=TEM_SIGNAL, return_=TEM_GROUND, domain=TEM_DOMAIN
    )
    alone = adapter.line_reading(tem_solution, signal=TEM_SIGNAL, domain=TEM_DOMAIN)
    assert alone.wall_mode is None
    assert alone.z0_ohm == pytest.approx(full.z0_ohm, rel=1e-6)
    assert "Z_PI" in alone.diagnostic
    assert "Z_PV" not in alone.diagnostic


def test_line_reading_index_selects_the_frequency(tem_solution):
    with pytest.raises(IndexError):
        adapter.line_reading(
            tem_solution,
            signal=TEM_SIGNAL,
            return_=TEM_GROUND,
            domain=TEM_DOMAIN,
            index=3,
        )


def test_solve_line_refuses_media_that_change_with_frequency(monkeypatch, pec_mesh):
    pytest.importorskip("gfem")

    def drifting(_mesh, _stack, *, frequency_hz, **_kwargs):
        return {OXIDE_REGION: Medium(eps_r=4.0 + frequency_hz * 1e-10)}

    monkeypatch.setattr(adapter, "region_media", drifting)
    with pytest.raises(ValueError, match="once per frequency"):
        adapter.solve_line(pec_mesh, _stack(), frequency_hz=[1e9, 2e9])


def test_finite_conductivity_conductor_tends_to_the_perfect_one(conductivity_mesh):
    pytest.importorskip("gfem")
    n_eff = {}
    for sigma in (SIGMA_AL, 100 * SIGMA_AL):
        result = adapter.solve_line(
            conductivity_mesh,
            _stack(conductor_layers=True, sigma=sigma),
            frequency_hz=FREQUENCY_HZ,
            metallic_boundaries=True,
        )
        assert np.asarray(result.solved).all()
        n_eff[sigma] = complex(np.asarray(result.n_eff)[0])
    pec = np.sqrt(EPS_OXIDE)
    weak, strong = n_eff[SIGMA_AL], n_eff[100 * SIGMA_AL]
    # The surface impedance adds internal inductance and loss (n'' > 0 in the
    # n' - i n'' convention); both shrink as the conductivity grows, towards
    # the perfect-conductor index.
    assert weak.real > strong.real > pec
    assert strong.real == pytest.approx(pec, rel=1e-2)
    assert 0.0 < -strong.imag < -weak.imag

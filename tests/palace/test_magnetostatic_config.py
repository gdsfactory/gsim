"""Tests for the magnetostatic Palace config (``SurfaceCurrent`` and ``PEC``).

The first tests write configs from synthetic physical groups; the last ones
mesh real coils with gmsh (no Palace run).
"""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import gdsfactory as gf
import pytest

from gsim.palace import DrivenSim, MagnetostaticSim
from gsim.palace.mesh.config_generator import generate_palace_config
from gsim.palace.mesh.validation import check_lumped_port_geometry, validate_mesh
from gsim.palace.models import MagnetostaticConfig, MagnetostaticSolverConfig
from gsim.palace.ports.config import PalacePort, PortGeometry, PortType
from tests._electrode_layouts import stack

# Group numbers: m1 shell 10 11, m2 shell 12 13, PEC block 14, via sides 15 16,
# source sheets 20 21, outer wall 30 31, volumes 1 2.
OUTER = [30, 31]


def _groups() -> dict:
    """A two-layer layout with a via, a PEC block and two source sheets."""
    return {
        "volumes": {
            "airbox": {"phys_group": 1, "tags": [1]},
            "v1": {"phys_group": 2, "tags": [2], "is_via": True},
        },
        "conductor_surfaces": {
            "m1_xy": {"phys_group": 10, "tags": []},
            "m1_z": {"phys_group": 11, "tags": []},
            "m2_xy": {"phys_group": 12, "tags": []},
            "m2_z": {"phys_group": 13, "tags": []},
        },
        "pec_surfaces": {"shield_xy": {"phys_group": 14, "tags": []}},
        "via_boundary_surfaces": {"v1": [15, 16]},
        "port_surfaces": {
            "P1": {"phys_group": 20, "tags": [], "type": "surface_current"},
            "P2": {"phys_group": 21, "tags": [], "type": "surface_current"},
        },
        "boundary_surfaces": {"absorbing": {"phys_group": OUTER, "tags": []}},
    }


def _sheet(name: str, orientation: float, layer: str = "m1") -> PalacePort:
    """A source sheet as ``ResolvedCurrentSource.to_palace_port`` makes it."""
    return PalacePort(
        name=name,
        port_type=PortType.SURFACE_CURRENT,
        geometry=PortGeometry.GAP,
        center=(0.0, 0.0),
        width=2.0,
        orientation=orientation,
        zmin=0.0,
        zmax=1.0,
        layer=layer,
    )


def _config(tmp_path, ports=None, **kwargs) -> dict:
    """Write the magnetostatic config of the layout and read it back."""
    ports = ports if ports is not None else [_sheet("A", 180.0), _sheet("B", 90.0)]
    kwargs.setdefault("hints", {"_metallic_boundaries": True})
    kwargs.setdefault("absorbing_boundary", False)
    path = generate_palace_config(
        groups=kwargs.pop("groups", _groups()),
        ports=ports,
        port_info=[],
        stack=stack(),
        output_path=tmp_path,
        model_name="palace",
        fmax=1e9,
        simulation_type="magnetostatic",
        numerical_config=MagnetostaticSolverConfig(),
        **kwargs,
    )
    return json.loads(path.read_text())


def test_the_problem_is_magnetostatic_with_ams_cg(tmp_path) -> None:
    config = _config(tmp_path)
    assert config["Problem"]["Type"] == "Magnetostatic"
    assert config["Solver"]["Magnetostatic"] == {"Save": 0}
    assert config["Solver"]["Linear"]["Type"] == "AMS"
    assert config["Solver"]["Linear"]["KSPType"] == "CG"


def test_problem_settings_are_written(tmp_path) -> None:
    config = _config(tmp_path, magnetostatic_config=MagnetostaticConfig(save_fields=2))
    assert config["Solver"]["Magnetostatic"] == {"Save": 2}


def test_without_a_solver_config_the_magnetostatic_defaults_apply(tmp_path) -> None:
    path = generate_palace_config(
        groups=_groups(),
        ports=[_sheet("A", 180.0)],
        port_info=[],
        stack=stack(),
        output_path=tmp_path,
        model_name="palace",
        fmax=1e9,
        simulation_type="magnetostatic",
        absorbing_boundary=False,
    )
    linear = json.loads(path.read_text())["Solver"]["Linear"]
    assert (linear["Type"], linear["KSPType"], linear["Tol"]) == ("AMS", "CG", 1e-8)


def test_pec_holds_conductors_pec_blocks_via_sides_and_the_outer_wall(
    tmp_path,
) -> None:
    pec = _config(tmp_path)["Boundaries"]["PEC"]["Attributes"]
    assert pec == [10, 11, 12, 13, 14, 15, 16, *OUTER]


def test_without_metallic_boundaries_the_outer_wall_is_left_pmc(tmp_path) -> None:
    pec = _config(tmp_path, hints={})["Boundaries"]["PEC"]["Attributes"]
    assert pec == [10, 11, 12, 13, 14, 15, 16]
    assert not set(OUTER) & set(pec)


def test_surface_currents_follow_the_source_order_and_direction(tmp_path) -> None:
    sources = _config(tmp_path)["Boundaries"]["SurfaceCurrent"]
    assert sources == [
        {"Index": 1, "Attributes": [20], "Direction": "-X"},
        {"Index": 2, "Attributes": [21], "Direction": "+Y"},
    ]


@pytest.mark.parametrize(
    ("orientation", "direction"),
    [(0.0, "+X"), (90.0, "+Y"), (180.0, "-X"), (270.0, "-Y")],
)
def test_the_direction_is_the_signed_cardinal_keyword(
    tmp_path, orientation, direction
) -> None:
    config = _config(tmp_path, ports=[_sheet("A", orientation)])
    assert config["Boundaries"]["SurfaceCurrent"][0]["Direction"] == direction


def test_only_pec_and_surface_currents_are_written(tmp_path) -> None:
    boundaries = _config(tmp_path)["Boundaries"]
    assert set(boundaries) == {"PEC", "SurfaceCurrent"}
    for entry in boundaries["SurfaceCurrent"]:
        assert set(entry) == {"Index", "Attributes", "Direction"}


def test_no_source_sheet_is_a_pec_attribute(tmp_path) -> None:
    boundaries = _config(tmp_path)["Boundaries"]
    sheets = {a for s in boundaries["SurfaceCurrent"] for a in s["Attributes"]}
    assert not sheets & set(boundaries["PEC"]["Attributes"])


def test_a_source_sheet_in_the_pec_set_is_refused(tmp_path) -> None:
    groups = _groups()
    groups["pec_surfaces"]["sheet"] = {"phys_group": 20, "tags": []}
    with pytest.raises(ValueError, match=r"source .*'A'.*PEC"):
        _config(tmp_path, groups=groups)


def test_a_port_that_is_not_a_surface_current_is_refused(tmp_path) -> None:
    lumped = PalacePort(name="o1", geometry=PortGeometry.GAP, width=2.0, zmax=1.0)
    with pytest.raises(ValueError, match=r"'o1'.*not a current source"):
        _config(tmp_path, ports=[lumped])


def test_a_source_without_a_mesh_sheet_is_refused(tmp_path) -> None:
    groups = _groups()
    del groups["port_surfaces"]["P2"]
    with pytest.raises(ValueError, match=r"'B'.*no sheet in the mesh"):
        _config(tmp_path, groups=groups)


def test_an_absorbing_wall_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"Absorbing.*magnetostatic"):
        _config(tmp_path, hints={}, absorbing_boundary=True)


# ---------------------------------------------------------------------------
# Meshed coils (gmsh, no Palace)
# ---------------------------------------------------------------------------

COARSE = {"refined_mesh_size": 2.0, "max_mesh_size": 30.0}


def _meshed(sim: MagnetostaticSim, output_dir) -> dict:
    """Mesh a sim, write its config and read back what the run would use."""
    sim.set_output_dir(output_dir)
    result = sim.mesh(**COARSE)
    sim.write_config()
    return {
        "mesh_hash": result.mesh_stats["mesh_hash"],
        "groups": sim.mesh_groups,
        "config": json.loads((output_dir / "config.json").read_text()),
        "ports": json.loads((output_dir / "port_information.json").read_text())[
            "ports"
        ],
        "metadata": json.loads((output_dir / "metadata.json").read_text()),
    }


@pytest.fixture(scope="module")
def stacked_pair(tmp_path_factory) -> dict[str, dict]:
    """The stacked IHP coil pair meshed twice, with source B one way then reversed."""
    pytest.importorskip("ihp")
    from tests._cloud_fixtures import inductor2_pair

    runs = {}
    for run in ("forward", "reversed"):
        sim = MagnetostaticSim()
        sim.set_geometry(inductor2_pair(stacked=True))
        sim.set_stack(substrate_thickness=2.0)
        sim.set_airbox(margin_x=20, margin_y=20, z_above=20, z_below=20)
        sim.add_current_source("A", plus_port="A_P1", minus_port="A_P2")
        sim.add_current_source("B", plus_port="B_P1", minus_port="B_P2")
        if run == "reversed":
            sim.add_current_source("B", plus_port="B_P2", minus_port="B_P1")
        runs[run] = _meshed(sim, tmp_path_factory.mktemp(run))
    return runs


def test_reversing_a_source_keeps_the_mesh_and_flips_only_its_direction(
    stacked_pair,
) -> None:
    forward, backward = stacked_pair["forward"], stacked_pair["reversed"]
    assert forward["mesh_hash"] == backward["mesh_hash"]
    a, b = forward["config"]["Boundaries"]["SurfaceCurrent"]
    assert (a["Direction"], b["Direction"]) == ("-X", "+X")
    expected = json.loads(json.dumps(forward["config"]))
    expected["Boundaries"]["SurfaceCurrent"][1]["Direction"] = "-X"
    assert backward["config"] == expected


def test_a_meshed_pair_has_pec_conductors_and_wall_and_one_sheet_per_source(
    stacked_pair,
) -> None:
    run = stacked_pair["forward"]
    boundaries = run["config"]["Boundaries"]
    groups = run["groups"]
    assert set(boundaries) == {"PEC", "SurfaceCurrent"}
    pec = set(boundaries["PEC"]["Attributes"])
    conductors = {info["phys_group"] for info in groups["conductor_surfaces"].values()}
    wall = set(groups["boundary_surfaces"]["absorbing"]["phys_group"])
    assert conductors <= pec
    assert wall <= pec
    sheets = [s["Attributes"] for s in boundaries["SurfaceCurrent"]]
    assert sheets == [
        [groups["port_surfaces"]["P1"]["phys_group"]],
        [groups["port_surfaces"]["P2"]["phys_group"]],
    ]
    assert not {a for (a,) in sheets} & pec


def test_the_cloud_sizes_a_meshed_pair_from_its_nd_dof_estimate(
    stacked_pair,
) -> None:
    metadata = stacked_pair["forward"]["metadata"]
    estimate = metadata["mesh"]["field_dofs"]
    assert metadata["simulation"]["problem_type"] == "Magnetostatic"
    assert "unavailable_reason" not in estimate
    assert estimate["estimated_field_dofs"] >= metadata["mesh"]["topology"]["edges"]


def test_the_port_information_names_surface_currents_without_z0(
    stacked_pair,
) -> None:
    ports = stacked_pair["forward"]["ports"]
    assert [(p["name"], p["type"]) for p in ports] == [
        ("A", "surface_current"),
        ("B", "surface_current"),
    ]
    assert all("Z0" not in p for p in ports)
    assert ports[0]["length"] == pytest.approx(2.1)  # the gap between the leads


def test_ports_a_driven_sim_configured_on_the_component_are_not_meshed(
    tmp_path,
) -> None:
    gf.gpdk.PDK.activate()
    component = gf.Component()
    component.add_ports((component << gf.components.inductor()).ports)
    driven = DrivenSim()
    driven.set_geometry(component)
    driven.set_stack()
    driven.add_port("o1", layer="metal3", length=2.0)
    driven._configure_ports_on_component(driven._resolve_stack())
    assert component.ports["o1"].info["palace_type"]  # the stale decoration

    sim = MagnetostaticSim()
    sim.set_geometry(component)
    sim.set_stack()
    sim.set_airbox(margin_x=20, margin_y=20, z_above=20, z_below=20)
    sim.add_current_source("L", plus_port="P1", minus_port="P2")
    run = _meshed(sim, tmp_path)
    assert [p["name"] for p in run["ports"]] == ["L"]
    assert list(run["groups"]["port_surfaces"]) == ["P1"]
    assert len(run["config"]["Boundaries"]["SurfaceCurrent"]) == 1
    assert "LumpedPort" not in run["config"]["Boundaries"]


def test_planar_conductors_are_refused(tmp_path) -> None:
    gf.gpdk.PDK.activate()
    sim = MagnetostaticSim()
    sim.set_geometry(gf.components.inductor())
    sim.set_stack()
    sim.set_airbox(margin_x=20, margin_y=20, z_above=20, z_below=20)
    sim.add_current_source("L", plus_port="P1", minus_port="P2")
    sim.set_output_dir(tmp_path)
    with pytest.raises(ValueError, match="planar_conductors"):
        sim.mesh(planar_conductors=True, **COARSE)
    assert not (tmp_path / "palace.msh").exists()


@pytest.fixture
def meshed_pair(tmp_path) -> MagnetostaticSim:
    """The stacked IHP coil pair, meshed and ready for ``validate_mesh``."""
    pytest.importorskip("ihp")
    from tests._cloud_fixtures import inductor2_pair

    sim = MagnetostaticSim()
    sim.set_geometry(inductor2_pair(stacked=True))
    sim.set_stack(substrate_thickness=2.0)
    sim.set_airbox(margin_x=20, margin_y=20, z_above=20, z_below=20)
    sim.add_current_source("A", plus_port="A_P1", minus_port="A_P2")
    sim.add_current_source("B", plus_port="B_P1", minus_port="B_P2")
    sim.set_output_dir(tmp_path)
    sim.mesh(**COARSE)
    return sim


def test_validate_mesh_accepts_a_meshed_pair(meshed_pair) -> None:
    result = validate_mesh(meshed_pair)
    assert result.valid, result.errors


def test_validate_mesh_flags_a_source_sheet_that_touches_no_pec(
    meshed_pair,
) -> None:
    # Coil A is TopMetal2; losing its surfaces leaves sheet A floating in
    # dielectric, as when a boolean cut consumes the leads.
    conductors = meshed_pair.mesh_groups["conductor_surfaces"]
    del conductors["topmetal2_xy"], conductors["topmetal2_z"]
    errors = validate_mesh(meshed_pair).errors
    assert any("surface currents [1]" in e and "share no nodes" in e for e in errors), (
        errors
    )


def test_the_sheet_geometry_check_covers_surface_currents(meshed_pair) -> None:
    sheets = meshed_pair._last_ports
    kwargs = {
        "mesh_path": meshed_pair._last_mesh_result.mesh_path,
        "port_surfaces": meshed_pair.mesh_groups["port_surfaces"],
    }
    assert check_lumped_port_geometry(palace_ports=sheets, **kwargs) == []
    turned = [dataclasses.replace(s, orientation=s.orientation + 90) for s in sheets]
    errors = check_lumped_port_geometry(palace_ports=turned, **kwargs)
    assert len(errors) == 2
    assert all("geometry check FAILED" in e for e in errors)


def test_a_mesh_without_source_sheets_is_an_error(meshed_pair) -> None:
    meshed_pair.mesh_groups["port_surfaces"].clear()
    assert "No port surfaces in mesh." in validate_mesh(meshed_pair).errors


def test_a_config_without_surface_currents_is_an_error(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"Boundaries": {"PEC": {"Attributes": [4]}}})
    )
    sim = SimpleNamespace(
        simulation_type="magnetostatic",
        _last_mesh_result=SimpleNamespace(
            groups={"volumes": {"air": {}}, "conductor_surfaces": {"m": {}}},
            mesh_path=tmp_path / "missing.msh",
        ),
        _output_dir=tmp_path,
        write_config=lambda **_kwargs: None,
    )
    errors = validate_mesh(sim).errors
    assert any("SurfaceCurrent" in e for e in errors), errors


def test_a_missing_mesh_file_reports_the_contact_check_as_not_run(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "Boundaries": {
                    "PEC": {"Attributes": [4]},
                    "SurfaceCurrent": [{"Index": 1, "Attributes": [5]}],
                }
            }
        )
    )
    sim = SimpleNamespace(
        simulation_type="magnetostatic",
        _last_mesh_result=SimpleNamespace(
            groups={"volumes": {"air": {}}, "conductor_surfaces": {"m": {}}},
            mesh_path=tmp_path / "missing.msh",
        ),
        _output_dir=tmp_path,
        write_config=lambda **_kwargs: None,
    )
    result = validate_mesh(sim)
    assert not any("PEC" in e for e in result.errors), result.errors
    assert any(
        "contact with PEC not checked" in w and "missing.msh" in w
        for w in result.warnings
    ), result.warnings

# Copyright 2026 GDSFactory
"""Geometry and finite-conductivity tests for the EIC benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gsim.palace.benchmarks.eic_ihp import IHP_PORT_SPECS, build_ihp_stack
from gsim.palace.benchmarks.eic_nist import (
    AIR_CHANNEL_LAYER,
    GAP_PARTITION_SUBSTRATE_LAYER,
    GAP_WIDTH_UM,
    NIST_MAX_TETRAHEDRA,
    PARYLENE_LAYER,
    PDMS_LAYER,
    PLATINUM_LAYER,
    SUBSTRATE_LAYER,
    build_nist_component,
    build_nist_stack,
    make_nist_simulation,
    nist_section_edges_um,
    require_nist_tetrahedron_budget,
)
from gsim.palace.mesh.config_generator import generate_palace_config
from gsim.palace.mesh.geometry import (
    GeometryData,
    _matching_volume_tags,
    resolve_dielectric_regions,
)
from gsim.palace.models import DrivenConfig


def test_ihp_stack_matches_emdesign2_and_uses_finite_conductivity() -> None:
    """The imported source layers map to the resolved native HFSS stack."""
    stack = build_ihp_stack()

    assert stack.validate_stack().valid
    assert stack.layers["Metal1"].gds_layer == (62, 0)
    assert stack.layers["Metal5"].gds_layer == (123, 0)
    assert stack.layers["TopVia1"].gds_layer == (125, 0)
    assert stack.layers["TopMetal1"].gds_layer == (126, 0)
    assert stack.layers["TopMetal1"].thickness == pytest.approx(2.0)
    assert stack.materials["topmetal1"]["conductivity"] == pytest.approx(2.78e7)
    assert stack.materials["topvia1"]["conductivity"] == pytest.approx(2.191e6)
    assert IHP_PORT_SPECS["port1"]["center"] == (-33.581, -263.073)
    assert IHP_PORT_SPECS["port2"]["center"] == (36.419, -263.073)


def test_nist_geometry_has_full_section_sequence_and_expected_layers() -> None:
    """The Palace component represents the complete symmetric 6.5 mm chip."""
    component = build_nist_component()
    edges = nist_section_edges_um()
    polygons = component.get_polygons(by="tuple")

    assert edges == pytest.approx(
        [
            -3250.25,
            -3190.25,
            -2500.25,
            -1767.85,
            -1347.75,
            1347.75,
            1767.85,
            2500.25,
            3190.25,
            3250.25,
        ]
    )
    np.testing.assert_allclose(
        component.bbox_np(), [[-3250.25, -682.5], [3250.25, 682.5]]
    )
    assert len(polygons[PLATINUM_LAYER]) == 3
    assert len(polygons[SUBSTRATE_LAYER]) == 1
    assert len(polygons[PARYLENE_LAYER]) == 1
    assert len(polygons[PDMS_LAYER]) == 1
    assert len(polygons[AIR_CHANNEL_LAYER]) == 3

    ports = {port.name: port for port in component.ports}
    assert tuple(ports["left"].center) == pytest.approx((-3250.25, 0.0))
    assert ports["left"].orientation == pytest.approx(180.0)
    assert tuple(ports["right"].center) == pytest.approx((3250.25, 0.0))
    assert ports["right"].orientation == pytest.approx(0.0)


def test_nist_stack_separates_ansys_and_measured_platinum_thicknesses() -> None:
    """The model-parity and measurement-comparison Pt cases stay explicit."""
    ansys_stack = build_nist_stack(platinum_thickness_um=0.2)
    measured_stack = build_nist_stack(platinum_thickness_um=0.405)

    assert ansys_stack.validate_stack().valid
    assert measured_stack.validate_stack().valid
    assert ansys_stack.layers["platinum"].thickness == pytest.approx(0.2)
    assert measured_stack.layers["platinum"].thickness == pytest.approx(0.405)
    assert ansys_stack.materials["platinum"]["conductivity"] == pytest.approx(9.04e6)
    assert ansys_stack.layers["air_channel"].zmax == pytest.approx(219.67)


def test_nist_simulation_uses_two_cpw_lumped_ports(tmp_path: Path) -> None:
    """The capped 3D model has explicit 50-ohm CPW terminations."""
    simulation = make_nist_simulation(tmp_path / "nist")

    assert simulation.validate_config().valid
    assert simulation.wave_ports == []
    assert [port.name for port in simulation.cpw_ports] == ["left", "right"]
    assert all(port.impedance == 50.0 for port in simulation.cpw_ports)
    assert all(port.gap_width == GAP_WIDTH_UM for port in simulation.cpw_ports)
    assert all(port.excited for port in simulation.cpw_ports)
    component = simulation.component
    stack = simulation.stack
    assert component is not None
    assert stack is not None
    polygons = component.get_polygons(by="tuple")
    assert len(polygons[GAP_PARTITION_SUBSTRATE_LAYER]) == 2
    assert stack.materials["fused_silica_partition"] == stack.materials["fused_silica"]
    assert simulation._airbox_config["margin_x"] == pytest.approx(50.0)
    assert simulation.driven.reference_impedance == pytest.approx(50.0)
    assert simulation.numerical.order == 1


def test_nist_numeric_wave_ports_remain_available(tmp_path: Path) -> None:
    """Numeric wave ports remain an explicit convergence-study option."""
    simulation = make_nist_simulation(tmp_path / "nist-wave", port_type="wave")

    assert simulation.cpw_ports == []
    assert [port.name for port in simulation.wave_ports] == ["left", "right"]
    assert all(port.max_size and port.excited for port in simulation.wave_ports)
    assert simulation._airbox_config["margin_x"] == pytest.approx(0.0)


def test_nist_tetrahedron_budget_blocks_solver_runs_above_limit() -> None:
    """The benchmark refuses any Palace execution above the hard cap."""
    accepted = SimpleNamespace(mesh_stats={"tetrahedra": NIST_MAX_TETRAHEDRA})
    rejected = SimpleNamespace(mesh_stats={"tetrahedra": NIST_MAX_TETRAHEDRA + 1})

    assert require_nist_tetrahedron_budget(accepted) == NIST_MAX_TETRAHEDRA
    with pytest.raises(RuntimeError, match="Palace execution is blocked"):
        require_nist_tetrahedron_budget(rejected)


@pytest.mark.parametrize(
    ("stack_factory", "conductor_name", "thickness_um", "conductivity"),
    [
        (build_ihp_stack, "TopMetal1", 2.0, 2.78e7),
        (build_nist_stack, "platinum", 0.2, 9.04e6),
    ],
)
def test_generated_palace_config_uses_conductivity_boundary(
    tmp_path: Path,
    stack_factory,
    conductor_name: str,
    thickness_um: float,
    conductivity: float,
) -> None:
    """Thick signal metals become finite-conductivity shell boundaries."""
    stack = stack_factory()
    groups = {
        "volumes": {"airbox": {"phys_group": 1}},
        "conductor_surfaces": {
            f"{conductor_name}_xy": {"phys_group": 2},
        },
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {},
    }
    config_path = generate_palace_config(
        groups=groups,
        ports=[],
        port_info=[],
        stack=stack,
        output_path=tmp_path,
        model_name="eic_config",
        fmax=20e9,
        driven_config=DrivenConfig(fmin=1e9, fmax=20e9, num_points=3),
    )
    config = json.loads(config_path.read_text())

    boundaries = config["Boundaries"]["Conductivity"]
    assert len(boundaries) == 1
    assert boundaries[0]["Conductivity"] == pytest.approx(conductivity)
    assert boundaries[0]["Thickness"] == pytest.approx(thickness_um)


def test_nist_section_edges_are_strictly_increasing() -> None:
    """Guard the section builder against zero-length or reordered regions."""
    assert np.all(np.diff(nist_section_edges_um()) > 0)


def test_split_conductor_volumes_are_recovered_by_contained_fragments() -> None:
    """Shaped dielectric boundaries may split a conductor during OCC dedup."""
    target = (-10.0, -2.0, 0.0, 10.0, 2.0, 0.2)
    candidates = {
        1: (-10.0, -2.0, 0.0, -4.0, 2.0, 0.2),
        2: (-4.0, -2.0, 0.0, 4.0, 2.0, 0.2),
        3: (4.0, -2.0, 0.0, 10.0, 2.0, 0.2),
        4: (-10.0, -2.0, 0.0, 10.0, 2.0, 6.67),
        5: (-20.0, -2.0, 0.0, 20.0, 2.0, 0.2),
    }

    assert _matching_volume_tags(target, candidates) == [1, 2, 3]


def test_airbox_encloses_tallest_patterned_dielectric() -> None:
    """Max-size ports and the explicit airbox share the full stack z envelope."""
    geometry = GeometryData(
        polygons=[],
        bbox=(-3250.25, -682.5, 3250.25, 682.5),
        layer_bboxes={},
    )
    regions = resolve_dielectric_regions(
        geometry,
        build_nist_stack(),
        margin_x=0.0,
        margin_y=0.0,
        airbox_margin_x=0.0,
        airbox_margin_y=0.0,
        airbox_z_above=250.0,
        airbox_z_below=0.0,
    )
    airbox = next(region for region in regions if region.name == "airbox")

    assert airbox.zmin == pytest.approx(-500.0)
    assert airbox.zmax == pytest.approx(756.67)

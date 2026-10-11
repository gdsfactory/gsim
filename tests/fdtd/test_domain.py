"""Directional domain padding, validation, and mesh/runtime consistency."""

from __future__ import annotations

import json
from dataclasses import replace

import meshio
import pytest
from pydantic import ValidationError

from gsim import fdtd
from gsim.fdtd.domain import background_bounds_nm


@pytest.mark.parametrize("field", ["padding", "padding_x", "padding_y", "padding_z"])
@pytest.mark.parametrize("value", [0, 0.5, (0, 2), [0.5, 1.5]])
def test_domain_accepts_scalar_and_directional_padding(field, value) -> None:
    domain = fdtd.Domain(**{field: value})
    expected = tuple(value) if isinstance(value, list) else value
    assert getattr(domain, field) == expected
    assert fdtd.Domain.model_validate(domain.model_dump()) == domain


@pytest.mark.parametrize("field", ["padding", "padding_x", "padding_y", "padding_z"])
@pytest.mark.parametrize(
    "value",
    [-1, (-0.1, 1), (1, -0.1), float("inf"), (0, float("nan")), (1,), (1, 2, 3)],
)
def test_domain_rejects_invalid_padding(field, value) -> None:
    domain = fdtd.Domain()
    before = domain.model_dump()
    with pytest.raises(ValidationError):
        domain(**{field: value})
    assert domain.model_dump() == before


def test_old_padding_argument_names_are_rejected() -> None:
    with pytest.raises(ValidationError, match="padding_um"):
        fdtd.Domain(padding_um=1)
    with pytest.raises(TypeError, match="background_padding_um"):
        fdtd.Simulation(background_padding_um=1)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_bounds_and_axis_padding_are_mutually_exclusive(axis) -> None:
    bounds = f"{axis}_bounds"
    padding = f"padding_{axis}"
    with pytest.raises(ValidationError, match="cannot be combined"):
        fdtd.Domain(**{bounds: (-1, 2), padding: 0.5})

    domain = fdtd.Domain(**{bounds: (-1, 2)})
    before = domain.model_dump()
    with pytest.raises(ValidationError, match="cannot be combined"):
        domain(**{padding: 0.5})
    with pytest.raises(ValidationError, match="cannot be combined"):
        setattr(domain, padding, 0.5)
    assert domain.model_dump() == before

    domain(**{bounds: None, padding: (0, 0.5)})
    before = domain.model_dump()
    with pytest.raises(ValidationError, match="cannot be combined"):
        setattr(domain, bounds, (-1, 2))
    assert domain.model_dump() == before
    domain(**{padding: None, bounds: (-1, 2)})


def test_axis_overrides_and_fallback_size_all_directions(fdtd_pdk_module) -> None:
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    resolved = replace(simulation.geometry("straight"), ports={})
    bounds = background_bounds_nm(
        resolved,
        "air",
        (0.2, 0.4),
        padding_x=0.5,
        padding_z=(0, 1),
    )
    assert bounds == pytest.approx((-500, -450, 0, 2500, 650, 1220))


def test_fallback_retains_port_alignment_and_background_stack(fdtd_pdk_module) -> None:
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module, domain={"padding": 0.5})
    resolved = simulation.geometry("straight")
    assert simulation.domain.padding == 0.5
    assert background_bounds_nm(resolved, "SiO2", 0.5) == pytest.approx(
        (0, -750, -1000, 2000, 750, 1000)
    )
    bounds = background_bounds_nm(resolved, "SiO2", 0.5, z_bounds=(-0.4, 0.6))
    assert bounds == pytest.approx((0, -750, -400, 2000, 750, 600))


@pytest.mark.parametrize("padding_x", [1, (0.5, 0), (0, 0.5)])
def test_guided_port_padding_reports_unsupported_extensions(
    tmp_path, fdtd_pdk_module, padding_x
) -> None:
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module, default_port="o1")
    simulation.geometry("straight")
    simulation.domain(padding_x=padding_x)
    with pytest.raises(
        fdtd.FDTDGeometryError, match=r"padding_x.*guided port.*extensions"
    ):
        simulation.write(tmp_path)
    assert not (tmp_path / "mesh.msh").exists()


def test_bend_can_pad_faces_without_guided_ports(fdtd_pdk_module) -> None:
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    resolved = simulation.geometry("straight")
    ports = dict(resolved.ports)
    ports["o2"] = replace(
        ports["o2"], center=(1, 0.25, 0.11), normal=(0, 1, 0), orientation=90
    )
    resolved = replace(resolved, ports=ports)
    bounds = background_bounds_nm(
        resolved,
        "SiO2",
        1,
        padding_x=(0, 0.5),
        padding_y=(0.7, 0),
        padding_z=(0.2, 0.3),
    )
    assert bounds == pytest.approx((0, -950, -200, 2500, 250, 520))
    with pytest.raises(fdtd.FDTDGeometryError, match=r"padding_y.*\+y.*o2"):
        background_bounds_nm(resolved, "SiO2", 1, padding_y=(0, 1))


def test_asymmetric_mesh_and_vertical_source_share_domain_bounds(
    tmp_path, fdtd_pdk_module
) -> None:
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module, default_port="o2")
    simulation.geometry("vertical_coupler", mesh_size_nm=750)
    simulation.domain(padding_x=(0, 0.5), padding_y=(0.4, 0.7), padding_z=(0.4, 0.6))

    artifacts = simulation.write(tmp_path)
    mesh = meshio.read(artifacts.mesh_path)
    assert mesh.points.min(axis=0) == pytest.approx([0, -650, -400])
    assert mesh.points.max(axis=0) == pytest.approx([2500, 950, 820])

    document = json.loads(artifacts.config_path.read_text())
    beam = document["excitation"]["gaussian_beam"]
    assert beam["region_min"] == pytest.approx([500, -500, 820])
    assert beam["region_max"] == pytest.approx([1500, 500, 820])
    assert document["monitors"][0]["region_min"][2] == pytest.approx(820)

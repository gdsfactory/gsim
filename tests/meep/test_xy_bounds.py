"""Exact PML-inner X/Y bounds for Meep simulations."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError


def _bounded_simulation(
    *, marker_bounds: tuple[float, float, float, float] | None = None
):
    import gdsfactory as gf

    from gsim.common.stack import Layer, LayerStack
    from gsim.meep import Simulation

    component = gf.Component()
    component.add_polygon(
        [(-5.0, -3.6), (5.0, -3.6), (5.0, 3.6), (-5.0, 3.6)],
        layer=(1, 0),
    )
    if marker_bounds is not None:
        left, bottom, right, top = marker_bounds
        component.add_polygon(
            [(left, bottom), (right, bottom), (right, top), (left, top)],
            layer=(99, 0),
        )
    component.add_port(
        name="o1",
        center=(-5.0, 0.0),
        orientation=180.0,
        width=0.6,
        layer=(1, 0),
    )
    component.add_port(
        name="o2",
        center=(5.0, 0.0),
        orientation=0.0,
        width=0.6,
        layer=(1, 0),
    )

    stack = LayerStack(
        pdk_name="test",
        layers={
            "core": Layer(
                name="core",
                gds_layer=(1, 0),
                zmin=0.0,
                zmax=0.22,
                thickness=0.22,
                material="si",
                layer_type="dielectric",
            )
        },
        dielectrics=[{"name": "clad", "zmin": 0.0, "zmax": 3.0, "material": "sio2"}],
    )

    simulation = Simulation()
    simulation.geometry(component=component, stack=stack)
    simulation.materials = {"si": 12.0, "sio2": 2.1}
    simulation.source(port="o1", wavelength=1.31, wavelength_span=0.01)
    simulation.monitors = ["o1", "o2"]
    return simulation


def test_domain_accepts_exact_bounds_with_other_axis_margin():
    from gsim.meep import Domain

    domain = Domain(
        margin_x=1.0,
        y_bounds=(-4.0, 4.0),
        z_bounds=(0.0, 3.0),
    )

    assert domain.x_bounds == "auto"
    assert domain.y_bounds == (-4.0, 4.0)
    assert domain.z_bounds == (0.0, 3.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"x_bounds": (-6.0, 6.0), "margin_x": 1.0}, "domain.x_bounds"),
        ({"y_bounds": (-4.0, 4.0), "margin_y": 1.0}, "domain.y_bounds"),
    ],
)
def test_domain_rejects_bound_and_explicit_same_axis_margin(kwargs, message):
    from gsim.meep import Domain

    with pytest.raises(ValidationError, match=message):
        Domain(**kwargs)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_domain_explicit_bounds_round_trip_without_inactive_margin(axis):
    from gsim.meep import Domain

    domain = Domain(**{f"{axis}_bounds": (-4.0, 4.0)})

    dumped = domain.model_dump()
    restored = Domain.model_validate(dumped)

    assert f"margin_{axis}" not in dumped
    assert restored == domain


def test_simulation_explicit_xy_bounds_model_round_trip():
    from gsim.meep import Domain, Simulation

    simulation = Simulation(domain=Domain(x_bounds=(-6.0, 6.0), y_bounds=(-4.0, 4.0)))

    dumped = simulation.model_dump()
    restored = Simulation.model_validate(dumped)

    assert "margin_x" not in dumped["domain"]
    assert "margin_y" not in dumped["domain"]
    assert restored == simulation


@pytest.mark.parametrize(
    ("initial", "field", "value"),
    [
        ({"x_bounds": (-6.0, 6.0)}, "margin_x", 1.0),
        ({"y_bounds": (-4.0, 4.0)}, "margin_y", 1.0),
        ({"margin_x": 1.0}, "x_bounds", (-6.0, 6.0)),
        ({"margin_y": 1.0}, "y_bounds", (-4.0, 4.0)),
    ],
)
def test_domain_assignment_rejects_competing_same_axis_control(initial, field, value):
    from gsim.meep import Domain

    domain = Domain(**initial)
    original = domain.model_copy(deep=True)
    original_fields_set = domain.model_fields_set.copy()

    with pytest.raises(ValidationError, match="cannot be combined"):
        setattr(domain, field, value)

    assert domain == original
    assert domain.model_fields_set == original_fields_set


def test_round_tripped_default_domain_accepts_new_explicit_xy_bounds():
    from gsim.meep import Domain

    dumped = Domain().model_dump()
    restored = Domain.model_validate(dumped)

    assert "margin_x" not in dumped
    assert "margin_y" not in dumped
    restored.x_bounds = (-6.0, 6.0)
    restored.y_bounds = (-4.0, 4.0)
    assert restored.x_bounds == (-6.0, 6.0)
    assert restored.y_bounds == (-4.0, 4.0)


def test_round_trip_preserves_explicit_default_valued_margin_intent():
    from gsim.meep import Domain

    dumped = Domain(margin_x=0.5).model_dump()
    restored = Domain.model_validate(dumped)
    original = restored.model_copy(deep=True)

    assert dumped["margin_x"] == 0.5
    with pytest.raises(ValidationError, match="cannot be combined"):
        restored.x_bounds = (-6.0, 6.0)
    assert restored == original


@pytest.mark.parametrize(
    "field_value",
    [(1.0, 1.0), (2.0, -2.0), (float("nan"), 1.0), (0.0, float("inf"))],
)
def test_domain_rejects_invalid_xy_bounds(field_value):
    from gsim.meep import Domain

    with pytest.raises(ValidationError):
        Domain(x_bounds=field_value)
    with pytest.raises(ValidationError):
        Domain(y_bounds=field_value)


def test_explicit_y_bounds_ignore_non_simulated_marker_bbox():
    simulation = _bounded_simulation(marker_bounds=(-5.0, -25.0, 5.0, 25.0))
    simulation.domain(
        pml=1.0,
        margin_x=1.0,
        port_margin=0.0,
        y_bounds=(-4.0, 4.0),
        z_bounds=(0.0, 3.0),
    )

    result = simulation.build_config()
    domain = result.config.domain

    assert domain.x_bounds is None
    assert domain.y_bounds == (-4.0, 4.0)
    assert domain.z_bounds == (0.0, 3.0)
    assert domain.margin_x_low == domain.margin_x_high == 1.0
    assert domain.margin_y_low == domain.margin_y_high == 0.0
    assert result.config.component_bbox == [-5.0, -25.0, 5.0, 25.0]
    assert result.component.dbbox().left == pytest.approx(-7.0)
    assert result.component.dbbox().right == pytest.approx(7.0)


def test_tight_bound_rejects_simulated_physical_geometry():
    simulation = _bounded_simulation(marker_bounds=(-5.0, -25.0, 5.0, 25.0))
    simulation.domain(y_bounds=(-3.0, 3.0))

    with pytest.raises(ValueError, match="simulated physical geometry"):
        simulation.build_config()


def test_far_explicit_bounds_rejected_before_port_extension(monkeypatch):
    import gdsfactory as gf

    simulation = _bounded_simulation()
    simulation.domain(x_bounds=(1_000_000.0, 1_000_010.0))

    def fail_if_extended(*_args, **_kwargs):
        raise AssertionError("port extension ran before bounds validation")

    monkeypatch.setattr(gf.components, "extend_ports", fail_if_extended)

    with pytest.raises(ValueError, match="simulated physical geometry"):
        simulation.build_config()


def test_tight_bound_rejects_port_mode_plane():
    simulation = _bounded_simulation()
    simulation.geometry.component = simulation.geometry.component.copy()
    core_layer = simulation.geometry.component.kcl.layer(1, 0)
    simulation.geometry.component.kdb_cell.clear(core_layer)
    simulation.geometry.component.add_polygon(
        [(-5.0, -0.25), (5.0, -0.25), (5.0, 0.25), (-5.0, 0.25)],
        layer=(1, 0),
    )
    simulation.domain(y_bounds=(-0.5, 0.5))

    with pytest.raises(ValueError, match="mode plane for port"):
        simulation.build_config()


def test_explicit_y_bounds_rejected_when_xz_collapses_y():
    simulation = _bounded_simulation()
    simulation.source.port = None
    simulation.solver(mode="2d", y_cut="auto")
    simulation.source_fiber(x=0.0, z=1.0, waist=1.0)
    simulation.domain(y_bounds=(-4.0, 4.0))

    with pytest.raises(ValueError, match="requires an active Y axis"):
        simulation.build_config()


def test_explicit_x_bounds_supported_in_xz():
    simulation = _bounded_simulation()
    simulation.source.port = None
    simulation.solver(mode="2d", y_cut=0.0)
    simulation.source_fiber(x=0.0, z=1.0, waist=1.0)
    simulation.domain(x_bounds=(-6.0, 6.0))

    result = simulation.build_config()

    assert result.config.domain.x_bounds == (-6.0, 6.0)
    assert result.config.domain.y_bounds is None


def test_explicit_xy_bounds_supported_in_xy_2d():
    simulation = _bounded_simulation()
    simulation.solver(mode="2d", z_cut="auto")
    simulation.domain(x_bounds=(-6.0, 6.0), y_bounds=(-4.0, 4.0))

    result = simulation.build_config()

    assert result.config.is_3d is False
    assert result.config.plane == "xy"
    assert result.config.domain.x_bounds == (-6.0, 6.0)
    assert result.config.domain.y_bounds == (-4.0, 4.0)


@pytest.mark.parametrize("fiber_x", [5.9, 5.95])
def test_explicit_x_bounds_reject_unsafe_fiber_source_position(fiber_x):
    simulation = _bounded_simulation()
    simulation.source.port = None
    simulation.solver(mode="2d", y_cut=0.0)
    simulation.source_fiber(x=fiber_x, z=1.0, waist=1.0)
    simulation.domain(x_bounds=(-6.0, 6.0))

    with pytest.raises(ValueError, match="fiber source placement"):
        simulation.build_config()


def test_serialized_bounds_are_hash_relevant(tmp_path):
    simulation = _bounded_simulation()
    simulation.domain(y_bounds=(-4.0, 4.0))
    result = simulation.build_config()
    path = result.config.to_json(tmp_path / "sim_config.json")

    domain_data = json.loads(path.read_text())["domain"]
    assert "x_bounds" not in domain_data
    assert domain_data["y_bounds"] == [-4.0, 4.0]
    assert domain_data["margin_y_low"] == 0.0
    assert domain_data["margin_y_high"] == 0.0


def test_input_hash_changes_when_exact_bounds_change(tmp_path):
    from gsim.hashing import compute_input_hash

    simulation = _bounded_simulation()
    auto_directory = simulation.write_config(tmp_path / "auto")
    auto_hash = compute_input_hash(auto_directory, job_type="meep")

    simulation.domain.y_bounds = (-4.0, 4.0)
    bounded_directory = simulation.write_config(tmp_path / "bounded")
    bounded_hash = compute_input_hash(bounded_directory, job_type="meep")

    assert bounded_hash != auto_hash


def test_auto_xy_bounds_keep_legacy_json_shape(tmp_path):
    simulation = _bounded_simulation()

    output_directory = simulation.write_config(tmp_path / "auto-bounds")
    domain_data = json.loads((output_directory / "sim_config.json").read_text())[
        "domain"
    ]

    assert "x_bounds" not in domain_data
    assert "y_bounds" not in domain_data
    assert domain_data["margin_x_low"] == 0.5
    assert domain_data["margin_x_high"] == 0.5
    assert domain_data["margin_y_low"] == 0.5
    assert domain_data["margin_y_high"] == 0.5


def test_auto_port_extension_accounts_for_marker_expanded_bbox():
    from gsim.meep.domain import automatic_port_extension_length

    simulation = _bounded_simulation(marker_bounds=(-10.0, -3.6, 10.0, 3.6))
    domain = simulation._domain_config()

    assert automatic_port_extension_length(simulation.geometry.component, domain) == 6.5

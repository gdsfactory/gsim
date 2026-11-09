"""Tests for the symmetry-plane model and the ``add_symmetry_plane`` API."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gsim.palace import (
    BoundaryModeSim,
    DrivenSim,
    EigenmodeSim,
    ElectrostaticSim,
    SymmetryPlaneConfig,
)


def test_defaults():
    """The default plane is a positive-side PMC plane at y=0."""
    plane = SymmetryPlaneConfig()

    assert (plane.axis, plane.position, plane.kind, plane.keep) == (
        "y",
        0.0,
        "pmc",
        "positive",
    )
    assert plane.verify_symmetry is True


def test_axis_z_rejected():
    """A layer stack is not mirror-symmetric in z."""
    with pytest.raises(ValidationError):
        SymmetryPlaneConfig(axis="z")


def test_position_grid_validation_is_deferred_until_layout_is_known():
    """A model cannot assume the owning layout has a 1 nm database unit."""
    assert SymmetryPlaneConfig(position=0.0004).position == 0.0004


@pytest.mark.parametrize("position", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_position_rejected(position):
    """Symmetry planes need a finite physical position."""
    with pytest.raises(ValidationError, match="finite"):
        SymmetryPlaneConfig(position=position)


def test_on_grid_position_accepted():
    """Finite positions are stored without rounding."""
    assert SymmetryPlaneConfig(position=12.345).position == 12.345


@pytest.mark.parametrize("sim_class", [DrivenSim, EigenmodeSim])
def test_add_symmetry_plane_stores_config(sim_class):
    """Supported simulations store the plane."""
    sim = sim_class()
    sim.add_symmetry_plane(axis="x", position=5.0, kind="pec", keep="negative")

    (plane,) = sim._symmetry_planes
    assert (plane.axis, plane.position, plane.kind, plane.keep) == (
        "x",
        5.0,
        "pec",
        "negative",
    )


def test_second_plane_raises():
    """Only one plane is supported."""
    sim = DrivenSim()
    sim.add_symmetry_plane()

    with pytest.raises(ValueError, match="Only one symmetry plane"):
        sim.add_symmetry_plane(axis="x")


@pytest.mark.parametrize("sim_class", [ElectrostaticSim, BoundaryModeSim])
def test_unsupported_simulations_raise(sim_class):
    """Electrostatic and boundarymode simulations reject symmetry planes."""
    sim = sim_class()

    with pytest.raises(NotImplementedError, match="driven and eigenmode"):
        sim.add_symmetry_plane()
    assert sim._symmetry_planes == []


def test_planes_are_per_instance():
    """The private plane list is not shared between simulations."""
    first, second = DrivenSim(), DrivenSim()
    first.add_symmetry_plane()

    assert second._symmetry_planes == []

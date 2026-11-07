"""Mesh grading is opt-in: without it a native-2D Palace mesh is unchanged."""

from __future__ import annotations

import gdsfactory as gf
import meshio
import pytest

from gsim.common import Layer, LayerStack
from gsim.palace import BoundaryModeSim, MeshGrading
from gsim.palace.mesh.generator import _graded_curve_sampling, generate_mesh
from gsim.palace.models import MeshConfig

BOX = (0.2, 0.6, 0.0, 0.2, 0.03)


def _geometry():
    gf.gpdk.PDK.activate()
    component = gf.Component()
    component.add_polygon([(0, 0), (2, 0), (2, 1), (0, 1)], layer=(1, 0))
    component.add_polygon([(0, 0), (2, 0), (2, 1), (0, 1)], layer=(2, 0))
    stack = LayerStack()
    stack.layers["core"] = Layer(
        name="core",
        gds_layer=(1, 0),
        zmin=0,
        zmax=0.2,
        thickness=0.2,
        material="silicon",
        layer_type="dielectric",
        mesh_resolution=0.05,
    )
    stack.layers["metal"] = Layer(
        name="metal",
        gds_layer=(2, 0),
        zmin=0.6,
        zmax=0.8,
        thickness=0.2,
        material="aluminum",
        layer_type="conductor",
    )
    return component, stack


def _simulation(tmp_path):
    component, stack = _geometry()
    sim = BoundaryModeSim()
    sim.set_geometry(component)
    sim.set_stack(stack)
    sim.set_output_dir(tmp_path)
    sim.set_cross_section("x=1")
    sim.set_airbox(margin_x=0.5, margin_y=0.5, z_above=0.5, z_below=0.5)
    return sim


def _counts(mesh_path) -> tuple[int, int]:
    mesh = meshio.read(str(mesh_path))
    triangles = sum(len(block.data) for block in mesh.cells if block.type == "triangle")
    return len(mesh.points), triangles


def test_default_boundary_mode_mesh_is_unchanged(tmp_path):
    # Recorded on the release before mesh grading existed: a conductor, a
    # layer with its own fine resolution and a refinement box.
    result = _simulation(tmp_path).mesh(
        refined_mesh_size=0.1,
        max_mesh_size=0.5,
        refinement_boxes=[BOX],
        verbose=False,
    )

    assert _counts(result.mesh_path) == (783, 1475)


def test_grading_changes_the_boundary_mode_mesh(tmp_path):
    result = _simulation(tmp_path).mesh(
        refined_mesh_size=0.1,
        max_mesh_size=0.5,
        refinement_boxes=[BOX],
        mesh_grading=MeshGrading(growth_per_um=0.25),
        verbose=False,
    )

    assert _counts(result.mesh_path) != (783, 1475)


class TestTheOptionOnTheMeshConfig:
    """Grading is kept on the mesh config like the refinement boxes."""

    GRADING = MeshGrading(growth_per_um=0.25, sources=["conductors"])

    @staticmethod
    def _build(sim, **overrides):
        settings = {
            "preset": "coarse",
            "refined_mesh_size": None,
            "max_mesh_size": None,
            "margin": None,
            "airbox_margin": None,
            "fmax": None,
            "planar_conductors": None,
            "show_gui": False,
        }
        return sim._build_mesh_config(**(settings | overrides))

    def test_grading_on_the_sims_mesh_config_survives_a_mesh_call(self):
        sim = BoundaryModeSim()
        sim.mesh_config = MeshConfig(mesh_grading=self.GRADING)

        assert self._build(sim).mesh_grading == self.GRADING

    def test_grading_given_to_the_call_replaces_it(self):
        sim = BoundaryModeSim()
        sim.mesh_config = MeshConfig(mesh_grading=self.GRADING)
        other = MeshGrading(growth_per_um=0.1)

        assert self._build(sim, mesh_grading=other).mesh_grading == other


def test_grading_is_refused_for_a_3d_mesh(tmp_path):
    component, stack = _geometry()
    with pytest.raises(ValueError, match="mesh_grading"):
        generate_mesh(
            component,
            stack,
            ports=[],
            output_dir=tmp_path,
            simulation_type="driven",
            mesh_grading=MeshGrading(growth_per_um=0.25),
            write_config=False,
        )


class TestCurveSampling:
    """The Distance-field samples are close enough for the grading rate."""

    @staticmethod
    def _size_between_samples(length, size, rate, sampling):
        # A point on the curve midway between two samples.
        return size + rate * (length / sampling) / 2

    @pytest.mark.parametrize("rate", [0.1, 0.25, 1.0, 2.0, 10.0])
    def test_size_on_the_curve_stays_within_a_tenth(self, rate):
        length, size = 10.0, 0.05
        sampling = _graded_curve_sampling(length, size, rate)

        assert self._size_between_samples(length, size, rate, sampling) <= 1.1 * size

    def test_slow_grading_keeps_half_size_spacing(self):
        # Below a rate of 0.4 per um the spacing is half the size, as before.
        assert _graded_curve_sampling(10.0, 0.1, 0.25) == 400
        assert _graded_curve_sampling(100.0, 0.1, 0.25) == 2000

    def test_cap_is_reported(self, caplog):
        length, size, rate = 1000.0, 0.01, 2.0
        with caplog.at_level("WARNING", logger="gsim.palace.mesh.generator"):
            sampling = _graded_curve_sampling(length, size, rate)

        assert sampling == 20000
        assert "Lower growth_per_um" in caplog.text
        assert self._size_between_samples(length, size, rate, sampling) > 1.1 * size

    def test_no_warning_below_the_cap(self, caplog):
        with caplog.at_level("WARNING", logger="gsim.palace.mesh.generator"):
            _graded_curve_sampling(10.0, 0.05, 2.0)

        assert caplog.text == ""

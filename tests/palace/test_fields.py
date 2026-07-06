"""Tests for the NaN-free field-visualization module (gsim.palace.fields)."""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers — build tiny synthetic meshes
# ---------------------------------------------------------------------------


def _make_triangle_mesh() -> tuple:
    """Create a tiny PyVista triangle mesh with a vector field + attribute cell data.

    Returns (mesh, vector_field_name, attribute_field_name).
    """
    import pyvista as pv

    # Two triangles forming a square
    # PyVista 0.46+ requires flat connectivity
    points = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
        ],
        dtype=float,
    )
    # Flat connectivity: for each cell: [n_nodes, id0, id1, id2]
    cells = np.array([3, 0, 1, 2, 3, 0, 2, 3], dtype=np.int64)
    celltypes = np.array([pv.CellType.TRIANGLE, pv.CellType.TRIANGLE], dtype=np.uint8)

    mesh = pv.UnstructuredGrid(cells, celltypes, points)

    # Vector point data (3-component)
    mesh.point_data["E"] = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=float,
    )

    # Attribute cell data
    mesh.cell_data["attribute"] = np.array([1, 2], dtype=int)

    return mesh, "E", "attribute"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestActivateVectorComponent:
    def test_magnitude(self):
        from gsim.palace.fields import activate_vector_component

        mesh, field, _ = _make_triangle_mesh()
        name = activate_vector_component(mesh, field, component="mag")
        assert name == "E_mag"
        assert "E_mag" in mesh.point_data
        vals = mesh.point_data["E_mag"]
        assert vals.shape == (4,)
        assert np.isfinite(vals).all()

    def test_x_component(self):
        from gsim.palace.fields import activate_vector_component

        mesh, field, _ = _make_triangle_mesh()
        name = activate_vector_component(mesh, field, component="x")
        assert name == "E_x"
        assert "E_x" in mesh.point_data

    def test_custom_output_name(self):
        from gsim.palace.fields import activate_vector_component

        mesh, field, _ = _make_triangle_mesh()
        name = activate_vector_component(mesh, field, component="mag", output_name="magnitude")
        assert name == "magnitude"
        assert "magnitude" in mesh.point_data

    def test_missing_field(self):
        from gsim.palace.fields import activate_vector_component

        mesh, _, _ = _make_triangle_mesh()
        with pytest.raises(KeyError, match="not found"):
            activate_vector_component(mesh, "nonexistent", component="mag")

    def test_invalid_component(self):
        from gsim.palace.fields import activate_vector_component

        mesh, field, _ = _make_triangle_mesh()
        with pytest.raises(ValueError, match="component must be one of"):
            activate_vector_component(mesh, field, component="w")


class TestExtractBoundaryCells:
    def test_basic_extraction(self):
        from gsim.palace.fields import extract_boundary_cells

        mesh, _, _ = _make_triangle_mesh()
        subset = extract_boundary_cells(mesh, [1])
        assert subset.n_cells == 1
        assert subset.n_points > 0

    def test_multiple_attributes(self):
        from gsim.palace.fields import extract_boundary_cells

        mesh, _, _ = _make_triangle_mesh()
        subset = extract_boundary_cells(mesh, {1, 2})
        assert subset.n_cells == 2

    def test_no_match(self):
        from gsim.palace.fields import extract_boundary_cells

        mesh, _, _ = _make_triangle_mesh()
        with pytest.raises(ValueError, match="No cells matched"):
            extract_boundary_cells(mesh, [99])

    def test_empty_attributes(self):
        from gsim.palace.fields import extract_boundary_cells

        mesh, _, _ = _make_triangle_mesh()
        with pytest.raises(ValueError, match="No attributes provided"):
            extract_boundary_cells(mesh, [])


class TestResolveScalarField:
    def test_scalar_direct(self):
        from gsim.palace.fields import resolve_scalar_field

        mesh, _, _ = _make_triangle_mesh()
        mesh.point_data["my_scalar"] = np.ones(mesh.n_points)
        name = resolve_scalar_field(mesh, scalar_field="my_scalar")
        assert name == "my_scalar"

    def test_vector_field(self):
        from gsim.palace.fields import resolve_scalar_field

        mesh, field, _ = _make_triangle_mesh()
        name = resolve_scalar_field(mesh, vector_field=field, component="mag")
        assert name == "E_mag"
        assert "E_mag" in mesh.point_data

    def test_both_provided(self):
        from gsim.palace.fields import resolve_scalar_field

        mesh, _, _ = _make_triangle_mesh()
        with pytest.raises(ValueError, match="not both"):
            resolve_scalar_field(mesh, scalar_field="a", vector_field="b")

    def test_neither_provided(self):
        from gsim.palace.fields import resolve_scalar_field

        mesh, _, _ = _make_triangle_mesh()
        with pytest.raises(ValueError, match="Provide one of"):
            resolve_scalar_field(mesh)


class TestPlotBoundaryField:
    """Smoke tests — create plotter but don't show."""

    def test_vector_field_render(self):
        from gsim.palace.fields import BoundaryFieldData, plot_boundary_field

        mesh, field, _ = _make_triangle_mesh()
        bnd = BoundaryFieldData(
            mesh=mesh,
            dataset_name="test",
            step_index=0,
            timestep=0.0,
            selected_attributes=(1, 2),
        )
        pl = plot_boundary_field(bnd, vector_field=field, component="mag", off_screen=True)
        assert pl is not None
        pl.close()

    def test_scalar_field_render(self):
        from gsim.palace.fields import BoundaryFieldData, plot_boundary_field

        mesh, _, _ = _make_triangle_mesh()
        mesh.point_data["my_scalar"] = np.ones(mesh.n_points)
        bnd = BoundaryFieldData(
            mesh=mesh,
            dataset_name="test",
            step_index=0,
            timestep=0.0,
            selected_attributes=(1, 2),
        )
        pl = plot_boundary_field(bnd, scalar_field="my_scalar", off_screen=True)
        assert pl is not None
        pl.close()

    def test_log_scale_positive(self):
        from gsim.palace.fields import BoundaryFieldData, plot_boundary_field

        mesh, field, _ = _make_triangle_mesh()
        # Ensure all values positive for log scale
        mesh.point_data[field] = np.abs(mesh.point_data[field]) + 0.1
        bnd = BoundaryFieldData(
            mesh=mesh,
            dataset_name="test",
            step_index=0,
            timestep=0.0,
            selected_attributes=(1, 2),
        )
        pl = plot_boundary_field(bnd, vector_field=field, component="mag", log_scale=True, off_screen=True)
        assert pl is not None
        pl.close()

    def test_no_nan_after_magnitude(self):
        """Core assertion: computed magnitude must have 0 NaN."""
        from gsim.palace.fields import BoundaryFieldData, activate_vector_component

        mesh, field, _ = _make_triangle_mesh()
        scalar_name = activate_vector_component(mesh, field, component="mag")
        assert np.isnan(mesh.point_data[scalar_name]).sum() == 0

    def test_log_scale_with_nonpositive(self):
        """Log scale with some zero values must clamp correctly, no errors."""
        from gsim.palace.fields import BoundaryFieldData, plot_boundary_field

        mesh, field, _ = _make_triangle_mesh()
        # Set one point to zero
        mesh.point_data[field][0] = [0.0, 0.0, 0.0]
        bnd = BoundaryFieldData(
            mesh=mesh,
            dataset_name="test",
            step_index=0,
            timestep=0.0,
            selected_attributes=(1, 2),
        )
        pl = plot_boundary_field(bnd, vector_field=field, component="mag", log_scale=True, off_screen=True)
        pl.close()


class TestSelectorContext:
    def test_build_from_dict(self):
        from gsim.palace.fields import build_selector_context

        config = {"Boundaries": {"PEC": {"Attributes": [1, 2]}}}
        pg_map = {"top_conductor": 1, "ground_plane": 2}
        ctx = build_selector_context(config, pg_map)
        assert ctx.pg_map["top_conductor"] == 1
        assert ctx.boundaries_by_type["PEC"] == (1, 2)

    def test_resolve_entity_attributes(self):
        from gsim.palace.fields import (
            SelectorContext,
            resolve_entity_attributes,
        )

        ctx = SelectorContext(pg_map={"a": 10, "b": 20}, boundaries_by_type={})
        attrs = resolve_entity_attributes(["a", "b"], ctx)
        assert attrs == (10, 20)

    def test_resolve_unknown_entity(self):
        from gsim.palace.fields import (
            SelectorContext,
            resolve_entity_attributes,
        )

        ctx = SelectorContext(pg_map={"a": 10}, boundaries_by_type={})
        with pytest.raises(KeyError, match="Unknown"):
            resolve_entity_attributes("missing", ctx)


class TestResolveBoundaryType:
    def test_resolve_boundary_type(self):
        from gsim.palace.fields import (
            SelectorContext,
            resolve_boundary_type_attributes,
        )

        ctx = SelectorContext(
            pg_map={},
            boundaries_by_type={"PEC": (1, 2)},
        )
        assert resolve_boundary_type_attributes("PEC", ctx) == (1, 2)

    def test_unknown_boundary_type(self):
        from gsim.palace.fields import (
            SelectorContext,
            resolve_boundary_type_attributes,
        )

        ctx = SelectorContext(pg_map={}, boundaries_by_type={})
        with pytest.raises(KeyError, match="not found"):
            resolve_boundary_type_attributes("ABC", ctx)


class TestPlotVolumeSlice:
    def test_slice_render(self):
        from gsim.palace.fields import VolumeFieldData, plot_volume_slice

        mesh, field, _ = _make_triangle_mesh()
        vol = VolumeFieldData(mesh=mesh, dataset_name="test", step_index=0, timestep=0.0)
        pl = plot_volume_slice(vol, vector_field=field, axis="z", value=0.0, off_screen=True)
        pl.close()


class TestExtractAxisSlice:
    def test_extract(self):
        from gsim.palace.fields import VolumeFieldData, extract_axis_slice

        mesh, _, _ = _make_triangle_mesh()
        vol = VolumeFieldData(mesh=mesh, dataset_name="test", step_index=0, timestep=0.0)
        sl = extract_axis_slice(vol, axis="z", value=0.0)
        assert sl.n_points > 0

    def test_invalid_axis(self):
        from gsim.palace.fields import VolumeFieldData, extract_axis_slice

        mesh, _, _ = _make_triangle_mesh()
        vol = VolumeFieldData(mesh=mesh, dataset_name="test", step_index=0, timestep=0.0)
        with pytest.raises(ValueError, match="axis must be one of"):
            extract_axis_slice(vol, axis="w", value=0.0)
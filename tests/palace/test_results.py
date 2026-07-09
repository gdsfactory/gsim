"""Tests for gsim.palace.results and gsim.palace.fields.

Covers S-parameter loading with port-name mapping (results) and the
NaN-free field-visualization routines (fields).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gsim.palace.results import (
    PalaceTextResults,
    SParams,
    get_port_map,
    load_sparams,
    load_text_results,
)


@pytest.fixture
def sim_dir(tmp_path: Path) -> Path:
    """Create a minimal Palace output directory."""
    palace_dir = tmp_path / "output" / "palace"
    palace_dir.mkdir(parents=True)

    port_info = {
        "ports": [
            {"portnumber": 1, "name": "o1", "Z0": 50.0, "type": "cpw"},
            {"portnumber": 2, "name": "o2", "Z0": 50.0, "type": "cpw"},
            {"portnumber": 3, "name": "o3", "Z0": 50.0, "type": "lumped"},
        ],
        "unit": 1e-6,
        "name": "palace",
    }
    (tmp_path / "port_information.json").write_text(json.dumps(port_info))

    csv_content = (
        "f (GHz), |S[1][1]| (dB), arg(S[1][1]) (deg.),"
        " |S[2][1]| (dB), arg(S[2][1]) (deg.),"
        " |S[3][1]| (dB), arg(S[3][1]) (deg.)\n"
        "1.0, -20.0, -45.0, -3.0, -90.0, -30.0, -120.0\n"
        "2.0, -18.0, -50.0, -2.5, -85.0, -28.0, -115.0\n"
    )
    (palace_dir / "port-S.csv").write_text(csv_content)

    return tmp_path


@pytest.fixture
def sim_dir_no_names(tmp_path: Path) -> Path:
    """Sim dir with port_information.json without name fields."""
    palace_dir = tmp_path / "output" / "palace"
    palace_dir.mkdir(parents=True)

    port_info = {
        "ports": [
            {"portnumber": 1, "Z0": 50.0, "type": "cpw"},
            {"portnumber": 2, "Z0": 50.0, "type": "cpw"},
        ],
        "unit": 1e-6,
        "name": "palace",
    }
    (tmp_path / "port_information.json").write_text(json.dumps(port_info))

    csv_content = (
        "f (GHz), |S[1][1]| (dB), arg(S[1][1]) (deg.),"
        " |S[2][1]| (dB), arg(S[2][1]) (deg.)\n"
        "1.0, -20.0, -45.0, -3.0, -90.0\n"
    )
    (palace_dir / "port-S.csv").write_text(csv_content)

    return tmp_path


class TestSParams:
    """Tests for the SParams result object."""

    def test_returns_sparams_object(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        assert isinstance(sp, SParams)

    def test_freq(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        assert sp.freq[0] == pytest.approx(1.0)
        assert sp.freq[1] == pytest.approx(2.0)
        assert len(sp.freq) == 2

    def test_port_names(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        assert sp.port_names == ["o1", "o2", "o3"]

    def test_bracket_access(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        s11 = sp["o1", "o1"]
        assert s11.db[0] == pytest.approx(-20.0)
        assert s11.deg[0] == pytest.approx(-45.0)

    def test_bracket_access_cross(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        s21 = sp["o2", "o1"]
        assert s21.db[0] == pytest.approx(-3.0)
        assert s21.deg[0] == pytest.approx(-90.0)

    def test_mag_property(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        s11 = sp["o1", "o1"]
        expected_mag = 10 ** (-20.0 / 20)
        assert s11.mag[0] == pytest.approx(expected_mag)

    def test_complex_property(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        s11 = sp["o1", "o1"]
        c = s11.complex[0]
        assert abs(c) == pytest.approx(10 ** (-20.0 / 20))
        assert np.rad2deg(np.angle(c)) == pytest.approx(-45.0)

    def test_rf_shorthand_s11(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        assert sp.s11.db[0] == pytest.approx(-20.0)

    def test_rf_shorthand_s21(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        assert sp.s21.db[0] == pytest.approx(-3.0)

    def test_rf_shorthand_s31(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        assert sp.s31.db[0] == pytest.approx(-30.0)

    def test_invalid_shorthand_raises(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        with pytest.raises(AttributeError):
            _ = sp.s99

    def test_invalid_bracket_raises(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        with pytest.raises(KeyError, match="not found"):
            _ = sp["o1", "o99"]

    def test_keys(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        keys = sp.keys()
        assert ("o1", "o1") in keys
        assert ("o2", "o1") in keys
        assert ("o3", "o1") in keys

    def test_repr(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        r = repr(sp)
        assert "3 ports" in r
        assert "o1" in r

    def test_to_dataframe(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        df = sp.to_dataframe()
        assert "freq_ghz" in df.columns
        assert "S_o1_o1_db" in df.columns

    def test_plot_runs(self, sim_dir: Path) -> None:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt

        sp = load_sparams(sim_dir)
        sp.plot()
        plt.close("all")

    def test_plot_interactive_runs(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        fig = sp.plot_interactive()
        assert len(fig.data) == len(sp.keys())

    def test_plot_interactive_labels(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        fig = sp.plot_interactive()
        names = [trace.name for trace in fig.data]
        # 3 ports -> S11, S21, S31 etc.
        assert "S11" in names
        assert "S21" in names

    def test_plot_interactive_visibility(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        fig = sp.plot_interactive()
        # First excitation column (Si1) should be visible
        for trace in fig.data:
            if trace.name.endswith("1"):  # S11, S21, S31
                assert trace.visible is True
            else:
                assert trace.visible == "legendonly"


class TestLoadSparamsSource:
    """Tests for source resolution (dir, subdir, dict)."""

    def test_accepts_dir(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir)
        assert len(sp.freq) == 2

    def test_accepts_palace_subdir(self, sim_dir: Path) -> None:
        sp = load_sparams(sim_dir / "output" / "palace")
        assert len(sp.freq) == 2

    def test_accepts_results_dict(self, sim_dir: Path) -> None:
        results = {
            "port-S.csv": sim_dir / "output" / "palace" / "port-S.csv",
        }
        sp = load_sparams(results)
        assert sp["o1", "o1"].db[0] == pytest.approx(-20.0)

    def test_missing_csv_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"port-S\.csv"):
            load_sparams(tmp_path)

    def test_results_dict_missing_csv_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="port-S"):
            load_sparams({"other.csv": Path("/nonexistent")})

    def test_fallback_numeric_names(self, sim_dir_no_names: Path) -> None:
        sp = load_sparams(sim_dir_no_names)
        assert sp.port_names == ["p1", "p2"]
        assert sp["p1", "p1"].db[0] == pytest.approx(-20.0)


class TestGetPortMap:
    """Tests for get_port_map."""

    def test_returns_mapping(self, sim_dir: Path) -> None:
        pm = get_port_map(sim_dir)
        assert pm == {1: "o1", 2: "o2", 3: "o3"}

    def test_legacy_numeric_fallback(self, sim_dir_no_names: Path) -> None:
        pm = get_port_map(sim_dir_no_names)
        assert pm == {1: "p1", 2: "p2"}


class TestSParamsSaveLoad:
    """Tests for SParams save_npz/from_file round-trip."""

    def test_round_trip(self, sim_dir: Path, tmp_path: Path) -> None:
        sp = load_sparams(sim_dir)
        out = sp.save_npz(tmp_path / "cached")
        assert out.suffix == ".npz"
        assert out.exists()

        loaded = SParams.from_file(out)
        assert loaded.port_names == sp.port_names
        assert len(loaded.freq) == len(sp.freq)
        np.testing.assert_allclose(loaded.freq, sp.freq)
        for key in sp._data:
            np.testing.assert_allclose(loaded[key].db, sp[key].db)
            np.testing.assert_allclose(loaded[key].deg, sp[key].deg)

    def test_adds_npz_suffix(self, sim_dir: Path, tmp_path: Path) -> None:
        sp = load_sparams(sim_dir)
        out = sp.save_npz(tmp_path / "no_ext")
        assert out.name == "no_ext.npz"

    def test_from_file_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            SParams.from_file(tmp_path / "nonexistent.npz")


# ---------------------------------------------------------------------------
# Field-visualization tests (gsim.palace.fields) — NaN-free direct mesh rendering
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

    def test_auto_clim_percentile(self):
        """Auto clim uses 98th percentile so peaks don't wash out the bulk."""
        from gsim.palace.fields import _auto_clim

        # bulk ~1, a few peaks ~1000 (current crowding)
        vals = np.concatenate([np.full(900, 1.0), np.full(100, 1000.0)])
        vmin, vmax = _auto_clim(vals, log_scale=False)
        assert vmin == 0.0
        # 98th percentile of this distribution is 1.0 (peaks are top 2%)
        assert vmax == pytest.approx(1.0)

    def test_auto_clim_log(self):
        from gsim.palace.fields import _auto_clim

        vals = np.concatenate([np.full(900, 1.0), np.full(100, 1000.0)])
        vmin, vmax = _auto_clim(vals, log_scale=True)
        assert vmin > 0.0
        assert vmax > vmin

    def test_auto_clim_signed_symmetric(self):
        """Signed data gets symmetric limits centered at zero."""
        from gsim.palace.fields import _auto_clim

        # E_y-like data: both polarities, peaks at edges
        vals = np.concatenate([np.full(400, -100.0), np.full(400, 100.0),
                               np.linspace(-500, 500, 200)])
        vmin, vmax = _auto_clim(vals, signed=True)
        assert vmin < 0.0 < vmax
        assert vmin == pytest.approx(-vmax)  # symmetric

    def test_auto_clim_unsigned_clamps_to_zero(self):
        """Unsigned data (magnitudes) clamps vmin to 0."""
        from gsim.palace.fields import _auto_clim

        vals = np.abs(np.concatenate([np.full(400, -100.0), np.full(100, 500.0)]))
        vmin, vmax = _auto_clim(vals, signed=False)
        assert vmin == 0.0
        assert vmax > 0.0


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


class TestLoadFieldContext:
    """Tests for the load_field_context convenience helper."""

    def test_loads_meshes_and_context(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from gsim.palace import fields as fields_mod

        # Build a fake simulation directory layout:
        #   <sim_dir>/config.json
        #   <sim_dir>/palace.msh
        #   <sim_dir>/output/palace/port-S.csv   <- results_dir
        sim_dir = tmp_path / "sim"
        results_dir = sim_dir / "output" / "palace"
        results_dir.mkdir(parents=True)
        (results_dir / "port-S.csv").write_text("f (GHz)\n1.0\n")
        (sim_dir / "palace.msh").write_text("mesh")
        (sim_dir / "config.json").write_text(
            json.dumps({"Boundaries": {"PEC": {"Attributes": [1, 2]}}})
        )

        # Fake meshio.read returning an object with field_data
        fake_mesh = SimpleNamespace(field_data={"topmetal2_xy": (5, 2), "ground": (6, 2)})
        import meshio as _real_meshio

        monkeypatch.setattr(_real_meshio, "read", lambda _: fake_mesh)

        # Fake load_fields returning distinct volume/boundary meshes
        fake_vol = SimpleNamespace(n_points=100, n_cells=10, _kind="vol")
        fake_bnd = SimpleNamespace(n_points=50, n_cells=5, _kind="bnd")

        def fake_load_fields(source, *, excitation=1, cycle=None, boundary=False):
            return fake_bnd if boundary else fake_vol

        monkeypatch.setattr("gsim.palace.results.load_fields", fake_load_fields)

        vol, bnd, ctx, pg_map = fields_mod.load_field_context(results_dir, excitation=1)

        assert vol is fake_vol
        assert bnd is fake_bnd
        assert pg_map == {"topmetal2_xy": 5, "ground": 6}
        assert ctx.boundaries_by_type["PEC"] == (1, 2)

    def test_missing_mesh_raises(self, tmp_path):
        from gsim.palace import fields as fields_mod

        sim_dir = tmp_path / "sim"
        results_dir = sim_dir / "output" / "palace"
        results_dir.mkdir(parents=True)
        (results_dir / "port-S.csv").write_text("f (GHz)\n1.0\n")
        (sim_dir / "config.json").write_text(json.dumps({}))

        with pytest.raises(FileNotFoundError, match="Mesh file not found"):
            fields_mod.load_field_context(results_dir)

    def test_missing_config_raises(self, tmp_path):
        from gsim.palace import fields as fields_mod

        sim_dir = tmp_path / "sim"
        results_dir = sim_dir / "output" / "palace"
        results_dir.mkdir(parents=True)
        (results_dir / "port-S.csv").write_text("f (GHz)\n1.0\n")
        (sim_dir / "palace.msh").write_text("mesh")

        with pytest.raises(FileNotFoundError, match="Palace config not found"):
            fields_mod.load_field_context(results_dir)
@pytest.fixture
def text_results_dir(tmp_path: Path) -> Path:
    """Create a minimal BoundaryMode-style text output directory."""
    palace_dir = tmp_path / "output" / "palace"
    palace_dir.mkdir(parents=True)

    (palace_dir / "mode-kn.csv").write_text(
        "m,Re{kn} (1/m),Im{kn} (1/m),Re{n_eff},Im{n_eff}\n"
        "1,2.0,0.0,1.50,0.00\n"
        "2,1.8,-0.1,1.20,-0.05\n"
    )
    (palace_dir / "domain-E.csv").write_text("domain,energy\nair,0.4\nsio2,0.6\n")
    (palace_dir / "error-indicators.csv").write_text("elem,error\n1,0.01\n2,0.02\n")
    (palace_dir / "palace.json").write_text(
        json.dumps({"problem": "BoundaryMode", "converged": True})
    )
    (palace_dir / "solver.log").write_text("iteration 1\niteration 2\n")
    return tmp_path


class TestTextResults:
    """Tests for non-S-parameter text result parsing."""

    def test_load_text_results_from_directory(self, text_results_dir: Path) -> None:
        out = load_text_results(text_results_dir)
        assert isinstance(out, PalaceTextResults)
        assert "mode-kn.csv" in out.csv_tables
        assert "palace.json" in out.json_data
        assert "solver.log" in out.text_data
        assert 1 in out.modes
        assert out["mode_1"]["k_n"] == complex(2.0, 0.0)

    def test_load_text_results_from_dict(self, text_results_dir: Path) -> None:
        results = {
            "mode-kn.csv": text_results_dir / "output" / "palace" / "mode-kn.csv",
            "palace.json": text_results_dir / "output" / "palace" / "palace.json",
        }
        out = load_text_results(results)
        assert "mode-kn.csv" in out.csv_tables
        assert "palace.json" in out.json_data

    def test_str_contains_tables(self, text_results_dir: Path) -> None:
        out = load_text_results(text_results_dir)
        text = str(out)
        assert "mode 1:" in text
        assert "k_n =" in text
        assert "n_eff =" in text
        assert "eta_eff ~=" in text

    def test_print_alias_emits_pretty_text(
        self, text_results_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = load_text_results(text_results_dir)
        out.print(max_rows=1, max_lines=2)
        captured = capsys.readouterr()
        assert "mode 1:" in captured.out
        assert "eta_eff ~=" in captured.out

    def test_dictionary_style_mode_access(self, text_results_dir: Path) -> None:
        out = load_text_results(text_results_dir)
        modes = out["modes"]
        assert 2 in modes
        assert modes[2]["k_n"] == complex(1.8, -0.1)
        assert out[1]["n_eff"] == complex(1.5, 0.0)

    def test_no_text_files_raises(self, tmp_path: Path) -> None:
        (tmp_path / "output" / "palace").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="parseable"):
            load_text_results(tmp_path)

"""Check legacy mode tables and multi-excitation result presentation."""

import numpy as np
import pytest

from gsim.palace.results import PalaceTextResults, SParam, SParams


@pytest.mark.parametrize("column", ["Re{kn}", "k_n", "absent"])
@pytest.mark.parametrize("effective_index", ["0", "2", "invalid"])
def test_legacy_mode_tables(tmp_path, column, effective_index):
    tables = {
        "mode-kn.csv": [{column: "3", "Re{n_eff}": effective_index}],
        "mode-V.csv": [{"Re{V[1]} (V)": "1", "Im{V[1]} (V)": "2"}],
    }
    results = PalaceTextResults(
        files={name: tmp_path / name for name in tables},
        csv_tables=tables,
        json_data={},
        text_data={},
    )
    mode = results[1]
    if column == "absent":
        assert np.isnan(mode["k_n"].real)
    else:
        assert mode["k_n"] == 3
    assert np.isfinite(mode["eta_eff"].real) == (effective_index == "2")
    assert results.mode_voltage() == 1 + 2j
    assert results.mode_voltage(mode=2) is None
    assert "1.00+2.00j V" in str(results)
    assert results.keys() == sorted(tables)
    with pytest.raises(KeyError):
        results["invalid"]
    assert "empty csv" in results._format_csv_table("empty", [], 1)
    preview = results._format_csv_table("values", [{"a": "long"}, {"a": "hidden"}], 1)
    assert "long" in preview
    assert "hidden" not in preview
    assert "1 more rows" in preview


@pytest.mark.parametrize("full", [False, True])
def test_multi_excitation_plot_data(full):
    frequency = np.array([1.0, 2.0])
    parameter = SParam(np.array([-3.0, -6.0]), np.array([30.0, -90.0]))
    data = {(to, source): parameter for source in ("in", "out") for to in ("in", "out")}
    results = SParams(frequency, data, ["in", "out"])
    figure = results.plot_plotly(full=full)
    assert len(figure.data) == (8 if full else 4)
    for trace in figure.data:
        np.testing.assert_array_equal(trace.x, frequency)
        np.testing.assert_array_equal(
            trace.y, parameter.db if trace.yaxis == "y" else parameter.deg
        )


def test_sparse_network_uses_reciprocity():
    transmission = SParam(np.array([-6.0]), np.array([90.0]))
    results = SParams(np.array([2.0]), {("in", "out"): transmission}, ["in", "out"])
    network = results.to_skrf()
    np.testing.assert_allclose(
        network.s[0], [[0, transmission.complex[0]], [transmission.complex[0], 0]]
    )
    with pytest.raises(KeyError, match="indexing"):
        results[("in",)]
    with pytest.raises(AttributeError):
        _ = results.s11


@pytest.mark.parametrize("invalid", [0, 2])
def test_mesh_summary_reports_quality_and_named_groups(tmp_path, invalid):
    from gsim.palace.models.results import SimulationResult

    result = SimulationResult(
        mesh_path=tmp_path / "mesh.msh",
        output_dir=tmp_path,
        config_path=tmp_path / "config.json",
        mesh_stats={
            "bbox": {"xmin": 0, "xmax": 2, "ymin": 0, "ymax": 3, "zmin": 0, "zmax": 4},
            "nodes": 20,
            "elements": 10,
            "tetrahedra": 10,
            "quality": {"min": 0.5, "mean": 0.8},
            "edge_length": {"min": 0.1, "max": 1},
            "sicn": {"mean": 0.7, "invalid": invalid},
            "groups": {
                "volumes": [{"name": "core", "tag": 1}, "clad"],
                "surfaces": [{"name": "port", "tag": 2}, "wall"],
            },
        },
    )
    text = str(result)
    for expected in [
        "2.0 x 3.0 x 4.0",
        "0.800",
        "0.10 - 1.00",
        "core [1]",
        "port [2]",
        "invalid!" if invalid else "all valid",
    ]:
        assert expected in text

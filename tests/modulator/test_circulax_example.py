"""The circulax-side reassembly example, replayed against gsim's answers.

The example (samples/circulax_reassembly.py) is the reference for how a
consumer wires the exported artifacts together, so it is held to the
same round-trip contract as the library readers: loaded as a module —
imports resolved by path, no gsim inside — its readers and network math
must reproduce the line Stage's driven response and the charge Stage's
junction branch from the files alone.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from gsim.common.circuit import line_smatrix, write_junction_model, write_touchstone

from .test_line_stage import rf_params

EXAMPLE = Path(__file__).parents[2] / "samples" / "circulax_reassembly.py"


@pytest.fixture(scope="module")
def example():
    """The example loaded as a module straight from its file."""
    spec = importlib.util.spec_from_file_location("circulax_reassembly", EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def artifacts(tmp_path):
    """Canned exports written with the library writers."""
    rf = rf_params()
    length_m = 3e-3
    touchstone = write_touchstone(
        tmp_path / "electrode.s2p",
        freq_hz=rf.freq_hz,
        s=line_smatrix(rf.gamma_per_m, rf.z0_ohm, length_m=length_m),
        comments=["length_m = 0.003"],
    )
    junction = write_junction_model(
        tmp_path / "junction.json",
        bias_v=[0.0, 1.0, 2.0],
        r_s_ohm_m=[1.2e-4, 1.1e-4, 1.0e-4],
        c_j_f_per_m=[3.3e-10, 2.8e-10, 2.4e-10],
        contact="cathode",
        freq_hz=1e9,
    )
    return rf, length_m, touchstone, junction


class TestExampleReassembly:
    def test_the_example_reproduces_the_library_driven_response(
        self, example, artifacts
    ):
        from gsim.common.circuit import line_driven_response

        rf, length_m, touchstone, _ = artifacts
        freq, s, z_ref = example.read_touchstone(touchstone)
        reassembled = example.driven_response(s, z_ref, z_gen_ohm=50.0, z_load_ohm=45.0)

        expected = line_driven_response(
            rf.gamma_per_m,
            rf.z0_ohm,
            length_m=length_m,
            z_gen_ohm=50.0,
            z_load_ohm=45.0,
        )
        np.testing.assert_allclose(freq, rf.freq_hz, rtol=1e-12)
        np.testing.assert_allclose(reassembled, expected, rtol=1e-8)

    def test_the_example_reads_the_junction_model_exactly(self, example, artifacts):
        *_, junction = artifacts

        model = example.read_junction_model(junction)

        assert model["contact"] == "cathode"
        assert model["r_s_ohm_m"].tolist() == [1.2e-4, 1.1e-4, 1.0e-4]
        assert model["c_j_f_per_m"].tolist() == [3.3e-10, 2.8e-10, 2.4e-10]

    def test_the_junction_admittance_inverts_the_series_rc_fit(self, example):
        from gsim.common.transmission_line import series_rc_from_admittance

        r_s, c_j = 1.1e-4, 2.8e-10
        y = example.junction_shunt_admittance(1e9, r_s, c_j)

        fitted = series_rc_from_admittance(y, freq_hz=1e9)
        assert fitted.r_s_ohm_m == pytest.approx(r_s, rel=1e-12)
        assert fitted.c_j_f_per_m == pytest.approx(c_j, rel=1e-12)

    def test_the_example_runs_end_to_end_and_prints_both_tables(
        self, example, artifacts, capsys
    ):
        *_, touchstone, junction = artifacts

        assert example.main([str(touchstone), str(junction)]) == 0

        out = capsys.readouterr().out
        assert "Driven response" in out
        assert "Junction branch" in out
        assert "cathode" in out

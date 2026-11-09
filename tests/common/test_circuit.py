"""The line two-port export: S-matrix, Touchstone writer, SAX callable.

Everything here is pure array math against the analytic lossy-line
S-parameters, so nothing solves anything. The independent reference the
S-matrix is checked against is the reflection-coefficient form of the
same network — a different derivation of the same physics, coded from
scratch in this file — plus the closed-form special cases (matched,
half-wave, lossless-unitary) where the answer is a number.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsim.common.circuit import (
    line_smatrix,
    sax_line_model,
    write_touchstone,
)

FREQ_HZ = np.asarray([1e9, 10e9, 40e9], dtype=np.float64)
LENGTH_M = 3e-3
Z_REF = 50.0


def lossy_line() -> tuple[np.ndarray, np.ndarray]:
    """A mismatched lossy line's gamma(f) and Z0(f) over FREQ_HZ."""
    n_rf = np.asarray([3.4, 3.2, 3.1])
    alpha = np.asarray([20.0, 80.0, 220.0])  # Np/m
    gamma = alpha + 1j * 2.0 * np.pi * FREQ_HZ * n_rf / 299792458.0
    z0 = np.asarray([42.0 + 4.0j, 44.0 + 2.0j, 46.0 + 1.0j])
    return gamma, z0


def reference_smatrix(gamma, z0, length_m, z_ref):
    """The same two-port from the reflection-coefficient derivation."""
    reflection = (z0 - z_ref) / (z0 + z_ref)
    phase = np.exp(-gamma * length_m)
    denom = 1.0 - reflection**2 * phase**2
    s11 = reflection * (1.0 - phase**2) / denom
    s21 = (1.0 - reflection**2) * phase / denom
    return s11, s21


class TestLineSmatrix:
    def test_matched_line_is_reflectionless_and_delays(self):
        gamma, _ = lossy_line()

        s = line_smatrix(gamma, Z_REF, length_m=LENGTH_M, z_ref_ohm=Z_REF)

        np.testing.assert_allclose(s[:, 0, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(s[:, 1, 0], np.exp(-gamma * LENGTH_M), rtol=1e-12)

    def test_matches_the_reflection_coefficient_derivation(self):
        gamma, z0 = lossy_line()

        s = line_smatrix(gamma, z0, length_m=LENGTH_M, z_ref_ohm=Z_REF)
        s11, s21 = reference_smatrix(gamma, z0, LENGTH_M, Z_REF)

        np.testing.assert_allclose(s[:, 0, 0], s11, rtol=1e-10)
        np.testing.assert_allclose(s[:, 1, 0], s21, rtol=1e-10)

    def test_the_network_is_reciprocal_and_symmetric(self):
        gamma, z0 = lossy_line()

        s = line_smatrix(gamma, z0, length_m=LENGTH_M, z_ref_ohm=Z_REF)

        np.testing.assert_allclose(s[:, 0, 1], s[:, 1, 0], rtol=1e-12)
        np.testing.assert_allclose(s[:, 0, 0], s[:, 1, 1], rtol=1e-12)

    def test_a_lossless_line_is_unitary(self):
        beta = 2.0 * np.pi * FREQ_HZ * 3.2 / 299792458.0
        gamma = 1j * beta

        s = line_smatrix(gamma, 42.0, length_m=LENGTH_M, z_ref_ohm=Z_REF)

        power = np.abs(s[:, 0, 0]) ** 2 + np.abs(s[:, 1, 0]) ** 2
        np.testing.assert_allclose(power, 1.0, rtol=1e-12)

    def test_a_lossless_half_wave_line_disappears(self):
        # beta L = pi: any lossless line is reflectionless and inverts.
        freq = 10e9
        n_rf = 299792458.0 / (2.0 * freq * LENGTH_M)
        gamma = 1j * 2.0 * np.pi * freq * n_rf / 299792458.0

        s = line_smatrix(gamma, 137.0, length_m=LENGTH_M, z_ref_ohm=Z_REF)

        np.testing.assert_allclose(s[0, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(s[1, 0], -1.0, rtol=1e-10)

    def test_a_nonpositive_length_is_refused(self):
        gamma, z0 = lossy_line()

        with pytest.raises(ValueError, match="length"):
            line_smatrix(gamma, z0, length_m=0.0, z_ref_ohm=Z_REF)

    def test_mismatched_shapes_are_refused(self):
        gamma, _ = lossy_line()

        with pytest.raises(ValueError, match="shape"):
            line_smatrix(gamma, np.asarray([50.0, 50.0]), length_m=LENGTH_M)


class TestTouchstone:
    def test_scikit_rf_reads_back_the_same_network(self, tmp_path):
        skrf = pytest.importorskip("skrf")
        gamma, z0 = lossy_line()
        s = line_smatrix(gamma, z0, length_m=LENGTH_M, z_ref_ohm=Z_REF)

        path = write_touchstone(
            tmp_path / "line.s2p", freq_hz=FREQ_HZ, s=s, z_ref_ohm=Z_REF
        )
        network = skrf.Network(str(path))

        np.testing.assert_allclose(network.f, FREQ_HZ)
        np.testing.assert_allclose(network.s, s, atol=1e-9)
        np.testing.assert_allclose(network.z0, Z_REF)

    def test_the_suffix_is_supplied_when_missing(self, tmp_path):
        gamma, z0 = lossy_line()
        s = line_smatrix(gamma, z0, length_m=LENGTH_M)

        path = write_touchstone(tmp_path / "line", freq_hz=FREQ_HZ, s=s)

        assert path.suffix == ".s2p"
        assert path.exists()

    def test_a_complex_reference_is_refused(self, tmp_path):
        gamma, z0 = lossy_line()
        s = line_smatrix(gamma, z0, length_m=LENGTH_M)

        with pytest.raises(ValueError, match="real"):
            write_touchstone(
                tmp_path / "line.s2p", freq_hz=FREQ_HZ, s=s, z_ref_ohm=50.0 + 1j
            )

    def test_a_descending_frequency_axis_is_refused(self, tmp_path):
        gamma, z0 = lossy_line()
        s = line_smatrix(gamma, z0, length_m=LENGTH_M)

        with pytest.raises(ValueError, match="ascending"):
            write_touchstone(tmp_path / "line.s2p", freq_hz=FREQ_HZ[::-1], s=s)

    def test_comment_lines_carry_the_provenance(self, tmp_path):
        gamma, z0 = lossy_line()
        s = line_smatrix(gamma, z0, length_m=LENGTH_M)

        path = write_touchstone(
            tmp_path / "line.s2p",
            freq_hz=FREQ_HZ,
            s=s,
            comments=["length_m = 0.003"],
        )

        from gsim.common.circuit import read_touchstone

        assert "length_m = 0.003" in read_touchstone(path).comments


class TestSaxLineModel:
    def test_the_default_frequencies_reproduce_the_solved_matrix(self):
        gamma, z0 = lossy_line()
        s = line_smatrix(gamma, z0, length_m=LENGTH_M, z_ref_ohm=Z_REF)

        model = sax_line_model(FREQ_HZ, gamma, z0, length_m=LENGTH_M, z_ref_ohm=Z_REF)
        sdict = model()

        np.testing.assert_allclose(sdict[("o1", "o1")], s[:, 0, 0], rtol=1e-12)
        np.testing.assert_allclose(sdict[("o2", "o1")], s[:, 1, 0], rtol=1e-12)
        np.testing.assert_allclose(sdict[("o1", "o2")], s[:, 0, 1], rtol=1e-12)
        np.testing.assert_allclose(sdict[("o2", "o2")], s[:, 1, 1], rtol=1e-12)

    def test_between_solved_points_the_line_parameters_interpolate(self):
        """The model resamples the way the line record does: the RF index,
        the loss and the complex impedance each linearly in frequency."""
        from gsim.common.transmission_line import line_params_from_gamma

        gamma, z0 = lossy_line()
        f_mid = np.asarray([15e9, 30e9])
        model = sax_line_model(FREQ_HZ, gamma, z0, length_m=LENGTH_M, z_ref_ohm=Z_REF)
        sdict = model(f=f_mid)

        at = line_params_from_gamma(FREQ_HZ, gamma, z0_ohm=z0).resampled(f_mid)
        expected = line_smatrix(
            at.gamma_per_m, at.z0_ohm, length_m=LENGTH_M, z_ref_ohm=Z_REF
        )
        np.testing.assert_allclose(sdict[("o2", "o1")], expected[:, 1, 0])
        np.testing.assert_allclose(sdict[("o1", "o1")], expected[:, 0, 0])

    def test_a_scalar_frequency_gives_scalar_entries(self):
        gamma, z0 = lossy_line()

        model = sax_line_model(FREQ_HZ, gamma, z0, length_m=LENGTH_M)
        sdict = model(f=10e9)

        assert np.shape(sdict[("o2", "o1")]) == ()

    def test_a_descending_solved_axis_is_refused(self):
        # np.interp on a descending axis returns silently wrong values,
        # so the model refuses to be built on one.
        gamma, z0 = lossy_line()

        with pytest.raises(ValueError, match="ascending"):
            sax_line_model(FREQ_HZ[::-1], gamma, z0, length_m=LENGTH_M)

    def test_the_callable_carries_no_solver_state(self):
        # The SAX convention is a plain function over numpy arrays: the
        # closure holds copies, so mutating the inputs cannot move it.
        gamma, z0 = lossy_line()
        model = sax_line_model(FREQ_HZ, gamma, z0, length_m=LENGTH_M)
        before = model()[("o2", "o1")].copy()

        gamma += 1e3
        z0 += 10.0

        np.testing.assert_allclose(model()[("o2", "o1")], before, rtol=1e-12)


class TestJunctionModelFile:
    """The junction compact model round-trips through its JSON file."""

    BIAS = np.asarray([0.0, 0.5, 1.0, 2.0])
    # Deliberately awkward floats: exact round-trip is the contract.
    R_S = np.asarray([1.2345678901234e-4, 1.1e-4, 0.9e-4, 1.0 / 3.0 * 1e-4])
    C_J = np.asarray([3.3e-10, 2.9e-10, 2.6e-10, 2.2e-10])

    def write(self, path, **overrides):
        from gsim.common.circuit import write_junction_model

        kwargs = dict(
            bias_v=self.BIAS,
            r_s_ohm_m=self.R_S,
            c_j_f_per_m=self.C_J,
            contact="cathode",
            freq_hz=1e9,
            provenance={"generator": "gsim test", "temperature_k": 300.0},
        )
        kwargs.update(overrides)
        return write_junction_model(path, **kwargs)

    def test_the_file_round_trips_exactly(self, tmp_path):
        from gsim.common.circuit import read_junction_model

        path = self.write(tmp_path / "junction.json")
        model = read_junction_model(path)

        assert model.bias_v.tolist() == self.BIAS.tolist()
        assert model.r_s_ohm_m.tolist() == self.R_S.tolist()
        assert model.c_j_f_per_m.tolist() == self.C_J.tolist()
        assert model.contact == "cathode"
        assert model.freq_hz == 1e9
        assert model.provenance["temperature_k"] == 300.0

    def test_the_file_is_plain_json_with_units(self, tmp_path):
        import json

        payload = json.loads(self.write(tmp_path / "junction.json").read_text())

        assert payload["format"] == "gsim-junction-model"
        assert payload["version"] == 1
        assert payload["units"]["c_j_f_per_m"] == "F/m"
        assert payload["units"]["r_s_ohm_m"] == "ohm*m"
        assert payload["contact"] == "cathode"

    def test_the_json_suffix_is_added(self, tmp_path):
        assert self.write(tmp_path / "junction").suffix == ".json"

    def test_a_dotted_stem_keeps_its_name(self, tmp_path):
        assert self.write(tmp_path / "sweep.2026-09").name == "sweep.2026-09.json"

    def test_mismatched_columns_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="per bias point"):
            self.write(tmp_path / "junction.json", r_s_ohm_m=self.R_S[:-1])

    def test_an_empty_sweep_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="non-empty"):
            self.write(
                tmp_path / "junction.json",
                bias_v=[],
                r_s_ohm_m=[],
                c_j_f_per_m=[],
            )

    def test_a_nonpositive_fit_frequency_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="freq_hz"):
            self.write(tmp_path / "junction.json", freq_hz=0.0)

    def test_a_foreign_file_is_refused_by_name(self, tmp_path):
        from gsim.common.circuit import read_junction_model

        path = tmp_path / "other.json"
        path.write_text('{"format": "something-else"}')

        with pytest.raises(ValueError, match="gsim-junction-model"):
            read_junction_model(path)

    def test_a_newer_schema_version_is_refused(self, tmp_path):
        import json

        from gsim.common.circuit import read_junction_model

        path = self.write(tmp_path / "junction.json")
        payload = json.loads(path.read_text())
        payload["version"] = 99
        path.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="version"):
            read_junction_model(path)


class TestTouchstoneReader:
    def test_what_gsim_writes_reads_back(self, tmp_path):
        from gsim.common.circuit import read_touchstone

        gamma, z0 = lossy_line()
        s = line_smatrix(gamma, z0, length_m=LENGTH_M, z_ref_ohm=Z_REF)
        path = write_touchstone(
            tmp_path / "line.s2p",
            freq_hz=FREQ_HZ,
            s=s,
            z_ref_ohm=Z_REF,
            comments=["length_m = 0.003"],
        )

        two_port = read_touchstone(path)

        np.testing.assert_allclose(two_port.freq_hz, FREQ_HZ, rtol=1e-12)
        np.testing.assert_allclose(two_port.s, s, rtol=1e-10, atol=1e-15)
        assert two_port.z_ref_ohm == Z_REF
        assert "length_m = 0.003" in two_port.comments

    def test_other_touchstone_flavors_read_as_the_same_network(self, tmp_path):
        from gsim.common.circuit import read_touchstone

        path = tmp_path / "ghz.s2p"
        path.write_text("# GHz S MA R 75\n1.0 0.5 0 1 -90 1 -90 0.5 0\n")

        two_port = read_touchstone(path)

        np.testing.assert_allclose(two_port.freq_hz, [1e9])
        np.testing.assert_allclose(two_port.s[0], [[0.5, -1j], [-1j, 0.5]], atol=1e-12)
        assert two_port.z_ref_ohm == 75.0

    def test_a_one_port_is_refused(self, tmp_path):
        from gsim.common.circuit import read_touchstone

        path = tmp_path / "load.s1p"
        path.write_text("# Hz S RI R 50\n1e9 0.1 0\n")

        with pytest.raises(ValueError, match="two-port"):
            read_touchstone(path)

    def test_a_malformed_row_is_refused(self, tmp_path):
        from gsim.common.circuit import read_touchstone

        path = tmp_path / "short.s2p"
        path.write_text("# Hz S RI R 50\n1e9 1 0 0\n")

        with pytest.raises(ValueError, match="not a readable Touchstone"):
            read_touchstone(path)


class TestDrivenResponse:
    """V_load/V_gen from the ABCD chain, checked against closed forms."""

    def test_matched_everything_halves_and_delays(self):
        from gsim.common.circuit import line_driven_response

        gamma, _ = lossy_line()
        h = line_driven_response(
            gamma, Z_REF, length_m=LENGTH_M, z_gen_ohm=Z_REF, z_load_ohm=Z_REF
        )

        # Generator divider gives 1/2; the line only delays and attenuates.
        np.testing.assert_allclose(h, 0.5 * np.exp(-gamma * LENGTH_M), rtol=1e-12)

    def test_the_quarter_wave_closed_form(self):
        from gsim.common.circuit import line_driven_response

        # A lossless quarter-wave line: A = D = 0, B = j Z0, C = j / Z0,
        # so H = Z_L Z0 / (j (Z0^2 + Z_g Z_L)).
        z0, z_gen, z_load = 60.0, 50.0, 75.0
        freq = 10e9
        beta = 2.0 * np.pi * freq / 299792458.0
        length = (np.pi / 2.0) / beta

        h = line_driven_response(
            np.asarray([1j * beta]),
            z0,
            length_m=length,
            z_gen_ohm=z_gen,
            z_load_ohm=z_load,
        )

        expected = z_load * z0 / (1j * (z0**2 + z_gen * z_load))
        np.testing.assert_allclose(h[0], expected, rtol=1e-12)

    def test_the_smatrix_route_agrees_with_the_telegrapher_route(self):
        from gsim.common.circuit import line_driven_response, terminated_response

        gamma, z0 = lossy_line()
        z_gen, z_load = 40.0 + 5.0j, 65.0 - 3.0j

        s = line_smatrix(gamma, z0, length_m=LENGTH_M, z_ref_ohm=Z_REF)
        via_s = terminated_response(
            s, z_ref_ohm=Z_REF, z_gen_ohm=z_gen, z_load_ohm=z_load
        )
        direct = line_driven_response(
            gamma, z0, length_m=LENGTH_M, z_gen_ohm=z_gen, z_load_ohm=z_load
        )

        np.testing.assert_allclose(via_s, direct, rtol=1e-10)

    def test_a_complex_reference_is_refused(self):
        from gsim.common.circuit import terminated_response

        s = np.zeros((3, 2, 2), dtype=np.complex128)
        s[:, 0, 1] = s[:, 1, 0] = 1.0

        with pytest.raises(ValueError, match="real"):
            terminated_response(s, z_ref_ohm=50 + 1j)  # type: ignore[arg-type]

    def test_a_wrongly_shaped_smatrix_is_refused(self):
        from gsim.common.circuit import terminated_response

        with pytest.raises(ValueError, match="2, 2"):
            terminated_response(np.zeros((3, 3)))

    def test_a_zero_transmission_two_port_is_refused(self):
        from gsim.common.circuit import terminated_response

        s = np.zeros((3, 2, 2), dtype=np.complex128)

        with pytest.raises(ValueError, match="S21"):
            terminated_response(s)

    def test_a_repeated_option_line_does_not_override_the_reference(self, tmp_path):
        from gsim.common.circuit import read_touchstone

        path = tmp_path / "echoed.s2p"
        path.write_text("# Hz S RI R 50\n1e9 0 0 1 0 1 0 0 0\n# Hz S RI R 75\n")

        assert read_touchstone(path).z_ref_ohm == 50.0

    def test_a_dotted_stem_keeps_its_name(self, tmp_path):
        gamma, z0 = lossy_line()
        s = line_smatrix(gamma, z0, length_m=LENGTH_M)

        path = write_touchstone(tmp_path / "line.v2", freq_hz=FREQ_HZ, s=s)

        assert path.name == "line.v2.s2p"

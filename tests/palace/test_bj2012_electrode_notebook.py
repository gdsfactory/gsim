"""Check the post-processing of the Baehr-Jones 2012 electrode notebook.

It runs on synthetic lines, with no cloud job.

Baehr-Jones et al., Opt. Express 20(11) 12014 (2012),
https://doi.org/10.1364/OE.20.012014. ``nbs/palace_bj2012_electrode.ipynb`` de-embeds a
200 um section from a 600 um one and fits one passive line to the 400 um remainder.
Here the notebook's own cells run, without any cloud job, on S-parameters of the
paper's metal-only line at 10 GHz (Z = 61 - 5j ohm, R = 4.9 kohm/m, L = 457 nH/m,
about 3.8 dB/cm) built from ABCD matrices with end fixtures and 1e-3 noise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import skrf as rf
from scipy.constants import speed_of_light

from gsim.common.transmission_line import section_abcd

NOTEBOOK_DIR = Path(__file__).resolve().parents[2] / "nbs"
FREQ_HZ = 10e9


@pytest.fixture(scope="module")
def functions():
    """Execute the notebook's analysis and synthetic-data cells (no cloud)."""
    notebook = json.loads(
        (NOTEBOOK_DIR / "palace_bj2012_electrode.ipynb").read_text(encoding="utf-8")
    )
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]
    reference = json.loads(
        (NOTEBOOK_DIR / "data" / "bj2012_reference.json").read_text(encoding="utf-8")
    )
    namespace = {
        "np": np,
        "rf": rf,
        "speed_of_light": speed_of_light,
        "section_abcd": section_abcd,
        "PAPER": reference["metal_only_line_10GHz"],
        "FREQ_HZ": FREQ_HZ,
        "LENGTHS_UM": (200, 600),
    }
    for marker in ("def extract_modal_parameters(", "def true_line("):
        source = next(s for s in sources if marker in s)
        exec(compile(source, marker, "exec"), namespace)  # noqa: S102
    return namespace


def _cell_sources():
    notebook = json.loads(
        (NOTEBOOK_DIR / "palace_bj2012_electrode.ipynb").read_text(encoding="utf-8")
    )
    return ["".join(cell["source"]) for cell in notebook["cells"]]


def _errors(found, truth):
    return {
        k: abs(found[k] - truth[k]) / abs(truth[k]) for k in truth if "G (" not in k
    }


class TestTruth:
    def test_the_truth_is_the_papers_line_at_10_ghz(self, functions):
        gamma, z0 = functions["true_gamma_z0"]([FREQ_HZ])
        assert z0[0] == pytest.approx(61 - 5j, rel=1e-12)
        n_rf = gamma[0].imag / (2 * np.pi * FREQ_HZ) * speed_of_light
        assert n_rf == pytest.approx(2.262, abs=0.002)
        assert gamma[0].real * 8.686 / 100 == pytest.approx(3.60, abs=0.01)
        rlgc = functions["true_line"]([FREQ_HZ])
        assert rlgc["R"][0] == pytest.approx(4900.0)
        assert rlgc["L"][0] == pytest.approx(457e-9)
        assert rlgc["C"][0] * 1e9 == pytest.approx(0.1238, abs=1e-3)

    def test_abcd_to_s_agrees_with_scikit_rf(self, functions):
        f = np.array([1e9, 10e9, 40e9])
        gamma, z0 = functions["true_gamma_z0"](f)
        abcd = section_abcd(gamma * 400e-6, z0)
        ours = functions["abcd_to_s"](abcd, 50.0)
        np.testing.assert_allclose(ours, rf.network.a2s(abcd, 50.0), atol=1e-12)

    def test_a_matched_lossless_line_only_delays(self, functions):
        theta = np.array([0.3j, 1.1j])
        s = functions["abcd_to_s"](section_abcd(theta, 50.0), 50.0)
        np.testing.assert_allclose(s[:, 0, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(s[:, 1, 0], np.exp(-theta), atol=1e-12)


class TestRecovery:
    @pytest.mark.parametrize(
        ("reference_ohm", "anchored"), [(50.0, False), (61.0, True)]
    )
    def test_noiseless_synthetic_line(self, functions, reference_ohm, anchored):
        """Without noise only the 2x-thru split's own bias (about 1 %) is left."""
        found, truth, _ = functions["recover"](
            reference_ohm, noise=0.0, anchored=anchored
        )
        errors = _errors(found, truth)
        assert errors["n_RF"] < 1e-3
        assert errors["alpha (dB/cm)"] < 5e-3
        for key in ("Z0 (ohm)", "L (nH/m)", "R (kohm/m)", "C (nF/m)"):
            assert errors[key] < 0.02, key

    @pytest.mark.parametrize(
        ("reference_ohm", "anchored"), [(50.0, False), (61.0, True)]
    )
    def test_noisy_synthetic_line(self, functions, reference_ohm, anchored):
        """1e-3 noise, four seeds: Z, n, L, C and the loss within 3 %, R within 10 %."""
        for seed in (1, 2, 3, 7):
            found, truth, out = functions["recover"](
                reference_ohm, noise=1e-3, seed=seed, anchored=anchored
            )
            errors = _errors(found, truth)
            assert out["fit"]["model"] == "sqrt_f"  # the model that generated the data
            for key in ("Z0 (ohm)", "n_RF", "alpha (dB/cm)", "L (nH/m)", "C (nF/m)"):
                assert errors[key] < 0.03, (seed, key, errors[key])
            assert errors["R (kohm/m)"] < 0.10, (seed, errors["R (kohm/m)"])

    def test_the_fit_does_not_return_negative_loss(self, functions):
        """The passive fit keeps every parameter non-negative."""
        _, _, out = functions["recover"](61.0, noise=1e-3, seed=3, anchored=True)
        assert all(value >= 0.0 for value in out["fit"]["params"].values())


class TestReferenceImpedance:
    """A wave port's S is normalised to its own mode; ``to_skrf`` only assigns an
    impedance."""

    def test_gamma_does_not_depend_on_the_assigned_reference(self, functions):
        """gamma and the index are the same for any reference given to the two equal
        ports.

        Only the extracted |Zc| scales with it, so the notebook takes the impedance
        from the
        port's own reference (the perfect-metal port mode), not from the assigned 50
        ohm and
        not from J1. A downloaded wall-mode run gave n = 6.436 at 5 GHz
        for every reference from 5 to 200 ohm: the wrong index was not a normalisation
        effect.
        """
        rng = np.random.default_rng(5)
        short, long_ = (
            functions["synthetic_network"](
                length, reference_ohm=61.0, noise=1e-3, rng=rng
            )
            for length in (200, 600)
        )
        out = {}
        for z0 in (20.0, 50.0, 200.0):
            networks = (
                rf.Network(frequency=net.frequency, s=net.s, z0=z0)
                for net in (short, long_)
            )
            out[z0] = functions["p370_extract"](*networks, 400e-6)
        for z0, result in out.items():
            np.testing.assert_allclose(result["gamma"], out[50.0]["gamma"], rtol=1e-6)
            np.testing.assert_allclose(
                np.abs(result["zc"]), np.abs(out[50.0]["zc"]) * z0 / 50.0, rtol=1e-6
            )


class TestPortReference:
    """The impedance from the de-embedded Zc' and the port's reference: Zc = Zc' Zp /
    50."""

    def test_noiseless_synthetic_line_with_the_port_reference(self, functions):
        """S in a 61 ohm reference, labelled 50: the right Zp recovers the line, no 2D
        anchors."""
        found, truth, _ = functions["recover"](61.0, noise=0.0, port_impedance_ohm=61.0)
        errors = _errors(found, truth)
        assert errors["n_RF"] < 1e-3
        assert errors["alpha (dB/cm)"] < 5e-3
        for key in ("Z0 (ohm)", "L (nH/m)", "R (kohm/m)", "C (nF/m)"):
            assert errors[key] < 0.02, key

    def test_noisy_synthetic_line_with_the_port_reference(self, functions):
        for seed in (1, 2, 3, 7):
            found, truth, out = functions["recover"](
                61.0, noise=1e-3, seed=seed, port_impedance_ohm=61.0
            )
            errors = _errors(found, truth)
            assert out["fit"]["model"] == "sqrt_f"
            for key in ("Z0 (ohm)", "n_RF", "alpha (dB/cm)", "L (nH/m)", "C (nF/m)"):
                assert errors[key] < 0.03, (seed, key, errors[key])

    def test_the_result_scales_with_the_port_impedance_and_gamma_does_not(
        self, functions
    ):
        """A wrong Zp moves Z and L up, C down by the same factor, and leaves n and the
        loss alone."""
        low, _, _ = functions["recover"](61.0, noise=0.0, port_impedance_ohm=50.0)
        high, _, _ = functions["recover"](61.0, noise=0.0, port_impedance_ohm=61.0)
        scale = 61.0 / 50.0
        assert abs(high["Z0 (ohm)"]) / abs(low["Z0 (ohm)"]) == pytest.approx(
            scale, rel=0.02
        )
        assert high["L (nH/m)"] / low["L (nH/m)"] == pytest.approx(scale, rel=0.02)
        assert low["C (nF/m)"] / high["C (nF/m)"] == pytest.approx(scale, rel=0.02)
        assert high["n_RF"] == pytest.approx(low["n_RF"], rel=1e-3)
        assert high["alpha (dB/cm)"] == pytest.approx(low["alpha (dB/cm)"], rel=5e-3)

    def test_the_real_analysis_uses_the_port_reference_not_j1(self):
        source = next(s for s in _cell_sources() if "palace = None" in s)
        assert 'port_impedance_ohm=perfect["z_pv"]' in source
        femwell = next(s for s in _cell_sources() if "def solve_line_mode(" in s)
        assert '"z_pv": float(z_defs["z_pv"])' in femwell
        assert "def impedance_definitions(" in femwell


class TestImpedanceAnchors:
    """Anchors below the fit band bias the fitted impedance; the notebook drops them."""

    F = np.arange(10, 81) * 0.5e9  # 5 to 40 GHz

    def _fit_z_at_10_ghz(self, functions, fa, za):
        gamma, _ = functions["true_gamma_z0"](self.F)
        best, _ = functions["fit_line_parameters"](self.F, [(400e-6, gamma)], fa, za)
        return functions["summarize"](functions["line_at"](best, 10e9))["Z0 (ohm)"]

    def test_anchors_in_band_drops_the_low_frequencies(self, functions):
        fa, za = functions["anchors_in_band"](
            np.array([1e9, 2.5e9, 5e9, 10e9]), np.array([41.0, 50.0, 53.0, 54.0]), 5.0
        )
        np.testing.assert_allclose(fa, [5e9, 10e9])
        np.testing.assert_allclose(za, [53.0, 54.0])
        with pytest.raises(ValueError, match="no impedance anchor"):
            functions["anchors_in_band"](np.array([1e9]), np.array([41.0]), 5.0)

    def test_a_low_frequency_dip_in_z_pulls_the_fit_down(self, functions):
        """A 1 GHz anchor 25 % under the line (substrate relaxation) moves Z at 10 GHz
        by 5 %."""
        fa = np.array([1e9, 5e9, 10e9, 20e9, 40e9])
        _, z_true = functions["true_gamma_z0"](fa)
        za = np.abs(z_true)
        za[0] *= 0.75
        raw = self._fit_z_at_10_ghz(functions, fa, za)
        kept = self._fit_z_at_10_ghz(
            functions, *functions["anchors_in_band"](fa, za, 5.0)
        )
        assert abs(raw) < 0.97 * abs(61 - 5j)
        assert abs(kept) == pytest.approx(abs(61 - 5j), rel=0.01)


class TestWavePortMode:
    def test_the_port_mode_number_is_the_line_modes_not_a_literal(self):
        """Mode 1 of this port face is the wall mode: the number comes from femwell."""
        sources = _cell_sources()
        settings = next(s for s in sources if "BM_FREQS_GHZ =" in s)
        assert "WAVE_MODE = PORT_LINE_MODE" in settings
        ranking = next(s for s in sources if "def solve_line_mode(" in s)
        assert (
            "key=lambda rm: -rm[0].n_eff.real" in ranking
        )  # decreasing index, as Palace

    def test_the_real_analysis_checks_the_port_log(self):
        source = next(s for s in _cell_sources() if "palace = None" in s)
        assert "validate_solution" in source
        assert "wave_number=complex(PORT_N_EFF[WAVE_MODE])" in source

    def test_the_fit_uses_only_anchors_in_the_fit_band(self):
        source = next(s for s in _cell_sources() if "palace = None" in s)
        assert "anchors_in_band(fa, za, FIT_MIN_GHZ)" in source
        assert "fit_min_ghz=FIT_MIN_GHZ" in source


class TestGeometryInputs:
    """The electrode inputs: no oxide over the strips, effective conductivity (see the
    notebook)."""

    def test_no_oxide_over_the_strips_and_the_effective_conductivity(self):
        source = next(s for s in _cell_sources() if "STRIP_WIDTH = REF" in s)
        assert "OXIDE_OVER_METAL = 0.0" in source
        assert "SIGMA_AL = 1.0e7" in source
        assert "SIGMA_AL_BULK = 3.77e7" in source

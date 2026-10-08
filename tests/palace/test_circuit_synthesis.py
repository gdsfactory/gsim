"""Tests for Palace AC circuit synthesis (AdaptiveCircuitSynthesis) support.

Covers the DrivenConfig flag wiring and the rom-*.csv parser / circuit
evaluation helpers in gsim.palace.circuit.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from gsim.palace import DrivenSim
from gsim.palace.circuit import (
    CircuitSynthesis,
    RLCFit,
    VectorFit,
    differential_impedance,
    fit_rlc,
    initial_guess_rlc,
    is_complete,
    load_circuit_synthesis,
    s_to_y,
    s_to_z,
    y_to_s,
    y_to_z,
    z_to_s,
    z_to_y,
)
from gsim.palace.models import DrivenConfig

# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------


def test_driven_config_emits_circuit_synthesis():
    """circuit_synthesis=True emits AdaptiveCircuitSynthesis in the JSON."""
    config = DrivenConfig(
        fmin=1e9, fmax=10e9, num_points=11, circuit_synthesis=True
    ).to_palace_config()
    assert config["AdaptiveCircuitSynthesis"] is True
    assert config["AdaptiveTol"] > 0


def test_driven_config_circuit_synthesis_requires_adaptive_sweep():
    """Circuit synthesis without an adaptive sweep raises (Palace rejects it)."""
    with pytest.raises(ValueError, match="adaptive"):
        DrivenConfig(adaptive_tol=0, circuit_synthesis=True).to_palace_config()


def test_driven_config_default_has_no_circuit_synthesis():
    """The flag is opt-in: default config has no AdaptiveCircuitSynthesis key."""
    config = DrivenConfig().to_palace_config()
    assert "AdaptiveCircuitSynthesis" not in config


def test_set_driven_circuit_synthesis():
    """DrivenSim.set_driven passes circuit_synthesis through to the config."""
    sim = DrivenSim()
    sim.set_driven(fmin=1e9, fmax=10e9, num_points=11, circuit_synthesis=True)
    assert sim.driven.circuit_synthesis is True
    palace_config = sim.driven.to_palace_config()
    assert palace_config["AdaptiveCircuitSynthesis"] is True


# ---------------------------------------------------------------------------
# Synthetic rom-*.csv parsing
# ---------------------------------------------------------------------------


def _write_matrix(path: Path, labels: list[str], values) -> None:
    """Write a Palace-style rom matrix CSV (header row of labels)."""
    values = np.asarray(values, dtype=float)
    lines = [",".join(labels)]
    lines.extend(",".join(f"{v:.17e}" for v in row) for row in values)
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def rom_dir(tmp_path: Path) -> Path:
    """Two-port + one-interior-node synthesized circuit output.

    Inductive network (proper graph Laplacian in L^-1): 1 nH between the two
    ports, 2 nH from port 1 to the interior node. Capacitors: 10 fF shunt at
    each port, 1 nF from the interior node to ground (dominates the interior
    shunt so the Schur complement in the tests is free of cancellation).
    Ports are terminated with 50 Ohm (R^-1 diagonal), exported as portload
    blocks in a separate test.
    """
    labels = ["port_1_re", "port_2_re", "sample_e1_s0_re"]
    g1 = 1.0 / 1e-9  # 1 nH port-to-port
    g2 = 1.0 / 2e-9  # 2 nH port1-to-interior
    L = np.array(
        [
            [g1 + g2, -g1, -g2],
            [-g1, g1, 0.0],
            [-g2, 0.0, g2],
        ]
    )
    R = np.diag([1.0 / 50.0, 1.0 / 50.0, 0.0])
    C = np.diag([10e-15, 10e-15, 1e-9])

    d = tmp_path / "output" / "palace"
    d.mkdir(parents=True)
    _write_matrix(d / "rom-Linv-re.csv", labels, L.real)
    _write_matrix(d / "rom-Rinv-re.csv", labels, R.real)
    _write_matrix(d / "rom-C-re.csv", labels, C.real)
    # Imaginary parts: only a loss-tangent-like contribution on C.
    _write_matrix(d / "rom-C-im.csv", labels, C.real * 1e-3)

    port_info = {
        "ports": [
            {"portnumber": 1, "name": "P1"},
            {"portnumber": 2, "name": "P2"},
        ],
        "unit": 1e-6,
        "name": "palace",
    }
    (tmp_path / "port_information.json").write_text(json.dumps(port_info))
    return tmp_path


def test_load_circuit_synthesis_from_dir(rom_dir: Path):
    """Directory loading parses labels, complex matrices and node kinds."""
    circuit = load_circuit_synthesis(rom_dir)
    assert circuit.nodes == ["port_1_re", "port_2_re", "sample_e1_s0_re"]
    assert circuit.port_labels == ["port_1_re", "port_2_re"]
    assert circuit.internal_indices == [2]
    assert circuit.L_inv.shape == (3, 3)
    assert circuit.R_inv[0, 0] == pytest.approx(1 / 50.0)
    # Complex assembly: re + 1j*im
    assert circuit.C[0, 0] == pytest.approx(10e-15 * (1 + 1e-3j))


def test_load_circuit_synthesis_port_map(rom_dir: Path):
    """port_map attaches gsim port names to palace port nodes."""
    circuit = load_circuit_synthesis(rom_dir, port_map={1: "P1", 2: "P2"})
    assert circuit.port_names == {
        "port_1_re": "P1",
        "port_2_re": "P2",
    }


def test_load_circuit_synthesis_missing_files(tmp_path: Path):
    """A directory without rom files raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="rom-"):
        load_circuit_synthesis(tmp_path)


def test_schur_complement_matches_analytic_ladder(rom_dir: Path):
    """Interior-node elimination reproduces the analytic network admittance.

    Synthetic network: 1 nH directly between the ports, 2 nH from port 1 to
    an interior node that carries a 1 nF shunt to ground, 10 fF shunts at
    each port. The fixture also writes an imaginary-C file (loss factor
    1e-3), so the closed-form expectation uses the same complex capacitors.
    """
    circuit = load_circuit_synthesis(rom_dir)
    f = np.array([1e9, 5e9, 20e9])
    w = 2 * np.pi * f
    Yp = circuit.port_admittance(f)

    loss = 1e-3  # fixture writes rom-C-im.csv = rom-C-re.csv * loss
    c_p = 10e-15 * (1 + 1j * loss)  # port shunt capacitors
    c_int = 1e-9 * (1 + 1j * loss)  # interior node capacitor

    y_1n = 1.0 / (1j * w * 1e-9)  # 1 nH port-to-port path
    y_2n = 1.0 / (1j * w * 2e-9)  # 2 nH port1-to-interior path
    y_int = y_2n + 1j * w * c_int  # interior node shunt

    expected = np.empty_like(Yp)
    expected[:, 0, 0] = 1 / 50.0 + 1j * w * c_p + y_1n + y_2n - y_2n * y_2n / y_int
    expected[:, 0, 1] = expected[:, 1, 0] = -y_1n
    expected[:, 1, 1] = 1 / 50.0 + 1j * w * c_p + y_1n

    np.testing.assert_allclose(Yp, expected, rtol=1e-10)


def test_port_load_subtraction_recovers_bare_device(rom_dir: Path):
    """Subtracting rom-portload blocks removes the 50 Ohm port terminations."""
    labels = ["port_1_re", "port_2_re", "sample_e1_s0_re"]
    d = rom_dir / "output" / "palace"
    load_r1 = np.zeros((3, 3))
    load_r1[0, 0] = 1.0 / 50.0
    load_r2 = np.zeros((3, 3))
    load_r2[1, 1] = 1.0 / 50.0
    _write_matrix(d / "rom-portload-port_1_re-Rinv-re.csv", labels, load_r1)
    _write_matrix(d / "rom-portload-port_2_re-Rinv-re.csv", labels, load_r2)

    circuit = load_circuit_synthesis(rom_dir)
    f = np.array([1e9])
    Y_loaded = circuit.port_admittance(f)
    Y_bare = circuit.port_admittance(f, subtract_port_loads=True)

    # The 50 Ohm terminations are removed from the terminal diagonal exactly;
    # the small residual real part comes from the fixture's lossy C.
    assert Y_loaded[0, 0, 0].real - Y_bare[0, 0, 0].real == pytest.approx(1 / 50.0)
    assert Y_bare[0, 0, 0].real == pytest.approx(0.0, abs=2e-6)
    assert Y_bare[0, 1, 0] == Y_loaded[0, 1, 0]  # off-diagonals untouched
    # Reference impedances come from the portload resistance.
    np.testing.assert_allclose(circuit.port_reference_impedances(), [50.0, 50.0])


def test_s_parameters_series_inductor_analytic():
    """S21 of a pure series inductor matches 2/(2 + Z/Z0)."""
    Ls = 1e-9
    g = 1.0 / Ls
    circuit = CircuitSynthesis(
        nodes=["port_1_re", "port_2_re"],
        L_inv=np.array([[g, -g], [-g, g]]),
        R_inv=np.zeros((2, 2)),
        C=np.zeros((2, 2)),
    )

    f = np.array([2e9, 10e9])
    S = circuit.s_parameters(f)
    omega = 2 * np.pi * f
    for k in range(len(f)):
        z_series = 1j * omega[k] * Ls
        assert S[k, 1, 0] == pytest.approx(2.0 / (2.0 + z_series / 50.0), rel=1e-10)
        # Series impedance between equal references: S11 = Z / (2 Z0 + Z).
        assert S[k, 0, 0] == pytest.approx(z_series / (2 * 50.0 + z_series), rel=1e-10)


def test_circuit_synthesis_class_validation():
    """Matrix shape mismatches are rejected."""
    with pytest.raises(ValueError, match="shape"):
        CircuitSynthesis(
            nodes=["port_1_re"],
            L_inv=np.zeros((2, 2)),
            R_inv=np.zeros((2, 2)),
            C=np.zeros((2, 2)),
        )


# ---------------------------------------------------------------------------
# One-pole and vector fitting (gsim.palace.circuit fit_rlc)
# ---------------------------------------------------------------------------

R_TRUE, L_TRUE, C_TRUE = 3.5, 110e-12, 8e-15


@pytest.fixture
def rlc_data():
    """Exact one-pole RLC impedance over a 10-200 GHz band."""
    f = np.linspace(10e9, 200e9, 200)
    f0 = 1.0 / (2.0 * np.pi * np.sqrt(L_TRUE * C_TRUE))
    model = RLCFit(
        R=R_TRUE,
        L=L_TRUE,
        C=C_TRUE,
        f0=f0,
        Q=2.0 * np.pi * f0 * L_TRUE / R_TRUE,
        rms_error=0.0,
    )
    return f, model.z(f), model


def test_differential_impedance_matches_manual_formula():
    rng = np.random.default_rng(42)
    z = rng.normal(size=(7, 3, 3)) + 1j * rng.normal(size=(7, 3, 3))
    expected = z[:, 0, 0] - z[:, 0, 1] - z[:, 1, 0] + z[:, 1, 1]
    np.testing.assert_allclose(differential_impedance(z), expected)


def test_differential_impedance_rejects_invalid_shape():
    with pytest.raises(ValueError, match="nf, N, N"):
        differential_impedance(np.zeros((5,)))


def test_differential_impedance_single_port_is_driving_point():
    """A one-port network is already differential: return Z11 directly."""
    rng = np.random.default_rng(7)
    z = rng.normal(size=(9, 1, 1)) + 1j * rng.normal(size=(9, 1, 1))
    np.testing.assert_allclose(differential_impedance(z), z[:, 0, 0])


def test_initial_guess_rlc_finds_peak_and_low_freq_resistance(rlc_data):
    f, z, _ = rlc_data
    f0, _, r = initial_guess_rlc(f, z)
    assert f0 == pytest.approx(f[int(np.argmax(np.abs(z)))])
    assert r == pytest.approx(R_TRUE, rel=0.5)


@pytest.mark.parametrize("solver", ["scipy", "jax"])
def test_fit_rlc_recovers_synthetic_rlc(rlc_data, solver):
    """Both solvers recover the true (R, L, C, f0, Q) from exact data."""
    if solver == "jax":
        pytest.importorskip("jax")
    f, z, model = rlc_data
    fit = fit_rlc(f, z, solver=solver)
    assert isinstance(fit, RLCFit)
    if solver == "jax":
        # Log-space optimization with a fixed iteration budget converges to
        # percent-level accuracy; the tolerances leave cross-platform
        # headroom (BLAS ordering differs on macOS/Windows CI).
        assert pytest.approx(R_TRUE, rel=5e-2) == fit.R
        assert pytest.approx(L_TRUE, rel=2e-2) == fit.L
        assert pytest.approx(C_TRUE, rel=2e-2) == fit.C
        assert fit.f0 == pytest.approx(model.f0, rel=5e-3)
        assert pytest.approx(model.Q, rel=5e-2) == fit.Q
        np.testing.assert_allclose(fit.z(f), z, rtol=5e-2)
    else:
        assert pytest.approx(R_TRUE, rel=2e-3) == fit.R
        assert pytest.approx(L_TRUE, rel=1e-3) == fit.L
        assert pytest.approx(C_TRUE, rel=1e-3) == fit.C
        assert fit.f0 == pytest.approx(model.f0, rel=2e-3)
        assert pytest.approx(model.Q, rel=3e-3) == fit.Q
        assert fit.rms_error == pytest.approx(0.0, abs=1e-6)
        np.testing.assert_allclose(fit.z(f), z, rtol=1e-4)


def test_fit_rlc_auto_solver_returns_valid_fit():
    """solver='auto' works regardless of whether the optional JAX deps exist."""
    f = np.linspace(10e9, 100e9, 50)
    model = RLCFit(
        R=R_TRUE,
        L=L_TRUE,
        C=C_TRUE,
        f0=1.0 / (2.0 * np.pi * np.sqrt(L_TRUE * C_TRUE)),
        Q=2.0 * np.pi * 1e9 * L_TRUE / R_TRUE,
        rms_error=0.0,
    )
    fit = fit_rlc(f, model.z(f), solver="auto")
    assert isinstance(fit, RLCFit)
    assert pytest.approx(L_TRUE, rel=1e-2) == fit.L


def test_rlc_fit_repr_and_dict(rlc_data):
    f, z, _ = rlc_data
    fit = fit_rlc(f, z, solver="scipy")
    assert isinstance(fit, RLCFit)
    assert "RLCFit" in repr(fit)
    d = fit.to_dict()
    assert set(d) == {"R", "L", "C", "f0", "Q", "rms_error"}


def test_circuit_fit_rlc_on_real_data():
    """End-to-end one-pole fit on the committed circuit-synthesis data for the
    notebook inductor (nbs/data/inductor/circuit_synthesis, Palace v0.18.0 run).

    Note: an exact *synthetic* two-node circuit cannot reproduce the one-pole
    model — in Palace's pencil Y(w) = L^-1/(iw) + R^-1 + iw*C the R^-1 and
    L^-1 matrices are parallel-admittance contributions, so a series R+L
    branch (whose admittance 1/(R + iwL) has a frequency-dependent mix) is not
    expressible as constant R^-1/L^-1 entries. Series behavior only emerges
    from the network, which is exactly what this real-data test exercises.
    """
    data_dir = (
        Path(__file__).resolve().parents[2] / "nbs/data/inductor/circuit_synthesis"
    )
    if not (data_dir / "rom-Linv-re.csv").exists():  # pragma: no cover - repo layout
        pytest.skip("circuit-synthesis data not present")

    circuit = load_circuit_synthesis(data_dir, port_map={1: "P1", 2: "P2"})
    # Evaluation grid: the sweep frequencies from port-S.csv.
    port_s = pd.read_csv(data_dir / "port-S.csv")
    f_hz = port_s.iloc[:, 0].to_numpy() * 1e9

    fit = circuit.fit_rlc(f_hz, model="rlc1p", solver="scipy")
    assert isinstance(fit, RLCFit)
    # Physically meaningful one-pole parameters within the validated ranges
    # (fit values are solver-dependent to ~5 %; the resonance is not).
    assert 2.0 < fit.R < 6.0
    assert 80e-12 < fit.L < 160e-12
    assert 4e-15 < fit.C < 12e-15
    assert 160e9 < fit.f0 < 180e9
    assert 20 < fit.Q < 60
    assert fit.rms_error < 150.0


# ---------------------------------------------------------------------------
# scikit-rf vector fitting (model="vector_fit")
# ---------------------------------------------------------------------------


def test_fit_rlc_vector_fit_on_real_circuit_data():
    """Multi-pole vector fit of the exported circuit's S-parameters.

    Skips when skrf is unavailable. Uses a modest pole count so the fit is
    fast and deterministic; the exported circuit reproduces the FEM response
    to ~5e-4 in S, so the rational model must be far below the 0.01 target.
    """
    pytest.importorskip("skrf")
    data_dir = (
        Path(__file__).resolve().parents[2] / "nbs/data/inductor/circuit_synthesis"
    )
    if not (data_dir / "rom-Linv-re.csv").exists():  # pragma: no cover
        pytest.skip("circuit-synthesis data not present")

    circuit = load_circuit_synthesis(data_dir, port_map={1: "P1", 2: "P2"})
    f_hz = pd.read_csv(data_dir / "port-S.csv").iloc[:, 0].to_numpy() * 1e9

    fit = circuit.fit_rlc(f_hz, model="vector_fit", n_poles_real=4)
    assert isinstance(fit, VectorFit)
    assert "VectorFit" in repr(fit)
    # skrf may prune non-contributing poles during fitting.
    assert 0 < fit.n_poles <= 2 * 4

    # Stability and passivity of the fitted rational model (passive lossy
    # device referenced to 50 Ohm).
    assert fit.is_stable
    assert fit.is_passive()

    # Rational response quality of the MODEL (fit.s(f_hz)); fit.s() with no
    # argument evaluates the model at the training frequencies - never the
    # raw training data (that lives on .network.s).
    s_model = fit.s(f_hz)
    s_target = circuit.s_parameters(f_hz)
    max_rel = np.max(np.abs(s_model - s_target) / (np.abs(s_target) + 1e-6))
    assert max_rel < 2e-3, f"vector fit S error {max_rel:.2e}"

    # The no-argument call must equal the model, not the data.
    np.testing.assert_allclose(fit.s(), s_model, rtol=1e-9)

    # Impedance evaluation round-trip (model Z vs the circuit's Z).
    z_model = fit.z(f_hz)
    z_target = differential_impedance(circuit.port_impedance(f_hz))
    rel = np.max(np.abs(differential_impedance(z_model) - z_target) / np.abs(z_target))
    assert rel < 5e-3, f"vector fit Z error {rel:.2e}"


def test_fit_rlc_vector_fit_detects_nonpassive_model():
    """A deliberately nonpassive response is detected.

    The issue's validation asks to detect deliberately nonpassive models.
    A negative-resistance inductor (Z = -R0 + jwL) yields |S11| > 1, so the
    fitted rational model must fail the passivity test before enforcement.

    Enforcement is attempted (skrf ``passivity_enforce``) but its OUTCOME is
    environment-dependent (it may fully restore passivity, e.g. on
    Windows/skrf 2.1.0, or refuse when the model's DC point is itself
    non-passive) - so only the detection facts are asserted here; the
    notebook-level quality checks consider the enforcement outcome directly.
    """
    pytest.importorskip("skrf")
    f = np.linspace(10e9, 200e9, 120)
    w = 2 * np.pi * f
    z_active = -0.5 + 1j * w * 100e-12  # active: Re(Z) < 0 -> |S11| > 1

    fit = fit_rlc(f, z_active, model="vector_fit", n_poles_real=2)
    assert isinstance(fit, VectorFit)
    assert not fit.is_passive(), "active impedance must fail the passivity test"
    assert fit.passivity_test().size > 0, "passivity violations must be reported"

    with warnings.catch_warnings():
        # skrf emits version/outcome-dependent warnings during enforcement.
        warnings.simplefilter("ignore")
        fit.passivity_enforce()


def test_fit_rlc_vector_fit_requires_skrf(monkeypatch):
    """Missing skrf yields an informative ImportError."""

    import builtins

    real_import = builtins.__import__

    def _no_skrf(name, *args, **kwargs):
        if name == "skrf" or name.startswith("skrf."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_skrf)
    f = np.linspace(10e9, 100e9, 20)
    with pytest.raises(ImportError, match="scikit-rf is required"):
        fit_rlc(f, np.ones(len(f)) * 50.0, model="vector_fit")


def test_fit_rlc_dispatch_errors():
    """Ambiguous or unknown model inputs raise clear errors."""
    f = np.linspace(10e9, 100e9, 10)
    z = np.ones(len(f)) * 50.0
    s = np.zeros((len(f), 2, 2))
    with pytest.raises(ValueError, match=r"either z or s, not both"):
        fit_rlc(f, z, s=s, model="vector_fit")
    with pytest.raises(ValueError, match=r"rlc1p.*impedance"):
        fit_rlc(f, s=s, model="rlc1p")
    with pytest.raises(ValueError, match=r"unknown fit model"):
        fit_rlc(f, z, model="nonexistent")
    with pytest.raises(ValueError, match=r"provide impedance data"):
        fit_rlc(f, model="rlc1p")


# ---------------------------------------------------------------------------
# Batched S <-> Z <-> Y conversions (gsim.palace.circuit)
# ---------------------------------------------------------------------------


@pytest.fixture
def causal_two_port() -> NDArray:
    """A passive 2-port: series branch + shunt reference cell.

    A small ground-referenced shunt makes the admittance invertible (a pure
    differential Laplacian is singular - no common-mode return path).
    """
    f = np.linspace(10e9, 200e9, 64)
    w = 2 * np.pi * f
    L, C, R = 110e-12, 8e-15, 3.5
    y_tot = 1.0 / (R + 1j * w * L) + 1j * w * C
    y = np.empty((len(w), 2, 2), dtype=complex)
    y[:, 0, 0] = y[:, 1, 1] = y_tot + 1j * w * 1e-15
    y[:, 0, 1] = y[:, 1, 0] = -y_tot
    return y


def test_round_trip_s_z_y_s(causal_two_port):
    """S -> Z -> S and S -> Y -> S round-trips to machine precision."""
    s = y_to_s(causal_two_port, z0=50.0)
    np.testing.assert_allclose(z_to_s(s_to_z(s, z0=50.0), z0=50.0), s, rtol=1e-9)
    np.testing.assert_allclose(y_to_s(s_to_y(s, z0=50.0), z0=50.0), s, rtol=1e-9)
    np.testing.assert_allclose(
        z_to_y(y_to_z(causal_two_port)), causal_two_port, rtol=1e-9
    )


def test_series_inductor_analytic():
    """1-port series-L behavior: Z = Z0 (1 + S)/(1 - S)."""
    f = np.array([2e9, 10e9])
    L = 1e-9
    z_true = 1j * 2 * np.pi * f * L
    s = ((z_true - 50.0) / (z_true + 50.0))[:, None, None]
    z = s_to_z(s, z0=50.0)[:, 0, 0]
    np.testing.assert_allclose(z, z_true, rtol=1e-12)
    # and the reverse
    s_back = z_to_s(z[:, None, None], z0=50.0)[:, 0, 0]
    np.testing.assert_allclose(s_back, s[:, 0, 0], rtol=1e-12)


def test_four_port_round_trip_with_per_port_z0():
    """4-port random passive-ish matrices survive round trips with per-port z0."""
    rng = np.random.default_rng(3)
    nf, n = 12, 4
    a = rng.normal(size=(nf, n, n)) + 1j * rng.normal(size=(nf, n, n))
    y = a + np.eye(n)[None] * (5.0 + 5j)
    z0 = np.array([50.0, 50.0, 25.0, 100.0])
    s = y_to_s(y, z0=z0)
    np.testing.assert_allclose(y_to_s(s_to_y(s, z0=z0), z0=z0), s, rtol=1e-8)
    np.testing.assert_allclose(z_to_s(s_to_z(s, z0=z0), z0=z0), s, rtol=1e-8)
    np.testing.assert_allclose(s_to_z(s, z0=z0), y_to_z(y), rtol=1e-8)


def test_conversions_match_scikit_rf_per_port_z0():
    """Anchor the per-port z0 formulas against scikit-rf (issue review).

    The wrong one-sided Z0 multiplication is self-consistent under round
    trips, so only an external anchor catches it.
    """
    pytest.importorskip("skrf")
    import skrf as rf

    n, nf = 3, 16
    rng = np.random.default_rng(0)
    a = rng.normal(size=(nf, n, n)) + 1j * rng.normal(size=(nf, n, n))
    S = 0.3 * (a + np.transpose(a, (0, 2, 1))) / 2
    z0 = np.array([50.0, 75.0, 100.0])
    f = np.linspace(1e9, 100e9, nf)
    net = rf.Network(
        frequency=rf.Frequency.from_f(f, unit="Hz"), s=S, z0=np.tile(z0, (nf, 1))
    )

    z = s_to_z(S, z0=z0)
    np.testing.assert_allclose(z, net.z, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(z_to_s(z, z0=z0), net.s, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(s_to_y(S, z0=z0), net.y, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(
        y_to_s(s_to_y(S, z0=z0), z0=z0), S, rtol=1e-12, atol=1e-12
    )


def test_incomplete_matrix_rejected():
    """The issue's 'reject incomplete S matrices' criterion."""
    s_ok = np.zeros((3, 2, 2), dtype=complex)
    with pytest.raises(ValueError, match="complete \\(nf, N, N\\)"):
        s_to_z(s_ok[:, :, :1], z0=50.0)
    with pytest.raises(ValueError, match="complete \\(nf, N, N\\)"):
        s_to_z(s_ok[0], z0=50.0)
    assert is_complete(s_ok)
    assert not is_complete(s_ok[:, :, :1])
    assert not is_complete(s_ok * np.nan)


def test_negative_resistance_units_are_preserved():
    """z0 scaling, frequency units and port order are preserved (issue #272)."""
    f = np.linspace(10e9, 200e9, 24)
    z_load = 3.0 + 1j * 2 * np.pi * f * 100e-12
    s = ((z_load - 25.0) / (z_load + 25.0))[:, None, None]
    z = s_to_z(s, z0=25.0)[:, 0, 0]
    np.testing.assert_allclose(z, z_load, rtol=1e-12)

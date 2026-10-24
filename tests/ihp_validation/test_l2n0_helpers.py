"""Tests for the IHP L2n0 benchmark helpers (no Palace execution)."""

from __future__ import annotations

import numpy as np
import pytest

from gsim.common.benchmarks import ihp_l2n0


def test_benchmark_source_files_are_pinned() -> None:
    assert ihp_l2n0.LAYOUT.is_file()
    assert set(ihp_l2n0.MEASURED) == {"die_x2y7", "die_x5y8"}
    for path in ihp_l2n0.MEASURED.values():
        assert path.is_file()


@pytest.mark.parametrize("label", ["die_x2y7", "die_x5y8"])
def test_measured_data_reaches_30_ghz(label: str) -> None:
    import skrf as rf

    network = rf.Network(str(ihp_l2n0.MEASURED[label]))
    assert network.f[0] <= 30e9 <= network.f[-1]
    point = network.interpolate(rf.Frequency(30, 30, 1, unit="GHz"))
    assert np.all(np.isfinite(point.s))  # ty: ignore[unresolved-attribute]


def test_component_and_stack_have_the_benchmark_ports_and_layers() -> None:
    pytest.importorskip("ihp")
    component = ihp_l2n0.load_benchmark_component()
    assert [port.name for port in component.ports] == ["P1", "P2"]
    assert (
        component.ports["P1"].layer_info.layer,
        component.ports["P1"].layer_info.datatype,
    ) == (126, 0)
    stack = ihp_l2n0.build_benchmark_stack(component)
    assert "subgnd" in stack.layers
    assert stack.layers["subgnd"].layer_type == "conductor"
    assert {"substrate", "epi", "ild"} <= {
        dielectric["name"] for dielectric in stack.dielectrics
    }


def test_lq_extraction_is_finite_for_recorded_run() -> None:
    run = ihp_l2n0.load_recorded_run(ihp_l2n0.RUN_ROOT / "l2n0_30ghz")
    inductance, q, impedance = ihp_l2n0.lq_at(run.sparams)
    assert math_is_finite(inductance)
    assert math_is_finite(q)
    assert math_is_finite(impedance.real)
    assert math_is_finite(impedance.imag)


def math_is_finite(value: float) -> bool:
    return bool(np.isfinite(value))


@pytest.mark.parametrize("label", ["die_x2y7", "die_x5y8"])
def test_measured_low_frequency_inductance_is_nominal_2nh(label: str) -> None:
    import skrf as rf

    network = rf.Network(str(ihp_l2n0.MEASURED[label]))
    point = network.interpolate(rf.Frequency(1, 1, 1, unit="GHz"))
    s = point.s[0]  # ty: ignore[unresolved-attribute]
    z = 2.0 * ihp_l2n0.Z0_OHM * (1.0 - s[1, 0]) / s[1, 0]
    inductance_nh = z.imag / (2 * np.pi * 1e9) * 1e9
    assert 1.8 < inductance_nh < 2.1


def test_recorded_sweep_extracts_finite_lq_at_1_and_30_ghz() -> None:
    run = ihp_l2n0.load_recorded_run(ihp_l2n0.RUN_ROOT / "l2n0_sweep")
    freqs = np.asarray(run.sparams.freq, dtype=float)
    assert freqs.size == 11
    for target in (1.0, 30.0):
        index = int(np.argmin(np.abs(freqs - target)))
        inductance, q, impedance = ihp_l2n0.lq_at(run.sparams, index)
        assert np.isfinite(inductance)
        assert np.isfinite(q)
        assert np.isfinite(impedance.real)
        assert np.isfinite(impedance.imag)


def test_measured_lq_is_finite_at_1_and_30_ghz() -> None:
    import skrf as rf

    for label in ("die_x2y7", "die_x5y8"):
        network = rf.Network(str(ihp_l2n0.MEASURED[label]))
        z0 = float(np.real(np.asarray(network.z0).reshape(-1)[0]))
        for target in (1.0, 30.0):
            point = network.interpolate(rf.Frequency(target, target, 1, unit="GHz"))
            s_point = point.s[0]  # ty: ignore[unresolved-attribute]
            l, q, z = ihp_l2n0.lq_from_s_matrix(s_point, target * 1e9, z0=z0)
            assert np.isfinite(l)
            assert np.isfinite(q)
            assert np.isfinite(z.real)
            assert np.isfinite(z.imag)


def test_best_variant_low_frequency_is_within_10_percent() -> None:
    run = ihp_l2n0.load_recorded_run(ihp_l2n0.RUN_ROOT / "l2n0_best")
    l_palace, _, _ = ihp_l2n0.lq_at(run.sparams, 0)
    measured = [1.9115, 1.9319]
    assert l_palace == pytest.approx(1.921, abs=0.01)
    assert all(abs(l_palace - m) <= 0.10 * m for m in measured)


def test_experiment_matrix_is_recorded() -> None:
    import json

    path = (
        ihp_l2n0.REPO
        / "nbs"
        / "data"
        / "ihp_validation"
        / "experiments"
        / "results.json"
    )
    matrix = json.loads(path.read_text())
    assert {
        "baseline",
        "shell_conformal",
        "ground0",
        "pec",
        "ground0_pec",
        "mesh1.5",
        "planar_pec",
        "substrate180",
        "openems_ground",
        "gap",
    } <= set(matrix)
    assert matrix["ground0_pec"]["ok"] is True


def test_committed_config_uses_conductor_thickness_bc() -> None:
    import json

    config = json.loads((ihp_l2n0.RUN_ROOT / "l2n0_best" / "config.json").read_text())
    conductivity = config["Boundaries"]["Conductivity"]
    assert conductivity
    assert all(
        "Thickness" in entry and "Conductivity" in entry for entry in conductivity
    )


def test_shell_variant_is_explicitly_modelled() -> None:
    pytest.importorskip("ihp")
    component = ihp_l2n0.load_benchmark_component()
    stack = ihp_l2n0.build_benchmark_stack(component, shell_variant=True)
    shell = stack.layers["tm2_shell"]
    tm2 = stack.layers["topmetal2"]
    assert shell.material == "sin"
    assert (shell.zmin, shell.zmax) == (tm2.zmin, tm2.zmax)
    import gdsfactory as gf

    polygons = component.get_polygons(layers=[(999, 0)])
    assert polygons
    shell_region = gf.kdb.Region(next(iter(polygons.values())))
    tm2_region = gf.kdb.Region(
        next(iter(component.get_polygons(layers=[(134, 0)]).values()))
    )
    assert len(shell_region & tm2_region) == 0
    assert shell_region.area() > 0.05 * tm2_region.area()
    assert len(shell_region.sized(-600)) == 0
    assert stack.layers["passiv"].zmin >= tm2.zmax  # codespell:ignore passiv


def test_verification_frequencies_are_recorded() -> None:
    import json

    root = ihp_l2n0.REPO / "nbs" / "data" / "ihp_validation" / "experiments"
    verify = json.loads((root / "verification" / "verification.json").read_text())
    assert set(verify) == {"baseline", "best"}
    for row in verify.values():
        assert set(row) == {"5", "9", "30"}
        for point in row.values():
            assert point["L_nH"] > 0

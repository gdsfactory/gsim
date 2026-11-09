"""Tests for the IHP L2n0 benchmark helpers (no Palace execution)."""

from __future__ import annotations

import json
from hashlib import sha256

import numpy as np
import pytest

from scripts import ihp_l2n0_benchmark as ihp_l2n0


def test_benchmark_source_files_are_pinned() -> None:
    assert ihp_l2n0.LAYOUT.is_file()
    assert set(ihp_l2n0.MEASURED) == {"die_x2y7", "die_x5y8"}
    for path in ihp_l2n0.MEASURED.values():
        assert path.is_file()


def test_pinned_ihp_touchstone_inventory_is_complete_and_intact() -> None:
    import skrf as rf

    source = ihp_l2n0.SOURCE
    manifest = json.loads((source / "provenance.json").read_text())
    expected_ports = {
        "die_x2y7.s2p": 2,
        "die_x5y8.s2p": 2,
        "diff_die_x2y7.s1p": 1,
        "diff_die_x5y8.s1p": 1,
        "openems_twoport_1p5um.s2p": 2,
        "openems_oneport_1p5um.s1p": 1,
        "openems_oneport_1um.s1p": 1,
        "openems_matlab_oneport_1p5um.s1p": 1,
        "openems_matlab_oneport_1um.s1p": 1,
        "openems_matlab_oneport_0p5um.s1p": 1,
        "openems_matlab_oneport_0p5um_finer_z.s1p": 1,
    }
    assert {
        name for name in manifest["files"] if name.lower().endswith((".s1p", ".s2p"))
    } == set(expected_ports)
    for name, count in expected_ports.items():
        path = source / name
        assert sha256(path.read_bytes()).hexdigest() == manifest["files"][name]
        network = rf.Network(str(path))
        assert network.nports == count
        assert np.allclose(network.z0, 50.0)


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


def test_openems_edge_variant_uses_gap_ports_and_ground_pillar(tmp_path) -> None:
    pytest.importorskip("ihp")
    sim = ihp_l2n0.make_simulation(tmp_path, port_variant="openems_edge")
    assert [(port.name, port.geometry) for port in sim._ports] == [
        ("P1edge", "gap"),
        ("P2edge", "gap"),
    ]
    assert [
        (block.gds_layer, block.from_layer, block.to_layer) for block in sim._pec_blocks
    ] == [((65000, 0), "metal1", "topmetal1")]
    ports = sim.geometry.component.ports
    assert tuple(ports["P1edge"].center) == pytest.approx((-16.1, 5.0))
    assert tuple(ports["P2edge"].center) == pytest.approx((16.1, 5.0))
    assert ports["P1edge"].width == pytest.approx(12.2)
    assert ports["P2edge"].width == pytest.approx(12.2)


@pytest.mark.parametrize("label", ["die_x2y7", "die_x5y8"])
def test_measured_low_frequency_differential_impedance(label: str) -> None:
    import skrf as rf

    network = rf.Network(str(ihp_l2n0.MEASURED[label]))
    point = network.interpolate(rf.Frequency(1, 1, 1, unit="GHz"))
    s = point.s[0]  # ty: ignore[unresolved-attribute]
    z = ihp_l2n0.differential_impedance_from_s(s)
    inductance_nh, q = ihp_l2n0.lq_from_impedance(z, 1e9)
    assert 1.6 < inductance_nh < 1.75
    assert 6.3 < q < 6.8


def test_differential_extraction_does_not_confuse_shunt_loading_with_series_l() -> None:
    # A passive symmetric two-port with appreciable port-to-ground coupling.
    z_matrix = np.array([[60 + 10j, 40 + 4j], [40 + 4j, 60 + 10j]])
    z0 = ihp_l2n0.Z0_OHM
    s = (z_matrix - z0 * np.eye(2)) @ np.linalg.inv(z_matrix + z0 * np.eye(2))
    recovered = ihp_l2n0.differential_impedance_from_s(s)
    assert recovered == pytest.approx(40 + 12j)
    series_only = 2 * z0 * (1 - s[1, 0]) / s[1, 0]
    assert abs(series_only - recovered) > 1


def test_recorded_best_disagrees_with_full_measured_s_matrices() -> None:
    import skrf as rf

    run = ihp_l2n0.load_recorded_run(ihp_l2n0.RUN_ROOT / "l2n0_best")
    frequencies = np.asarray(run.sparams.freq, dtype=float)
    simulated = np.stack(
        [ihp_l2n0.s_matrix(run.sparams, i) for i in range(len(frequencies))]
    )
    assert simulated.shape == (11, 2, 2)
    for path in ihp_l2n0.MEASURED.values():
        measured = rf.Network(str(path))
        assert np.allclose(measured.z0, ihp_l2n0.Z0_OHM)
        sampled = np.empty_like(simulated)
        for i in range(2):
            for j in range(2):
                values = measured.s[:, i, j]
                sampled[:, i, j] = np.interp(
                    frequencies, measured.f / 1e9, values.real
                ) + 1j * np.interp(frequencies, measured.f / 1e9, values.imag)
        rms = np.sqrt(np.mean(np.abs(simulated - sampled) ** 2, axis=0))
        # The old 1 GHz series-inductance match cannot be used as a validation.
        assert np.all(rms > 0.1), rms


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
        "substrate180",
        "openems_ground",
        "gap",
    } <= set(matrix)
    assert matrix["ground0_pec"]["ok"] is True
    assert matrix["planar_pec"]["pass"] is False
    assert matrix["planar_pec"]["L_nH"] < 0


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

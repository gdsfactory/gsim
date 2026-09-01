# Copyright 2026 GDSFactory
"""Regression tests for the public EIC benchmark references."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gsim.palace.benchmarks.eic_nist import (
    cascade_nist_rlcg,
    load_nist_air_reference,
)
from gsim.palace.benchmarks.reference import (
    differential_inductance_quality,
    load_touchstone_2port,
    maximum_singular_value,
    power_loss_fraction,
    reciprocity_error,
    sha256_file,
)

NIST_DATA = Path(__file__).parents[1] / "data" / "eic" / "nist"


@pytest.mark.parametrize(
    ("filename", "expected_sha256"),
    [
        (
            "1_NP_25umgap.csv",
            "16bbaf0d68dcf4f3aee44371f52c723f9233d41cbc4641a588df57c29cb93566",
        ),
        (
            "2_YP_25umgap.csv",
            "d598f44989c3cd06bb5eade14da13dcb259eeab7035d7ffb16cc4c09f8506c9f",
        ),
        (
            "3_AirPDMS_25umgap.csv",
            "3535574abae7017956ea9b1343437fb5776a9f2829c7caecdb606989fe80bc74",
        ),
        (
            "4_PDMS_25umgap.csv",
            "6779f8ff1077debc149eb8c97c62ba1fff87ed0b412eeeca878a38e2e1a8eaf4",
        ),
        (
            "5b_AirChannel.csv",
            "400899e0767ee9c2ea35265c1df74fe9653a38f0f3f3402e6b062ccde18bef6f",
        ),
        (
            "air_data_vs_sim.npz",
            "8b345f9ce60993d356ad1e5cbaf337f56e4d290191db2897f4ccd9043c79103a",
        ),
    ],
)
def test_nist_curated_file_checksum(filename: str, expected_sha256: str) -> None:
    """Curated files remain byte-identical to the documented sources."""
    assert sha256_file(NIST_DATA / filename) == expected_sha256


def test_nist_complex_arrays_preserve_shape_and_column_order() -> None:
    """The curated NPZ preserves both frequency grids and S-matrix order."""
    reference = load_nist_air_reference(NIST_DATA / "air_data_vs_sim.npz")

    assert reference.simulation.s.shape == (638, 2, 2)
    assert reference.measurement.s.shape == (640, 2, 2)
    assert reference.simulation.frequency_hz[[0, -1]] == pytest.approx([40e3, 110e9])
    assert reference.measurement.frequency_hz[[0, -1]] == pytest.approx([40e3, 110e9])
    np.testing.assert_allclose(
        reference.simulation.s[0],
        np.asarray(
            [
                [
                    0.4472512551845168 - 4.758251607877854e-6j,
                    0.5527487385218444 - 8.558372542961742e-6j,
                ],
                [
                    0.5527487385218702 - 8.558372562796179e-6j,
                    0.44725125518454834 - 4.758251632479014e-6j,
                ],
            ]
        ),
        rtol=1e-13,
        atol=1e-12,
    )


def test_nist_rlcg_cascade_matches_published_mat_with_documented_drift() -> None:
    """The final CSVs reproduce the older stored cascade within archive drift."""
    reconstructed = cascade_nist_rlcg(NIST_DATA)
    stored = load_nist_air_reference(NIST_DATA / "air_data_vs_sim.npz").simulation
    absolute_error = np.abs(reconstructed.s - stored.s)

    assert np.max(absolute_error) == pytest.approx(0.0017645356, rel=1e-6)
    assert reciprocity_error(reconstructed) < 1e-12
    assert maximum_singular_value(reconstructed) <= 1.0 + 1e-12


def test_touchstone_parser_and_ihp_derived_metrics(tmp_path: Path) -> None:
    """RI ordering is S11,S21,S12,S22 and yields the published IHP metrics."""
    touchstone_path = tmp_path / "ihp_point.s2p"
    touchstone_path.write_text(
        "# GHz S RI R 50\n"
        "2.45 "
        "0.2732186941 0.3886981271 "
        "0.6971307009 -0.4564828412 "
        "0.6971307009 -0.4564828412 "
        "0.2727959136 0.3903455789\n",
        encoding="utf-8",
    )

    reference = load_touchstone_2port(touchstone_path)
    inductance_h, quality_factor = differential_inductance_quality(reference)

    assert reference.s[0, 1, 0] == pytest.approx(0.6971307009 - 0.4564828412j)
    assert reference.s[0, 0, 1] == pytest.approx(reference.s[0, 1, 0])
    assert inductance_h[0] * 1e9 == pytest.approx(4.005845, rel=1e-6)
    assert quality_factor[0] == pytest.approx(16.2104, rel=1e-5)
    assert power_loss_fraction(reference).shape == (1, 2)
    assert np.all(power_loss_fraction(reference) > 0)

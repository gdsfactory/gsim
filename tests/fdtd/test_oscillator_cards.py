"""FDTD serialization must retain schema oscillator loss or reject it clearly."""

import json
from math import pi

import pytest
from pdk_schema import Lorentz, LorentzTerm

from gsim import fdtd


@pytest.fixture
def oscillator_simulation(fdtd_pdk_module):
    angular_frequency = 2 * pi * 299_792_458 / 1e-6
    base = fdtd_pdk_module.MATERIAL_CARDS["Si"]
    assert base.optical is not None
    model = Lorentz(
        validity=None,
        variation=None,
        eps_inf=3.99,
        terms=(
            LorentzTerm(
                delta_eps=0.08,
                omega_0=angular_frequency,
                gamma=0.2 * angular_frequency,
            ),
        ),
    )
    fdtd_pdk_module.MATERIAL_CARDS["Si"] = base.model_copy(
        update={"optical": base.optical.model_copy(update={"permittivity": model})}
    )
    simulation = fdtd.Simulation(pdk=fdtd_pdk_module)
    simulation.source(port="o1", wavelength_um=1, wavelength_span_um=0.02)
    simulation.geometry("ten_nm_feature", mesh_size_nm=750)
    return simulation


def test_lossy_oscillator_is_not_silently_made_lossless(
    tmp_path, oscillator_simulation
):
    with pytest.raises(fdtd.FDTDConfigError, match="extinction coefficient"):
        oscillator_simulation.write(tmp_path)
    assert not (tmp_path / "config.json").exists()


def test_dispersive_oscillator_table_retains_known_complex_index(
    tmp_path, oscillator_simulation
):
    oscillator_simulation.materials(dispersion=True)
    artifacts = oscillator_simulation.write(tmp_path)
    config = json.loads(artifacts.config_path.read_text())
    table = config["materials"]["Si"]["dispersion"]["table"]
    center = table["wavelength_nm"].index(1000)
    assert table["n"][center] == pytest.approx(2)
    assert table["k"][center] == pytest.approx(0.1)

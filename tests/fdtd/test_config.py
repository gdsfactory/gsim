"""Tests for the strict GDSFactory FDTD configuration boundary."""

from __future__ import annotations

import warnings

import pytest
from pdk_schema import (
    Coord,
    Index,
    ScalarValue,
    Sellmeier,
    SellmeierTerm,
    TableData,
    TabulatedValue,
)
from pydantic import ValidationError

from gsim.common.materials import (
    GSIM_MATERIAL_CARDS,
    evaluate_material_card,
    resolve_material_snapshot,
)
from gsim.common.materials._helpers import material_card, wavelength_validity
from gsim.fdtd.config import (
    DispersionConfig,
    FDTDConfig,
    FiberModeConfig,
    GaussianBeamConfig,
    IndexTableConfig,
    MaterialConfig,
    PlaneMonitorConfig,
    SellmeierConfig,
    build_fdtd_config,
)
from gsim.fdtd.models import (
    FDTDConfigError,
    MeshGroup,
    MeshManifest,
    PortMeshGroup,
)


def _manifest() -> MeshManifest:
    return MeshManifest(
        volumes={
            "background": MeshGroup(
                name="background",
                physical_tag=11,
                material="SiO2",
                priority=0,
            )
        },
        layers={
            "core": MeshGroup(
                name="core",
                physical_tag=17,
                material="Si",
                priority=1,
            )
        },
        ports={
            "o1": PortMeshGroup(
                name="o1",
                physical_name="port_o1",
                physical_tag=23,
                layer="core",
                normal=(-1, 0, 0),
            )
        },
        domain_min_nm=(0.0, -1000.0, -500.0),
        domain_max_nm=(2000.0, 1000.0, 720.0),
    )


def _config(
    snapshots: dict | None = None,
    *,
    wavelength_halfspan_nm: float = 50,
    dispersion: bool = True,
) -> FDTDConfig:
    if snapshots is None:
        snapshots = {
            name: resolve_material_snapshot(name, 1.55, {}) for name in ("Si", "SiO2")
        }
    return build_fdtd_config(
        _manifest(),
        snapshots,
        background_material="SiO2",
        center_wavelength_nm=1550,
        wavelength_halfspan_nm=wavelength_halfspan_nm,
        num_wavelengths=3,
        default_port="o1",
        nanometers_per_cell=31.25,
        pml_cells=16,
        max_timesteps=None,
        energy_decay_fraction=1e-6,
        max_wall_seconds=3600,
        dispersion=dispersion,
    )


def _silica_with(card) -> dict:
    """Return snapshots with a custom Si card and the built-in SiO2."""
    return {
        "Si": resolve_material_snapshot("Si", 1.55, {"Si": card}),
        "SiO2": resolve_material_snapshot("SiO2", 1.55, {}),
    }


def _dispersion(material: MaterialConfig) -> DispersionConfig:
    assert material.dispersion is not None
    return material.dispersion


def _table(material: MaterialConfig) -> IndexTableConfig:
    table = _dispersion(material).table
    assert table is not None
    return table


def _sellmeier(material: MaterialConfig) -> SellmeierConfig:
    sellmeier = _dispersion(material).sellmeier
    assert sellmeier is not None
    return sellmeier


def _constant_card(n: float, k: float | None = None, *, name: str = "Si"):
    return material_card(
        name=name,
        temperature_ref=None,
        permittivity=Index(
            validity=None,
            variation=None,
            conductivity=None,
            n=ScalarValue(unit="", value=n),
            k=None if k is None else ScalarValue(unit="", value=k),
        ),
    )


def _narrow_sellmeier_card(minimum_um: float, maximum_um: float):
    return material_card(
        name="Si",
        temperature_ref=None,
        permittivity=Sellmeier(
            validity=wavelength_validity(minimum_um, maximum_um),
            variation=None,
            conductivity=None,
            offset=0.0,
            terms=[SellmeierTerm(b=10.6684293, c_um=0.301516485)],
        ),
    )


def test_config_uses_manifest_tags_and_rejects_extra_fields() -> None:
    config = _config()

    assert config.length_scale_meters == 1e-9
    assert config.geometry.volumes["background"].phys_group == 11
    assert config.geometry.layers["core"].phys_group == 17
    assert config.geometry.ports["o1"].phys_group == 23
    assert config.grid.non_pml_bounds.min == (0.0, -1000.0, -500.0)
    assert config.grid.non_pml_bounds.max == (2000.0, 1000.0, 720.0)

    document = config.model_dump()
    document["unsupported"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        FDTDConfig.model_validate(document)

    document = config.model_dump()
    document["excitation"]["default_port"] = "missing"
    with pytest.raises(ValidationError, match="is not declared"):
        FDTDConfig.model_validate(document)


def test_dispersive_cards_are_written_over_the_source_band() -> None:
    li_card = GSIM_MATERIAL_CARDS["Si-Li-293K"].model_copy(update={"name": "Si"})
    config = _config(_silica_with(li_card))
    silicon = config.materials["Si"]
    silica = config.materials["SiO2"]

    # A tabulated card is sampled over the band plus a 10 % margin.
    assert silicon.refractive_index is None
    table = _table(silicon)
    assert table.wavelength_nm[0] == pytest.approx(1490)
    assert table.wavelength_nm[-1] == pytest.approx(1610)
    assert table.n[0] > table.n[-1]
    for wavelength_nm, n, k in zip(table.wavelength_nm, table.n, table.k, strict=True):
        assert (n, k) == pytest.approx(
            evaluate_material_card(li_card, wavelength_nm / 1000)
        )

    # A Sellmeier card's coefficients reach the solver unchanged.
    assert silica.refractive_index is None
    assert _dispersion(silica).wavelength_range_nm == pytest.approx((1490, 1610))
    assert _sellmeier(silica).b == [0.6961663, 0.4079426, 0.8974794]
    assert _sellmeier(silica).c_um2 == pytest.approx(
        [0.0684043**2, 0.1162414**2, 9.896161**2]
    )
    assert config.background_refractive_index == pytest.approx(1.4440236217)

    FDTDConfig.model_validate_json(config.model_dump_json(exclude_none=True))


def test_constant_cards_and_single_wavelength_runs_stay_scalar() -> None:
    config = _config(_silica_with(_constant_card(3.47)))
    assert config.materials["Si"].refractive_index == 3.47
    assert config.materials["Si"].dispersion is None

    config = _config(wavelength_halfspan_nm=0)
    assert config.materials["Si"].refractive_index == pytest.approx(3.4777, abs=1e-4)
    assert config.materials["SiO2"].refractive_index == pytest.approx(1.4440236217)


def test_lossy_cards_need_a_band_and_dispersion() -> None:
    lossy = _silica_with(_constant_card(3.47, 0.01))

    table = _table(_config(lossy).materials["Si"])
    assert set(table.n) == {3.47}
    assert set(table.k) == {0.01}

    with pytest.raises(FDTDConfigError, match=r"pulsed source .*\(dispersion=True\)"):
        _config(lossy, wavelength_halfspan_nm=0)
    with pytest.raises(FDTDConfigError, match=r"dispersion=False\), the default"):
        _config(lossy, dispersion=False)


def test_table_samples_include_the_card_knots_inside_the_band() -> None:
    # A 3 nm wide absorption line between two evenly spaced samples.
    knots_um = [1.4, 1.553, 1.5545, 1.556, 1.7]
    absorbing = material_card(
        name="Si",
        temperature_ref=None,
        permittivity=Index(
            validity=None,
            variation=None,
            conductivity=None,
            n=ScalarValue(unit="", value=3.47),
            k=TabulatedValue(
                unit="",
                data=TableData(
                    dims=("wavelength",),
                    coords={"wavelength": Coord(values=knots_um, unit="um")},
                    values=[0.0, 0.0, 0.05, 0.0, 0.0],
                    attrs={},
                    interp="linear",
                ),
            ),
        ),
    )

    table = _table(_config(_silica_with(absorbing)).materials["Si"])

    assert {1553.0, 1554.5, 1556.0} <= set(table.wavelength_nm)
    assert max(table.k) == pytest.approx(0.05)
    assert 1400.0 not in table.wavelength_nm


def test_card_validity_must_cover_the_source_band() -> None:
    # Each side keeps its margin only while it stays inside the card's validity.
    config = _config(_silica_with(_narrow_sellmeier_card(1.5, 1.6)))
    assert _dispersion(config.materials["Si"]).wavelength_range_nm == (1500, 1600)
    config = _config(_silica_with(_narrow_sellmeier_card(1.5, 1.65)))
    assert _dispersion(config.materials["Si"]).wavelength_range_nm == (
        1500,
        pytest.approx(1610),
    )

    with pytest.raises(FDTDConfigError, match="source band 1500 to 1600 nm"):
        _config(_silica_with(_narrow_sellmeier_card(1.52, 1.6)))


def test_constant_indices_warn_once_about_flattened_cards() -> None:
    with pytest.warns(UserWarning, match="dispersion is not sent") as caught:
        config = _config(dispersion=False)
    assert len(caught) == 1
    assert caught[0].filename == __file__
    message = str(caught[0].message)
    assert "Si (0.3" in message
    assert "SiO2 (0.1" in message
    assert config.materials["Si"].refractive_index == pytest.approx(3.4777, abs=1e-4)
    assert config.materials["SiO2"].refractive_index == pytest.approx(1.4440236217)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        constants = {
            "Si": _constant_card(3.47),
            "SiO2": _constant_card(1.444, name="SiO2"),
        }
        _config(
            {
                name: resolve_material_snapshot(name, 1.55, constants)
                for name in constants
            },
            dispersion=False,
        )


def test_material_schema_requires_one_representation() -> None:
    document = _config().model_dump(exclude_none=True)
    document["materials"]["Si"]["refractive_index"] = 3.47
    with pytest.raises(ValidationError, match="exactly one of refractive_index"):
        FDTDConfig.model_validate(document)

    document = _config().model_dump(exclude_none=True)
    document["materials"]["SiO2"]["dispersion"].pop("wavelength_range_nm")
    with pytest.raises(ValidationError, match="needs wavelength_range_nm"):
        FDTDConfig.model_validate(document)


def test_config_supports_gaussian_beam_and_fiber_monitor() -> None:
    beam = GaussianBeamConfig(
        region_min=(0, 0, 3000),
        region_max=(10000, 10000, 3000),
        aperture_normal="-z",
        propagation_direction=(0, 0, -1),
        e_polarization=(0, 1, 0),
        focal_point=(5000, 5000, 220),
        waist_radius=5000,
        refractive_index=1.444,
    )
    monitor = PlaneMonitorConfig(
        name="fiber",
        region_min=(0, 0, 3000),
        region_max=(10000, 10000, 3000),
        normal="+z",
        fiber_mode=FiberModeConfig(
            propagation_direction=(0, 0, 1),
            e_polarization=(0, 1, 0),
            focal_point=(5000, 5000, 220),
            waist_radius=5000,
            refractive_index=1.444,
        ),
    )
    snapshots = {
        name: resolve_material_snapshot(name, 1.55, {}) for name in ("Si", "SiO2")
    }

    config = build_fdtd_config(
        _manifest(),
        snapshots,
        background_material="SiO2",
        center_wavelength_nm=1550,
        wavelength_halfspan_nm=50,
        num_wavelengths=3,
        default_port=None,
        nanometers_per_cell=31.25,
        pml_cells=16,
        max_timesteps=None,
        energy_decay_fraction=1e-6,
        max_wall_seconds=3600,
        gaussian_beam=beam,
        monitors=[monitor],
    )

    assert config.excitation.type == "gaussian_beam"
    assert config.excitation.gaussian_beam == beam
    assert config.excitation.default_port is None
    assert config.monitors == [monitor]


def test_config_rejects_empty_non_pml_bounds() -> None:
    document = _config().model_dump()
    document["grid"]["non_pml_bounds"]["max"] = (2000.0, -1000.0, 720.0)

    with pytest.raises(ValidationError, match="min must be smaller than max"):
        FDTDConfig.model_validate(document)

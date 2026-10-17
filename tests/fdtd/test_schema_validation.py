"""Reject malformed solver inputs before serializing a configuration."""

import pytest
from pydantic import ValidationError

from gsim.fdtd.config import (
    DispersionConfig,
    ExcitationConfig,
    FDTDConfig,
    IndexTableConfig,
    PlaneMonitorConfig,
    PortConfig,
    SellmeierConfig,
)
from tests.fdtd.test_config import _config


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"n": [1, 1, 1]}, "equal lengths"),
        ({"wavelength_nm": [-1, 2]}, "positive and increasing"),
        ({"wavelength_nm": [2, 1]}, "positive and increasing"),
        ({"n": [0, 1]}, "indices must be positive"),
        ({"k": [-1, 0]}, "cannot be negative"),
    ],
)
def test_invalid_index_tables(options, message):
    with pytest.raises(ValidationError, match=message):
        IndexTableConfig(
            **({"wavelength_nm": [1, 2], "n": [1, 1], "k": [0, 0]} | options)
        )


@pytest.mark.parametrize(
    "options",
    [
        {},
        {
            "table": {"wavelength_nm": [1, 2], "n": [1, 1], "k": [0, 0]},
            "wavelength_range_nm": [1, 2],
        },
        {"sellmeier": {"b": [1], "c_um2": [1]}, "wavelength_range_nm": [2, 1]},
    ],
)
def test_invalid_dispersion_models(options):
    with pytest.raises(ValidationError):
        DispersionConfig(**options)
    with pytest.raises(ValidationError, match="equal lengths"):
        SellmeierConfig(b=[1], c_um2=[1, 2])


@pytest.mark.parametrize("normal", [(0, 0, 0), (1, 1, 0), (2, 0, 0)])
def test_invalid_port_normals(normal):
    with pytest.raises(ValidationError, match="unit axis"):
        PortConfig(phys_group=1, layer="core", normal=normal)


@pytest.mark.parametrize(
    "options",
    [
        {"wavelength_halfspan": 2000},
        {"waveform": "continuous_wave"},
        {"amplitude": 0},
        {"default_port": None},
        {"type": "dipole"},
        {"type": "line_current", "default_port": None},
        {"dipole": {"position": [0, 0, 0], "polarization": [1, 0, 0]}},
    ],
)
def test_invalid_excitation_combinations(options):
    with pytest.raises(ValidationError):
        ExcitationConfig(
            **(
                {
                    "center_wavelength": 1550,
                    "wavelength_halfspan": 50,
                    "num_wavelengths": 3,
                    "default_port": "o1",
                }
                | options
            )
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"region_min": [2, 0, 0]}, "cannot exceed"),
        ({"region_max": [1, 1, 1]}, "must be planar"),
        ({"wavelengths": [0]}, "must be positive"),
    ],
)
def test_invalid_monitor_planes(options, message):
    with pytest.raises(ValidationError, match=message):
        PlaneMonitorConfig(
            **(
                {
                    "name": "m",
                    "region_min": [0, 0, 0],
                    "region_max": [1, 1, 0],
                    "normal": "+z",
                }
                | options
            )
        )


@pytest.mark.parametrize("invalid", ["material", "layer", "monitors"])
def test_configuration_references_are_consistent(invalid):
    document = _config().model_dump(mode="json")
    if invalid == "material":
        document["geometry"]["layers"]["core"]["material"] = "absent"
    elif invalid == "layer":
        document["geometry"]["ports"]["o1"]["layer"] = "absent"
    else:
        monitor = {
            "name": "duplicate",
            "region_min": [0, 0, 0],
            "region_max": [1, 1, 0],
            "normal": "+z",
        }
        document["monitors"] = [monitor, monitor]
    with pytest.raises(ValidationError):
        FDTDConfig.model_validate(document)

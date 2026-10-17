"""Check interpolation boundaries independently of database values."""

import pytest

from gsim.common.materials import snapshots


@pytest.mark.parametrize(
    ("coordinates", "values", "wavelength", "method", "error"),
    [
        ([1, 2], [1], 1.5, "linear", "inconsistent lengths"),
        ([2, 1], [1, 2], 1.5, "linear", "strictly increasing"),
        ([1, 2], [1, 2], 3, "linear", "does not include"),
        ([1, 2], [1, 2], 1.5, "cubic", "Unsupported table interpolation"),
    ],
)
def test_invalid_interpolation_grid(coordinates, values, wavelength, method, error):
    with pytest.raises(snapshots.MaterialResolutionError, match=error):
        snapshots._interpolate(coordinates, values, wavelength, method)


@pytest.mark.parametrize(
    ("wavelength", "expected"), [(1, 2), (1.4, 2), (1.6, 4), (2, 4)]
)
def test_nearest_interpolation(wavelength, expected):
    assert snapshots._interpolate([1, 2], [2, 4], wavelength, "nearest") == expected


def test_unsupported_wavelength_unit():
    with pytest.raises(
        snapshots.MaterialModelError, match="Unsupported wavelength unit"
    ):
        snapshots._wavelength_to_um(1, "mm")

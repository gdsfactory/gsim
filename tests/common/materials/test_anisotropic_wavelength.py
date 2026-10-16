"""Resolve tensor dispersion at a wavelength, including incomplete axis data."""

import numpy as np
import pytest

from gsim.common.stack.materials import DispersionModel, MaterialProperties


@pytest.mark.parametrize("optical", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("specified", [False, True])
def test_axis_models_use_isotropic_fallback_without_losing_metadata(
    optical, fallback, specified
):
    validity = {"valid_wavelength": (1, 2)} if specified else {}
    axis_model = DispersionModel(
        type="cauchy" if optical else "constant",
        axis="xx",
        cauchy_terms=[3.0] if optical else [],
        permittivity=4,
        source="axis measurement",
        validity=validity,
    )
    models = [axis_model]
    if fallback:
        models.append(
            DispersionModel(
                type="constant",
                permittivity=9,
                source="bulk measurement",
                validity=validity,
            )
        )
    material = MaterialProperties(
        conductivity=100, loss_tangent=0.01, dispersion_models=models
    )
    result = material.evaluate_at_wavelength(1.5)
    assert isinstance(result.permittivity, list)
    np.testing.assert_allclose(
        result.permittivity,
        [1 + 3 / 1.5**2 if optical else 4, 9 if fallback else 1, 9 if fallback else 1],
    )
    assert result.conductivity == (None if optical else 100)
    assert result.loss_tangent == 0.01
    assert result.within_validity == fallback
    assert "xx(axis measurement)" in result.model_source
    if not fallback:
        assert "yy: no model found" in result.validity_note
    if not specified:
        assert "validity unspecified" in result.validity_note

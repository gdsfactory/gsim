"""One spec-validation vocabulary the Backends share."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from gsim.common.validation import AscendingInterval


class Windowed(BaseModel):
    """A model carrying the annotated interval, as a Backend's does."""

    window: AscendingInterval | None = None


class TestAscendingInterval:
    def test_an_ascending_interval_passes_through(self):
        assert Windowed(window=(-1.5, 2.0)).window == (-1.5, 2.0)

    def test_an_omitted_interval_is_not_checked(self):
        assert Windowed().window is None

    def test_a_descending_interval_is_rejected(self):
        with pytest.raises(ValidationError, match="ascending"):
            Windowed(window=(2.0, -1.0))

    def test_an_empty_interval_is_rejected(self):
        with pytest.raises(ValidationError, match="ascending"):
            Windowed(window=(1.0, 1.0))

    def test_a_non_finite_bound_is_rejected(self):
        with pytest.raises(ValidationError, match="bounds must be finite"):
            Windowed(window=(0.0, float("inf")))

    def test_the_field_name_is_in_the_error_location(self):
        with pytest.raises(ValidationError) as raised:
            Windowed(window=(2.0, -1.0))
        assert raised.value.errors()[0]["loc"] == ("window",)

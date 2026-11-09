"""Deprecation of Palace's former coordinate-path sample-count setting."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Annotated

import pydantic
from pydantic import Field, TypeAdapter

_SAMPLE_COUNT = TypeAdapter(Annotated[int, Field(ge=1)])
_WARNING_SKIP_PREFIXES = (
    str(Path(__file__).resolve().parents[1]) + os.sep,
    str(Path(pydantic.__file__).resolve().parent) + os.sep,
)


def warn_legacy_path_sampling(nsamples: int | None) -> None:
    """Validate an explicit legacy count and warn at its caller; ignore omission."""
    if nsamples is None:
        return
    _SAMPLE_COUNT.validate_python(nsamples)
    warnings.warn(
        "nsamples is deprecated for Palace postprocessing and is ignored; "
        "remove it. Palace controls coordinate-path integration.",
        DeprecationWarning,
        skip_file_prefixes=_WARNING_SKIP_PREFIXES,
    )

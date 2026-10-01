"""Validated transmission-line extraction and simulated TRL calibration.

Install ``gsim[rf]`` for network operations. Lengths are in metres, frequencies
in Hz, and propagation constants in inverse metres. Calibrated results use a
normalized line-wave basis; no physical characteristic impedance is inferred.
"""

from .calibration import LineCalibration, calibrate_multiline_trl, calibrate_trl
from .propagation import (
    LineConditioning,
    PropagationResult,
    extract_propagation,
    line_conditioning,
    predict_line,
)

__all__ = [
    "LineCalibration",
    "LineConditioning",
    "PropagationResult",
    "calibrate_multiline_trl",
    "calibrate_trl",
    "extract_propagation",
    "line_conditioning",
    "predict_line",
]

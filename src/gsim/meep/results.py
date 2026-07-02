"""Aggregated eigenmode sweep result wrapper."""

from __future__ import annotations

from gsim.meep.models.results import ModeResult


class ModeSweepResult:
    """Collection of :class:`ModeResult` items from an eigenmode sweep.

    Provides fluent accessors for filtering and key-based lookup.

    Example::

        result = sim.solve_modes()
        mode_at_1550 = result.at(1.55).band(3)
        df = pd.DataFrame(result.to_dict())
    """

    def __init__(self, results: list[ModeResult]) -> None:
        """Create a sweep result from a flat list of mode results."""
        self.results: list[ModeResult] = results

    def at(self, wavelength: float) -> ModeSweepResult:
        """Filter results to those at a specific wavelength."""
        return ModeSweepResult([r for r in self.results if r.wavelength == wavelength])

    def band(self, n: int) -> ModeResult | None:
        """Return the result for a specific band number, or None."""
        for r in self.results:
            if r.band_num == n:
                return r
        return None

    def to_dict(self) -> dict[str, list]:
        """Convert results to a dict-of-lists suitable for ``pd.DataFrame``."""
        return {
            "wavelength": [r.wavelength for r in self.results],
            "frequency": [r.frequency for r in self.results],
            "band_num": [r.band_num for r in self.results],
            "n_eff": [r.n_eff for r in self.results],
            "n_group": [r.n_group for r in self.results],
            "parity": [r.parity for r in self.results],
            "kdom": [r.kdom for r in self.results],
            "fields": [r.fields for r in self.results],
        }

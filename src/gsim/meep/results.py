"""Aggregated eigenmode sweep result wrapper."""

from __future__ import annotations

from typing import Any

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

    def _find_result(self, wavelength: float, band: int | None = None) -> ModeResult:
        """Return the ModeResult matching wavelength and optional band, or error."""
        candidates = [r for r in self.results if r.wavelength == wavelength]
        if not candidates:
            available = sorted({r.wavelength for r in self.results})
            raise ValueError(
                f"No mode result found at wavelength={wavelength} µm. "
                f"Available: {available}"
            )
        if band is not None:
            for r in candidates:
                if r.band_num == band:
                    return r
            available_bands = sorted({r.band_num for r in candidates})
            raise ValueError(
                f"No mode result found at wavelength={wavelength} µm, "
                f"band={band}. Available bands: {available_bands}"
            )
        return candidates[0]

    def plot_mode(
        self,
        wavelength: float,
        band: int | None = None,
        *,
        components: str | list[str] = "auto",
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """Delegate to ModeResult.plot_mode for mode at wavelength and optional band.

        Args:
            wavelength: Free-space wavelength in µm.
            band: Mode band index (``None`` = first match).
            components: ``"auto"``, ``"all"``, or list of component names.
            **kwargs: Forwarded to :meth:`ModeResult.plot_mode`.

        Returns:
            ``(fig, ax)`` or ``(fig, axes)`` as from
            :meth:`ModeResult.plot_mode`.
        """
        result = self._find_result(wavelength, band=band)
        return result.plot_mode(components=components, **kwargs)

    def plot_index(self, wavelength: float, **kwargs: Any) -> tuple[Any, Any]:
        """Delegate to ModeResult.plot_index for the mode at wavelength.

        Args:
            wavelength: Free-space wavelength in µm.
            **kwargs: Forwarded to :meth:`ModeResult.plot_index`.

        Returns:
            ``(fig, ax)`` as from :meth:`ModeResult.plot_index`.
        """
        result = self._find_result(wavelength)
        return result.plot_index(**kwargs)

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

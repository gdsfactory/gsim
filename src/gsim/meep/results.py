"""Aggregated eigenmode sweep result wrapper."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import numpy as np

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

    @classmethod
    def from_directory(cls, directory: str | Path) -> ModeSweepResult:
        """Load a :class:`ModeSweepResult` from a cloud runner output directory.

        Reads ``mode_results.json`` and all referenced ``.npy`` field files
        from *directory*.  The JSON must have a ``"results"`` key containing
        a list of per-mode metadata dicts, each optionally carrying a
        ``"wl_idx"`` and ``"band_num"`` for file-name resolution.

        Args:
            directory: Path to the directory containing ``mode_results.json``
                and ``.npy`` field files.

        Returns:
            :class:`ModeSweepResult` with reconstructed :class:`ModeResult`
            objects.
        """
        directory = Path(directory)
        json_path = directory / "mode_results.json"
        if not json_path.exists():
            raise FileNotFoundError(f"mode_results.json not found in {directory}")

        data = json.loads(json_path.read_text())

        z_grid = None
        z_grid_path = directory / "mode_z_grid.npy"
        if z_grid_path.exists():
            with contextlib.suppress(OSError, ValueError):
                z_grid = np.load(z_grid_path)

        results: list[ModeResult] = []
        for entry in data.get("results", []):
            wl_idx = entry.get("wl_idx", 0)
            band_num = entry.get("band_num", 1)

            fields: dict[str, np.ndarray] = {}
            for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                fname = f"mode_f_band{band_num}_wl{wl_idx}_{comp}.npy"
                fpath = directory / fname
                if fpath.exists():
                    with contextlib.suppress(OSError, ValueError):
                        fields[comp] = np.load(fpath)

            results.append(
                ModeResult(
                    n_eff=entry["n_eff"],
                    wavelength=entry["wavelength"],
                    frequency=entry["frequency"],
                    fields=fields,
                    kdom=entry.get("kdom", []),
                    n_group=entry.get("n_group"),
                    band_num=band_num,
                    parity=entry.get("parity", "NO_PARITY"),
                    z_grid=z_grid,
                    cross_section_plane=entry.get("cross_section_plane"),
                )
            )

        return cls(results)

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

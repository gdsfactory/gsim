"""Driven simulation class for frequency-domain S-parameter extraction.

This module provides the DrivenSim class for running frequency-sweep
simulations to extract S-parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from gsim.common import Geometry, LayerStack
from gsim.palace.base import PalaceSimMixin
from gsim.palace.models import (
    CPWPortConfig,
    DrivenConfig,
    MaterialConfig,
    MeshConfig,
    NumericalConfig,
    PortConfig,
    WavePortConfig,
)

if TYPE_CHECKING:
    from gsim.palace.results import SParams


class DrivenSim(PalaceSimMixin, BaseModel):
    """Frequency-domain driven simulation for S-parameter extraction.

    This class configures and runs driven simulations that sweep through
    frequencies to compute S-parameters. Uses composition (no inheritance)
    with shared Geometry and Stack components from gsim.common.

    Example:
        >>> from gsim.palace import DrivenSim
        >>>
        >>> sim = DrivenSim()
        >>> sim.set_geometry(component)
        >>> sim.set_stack(air_above=300.0)
        >>> sim.add_cpw_port("o1", layer="topmetal2", s_width=10, gap_width=6)
        >>> sim.add_cpw_port("o2", layer="topmetal2", s_width=10, gap_width=6)
        >>> sim.set_driven(fmin=1e9, fmax=100e9, num_points=40)
        >>> sim.set_output_dir("./sim")
        >>> sim.mesh(preset="default")
        >>> sp = sim.run()  # SParams
        >>> sp.s21.db  # dB magnitude of S21 vs frequency

    Attributes:
        geometry: Wrapped gdsfactory Component (from common)
        stack: Layer stack configuration (from common)
        ports: List of single-element port configurations
        cpw_ports: List of CPW (two-element) port configurations
        driven: Driven simulation configuration (frequencies, etc.)
        mesh: Mesh configuration
        materials: Material property overrides
        numerical: Numerical solver configuration
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    simulation_type: Literal["driven"] = "driven"

    # Composed objects (from common)
    geometry: Geometry | None = None
    stack: LayerStack | None = None

    # Port configurations
    ports: list[PortConfig] = Field(default_factory=list)
    cpw_ports: list[CPWPortConfig] = Field(default_factory=list)
    wave_ports: list[WavePortConfig] = Field(default_factory=list)
    terminals: None = None

    # Driven simulation config
    driven: DrivenConfig = Field(default_factory=DrivenConfig)
    eigenmode: None = None
    absorbing_boundary: bool = True

    # Mesh config
    mesh_config: MeshConfig = Field(default_factory=MeshConfig.default)

    # Material overrides and numerical config
    materials: dict[str, MaterialConfig] = Field(default_factory=dict)
    numerical: NumericalConfig = Field(default_factory=NumericalConfig)

    # Stack configuration (stored as kwargs until resolved)
    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _pec_blocks: list = PrivateAttr(default_factory=list)
    _hints: dict[str, Any] = PrivateAttr(default_factory=dict)
    _airbox_config: dict[str, float] = PrivateAttr(default_factory=dict)

    # Internal state
    _output_dir: Path | None = PrivateAttr(default=None)
    _configured_ports: bool = PrivateAttr(default=False)
    _last_mesh_result: Any = PrivateAttr(default=None)
    _last_ports: list = PrivateAttr(default_factory=list)

    # Cloud job state (set by upload/run)
    _job_id: str | None = PrivateAttr(default=None)

    # -------------------------------------------------------------------------
    # Cloud run (narrowed return type)
    # -------------------------------------------------------------------------

    def run(
        self,
        parent_dir: str | Path | None = None,
        *,
        verbose: Literal["quiet", "status", "full"] = "status",
        wait: bool = True,
    ) -> SParams | str:
        """Run the driven sim on GDSFactory+ cloud.

        Thin wrapper over :meth:`PalaceSimMixin.run` that narrows the
        return type: the palace result parser always turns a completed
        driven run (which has ``port-S.csv``) into an
        :class:`~gsim.palace.results.SParams` object.

        Returns:
            :class:`SParams` when ``wait=True`` (the default), or the
            ``job_id`` string when ``wait=False``.
        """
        from gsim.palace.results import SParams as _SParams

        result = super().run(parent_dir, verbose=verbose, wait=wait)
        if isinstance(result, (_SParams, str)):
            return result
        msg = (
            f"DrivenSim.run expected SParams (driven sweep with "
            f"port-S.csv) but got {type(result).__name__}. "
            "This usually means the cloud job did not produce S-params."
        )
        raise RuntimeError(msg)

    # -------------------------------------------------------------------------
    # Driven configuration
    # -------------------------------------------------------------------------

    def set_driven(
        self,
        *,
        fmin: float = 1e9,
        fmax: float = 100e9,
        num_points: int = 40,
        scale: Literal["linear", "log"] = "linear",
        adaptive_tol: float = 0.02,
        adaptive_max_samples: int = 20,
        compute_s_params: bool = True,
        reference_impedance: float = 50.0,
        excitation_port: str | None = None,
        save_step: int = 0,
        save_fields_at: list[float] | None = None,
        save_freq: str | None = None,
    ) -> None:
        """Configure driven (frequency sweep) simulation.

        Args:
            fmin: Minimum frequency in Hz
            fmax: Maximum frequency in Hz
            num_points: Number of frequency points
            scale: "linear" or "log" frequency spacing
            adaptive_tol: Relative error tolerance (unitless fraction) for
                adaptive frequency sampling. Palace builds a reduced-order
                model from a few full solves and interpolates the rest.
                0 = disabled (full solve at every point), 0.02 = 2% error
                (fast default), 1e-3 = 0.1% (accurate), 1e-4 = publication
                quality.
            adaptive_max_samples: Maximum number of additional frequency
                points that adaptive refinement may insert between the
                points defined by fmin, fmax, and num_points.
            compute_s_params: Compute S-parameters
            reference_impedance: Reference impedance for S-params (Ohms)
            excitation_port: Port to excite (None = first port)
            save_step: Save fields every N frequency steps for ParaView
                (0 = disabled)
            save_fields_at: Specific frequencies (Hz) at which to save
                fields for ParaView visualisation.
            save_freq: Convenience shorthand for saving fields at a named
                frequency.  ``"center"`` saves at ``(fmin + fmax) / 2``.
                Appended to *save_fields_at* if both are given.

        Example:
            >>> sim.set_driven(fmin=1e9, fmax=100e9, num_points=40)
            >>> sim.set_driven(fmin=1e9, fmax=100e9, save_freq="center")
        """
        fields_at: list[float] = list(save_fields_at or [])

        if save_freq is not None:
            if save_freq == "center":
                fields_at.append((fmin + fmax) / 2)
            else:
                raise ValueError(
                    f"Unknown save_freq value: {save_freq!r}. "
                    "Supported values: 'center'."
                )

        self.driven = DrivenConfig(
            fmin=fmin,
            fmax=fmax,
            num_points=num_points,
            scale=scale,
            adaptive_tol=adaptive_tol,
            adaptive_max_samples=adaptive_max_samples,
            compute_s_params=compute_s_params,
            reference_impedance=reference_impedance,
            excitation_port=excitation_port,
            save_step=save_step,
            save_fields_at=fields_at,
        )


__all__ = ["DrivenSim"]

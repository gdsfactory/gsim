"""The femwell Route: the default Backend, solving the shared mesh in-process.

femwell reads the Mode's fields as it solves, so it is the Route that can
also say how much of a Mode sits on the Window wall or outside the
Strips, and the one that can carry a continuous permittivity per element.
Its electrode metal defaults to a Region of lossy metal (ADR 0003),
because it is the Route that can carry the metal's own loss.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from gsim.modulator.route import EMRoute, Route

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from gsim.common.modes import Conductor, LineReading
    from gsim.modulator.staircase import ConductorModel
    from gsim.palace import BoundaryModeSim

__all__ = ["FemwellRoute"]


class FemwellRoute(Route):
    """The Route interface on femwell."""

    name: ClassVar[EMRoute] = "femwell"
    conductor_model: ClassVar[ConductorModel] = "volume"
    continuous_materials: ClassVar[bool] = True

    def require(self, *, stage_name: str) -> None:  # noqa: ARG002 - uniform signature
        """Check the femwell extra is installed.

        Raises:
            ImportError: Naming the ``femwell`` extra when it is not.
        """
        from gsim.femwell.runtime import require_femwell, require_skfem

        require_femwell()
        require_skfem()

    def check_line_settings(
        self,
        *,
        conductor_model: ConductorModel,
        metallic_boundaries: bool,
        order: int,
        stage_name: str,
    ) -> None:
        """What a perfect electrode needs of a femwell solve.

        femwell has one perfect-conductor condition and applies it to
        every facet of the domain boundary at once, so a ``"pec"``
        electrode — which is a hole in that boundary — is a conductor
        only while ``metallic_boundaries`` is on. Off, the same hole
        takes the natural condition and the Stage would quietly solve a
        cross-section with open slots where its electrodes should be.

        And femwell solves for ``E`` and derives ``H`` from its curl, one
        order lower, so a first-order solve leaves ``H`` piecewise
        constant exactly where a ``"pec"`` conductor's contour integral
        reads it. On the shipped coax benchmark that is 8-29% of the
        impedance depending on the mesh, and it does not converge with
        refinement — only with order.

        Raises:
            ValueError: When a perfect electrode is asked for with the
                wall off.

        Warns:
            UserWarning: When a perfect electrode's impedance would be
                read off a first-order field.
        """
        if conductor_model != "pec":
            return
        if not metallic_boundaries:
            raise ValueError(
                f"The {stage_name} stage's femwell route cannot leave its "
                "electrodes perfect while metallic_boundaries is off: femwell "
                "applies that one condition to the whole domain boundary, and a "
                "perfect electrode is a hole in it, so the electrodes would come "
                "out as open slots. Turn the wall back on with "
                f"study.{stage_name}(metallic_boundaries=True), or mesh the "
                "electrodes as lossy volumes with "
                f"study.{stage_name}(conductor_model='volume')."
            )
        if order < 2:
            warnings.warn(
                f"The {stage_name} stage's femwell route reads its "
                "characteristic impedance off the field around a perfect "
                f"conductor at order {order}, where femwell's h field is "
                "piecewise constant: the impedance is biased high by tens of "
                f"percent. Solve with study.{stage_name}(order=2), or mesh "
                "the electrodes as lossy volumes with "
                f"study.{stage_name}(conductor_model='volume').",
                stacklevel=3,
            )

    def solve(
        self,
        sim: BoundaryModeSim,
        *,
        freq_hz: float,
        num_modes: int,
        target: float | None,
        order: int,
        verbose: bool,  # noqa: ARG002 - femwell has nothing to stream
        stage_name: str,  # noqa: ARG002 - nothing to warn about
        epsilon: ArrayLike | None = None,
    ) -> Sequence[Any]:
        """Solve the meshed simulation with femwell at one frequency.

        The materials are the simulation's own stack, resolved at the
        frequency, unless a continuous per-element permittivity replaces
        them.
        """
        from scipy.constants import speed_of_light as c0

        from gsim.femwell.adapter import epsilon_by_region, solve_modes

        mesh_path = sim.mesh_path
        if epsilon is not None:
            materials: Any = epsilon
        else:
            if sim.stack is None:
                raise ValueError(
                    "The simulation carries no layer stack to resolve materials "
                    "from; call set_stack() before solving."
                )
            materials = epsilon_by_region(mesh_path, sim.stack, frequency_hz=freq_hz)
        return solve_modes(
            mesh_path,
            epsilon=materials,
            wavelength_um=c0 / freq_hz * 1e6,
            num_modes=num_modes,
            order=order,
            metallic_boundaries=sim.metallic_boundaries,
            n_guess=target,
        )

    def boundary_ratio(self, mode: Any) -> float:
        """Peak field at the Window boundary over peak field overall."""
        from gsim.femwell.adapter import boundary_field_ratio

        return boundary_field_ratio(mode)

    def strip_fraction_outside(
        self,
        mode: Any,
        span: tuple[float, float],
        *,
        stage_name: str,  # noqa: ARG002 - femwell measures it, nothing to warn
    ) -> float:
        """How much of the Mode's power sits outside the Strip extent."""
        from gsim.femwell.adapter import field_fraction_outside

        return field_fraction_outside(mode, span)

    def read_line(
        self,
        sim: BoundaryModeSim,
        mode: Any,
        *,
        freq_hz: float,
        signal: Conductor,
        return_: Conductor | None,
        stage_name: str,  # noqa: ARG002 - the reading names no Stage
    ) -> LineReading:
        """Index and impedance off the fields, the wall Mode off the currents."""
        from gsim.femwell.adapter import line_reading

        return line_reading(
            mode,
            frequency_hz=freq_hz,
            mesh=sim.mesh_path,
            signal=signal,
            return_=return_,
        )

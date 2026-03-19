"""GMSH-based mesh generation for photonic simulations.

Mirrors Palace's DrivenSim pattern: configure → mesh() → write_config().
Produces a GMSH ``.msh`` + ``mesh_config.json`` that fully describes a
photonic problem. This is an alternative to Meep's existing prism-based
workflow.

Example::

    from gsim.meep import MeepMeshSim

    sim = MeepMeshSim()
    sim.geometry(component=c, z_crop="auto")
    sim.materials = {"si": 3.47, "SiO2": 1.44}
    sim.source(port="o1", wavelength=1.55, wavelength_span=0.1, num_freqs=11)
    sim.monitors = ["o1", "o2", "o3"]
    sim.domain(pml=1.0, margin=0.5)
    sim.solver(resolution=32)

    result = sim.mesh("./mesh-ybranch")
    sim.write_config()
    sim.plot_mesh(interactive=False)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from gsim.meep.models.api import (
    FDTD,
    Domain,
    Geometry,
    Material,
    ModeSource,
)

logger = logging.getLogger(__name__)


class MeepMeshSim(BaseModel):
    """GMSH-based mesh generation for photonic problems.

    Collects photonic simulation parameters and generates a GMSH mesh +
    config JSON that together fully describe the problem.

    Uses the same API models as :class:`gsim.meep.Simulation` (``Geometry``,
    ``ModeSource``, ``Domain``, ``FDTD``, ``Material``).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    geometry: Geometry = Field(default_factory=Geometry)
    materials: dict[str, float | Material] = Field(default_factory=dict)
    source: ModeSource = Field(default_factory=ModeSource)
    monitors: list[str] = Field(default_factory=list)
    domain: Domain = Field(default_factory=Domain)
    solver: FDTD = Field(default_factory=FDTD)

    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _output_dir: Path | None = PrivateAttr(default=None)
    _last_mesh_result: Any = PrivateAttr(default=None)

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------

    @field_validator("materials", mode="before")
    @classmethod
    def _normalize_materials(
        cls,
        v: dict[str, float | Material | dict],
    ) -> dict[str, float | Material]:
        """Accept float shorthand: ``{"si": 3.47}`` → ``Material(n=3.47)``."""
        out: dict[str, float | Material] = {}
        for name, val in v.items():
            if isinstance(val, (int, float)):
                out[name] = Material(n=float(val))
            elif isinstance(val, dict):
                out[name] = Material(**val)
            else:
                out[name] = val
        return out

    # -------------------------------------------------------------------------
    # Internal: stack resolution
    # -------------------------------------------------------------------------

    def _ensure_stack(self) -> None:
        """Lazily resolve the layer stack if not yet built."""
        if self.geometry.stack is not None:
            return

        from gsim.common.stack import get_stack

        if self._stack_kwargs:
            yaml_path = self._stack_kwargs.pop("yaml_path", None)
            self.geometry.stack = get_stack(yaml_path=yaml_path, **self._stack_kwargs)
            self._stack_kwargs["yaml_path"] = yaml_path
        else:
            self.geometry.stack = get_stack()

    # -------------------------------------------------------------------------
    # Internal: z-crop (reuses same logic as Simulation)
    # -------------------------------------------------------------------------

    def _apply_z_crop(self) -> None:
        """Apply z-crop to the stack if geometry.z_crop is set."""
        if self.geometry.z_crop is None:
            return

        from gsim.common.stack.extractor import Layer, LayerStack
        from gsim.meep.ports import _find_highest_n_layer

        stack = self.geometry.stack
        if stack is None:
            raise ValueError("No stack configured for z-crop.")

        ref: Layer | None = None
        if self.geometry.z_crop == "auto":
            ref, best_n = _find_highest_n_layer(stack)
            if ref is None or best_n <= 1.5:
                raise ValueError(
                    "Could not auto-detect core layer (no layer with n > 1.5). "
                    "Set geometry.z_crop to an explicit layer name."
                )
        else:
            layer_name = self.geometry.z_crop
            if layer_name not in stack.layers:
                raise ValueError(
                    f"Layer '{layer_name}' not found. "
                    f"Available: {list(stack.layers.keys())}"
                )
            ref = stack.layers[layer_name]

        z_lo = ref.zmin - self.domain.margin_z_below
        z_hi = ref.zmax + self.domain.margin_z_above

        cropped: dict[str, Layer] = {}
        for name, layer in stack.layers.items():
            if layer.zmax <= z_lo or layer.zmin >= z_hi:
                continue
            new_zmin = max(layer.zmin, z_lo)
            new_zmax = min(layer.zmax, z_hi)
            cropped[name] = layer.model_copy(
                update={
                    "zmin": new_zmin,
                    "zmax": new_zmax,
                    "thickness": new_zmax - new_zmin,
                }
            )

        cropped_dielectrics = []
        for diel in stack.dielectrics:
            if diel["zmax"] <= z_lo or diel["zmin"] >= z_hi:
                continue
            cropped_dielectrics.append(
                {
                    **diel,
                    "zmin": max(diel["zmin"], z_lo),
                    "zmax": min(diel["zmax"], z_hi),
                }
            )

        self.geometry.stack = LayerStack(
            pdk_name=stack.pdk_name,
            units=stack.units,
            layers=cropped,
            materials=stack.materials,
            dielectrics=cropped_dielectrics,
            simulation=stack.simulation,
        )
        self.geometry.z_crop = None

    # -------------------------------------------------------------------------
    # mesh()
    # -------------------------------------------------------------------------

    def mesh(
        self,
        output_dir: str | Path,
        *,
        refined_mesh_size: float = 0.05,
        max_mesh_size: float = 1.0,
        margin: float | None = None,
        air_margin: float = 2.0,
        include_airbox: bool = False,
    ) -> Any:
        """Generate a GMSH mesh for this simulation.

        Creates ``mesh.msh`` inside *output_dir*.

        Args:
            output_dir: Directory for mesh output files.
            refined_mesh_size: Fine mesh size near waveguide boundaries (um).
            max_mesh_size: Coarse mesh size in cladding/substrate (um).
            margin: XY margin around geometry (um). Defaults to
                ``domain.margin + domain.pml``.
            air_margin: Airbox margin around dielectric envelope (um).
            include_airbox: Whether to add a surrounding airbox volume.
                Not needed for photonic simulations (default ``False``).

        Returns:
            :class:`~gsim.common.mesh.types.MeshResult` with mesh path,
            statistics, and physical group info.

        Raises:
            ValueError: If no component or stack is configured.
        """
        from gsim.common.mesh import generate_mesh

        if self.geometry.component is None:
            raise ValueError(
                "No component set. Assign sim.geometry(component=...) first."
            )

        self._ensure_stack()
        if self.geometry.stack is None:
            raise ValueError("Stack resolution failed.")

        self._apply_z_crop()

        if margin is None:
            margin = self.domain.margin + self.domain.pml

        output_dir = Path(output_dir)
        self._output_dir = output_dir

        result = generate_mesh(
            component=self.geometry.component,
            stack=self.geometry.stack,
            output_dir=output_dir,
            model_name="mesh",
            refined_mesh_size=refined_mesh_size,
            max_mesh_size=max_mesh_size,
            margin=margin,
            air_margin=air_margin,
            include_airbox=include_airbox,
            mesh_scale=1000.0,  # um → nm
        )

        self._last_mesh_result = result
        logger.info("Mesh generated: %s", result.mesh_path)
        return result

    # -------------------------------------------------------------------------
    # write_config()
    # -------------------------------------------------------------------------

    def write_config(self) -> Path:
        """Write ``mesh_config.json`` alongside the mesh.

        Must be called after :meth:`mesh`.

        Returns:
            Path to the written config file.

        Raises:
            ValueError: If :meth:`mesh` has not been called.
        """
        from gsim.meep.mesh_config import write_mesh_config

        if self._last_mesh_result is None or self._output_dir is None:
            raise ValueError("Call mesh() first to generate a mesh.")

        return write_mesh_config(
            mesh_result=self._last_mesh_result,
            materials=self.materials,
            source=self.source,
            monitors=self.monitors,
            domain=self.domain,
            solver=self.solver,
            output_dir=self._output_dir,
        )

    # -------------------------------------------------------------------------
    # plot_mesh()
    # -------------------------------------------------------------------------

    def plot_mesh(
        self,
        show_groups: list[str] | None = None,
        interactive: bool = True,
    ) -> None:
        """Visualize the generated mesh.

        Delegates to :func:`gsim.viz.plot_mesh`.

        Args:
            show_groups: Physical group names to display (None = all).
            interactive: Show interactive 3D viewer.

        Raises:
            ValueError: If :meth:`mesh` has not been called.
        """
        from gsim.viz import plot_mesh

        if self._last_mesh_result is None:
            raise ValueError("Call mesh() first to generate a mesh.")

        plot_mesh(
            self._last_mesh_result.mesh_path,
            show_groups=show_groups,
            interactive=interactive,
        )

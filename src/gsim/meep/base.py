"""Base mixin for MEEP simulation classes.

Provides common fluent API methods following the PalaceSimMixin pattern,
adapted for photonic (MEEP) simulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from gdsfactory.component import Component

    from gsim.common import Geometry, LayerStack
    from gsim.common.geometry_model import GeometryModel
    from gsim.common.stack.materials import MaterialProperties


class MeepSimMixin:
    """Mixin providing common methods for MEEP simulation classes.

    Subclasses must define these attributes (typically via Pydantic fields):
        - geometry: Geometry | None
        - stack: LayerStack | None
        - materials: dict[str, MaterialProperties]
        - _output_dir: Path | None (private)
        - _stack_kwargs: dict[str, Any] (private)
    """

    geometry: Geometry | None
    stack: LayerStack | None
    materials: dict[str, MaterialProperties]
    _output_dir: Path | None
    _stack_kwargs: dict[str, Any]

    # -------------------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------------------

    def set_output_dir(self, path: str | Path) -> None:
        """Set the output directory for simulation files.

        Args:
            path: Directory path for output files

        Example:
            >>> sim.set_output_dir("./meep-sim")
        """
        self._output_dir = Path(path)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path | None:
        """Get the current output directory."""
        return self._output_dir

    # -------------------------------------------------------------------------
    # Geometry methods
    # -------------------------------------------------------------------------

    def set_geometry(self, component: Component) -> None:
        """Set the gdsfactory component for simulation.

        Args:
            component: gdsfactory Component to simulate

        Example:
            >>> sim.set_geometry(my_component)
        """
        from gsim.common import Geometry

        self.geometry = Geometry(component=component)

    @property
    def component(self) -> Component | None:
        """Get the current component."""
        return self.geometry.component if self.geometry else None

    # -------------------------------------------------------------------------
    # Stack methods
    # -------------------------------------------------------------------------

    def set_stack(
        self,
        *,
        yaml_path: str | Path | None = None,
        air_above: float = 1.0,
        substrate_thickness: float = 2.0,
        include_substrate: bool = False,
        **kwargs: Any,
    ) -> None:
        """Configure the layer stack.

        If yaml_path is provided, loads stack from YAML file.
        Otherwise, extracts from active PDK with given parameters.

        Note: Default air_above is 1.0 um (photonic), vs 200.0 um for RF.

        Args:
            yaml_path: Path to custom YAML stack file
            air_above: Air box height above top metal in um
            substrate_thickness: Thickness below z=0 in um
            include_substrate: Include lossy silicon substrate
            **kwargs: Additional args passed to extract_layer_stack

        Example:
            >>> sim.set_stack(air_above=2.0)
        """
        self._stack_kwargs = {
            "yaml_path": yaml_path,
            "air_above": air_above,
            "substrate_thickness": substrate_thickness,
            "include_substrate": include_substrate,
            **kwargs,
        }
        # Stack will be resolved lazily
        self.stack = None

    # -------------------------------------------------------------------------
    # Z-crop
    # -------------------------------------------------------------------------

    def set_z_crop(
        self,
        *,
        reference_layer: str | None = None,
    ) -> None:
        """Crop the layer stack along z using ``domain_config`` margins.

        Keeps materials within ``[ref.zmin - margin_z_below, ref.zmax + margin_z_above]``
        around the reference layer, then clips layers that partially overlap.
        Must be called after ``set_stack()``.

        The crop window is derived from ``domain_config`` (set via
        ``set_domain()``).  Call ``set_domain()`` *before* ``set_z_crop()``
        if you need non-default values.

        By default, auto-detects the core layer (highest refractive index).

        Args:
            reference_layer: Explicit layer name to crop around. If None,
                auto-detects the layer with the highest refractive index.

        Raises:
            ValueError: If stack is not configured or reference layer not found.

        Example:
            >>> sim.set_stack()
            >>> sim.set_z_crop()  # keeps margin_z_above/below of material around core
        """
        from gsim.common.stack.extractor import Layer, LayerStack
        from gsim.common.stack.materials import get_material_properties
        from gsim.meep.models import DomainConfig

        if self.stack is None:
            if self._stack_kwargs:
                self._resolve_stack()
            else:
                raise ValueError(
                    "No stack configured. Call set_stack() before set_z_crop()."
                )

        assert self.stack is not None  # for type checker

        # Find reference layer
        ref: Layer | None = None
        if reference_layer is not None:
            if reference_layer not in self.stack.layers:
                raise ValueError(
                    f"Layer '{reference_layer}' not found. "
                    f"Available: {list(self.stack.layers.keys())}"
                )
            ref = self.stack.layers[reference_layer]
        else:
            # Auto-detect: highest refractive index (same logic as port z-center)
            best_n = 0.0
            for layer in self.stack.layers.values():
                props = get_material_properties(layer.material)
                if (
                    props is not None
                    and props.refractive_index is not None
                    and props.refractive_index > best_n
                ):
                    best_n = props.refractive_index
                    ref = layer

            if ref is None or best_n <= 1.5:
                raise ValueError(
                    "Could not auto-detect core layer (no layer with n > 1.5). "
                    "Specify reference_layer explicitly."
                )

        # Use domain_config to determine how much material to keep
        dcfg: DomainConfig = getattr(self, "domain_config", DomainConfig())
        z_lo = ref.zmin - dcfg.margin_z_below
        z_hi = ref.zmax + dcfg.margin_z_above

        # Filter and clip layers
        cropped: dict[str, Layer] = {}
        for name, layer in self.stack.layers.items():
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

        # Crop dielectrics list too
        cropped_dielectrics = []
        for diel in self.stack.dielectrics:
            if diel["zmax"] <= z_lo or diel["zmin"] >= z_hi:
                continue
            cropped_dielectrics.append({
                **diel,
                "zmin": max(diel["zmin"], z_lo),
                "zmax": min(diel["zmax"], z_hi),
            })

        self.stack = LayerStack(
            pdk_name=self.stack.pdk_name,
            units=self.stack.units,
            layers=cropped,
            materials=self.stack.materials,
            dielectrics=cropped_dielectrics,
            simulation=self.stack.simulation,
        )

        print(
            f"Z-crop around '{ref.name}' [{ref.zmin:.2f}, {ref.zmax:.2f}] um: "
            f"window [{z_lo:.2f}, {z_hi:.2f}], "
            f"{len(cropped)} layers kept"
        )

    # -------------------------------------------------------------------------
    # Material methods
    # -------------------------------------------------------------------------

    def set_material(
        self,
        name: str,
        *,
        refractive_index: float | None = None,
        extinction_coeff: float | None = None,
    ) -> None:
        """Override or add optical material properties.

        Args:
            name: Material name (e.g., "si", "SiO2")
            refractive_index: Refractive index (n)
            extinction_coeff: Extinction coefficient (k)

        Example:
            >>> sim.set_material("si", refractive_index=3.47)
            >>> sim.set_material("SiO2", refractive_index=1.44)
        """
        from gsim.common.stack.materials import MaterialProperties

        self.materials[name] = MaterialProperties(
            type="dielectric",
            refractive_index=refractive_index,
            extinction_coeff=extinction_coeff,
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _resolve_stack(self) -> LayerStack:
        """Resolve the layer stack from PDK or YAML.

        Returns:
            LayerStack object
        """
        from gsim.common.stack import get_stack

        yaml_path = self._stack_kwargs.pop("yaml_path", None)
        legacy_stack = get_stack(yaml_path=yaml_path, **self._stack_kwargs)

        # Restore yaml_path for potential re-resolution
        self._stack_kwargs["yaml_path"] = yaml_path

        self.stack = legacy_stack
        return legacy_stack

    # -------------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------------

    def show_stack(self) -> None:
        """Print the layer stack table.

        Example:
            >>> sim.show_stack()
        """
        from gsim.common.stack import print_stack_table

        if self.stack is None:
            self._resolve_stack()

        if self.stack is not None:
            print_stack_table(self.stack)

    def plot_stack(self) -> None:
        """Plot the layer stack visualization.

        Example:
            >>> sim.plot_stack()
        """
        from gsim.common.stack import plot_stack

        if self.stack is None:
            self._resolve_stack()

        if self.stack is not None:
            plot_stack(self.stack)

    # -------------------------------------------------------------------------
    # Visualization (via common viz)
    # -------------------------------------------------------------------------

    def _build_geometry_model(self) -> GeometryModel:
        """Build a GeometryModel from the current component + stack.

        Uses the gdsfactory LayerStack from the active PDK (not the gsim
        LayerStack stored on ``self.stack``), because
        ``LayeredComponentBase`` needs gdsfactory's ``LayerStack`` for
        polygon extraction via ``DerivedLayer.get_shapes()``.

        Returns:
            GeometryModel ready for visualization.

        Raises:
            ValueError: If no geometry is configured or no active PDK.
        """
        import gdsfactory as gf

        from gsim.common.geometry_model import extract_geometry_model
        from gsim.common.layered_component import LayeredComponentBase

        if self.geometry is None:
            raise ValueError("No geometry set. Call set_geometry(component) first.")

        pdk = gf.get_active_pdk()
        gf_layer_stack = pdk.layer_stack
        if gf_layer_stack is None:
            raise ValueError(
                "Active PDK has no layer_stack. "
                "Activate a PDK with a layer stack first."
            )

        lc = LayeredComponentBase(
            component=self.geometry.component,
            layer_stack=gf_layer_stack,
        )
        gm = extract_geometry_model(lc)

        # If the stack has been z-cropped, clip the geometry model to match
        if self.stack is not None:
            gm = self._crop_geometry_model(gm)

        return gm

    def _crop_geometry_model(self, gm: GeometryModel) -> GeometryModel:
        """Clip a GeometryModel's prisms and bbox to self.stack z-range."""
        from gsim.common.geometry_model import GeometryModel, Prism

        z_lo = min(l.zmin for l in self.stack.layers.values())
        z_hi = max(l.zmax for l in self.stack.layers.values())

        cropped_prisms: dict[str, list[Prism]] = {}
        for layer_name, prism_list in gm.prisms.items():
            clipped = []
            for p in prism_list:
                if p.z_top <= z_lo or p.z_base >= z_hi:
                    continue
                clipped.append(
                    Prism(
                        vertices=p.vertices,
                        z_base=max(p.z_base, z_lo),
                        z_top=min(p.z_top, z_hi),
                        layer_name=p.layer_name,
                        material=p.material,
                        sidewall_angle=p.sidewall_angle,
                        original_polygon=p.original_polygon,
                    )
                )
            if clipped:
                cropped_prisms[layer_name] = clipped

        # Recompute bbox
        old_min, old_max = gm.bbox
        new_bbox = (
            (old_min[0], old_min[1], z_lo),
            (old_max[0], old_max[1], z_hi),
        )

        return GeometryModel(
            prisms=cropped_prisms,
            bbox=new_bbox,
            layer_bboxes=gm.layer_bboxes,
            layer_mesh_orders=gm.layer_mesh_orders,
        )

    def plot_3d(self, backend: str = "open3d", **kwargs: Any) -> Any:
        """Create interactive 3D visualisation of the geometry.

        Args:
            backend: "open3d" (Jupyter/VS Code) or "pyvista" (desktop).
            **kwargs: Extra args forwarded to the backend renderer.

        Example:
            >>> sim.plot_3d()
        """
        from gsim.common.viz import plot_prisms_3d, plot_prisms_3d_open3d

        gm = self._build_geometry_model()
        if backend == "pyvista":
            return plot_prisms_3d(gm, **kwargs)
        if backend == "open3d":
            return plot_prisms_3d_open3d(gm, **kwargs)
        raise ValueError(f"Unsupported backend: {backend}. Use 'open3d' or 'pyvista'")

    def _build_overlay(self, geometry_model: GeometryModel) -> Any:
        """Build a SimOverlay from current config, if available.

        Returns SimOverlay or None if ports/stack aren't configured yet.
        """
        from gsim.meep.models import DomainConfig
        from gsim.meep.overlay import build_sim_overlay

        if self.geometry is None:
            return None

        if self.stack is None and self._stack_kwargs:
            self._resolve_stack()

        if self.stack is None:
            return None

        try:
            from gsim.meep.ports import extract_port_info

            component = self.geometry.component.copy()
            port_data = extract_port_info(
                component, self.stack, source_port=getattr(self, "source_port", None)
            )
        except Exception:
            port_data = []

        domain_config = getattr(self, "domain_config", DomainConfig())
        dielectrics = self.stack.dielectrics if self.stack else []
        return build_sim_overlay(
            geometry_model, domain_config, port_data, dielectrics=dielectrics
        )

    def plot_2d(
        self,
        x: float | str | None = None,
        y: float | str | None = None,
        z: float | str = "core",
        ax: plt.Axes | None = None,
        legend: bool = True,
        slices: str = "z",
    ) -> plt.Axes | None:
        """Plot 2D cross-sections of the geometry.

        Args:
            x: X-coordinate or layer name for the slice plane.
            y: Y-coordinate or layer name for the slice plane.
            z: Z-coordinate or layer name for the slice plane.
            ax: Axes to draw on.  If ``None``, a new figure is created.
            legend: Whether to show the legend.
            slices: Slice direction(s) -- "x", "y", "z", or combinations.

        Returns:
            ``plt.Axes`` when *ax* was provided, otherwise ``None``.

        Example:
            >>> sim.plot_2d(slices="xyz")
        """
        from gsim.common.viz import plot_prism_slices

        gm = self._build_geometry_model()
        overlay = self._build_overlay(gm)
        return plot_prism_slices(gm, x, y, z, ax, legend, slices, overlay=overlay)

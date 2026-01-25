"""Legacy FDTD functions.

.. deprecated::
    These functions are deprecated and will be removed in a future version.
    Use the new modular FDTDSimulation class instead.
"""

from __future__ import annotations

import pathlib
import time
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td
from gdsfactory.component import Component
from gdsfactory.pdk import get_layer_stack
from gdsfactory.technology import LayerStack
from pydantic import NonNegativeFloat
from tidy3d.components.types import Symmetry

from gsim.fdtd.geometry import Geometry
from gsim.fdtd.materials import Tidy3DElementMapping, Tidy3DMedium, material_name_to_medium
from gsim.fdtd.materials.types import Sparameters
from gsim.fdtd.util import get_mode_solvers

PathType = pathlib.Path | str

home = pathlib.Path.home()
dirpath_default = home / ".gdsfactory" / "sparameters"


def write_sparameters(
    component: Component,
    layer_stack: LayerStack | None = None,
    material_mapping: dict[str, Tidy3DMedium] = material_name_to_medium,
    extend_ports: NonNegativeFloat = 0.5,
    port_offset: float = 0.2,
    pad_xy_inner: NonNegativeFloat = 2.0,
    pad_xy_outer: NonNegativeFloat = 2.0,
    pad_z_inner: float = 0.0,
    pad_z_outer: NonNegativeFloat = 0.0,
    dilation: float = 0.0,
    wavelength: float = 1.55,
    bandwidth: float = 0.2,
    num_freqs: int = 21,
    min_steps_per_wvl: int = 30,
    center_z: float | str | None = None,
    sim_size_z: float = 4.0,
    port_size_mult: float | tuple[float, float] = (4.0, 3.0),
    run_only: tuple[tuple[str, int], ...] | None = None,
    element_mappings: Tidy3DElementMapping = (),
    extra_monitors: tuple[Any, ...] | None = None,
    mode_spec: td.ModeSpec = td.ModeSpec(num_modes=1, filter_pol="te"),
    boundary_spec: td.BoundarySpec = td.BoundarySpec.all_sides(boundary=td.PML()),
    symmetry: tuple[Symmetry, Symmetry, Symmetry] = (0, 0, 0),
    run_time: float = 1e-12,
    shutoff: float = 1e-5,
    folder_name: str = "default",
    dirpath: PathType = dirpath_default,
    verbose: bool = True,
    plot_simulation_layer_name: str | None = None,
    plot_simulation_port_index: int = 0,
    plot_simulation_z: float | None = None,
    plot_simulation_x: float | None = None,
    plot_mode_index: int | None = 0,
    plot_mode_port_name: str | None = None,
    plot_epsilon: bool = False,
    filepath: PathType | None = None,
    overwrite: bool = False,
    **kwargs: Any,
) -> Sparameters:
    """Writes the S-parameters for a component.

    .. deprecated::
        This function represents the legacy monolithic approach.
        Use the new modular FDTDSimulation class instead:

        ```python
        from gsim.fdtd import FDTDSimulation, Geometry, Material, Physics

        sim = FDTDSimulation()
        sim.geometry = Geometry(component=component, layer_stack=layer_stack)
        sim.material = Material(mapping=material_mapping)
        result = sim.run()
        ```

        This function will be removed in a future version.

    Args:
        component: gdsfactory component to write the S-parameters for.
        layer_stack: The layer stack for the component.
        material_mapping: A mapping of material names to Tidy3DMedium instances.
        extend_ports: The extension length for ports.
        port_offset: The offset for ports.
        pad_xy_inner: The inner padding in the xy-plane.
        pad_xy_outer: The outer padding in the xy-plane.
        pad_z_inner: The inner padding in the z-direction.
        pad_z_outer: The outer padding in the z-direction.
        dilation: Dilation of the polygon.
        wavelength: The wavelength for the ComponentModeler.
        bandwidth: The bandwidth for the ComponentModeler.
        num_freqs: The number of frequencies for the ComponentModeler.
        min_steps_per_wvl: The minimum number of steps per wavelength.
        center_z: The z-coordinate for the center.
        sim_size_z: simulation size um in the z-direction.
        port_size_mult: The size multiplier for the ports.
        run_only: The run only specification.
        element_mappings: The element mappings.
        extra_monitors: The extra monitors.
        mode_spec: The mode specification.
        boundary_spec: The boundary specification.
        symmetry: The symmetry for the simulation.
        run_time: The run time.
        shutoff: The shutoff value.
        folder_name: The folder name.
        dirpath: Optional directory path for writing the Sparameters.
        verbose: Whether to print verbose output.
        plot_simulation_layer_name: Optional layer name to plot.
        plot_simulation_port_index: which port index to plot.
        plot_simulation_z: which z coordinate to plot.
        plot_simulation_x: which x coordinate to plot.
        plot_mode_index: which mode index to plot.
        plot_mode_port_name: which port name to plot.
        plot_epsilon: whether to plot epsilon.
        filepath: Optional file path for the S-parameters.
        overwrite: Whether to overwrite existing S-parameters.
        kwargs: Additional keyword arguments.

    Returns:
        Dictionary of S-parameters.
    """
    warnings.warn(
        "write_sparameters is deprecated. Use FDTDSimulation class instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    layer_stack = layer_stack or get_layer_stack()

    c = Geometry(
        component=component,
        layer_stack=layer_stack,
        material_mapping=material_mapping,
        extend_ports=extend_ports,
        port_offset=port_offset,
        pad_xy_inner=pad_xy_inner,
        pad_xy_outer=pad_xy_outer,
        pad_z_inner=pad_z_inner,
        pad_z_outer=pad_z_outer,
        dilation=dilation,
    )

    modeler = c.get_component_modeler(
        wavelength=wavelength,
        bandwidth=bandwidth,
        num_freqs=num_freqs,
        min_steps_per_wvl=min_steps_per_wvl,
        center_z=center_z,
        sim_size_z=sim_size_z,
        port_size_mult=port_size_mult,
        run_only=run_only,
        element_mappings=element_mappings,
        extra_monitors=extra_monitors,
        mode_spec=mode_spec,
        boundary_spec=boundary_spec,
        run_time=run_time,
        shutoff=shutoff,
        folder_name=folder_name,
        verbose=verbose,
        symmetry=symmetry,
        **kwargs,
    )

    path_dir = pathlib.Path(dirpath) / modeler._hash_self()
    modeler = modeler.updated_copy(path_dir=str(path_dir))

    sp = {}

    if plot_simulation_layer_name or plot_simulation_z or plot_simulation_x:
        if plot_simulation_layer_name is None and plot_simulation_z is None:
            raise ValueError(
                "You need to specify plot_simulation_z or plot_simulation_layer_name"
            )
        z = plot_simulation_z or c.get_layer_center(plot_simulation_layer_name)[2]
        x = plot_simulation_x or c.ports[plot_simulation_port_index].dcenter[0]

        modeler = c.get_component_modeler(
            center_z=plot_simulation_layer_name,
            port_size_mult=port_size_mult,
            sim_size_z=sim_size_z,
        )
        _, ax = plt.subplots(2, 1)
        if plot_epsilon:
            modeler.plot_sim_eps(z=z, ax=ax[0])
            modeler.plot_sim_eps(x=x, ax=ax[1])

        else:
            modeler.plot_sim(z=z, ax=ax[0])
            modeler.plot_sim(x=x, ax=ax[1])
        plt.show()
        return sp

    elif plot_mode_index is not None and plot_mode_port_name:
        modes = get_mode_solvers(modeler, port_name=plot_mode_port_name)
        mode_solver = modes[f"smatrix_{plot_mode_port_name}_{plot_mode_index}"]
        mode_data = mode_solver.solve()

        _, ax = plt.subplots(1, 3, tight_layout=True, figsize=(10, 3))
        abs(mode_data.Ex.isel(mode_index=plot_mode_index, f=0)).plot(
            x="y", y="z", ax=ax[0], cmap="magma"
        )
        abs(mode_data.Ey.isel(mode_index=plot_mode_index, f=0)).plot(
            x="y", y="z", ax=ax[1], cmap="magma"
        )
        abs(mode_data.Ez.isel(mode_index=plot_mode_index, f=0)).plot(
            x="y", y="z", ax=ax[2], cmap="magma"
        )
        ax[0].set_title("|Ex(x, y)|")
        ax[1].set_title("|Ey(x, y)|")
        ax[2].set_title("|Ez(x, y)|")
        plt.setp(ax, aspect="equal")
        plt.show()
        return sp

    dirpath = pathlib.Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    filepath = filepath or dirpath / f"{modeler._hash_self()}.npz"
    filepath = pathlib.Path(filepath)
    if filepath.suffix != ".npz":
        filepath = filepath.with_suffix(".npz")

    if filepath.exists() and not overwrite:
        print(f"Simulation loaded from {filepath!r}")
        return dict(np.load(filepath))
    else:
        time.sleep(0.2)
        s = modeler.run()
        for port_in in s.port_in.values:
            for port_out in s.port_out.values:
                for mode_index_in in s.mode_index_in.values:
                    for mode_index_out in s.mode_index_out.values:
                        sp[f"{port_in}@{mode_index_in},{port_out}@{mode_index_out}"] = (
                            s.sel(
                                port_in=port_in,
                                port_out=port_out,
                                mode_index_in=mode_index_in,
                                mode_index_out=mode_index_out,
                            ).values
                        )

        frequency = s.f.values
        sp["wavelengths"] = td.constants.C_0 / frequency
        np.savez_compressed(filepath, **sp)
        print(f"Simulation saved to {filepath!r}")
        return sp

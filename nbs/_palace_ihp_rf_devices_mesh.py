# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Palace Mesh Generation: IHP RF Devices
#
# This notebook discovers RF components from `ihp.cells.rf_devices`, prints their ports, and generates Palace meshes for each component.
#
# **Requirements:**
# - IHP PDK from GitHub (`pip install git+https://github.com/gdsfactory/IHP.git`)
# - gsim environment with Palace meshing dependencies

# %%
from __future__ import annotations

from pathlib import Path

import ihp.cells.rf_devices as rf_devices
import pandas as pd
from ihp import PDK

from gsim.common.stack import get_stack
from gsim.palace import DrivenSim

PDK.activate()

# %%
# Discover RF component factories defined in ihp.cells.rf_devices and registered in the active PDK.
rf_module_factories = {
    name
    for name, obj in vars(rf_devices).items()
    if callable(obj) and not name.startswith("_")
}

rf_keywords = (
    "wilkinson",
    "filter",
    "coupler",
    "divider",
    "directional",
    "hybrid",
    "branch",
    "transformer",
)

component_names = sorted(
    name
    for name in PDK.cells
    if name in rf_module_factories and any(k in name.lower() for k in rf_keywords)
)

print("Discovered RF components:")
for name in component_names:
    print(f"- {name}")

if not component_names:
    raise RuntimeError(
        "No RF device components were discovered in ihp.cells.rf_devices"
    )

# %%
# Build each component and tabulate ports.
port_rows = []
components = {}

for name in component_names:
    c = PDK.cells[name]()
    components[name] = c
    port_rows.extend(
        {
            "component": name,
            "port": p.name,
            "orientation_deg": p.orientation,
            "width_um": p.width,
            "center_um": tuple(float(x) for x in p.center),
            "layer": str(p.layer),
        }
        for p in c.ports
    )

ports_df = (
    pd.DataFrame(port_rows).sort_values(["component", "port"]).reset_index(drop=True)
)
ports_df

# %%
# Generate a mesh per component under ./palace-sim-ihp-rf/<component_name>.
stack = get_stack()
base_dir = Path("./palace-sim-ihp-rf")
base_dir.mkdir(parents=True, exist_ok=True)

mesh_rows = []

for name, c in components.items():
    out_dir = base_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        sim = DrivenSim()
        sim.set_output_dir(out_dir)
        sim.set_geometry(c)
        sim.set_stack(stack)

        # Add all component ports as via ports so port geometries are meshed.
        for port in c.ports:
            sim.add_port(
                port.name,
                from_layer="metal3",
                to_layer="topmetal2",
                geometry="via",
            )

        sim.set_driven(fmin=1e9, fmax=110e9, num_points=21)
        result = sim.mesh(preset="default")

        mesh_rows.append(
            {
                "component": name,
                "status": "ok",
                "mesh_path": str(result.mesh_path),
            }
        )
    except Exception as e:
        mesh_rows.append(
            {
                "component": name,
                "status": "error",
                "mesh_path": "",
                "error": str(e),
            }
        )

mesh_df = (
    pd.DataFrame(mesh_rows).sort_values(["status", "component"]).reset_index(drop=True)
)
mesh_df

# %%
from IPython.display import Markdown, display

from gsim.viz import plot_mesh

# Prefer an inline interactive backend in notebooks when available.
try:
    import pyvista as pv

    pv.set_jupyter_backend("trame")
except Exception:
    pass

transparent_groups = ["sin__None", "si__None", "Absorbing_boundary"]

shown = 0
for row in mesh_df.to_dict(orient="records"):
    if row.get("status") != "ok":
        continue

    component = str(row["component"])
    msh_path = Path(row["mesh_path"])

    if not msh_path.exists():
        continue

    shown += 1
    display(Markdown(f"### {component}"))
    plot_mesh(
        msh_path,
        interactive=True,
        style="solid",
        transparent_groups=transparent_groups,
    )

if shown == 0:
    raise RuntimeError(
        "No successful meshes found to display. Run the meshing cell first."
    )

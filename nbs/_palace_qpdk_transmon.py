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

# %%
import gdsfactory as gf
from qpdk import PDK, cells
from qpdk.tech import LAYER
from qpdk.utils import apply_additive_metals

PDK.activate()


@gf.cell
def transmon_component() -> gf.Component:
    """Create a qubit with resonator layout."""
    c = gf.Component()

    ref = c << cells.transmon_with_resonator(
        qubit="double_pad_transmon_with_bbox",
        resonator_length=5000.0,
        resonator_meanders=5,
        qubit_rotation=90,
    )
    c.add_ports(ref.ports)

    # Add simulation area around the component
    c.kdb_cell.shapes(LAYER.SIM_AREA).insert(c.bbox().enlarged(100, 100))

    return c


component = transmon_component()
_c = component.copy()
_c.draw_ports()
_c

# %% [markdown]
# ### Inspect raw layers and apply additive metals

# %%
# inspect_layers(component, filename="transmon_raw_layers.png")

# Apply additive metals processing (QPDK-specific step)
processed = apply_additive_metals(component.copy())

# inspect_layers(processed, filename="transmon_processed_layers.png")

# %% [markdown]
# ### Convert QPDK etch layers to conductor geometry

# %%
import warnings

import klayout.db as kdb
from qpdk.tech import LAYER as QPDK_LAYER

from gsim.common.polygon_utils import decimate

sim_area_layer = (QPDK_LAYER.SIM_AREA[0], QPDK_LAYER.SIM_AREA[1])
etch_layer = (QPDK_LAYER.M1_ETCH[0], QPDK_LAYER.M1_ETCH[1])

CPW_LAYERS = {"SUBSTRATE": (1, 0), "SUPERCONDUCTOR": (2, 0), "VACUUM": (3, 0)}

layout = processed.kdb_cell.layout()
sim_region = kdb.Region(
    processed.kdb_cell.begin_shapes_rec(layout.layer(*sim_area_layer))
)
etch_region = kdb.Region(processed.kdb_cell.begin_shapes_rec(layout.layer(*etch_layer)))

etch_polys = decimate(list(etch_region.each()))
etch_region = kdb.Region()
for poly in etch_polys:
    etch_region.insert(poly)

if sim_region.is_empty():
    warnings.warn("No polygons found on SIM_AREA", stacklevel=2)
if etch_region.is_empty():
    warnings.warn("No polygons found on M1_ETCH", stacklevel=2)

conductor_region = sim_region - etch_region

etched = gf.Component("etched_component")
el = etched.kdb_cell.layout()
for name, region in [
    ("SUPERCONDUCTOR", conductor_region),
    ("SUBSTRATE", sim_region),
    ("VACUUM", sim_region),
]:
    idx = el.layer(*CPW_LAYERS[name])
    etched.kdb_cell.shapes(idx).insert(region)

for port in processed.ports:
    etched.add_port(name=port.name, port=port)

# inspect_layers(etched, filename="transmon_etched_layers.png")
etched

# %% [markdown]
# ### Configure Simulation

# %%
from gsim.common.stack import Layer, LayerStack
from gsim.common.stack.materials import MATERIALS_DB
from gsim.palace import DrivenSim

# Build a CPW stack matching the etched component layers
substrate_thickness = 500
vacuum_thickness = 500

stack = LayerStack(pdk_name="qpdk")
stack.layers["SUBSTRATE"] = Layer(
    name="SUBSTRATE",
    gds_layer=(1, 0),
    zmin=0.0,
    zmax=substrate_thickness,
    thickness=substrate_thickness,
    material="sapphire",
    layer_type="dielectric",
)
stack.layers["SUPERCONDUCTOR"] = Layer(
    name="SUPERCONDUCTOR",
    gds_layer=(2, 0),
    zmin=substrate_thickness,
    zmax=substrate_thickness,
    thickness=0,
    material="aluminum",
    layer_type="conductor",
)
stack.layers["VACUUM"] = Layer(
    name="VACUUM",
    gds_layer=(3, 0),
    zmin=substrate_thickness,
    zmax=substrate_thickness + vacuum_thickness,
    thickness=vacuum_thickness,
    material="vacuum",
    layer_type="dielectric",
)
stack.dielectrics = [
    {
        "name": "substrate",
        "zmin": 0.0,
        "zmax": substrate_thickness,
        "material": "sapphire",
    },
    {
        "name": "vacuum",
        "zmin": substrate_thickness,
        "zmax": substrate_thickness + vacuum_thickness,
        "material": "vacuum",
    },
]
stack.materials = {
    "sapphire": MATERIALS_DB["sapphire"].to_dict(),
    "aluminum": MATERIALS_DB["aluminum"].to_dict(),
    "vacuum": MATERIALS_DB["vacuum"].to_dict(),
}

sim = DrivenSim()
sim.set_geometry(etched)
sim.set_stack(stack)

# %% [markdown]
# ### Configure eigenmode simulation
#
# The junction port is modelled as a lumped element with a 10 nH inductance.

# %%
# Junction port with 10 nH inductance
sim.add_port("junction", layer="SUPERCONDUCTOR", length=5.0, inductance=10e-9)

# CPW feed ports
sim.add_cpw_port(
    "o1", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, length=5.0, offset=-30
)

sim.set_driven(fmin=7.75e9, fmax=7.8e9, num_points=100)

# %% [markdown]
# ### Mesh and run

# %%
sim.set_output_dir("./sim_qpdk_qubit_resonator")
sim.mesh(preset="fine", margin=0)

print(sim.validate_mesh())

# %%
sim.plot_mesh(
    style="solid",
    interactive=True,
    transparent_groups=["vacuum__None", "sapphire__None"],
)

# %%
print(sim.write_config())
results = sim.run()

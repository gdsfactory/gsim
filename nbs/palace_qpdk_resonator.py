# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% papermill={"duration": 3.114078, "end_time": "2026-04-04T11:56:45.509802", "exception": false, "start_time": "2026-04-04T11:56:42.395724", "status": "completed"}
import gdsfactory as gf
from qpdk import PDK, cells
from qpdk.cells.airbridge import cpw_with_airbridges
from qpdk.tech import LAYER, route_bundle_sbend_cpw

PDK.activate()


@gf.cell
def resonator_compact(coupling_gap: float = 20.0) -> gf.Component:
    """Compact coupled resonator with S-bend CPW routes."""
    c = gf.Component()

    res = c << cells.resonator_coupled(
        coupling_straight_length=300, coupling_gap=coupling_gap
    )
    res.movex(-res.size_info.width / 4)

    left = c << cells.straight()
    right = c << cells.straight()

    w = res.size_info.width + 100
    left.move((-w, 0))
    right.move((w, 0))

    route_bundle_sbend_cpw(
        c,
        [left["o2"], right["o1"]],
        [res["coupling_o1"], res["coupling_o2"]],
        cross_section=cpw_with_airbridges(
            airbridge_spacing=250.0, airbridge_padding=20.0
        ),
    )

    c.kdb_cell.shapes(LAYER.SIM_AREA).insert(c.bbox().enlarged(0, 100))

    c.add_port(name="o1", port=left["o1"])
    c.add_port(name="o2", port=right["o2"])
    return c


component = resonator_compact(coupling_gap=15.0)
_c = component.copy()
_c.draw_ports()
_c

# %% [markdown] papermill={"duration": 0.005219, "end_time": "2026-04-04T11:56:42.391968", "exception": false, "start_time": "2026-04-04T11:56:42.386749", "status": "completed"}
# ### QPDK Coupled Resonator — Driven Simulation
#
# This notebook builds a compact coupled resonator from QPDK cells, converts etch
# layers to conductor geometry, and runs a Palace driven simulation to extract
# S-parameters and visualize the electric field at resonance (~7.78 GHz).
#
# [Palace](https://awslabs.github.io/palace/) is an open-source 3D electromagnetic simulator supporting eigenmode, driven (S-parameter), and electrostatic simulations.
#
# **Requirements:**
#
# - Quantum PDK: `uv pip install qpdk`
# - [GDSFactory+](https://gdsfactory.com) account for cloud simulation

# %% [markdown] papermill={"duration": 0.001139, "end_time": "2026-04-04T11:56:45.512022", "exception": false, "start_time": "2026-04-04T11:56:45.510883", "status": "completed"}
# ### Convert QPDK etch layers to conductor geometry

# %% papermill={"duration": 0.09865, "end_time": "2026-04-04T11:56:45.611594", "exception": false, "start_time": "2026-04-04T11:56:45.512944", "status": "completed"}
import warnings

import klayout.db as kdb
from qpdk.tech import LAYER as QPDK_LAYER

from gsim.common.polygon_utils import decimate

sim_area_layer = (QPDK_LAYER.SIM_AREA[0], QPDK_LAYER.SIM_AREA[1])
etch_layer = (QPDK_LAYER.M1_ETCH[0], QPDK_LAYER.M1_ETCH[1])

CPW_LAYERS = {"SUBSTRATE": (1, 0), "SUPERCONDUCTOR": (2, 0), "VACUUM": (3, 0)}

layout = component.kdb_cell.layout()
sim_region = kdb.Region(
    component.kdb_cell.begin_shapes_rec(layout.layer(*sim_area_layer))
)
etch_region = kdb.Region(component.kdb_cell.begin_shapes_rec(layout.layer(*etch_layer)))

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

for port in component.ports:
    etched.add_port(name=port.name, port=port)

etched

# %% [markdown] papermill={"duration": 0.001001, "end_time": "2026-04-04T11:56:45.614228", "exception": false, "start_time": "2026-04-04T11:56:45.613227", "status": "completed"}
# ### Configure simulation

# %% papermill={"duration": 0.614954, "end_time": "2026-04-04T11:56:46.230114", "exception": false, "start_time": "2026-04-04T11:56:45.615160", "status": "completed"}
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
sim.add_cpw_port("o1", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, offset=2.5)
sim.add_cpw_port("o2", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, offset=2.5)
sim.set_driven(fmin=7.75e9, fmax=7.8e9, num_points=300, save_fields_at=[7.78e9])

# %% papermill={"duration": 7.168132, "end_time": "2026-04-04T11:56:53.399620", "exception": false, "start_time": "2026-04-04T11:56:46.231488", "status": "completed"}
sim.set_output_dir("./sim_qpdk_resonator")
sim.mesh(preset="default")
sim.plot_mesh(show_groups=["superconductor", "P", "sapphire", "vacuum"])

# %% papermill={"duration": 536.401462, "end_time": "2026-04-04T12:05:49.805289", "exception": false, "start_time": "2026-04-04T11:56:53.403827", "status": "completed"}
sim.write_config()
results = sim.run()

# %% papermill={"duration": 0.186505, "end_time": "2026-04-04T12:05:49.995902", "exception": false, "start_time": "2026-04-04T12:05:49.809397", "status": "completed"}
results.plot_interactive()

# %% papermill={"duration": 0.016226, "end_time": "2026-04-04T12:05:50.015886", "exception": false, "start_time": "2026-04-04T12:05:49.999660", "status": "completed"}
results.plot_interactive(phase=True)

# %% [markdown] papermill={"duration": 0.01041, "end_time": "2026-04-04T12:05:50.033797", "exception": false, "start_time": "2026-04-04T12:05:50.023387", "status": "completed"}
# ### Field visualization — top view at conductor layer

# %% papermill={"duration": 0.866275, "end_time": "2026-04-04T12:05:50.904463", "exception": false, "start_time": "2026-04-04T12:05:50.038188", "status": "completed"}
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.interpolate import griddata

pv.OFF_SCREEN = True

results_dir = Path(results.files["port-S.csv"]).parent

# Find the resonance frequency from S21 minimum
res_idx = np.argmin(results.s21.db)
freq_ghz = results.freq[res_idx]
print(f"Resonance at {freq_ghz:.4f} GHz (S21 = {results.s21.db[res_idx]:.1f} dB)")

# Extract quality factor from the resonance dip
s21_db = results.s21.db
s21_mag = 10 ** (s21_db / 20.0)

# Method 1: -3 dB bandwidth (half-power points)
s21_min_mag = s21_mag[res_idx]
baseline = np.median(s21_mag)  # transmission away from resonance
dip_depth = baseline - s21_min_mag
half_depth = s21_min_mag + dip_depth / 2
half_power_db = 20 * np.log10(half_depth)

# Find the left and right half-power indices
left_idx = np.argmin(np.abs(s21_db[:res_idx] - half_power_db))
right_idx = res_idx + np.argmin(np.abs(s21_db[res_idx:] - half_power_db))

f_left = results.freq[left_idx]
f_right = results.freq[right_idx]
bw_ghz = f_right - f_left

Q_bw = freq_ghz / bw_ghz if bw_ghz > 0 else np.inf
print(f"-3 dB bandwidth: {bw_ghz:.6f} GHz")
print(f"Loaded quality factor Q ~ {Q_bw:.0f}")

# Method 2: Lorentzian fit to |S21| for refined Q estimate
from scipy.optimize import curve_fit


def lorentzian_dip(f, f0, Q, dip_depth, baseline):
    """Lorentzian dip: baseline - dip_depth / (1 + (2*Q*(f-f0)/f0)**2)"""
    return baseline - dip_depth / (1 + (2 * Q * (f - f0) / f0) ** 2)


freq = results.freq
mask = np.abs(freq - freq_ghz) < 0.01  # ±10 MHz fit window
if mask.sum() > 5:
    popt, pcov = curve_fit(
        lorentzian_dip,
        freq[mask],
        s21_mag[mask],
        p0=[freq_ghz, Q_bw, dip_depth, baseline],
        bounds=(
            [freq_ghz - 0.01, 100, 0, 0],
            [freq_ghz + 0.01, 1e6, 2 * baseline, 2 * baseline],
        ),
    )
    f0_fit, Q_fit, dip_fit, base_fit = popt
    Q_err = np.sqrt(np.diag(pcov))[1]
    print(f"Lorentzian fit: f0 = {f0_fit:.4f} GHz, Q = {Q_fit:.0f} ± {Q_err:.0f}")
else:
    print("Insufficient points for Lorentzian fit.")

# Load volume field data
vol_dir = results_dir / "paraview/driven/excitation_1"
vol_path = sorted(vol_dir.rglob("*.pvtu"))[-1]
vol = pv.read(str(vol_path))
print(f"Volume: {vol.n_points:,} points, {vol.n_cells:,} cells")

# %% papermill={"duration": 0.310541, "end_time": "2026-04-04T12:05:51.221344", "exception": false, "start_time": "2026-04-04T12:05:50.910803", "status": "completed"}
import matplotlib.pyplot as plt

# Slice volume at conductor plane (z = substrate_thickness)
vol_slice = vol.slice(normal="z", origin=(0, 0, substrate_thickness))

# Build interpolation grid from data bounds
vpts = vol_slice.points
x_pad, y_pad = 5, 5
xi = np.linspace(vpts[:, 0].min() - x_pad, vpts[:, 0].max() + x_pad, 800)
yi = np.linspace(vpts[:, 1].min() - y_pad, vpts[:, 1].max() + y_pad, 400)
Xi, Yi = np.meshgrid(xi, yi)


def plot_topview(pts, data, title, cmap="turbo"):
    gi = griddata(pts[:, :2], data, (Xi, Yi), method="linear")
    fig, ax = plt.subplots(figsize=(14, 6))
    vmax = np.nanpercentile(gi, 98)
    im = ax.pcolormesh(Xi, Yi, gi, cmap=cmap, shading="auto", vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    valid = ~np.isnan(gi)
    if valid.any():
        rows = np.any(valid, axis=1)
        cols = np.any(valid, axis=0)
        ax.set_xlim(xi[cols][0], xi[cols][-1])
        ax.set_ylim(yi[rows][0], yi[rows][-1])
    fig.tight_layout(pad=0.5)
    plt.show()


# %% papermill={"duration": 0.365346, "end_time": "2026-04-04T12:05:51.591418", "exception": false, "start_time": "2026-04-04T12:05:51.226072", "status": "completed"}
plot_topview(
    vpts,
    np.linalg.norm(vol_slice.point_data["E_real"], axis=1),
    f"Electric field |E| at {freq_ghz:.4f} GHz — (V/m)",
)

# %%

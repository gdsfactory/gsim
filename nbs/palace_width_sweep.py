# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: gsim
#     language: python
#     name: python3
# ---

# %% [markdown] papermill={"duration": 0.003998, "end_time": "2026-04-03T18:03:01.257043", "exception": false, "start_time": "2026-04-03T18:03:01.253045", "status": "completed"}
# # Palace Width Sweep: Parallel Simulation
#
# Sweep the microstrip width from 5 to 50 um (5 points), run all simulations
# in parallel on GDSFactory+ cloud using the non-blocking API, then compare
# S11 and S21 across widths.
#
# **Requirements:**
#
# - IHP PDK: `uv pip install ihp-gdsfactory`
# - [GDSFactory+](https://gdsfactory.com) account for cloud simulation

# %% [markdown] papermill={"duration": 0.001322, "end_time": "2026-04-03T18:03:01.260961", "exception": false, "start_time": "2026-04-03T18:03:01.259639", "status": "completed"}
# ### Define the sweep

# %% papermill={"duration": 0.035809, "end_time": "2026-04-03T18:03:01.298312", "exception": false, "start_time": "2026-04-03T18:03:01.262503", "status": "completed"}
import numpy as np

widths = np.arange(2, 21, 4)

# %% [markdown] papermill={"duration": 0.000672, "end_time": "2026-04-03T18:03:01.299809", "exception": false, "start_time": "2026-04-03T18:03:01.299137", "status": "completed"}
# ### Build components and configure simulations

# %% papermill={"duration": 6.720991, "end_time": "2026-04-03T18:03:08.021430", "exception": false, "start_time": "2026-04-03T18:03:01.300439", "status": "completed"}
import gdsfactory as gf
from ihp import LAYER, PDK, cells

from gsim.common.stack import get_stack
from gsim.palace import DrivenSim

PDK.activate()

stack = get_stack(air_above=300.0)  # auto-detects active PDK

sims = []

for w in widths:
    # Build component
    c = gf.Component()
    r1 = c << cells.straight_metal(length=1000, width=w)

    r = c.get_region(layer=LAYER.TopMetal2drawing)
    r_sized = r.sized(+20000)
    c.add_polygon(r_sized, layer=LAYER.Metal1drawing)
    c.add_ports(r1.ports)

    # Configure simulation
    sim = DrivenSim()
    sim.set_output_dir(f"./palace-sim-w{w:.1f}")
    sim.set_geometry(c)
    sim.set_stack(stack)

    for port in c.ports:
        sim.add_port(
            port.name, from_layer="metal1", to_layer="topmetal2", geometry="via"
        )

    sim.set_driven(fmin=1e9, fmax=100e9, num_points=80)
    sim.mesh(preset="default")

    sims.append(sim)

print(f"Configured {len(sims)} simulations")

# %% [markdown] papermill={"duration": 0.000636, "end_time": "2026-04-03T18:03:08.023074", "exception": false, "start_time": "2026-04-03T18:03:08.022438", "status": "completed"}
# ### Upload and start all jobs (non-blocking)

# %% papermill={"duration": 17.189773, "end_time": "2026-04-03T18:03:25.213421", "exception": false, "start_time": "2026-04-03T18:03:08.023648", "status": "completed"}
# Upload and start all jobs without waiting
job_ids = []
for sim in sims:
    job_id = sim.run(wait=False)
    job_ids.append(job_id)

print(f"Started {len(job_ids)} jobs: {job_ids}")

# %% [markdown] papermill={"duration": 0.001071, "end_time": "2026-04-03T18:03:25.216657", "exception": false, "start_time": "2026-04-03T18:03:25.215586", "status": "completed"}
# ### Wait for all jobs to complete

# %% papermill={"duration": 283.580034, "end_time": "2026-04-03T18:08:08.797695", "exception": false, "start_time": "2026-04-03T18:03:25.217661", "status": "completed"}
import gsim

# Poll all jobs concurrently, download and parse results
results = gsim.wait_for_results(job_ids)

# %% [markdown] papermill={"duration": 0.001049, "end_time": "2026-04-03T18:08:08.800379", "exception": false, "start_time": "2026-04-03T18:08:08.799330", "status": "completed"}
# ### Plot S11 and S21 comparison

# %% papermill={"duration": 0.109339, "end_time": "2026-04-03T18:08:08.910656", "exception": false, "start_time": "2026-04-03T18:08:08.801317", "status": "completed"}
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

for w, sp in zip(widths, results):
    ax1.plot(sp.freq, sp.s11.db, label=f"w={w:.1f} um")
    ax2.plot(sp.freq, sp.s21.db, label=f"w={w:.1f} um")

ax1.set(
    xlabel="Frequency (GHz)", ylabel="|S11| (dB)", title="S11 — Return Loss vs Width"
)
ax1.legend()
ax1.grid(True)

ax2.set(
    xlabel="Frequency (GHz)", ylabel="|S21| (dB)", title="S21 — Insertion Loss vs Width"
)
ax2.legend()
ax2.grid(True)

plt.tight_layout()

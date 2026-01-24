# Palace CPW Simulation

This example demonstrates S-parameter extraction for a CPW (Coplanar Waveguide) structure using the IHP SG13G2 PDK.

## Setup

```python
from gsim.palace import DrivenSim

# Install IHP PDK if needed:
# !uv pip install git+https://github.com/gdsfactory/IHP.git
```

## Load IHP PDK and Create Component

```python
from ihp import LAYER, PDK
import gdsfactory as gf

PDK.activate()


@gf.cell
def gsg_electrode(
    length: float = 100,
    s_width: float = 20,
    g_width: float = 40,
    gap_width: float = 15,
    layer=LAYER.TopMetal2drawing,
) -> gf.Component:
    """GSG (Ground-Signal-Ground) electrode."""
    c = gf.Component()

    # Top ground electrode
    r1 = c << gf.c.rectangle((length, g_width), centered=True, layer=layer)
    r1.move((0, (g_width + s_width) / 2 + gap_width))

    # Center signal electrode
    _r2 = c << gf.c.rectangle((length, s_width), centered=True, layer=layer)

    # Bottom ground electrode
    r3 = c << gf.c.rectangle((length, g_width), centered=True, layer=layer)
    r3.move((0, -(g_width + s_width) / 2 - gap_width))

    # Add ports at the gaps
    c.add_port(
        name="P1",
        center=(-length / 2, -(s_width + gap_width) / 2),
        width=gap_width,
        orientation=0,
        port_type="electrical",
        layer=layer,
    )
    c.add_port(
        name="P2",
        center=(-length / 2, (s_width + gap_width) / 2),
        width=gap_width,
        orientation=0,
        port_type="electrical",
        layer=layer,
    )
    c.add_port(
        name="P3",
        center=(length / 2, (s_width + gap_width) / 2),
        width=gap_width,
        orientation=180,
        port_type="electrical",
        layer=layer,
    )
    c.add_port(
        name="P4",
        center=(length / 2, -(s_width + gap_width) / 2),
        width=gap_width,
        orientation=180,
        port_type="electrical",
        layer=layer,
    )

    return c


c = gsg_electrode()
c
```

## Configure Simulation

```python
sim = DrivenSim()

# Set the component geometry
sim.set_geometry(c)

# Configure layer stack from active PDK
sim.set_stack(substrate_thickness=2.0, air_above=300.0)

# Configure CPW ports (upper, lower gap ports)
sim.add_cpw_port("P2", "P1", layer="topmetal2", length=5.0)
sim.add_cpw_port("P3", "P4", layer="topmetal2", length=5.0)

# Configure driven simulation (frequency sweep)
sim.set_driven(fmin=1e9, fmax=100e9, num_points=40)

# Validate configuration
print(sim.validate())
```

## Generate Mesh

```python
OUTPUT_DIR = "./palace-sim-cpw"

# Generate mesh (presets: "coarse", "default", "fine")
sim.mesh(OUTPUT_DIR, preset="default")
```

## Visualize Mesh

```python
from gsim.palace import plot_mesh

plot_mesh(
    f"{OUTPUT_DIR}/palace.msh",
    output=f"{OUTPUT_DIR}/mesh.png",
    show_groups=["metal", "P"],
    interactive=False,
)
```

## Run Simulation

```python
# Run simulation on GDSFactory+ cloud
results = sim.simulate()
```

## Plot Results

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(results["port-S.csv"])
df.columns = df.columns.str.strip()

freq = df["f (GHz)"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

# Magnitude
ax1.plot(freq, df["|S[1][1]| (dB)"], marker=".", label="S11")
ax1.plot(freq, df["|S[2][1]| (dB)"], marker=".", label="S21")
ax1.set_xlabel("Frequency (GHz)")
ax1.set_ylabel("Magnitude (dB)")
ax1.set_title("S-Parameters")
ax1.legend()
ax1.grid(True)

# Phase
ax2.plot(freq, df["arg(S[1][1]) (deg.)"], marker=".", label="S11")
ax2.plot(freq, df["arg(S[2][1]) (deg.)"], marker=".", label="S21")
ax2.set_xlabel("Frequency (GHz)")
ax2.set_ylabel("Phase (deg)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
```

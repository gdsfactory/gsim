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

# %% [markdown] papermill={"duration": 0.003779, "end_time": "2026-04-04T06:14:35.168197", "exception": false, "start_time": "2026-04-04T06:14:35.164418", "status": "completed"}
# # Running Palace Simulations: Branch Line Coupler
#
# [Palace](https://awslabs.github.io/palace/) is an open-source 3D electromagnetic simulator supporting eigenmode, driven (S-parameter), and electrostatic simulations. This notebook demonstrates using the `gsim.palace` API to run a driven simulation on a branch line coupler.
#
# **Requirements:**
#
# - IHP PDK: `uv pip install ihp-gdsfactory`
# - [GDSFactory+](https://gdsfactory.com) account for cloud simulation

# %% [markdown] papermill={"duration": 0.001942, "end_time": "2026-04-04T06:14:35.173615", "exception": false, "start_time": "2026-04-04T06:14:35.171673", "status": "completed"}
# ### Load a pcell from IHP PDK

# %% papermill={"duration": 1.713285, "end_time": "2026-04-04T06:14:36.888672", "exception": false, "start_time": "2026-04-04T06:14:35.175387", "status": "completed"}
import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec
from ihp import LAYER, PDK

PDK.activate()


# `branch_line_coupler` used to live in `ihp.cells` but was removed when the
# IHP PDK migrated its schematic metadata. The PCell is inlined here so this
# notebook is self-contained.
@gf.cell
def tline1(
    length: float = 100,
    width: float = 14,
    signal_cross_section: CrossSectionSpec = "topmetal2_routing",
    ground_cross_section: CrossSectionSpec = "metal3_routing",
    npoints: int = 2,
) -> gf.Component:
    """Coplanar transmission line: signal straight with a wider ground straight around it."""
    c = gf.Component()
    signal = c.add_ref(
        gf.c.straight(
            length=length,
            cross_section=signal_cross_section,
            width=width,
            npoints=npoints,
        )
    )
    c.add_ports(signal.ports)
    ground = c.add_ref(
        gf.c.straight(
            length=length + 6 * width,
            cross_section=ground_cross_section,
            width=7 * width,
            npoints=npoints,
        )
    )
    ground.move((-3 * width, 0))
    return c


@gf.cell
def branch_line_coupler(
    width: float = 10,
    width_coupled: float = 14,
    quarter_wave_length: float = 500,
    connection_length: float = 100,
    signal_cross_section: CrossSectionSpec = "topmetal2_routing",
    ground_cross_section: CrossSectionSpec = "metal3_routing",
) -> gf.Component:
    """Four-port branch-line coupler made of coplanar quarter-wave sections."""
    c = gf.Component()
    signal_layer = gf.get_cross_section(signal_cross_section).layer

    corner = gf.Component()
    corner.add_polygon(
        points=[
            (0, 0),
            (0, width),
            (width - (width_coupled - width), width),
            (width, width_coupled),
            (width, 0),
        ],
        layer=signal_layer,
    )
    corner.add_port(
        name="e1",
        center=(width / 2, 0),
        width=width,
        orientation=270,
        port_type="electrical",
        layer=signal_layer,
    )
    corner.add_port(
        name="e2",
        center=(width, width_coupled / 2),
        width=width_coupled,
        orientation=0,
        port_type="electrical",
        layer=signal_layer,
    )
    corner.add_port(
        name="e3",
        center=(0, width / 2),
        width=width,
        orientation=180,
        port_type="electrical",
        layer=signal_layer,
    )

    corner_nw = c.add_ref(corner)
    tline_top = c.add_ref(
        tline1(
            length=quarter_wave_length - width,
            signal_cross_section=signal_cross_section,
            ground_cross_section=ground_cross_section,
            width=width_coupled,
        )
    )
    tline_top.connect("e1", corner_nw.ports["e2"])

    corner_ne = c.add_ref(corner).mirror(p1=(0, 0), p2=(0, 1))
    corner_ne.connect("e2", tline_top.ports["e2"])

    tline_left = c.add_ref(
        tline1(
            length=quarter_wave_length - width_coupled,
            signal_cross_section=signal_cross_section,
            ground_cross_section=ground_cross_section,
            width=width,
        )
    )
    tline_left.connect("e1", corner_nw.ports["e1"])

    corner_sw = c.add_ref(corner).mirror(p1=(0, 0), p2=(1, 0))
    corner_sw.connect("e1", tline_left.ports["e2"])

    tline_bottom = c.add_ref(
        tline1(
            length=quarter_wave_length - width,
            signal_cross_section=signal_cross_section,
            ground_cross_section=ground_cross_section,
            width=width_coupled,
        )
    )
    tline_bottom.connect("e1", corner_sw.ports["e2"])

    corner_se = (
        c.add_ref(corner).mirror(p1=(0, 0), p2=(1, 0)).mirror(p1=(0, 0), p2=(0, 1))
    )
    corner_se.connect("e2", tline_bottom.ports["e2"])

    tline_right = c.add_ref(
        tline1(
            length=quarter_wave_length - width_coupled,
            signal_cross_section=signal_cross_section,
            ground_cross_section=ground_cross_section,
            width=width,
        )
    )
    tline_right.connect("e1", corner_ne.ports["e1"])

    for port, name in [
        (corner_nw.ports["e3"], "e1"),
        (corner_ne.ports["e3"], "e2"),
        (corner_se.ports["e3"], "e3"),
        (corner_sw.ports["e3"], "e4"),
    ]:
        feed = c.add_ref(
            tline1(
                length=connection_length,
                signal_cross_section=signal_cross_section,
                ground_cross_section=ground_cross_section,
                width=width,
            )
        )
        feed.connect("e1", port)
        c.add_port(name=name, port=feed.ports["e2"])

    c.move((0, -width))
    return c


c = gf.Component()
r1 = c << branch_line_coupler(
    width=8.85, width_coupled=14.96, quarter_wave_length=769.235, connection_length=50
)
c.add_ports(r1.ports)

# Save port info before flatten
ports = [(p.name, p.center, p.width, p.orientation, p.layer) for p in c.ports]

c.flatten()

# Fill holes in Metal3 ground plane
r = c.get_region(layer=LAYER.Metal3drawing)
r_filled = gf.kdb.Region([gf.kdb.Polygon(list(p.each_point_hull())) for p in r.each()])
c.remove_layers(layers=[LAYER.Metal3drawing])
c.add_polygon(r_filled, layer=LAYER.Metal3drawing)

# Re-add ports
for name, center, width, orientation, layer in ports:
    c.add_port(
        name=name, center=center, width=width, orientation=orientation, layer=layer
    )

cc = c.copy()
cc.draw_ports()
cc

# %% [markdown] papermill={"duration": 0.001014, "end_time": "2026-04-04T06:14:36.890928", "exception": false, "start_time": "2026-04-04T06:14:36.889914", "status": "completed"}
# ### Configure and run simulation with DrivenSim

# %% papermill={"duration": 0.482865, "end_time": "2026-04-04T06:14:37.374923", "exception": false, "start_time": "2026-04-04T06:14:36.892058", "status": "completed"}
from gsim.common.stack import get_stack
from gsim.palace import DrivenSim

# Create simulation object
sim = DrivenSim()

# Set output directory
sim.set_output_dir("./palace-sim-branch-coupler")

# Set the component geometry
sim.set_geometry(c)

# Configure layer stack from active PDK
stack = get_stack(air_above=300.0)  # auto-detects active PDK
sim.set_stack(stack)

# Configure via ports (Metal3 ground plane to TopMetal2 signal)
for port in c.ports:
    sim.add_port(port.name, from_layer="metal3", to_layer="topmetal2", geometry="via")

# Configure driven simulation (frequency sweep for S-parameters)
sim.set_driven(fmin=1e9, fmax=100e9, num_points=300)

# Validate configuration
print(sim.validate_config())

# %% papermill={"duration": 3.602926, "end_time": "2026-04-04T06:14:40.978728", "exception": false, "start_time": "2026-04-04T06:14:37.375802", "status": "completed"}
# Generate mesh (presets: "coarse", "default", "fine")
sim.mesh(preset="default")

# %% papermill={"duration": 1.02794, "end_time": "2026-04-04T06:14:42.007639", "exception": false, "start_time": "2026-04-04T06:14:40.979699", "status": "completed"}
# Static PNG
sim.plot_mesh(show_groups=["metal", "P"])

# %% [markdown] papermill={"duration": 0.002874, "end_time": "2026-04-04T06:14:42.013971", "exception": false, "start_time": "2026-04-04T06:14:42.011097", "status": "completed"}
# ### Run simulation on GDSFactory+ Cloud

# %% papermill={"duration": 2449.138372, "end_time": "2026-04-04T06:55:31.154320", "exception": false, "start_time": "2026-04-04T06:14:42.015948", "status": "completed"}
# Run simulation on GDSFactory+ cloud
results = sim.run()

# %% papermill={"duration": 0.185189, "end_time": "2026-04-04T06:55:31.341798", "exception": false, "start_time": "2026-04-04T06:55:31.156609", "status": "completed"}
results.plot_interactive()

# %% papermill={"duration": 0.014931, "end_time": "2026-04-04T06:55:31.360425", "exception": false, "start_time": "2026-04-04T06:55:31.345494", "status": "completed"}
results.plot_interactive(phase=True)

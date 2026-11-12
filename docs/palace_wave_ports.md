# Wave ports in Palace

A wave port is a face of the simulation domain on which Palace solves a 2D eigenproblem for the guided mode of the
cross-section, at every frequency, and uses that mode to drive and terminate the line. This page is the entry point for
wave ports in gsim: when to choose one over a lumped port, how to declare one, what it does to the outer wall of the
domain, how several ports on one face behave, and where the deeper pages are.

1. [Lumped port or wave port](#1-lumped-port-or-wave-port)
2. [A single-mode wave port](#2-a-single-mode-wave-port)
3. [The outer wall is PEC by default](#3-the-outer-wall-is-pec-by-default)
4. [The port eigenproblem: `waveport_boundary`](#4-the-port-eigenproblem-waveport_boundary)
5. [Several modes on one face](#5-several-modes-on-one-face)
6. [Stacked ports](#6-stacked-ports)
7. [Terminal S-parameters](#7-terminal-s-parameters)
8. [Check the port mode](#8-check-the-port-mode)

The code blocks use the generic PDK (`gf.gpdk`) and are built up to `write_config()`: they mesh locally and write
`config.json`, and none of them submits a job. They were run with gsim 0.9.0 on a Windows laptop with the `coarse` mesh preset, which is
meant for checking a setup and not for results.

## 1. Lumped port or wave port

| | Lumped port | Wave port |
| --- | --- | --- |
| gsim calls | `add_port`, `add_cpw_port`, `add_two_terminal_port` | `add_wave_port`, `add_multimode_wave_port` |
| What it is | A sheet in the model: across a gap, between two layers or across the CPW gaps | A domain face where Palace solves the 2D mode |
| S-parameter reference | A real port resistance (`impedance`, 50 ohm by default) | Power-normalised; `to_skrf(z0=50.0)` only assigns 50 ohm |
| Outer wall, if you set nothing | Absorbing | PEC (section 3) |
| Airbox in the CPW notebooks | `margin_x=50`, `margin_y=0` | `margin_x=0`, `margin_y=50` |
| Several modes, terminal S | No | Yes (sections 5 and 7) |

The rows come from the signatures and docstrings of the calls, from the
[CPW length sweep and de-embedding](nbs/palace_cpw_deembedding.md) notebook (the S-parameter reference) and from the CPW notebooks
(the airbox). In those notebooks the lumped-port line has an airbox margin along the line (`margin_x=50`) and the wave-port line
has none (`margin_x=0`), so that the port faces are the end faces of the domain (section 2).

**Which one to use.** This is guidance from how the examples are built, not a measured comparison:

- Use a **wave port** when you need the guided mode itself: the effective index, the modal S-parameters of a line, several
  modes on one face, or terminal S-parameters. The face has to be a cross-section of a line that continues beyond the model.
- Use a **lumped port** for a device that has no uniform line to take a mode from, such as an inductor fed between two metal
  layers or across a gap ([Inductor](nbs/palace_inductor.md)).
- The same CPW line with both port types is in [CPW (Lumped Ports)](nbs/palace_cpw_lumped.md),
  [CPW (Wave Ports)](nbs/palace_cpw_waveport.md) and, side by side, in the
  [length sweep and de-embedding](nbs/palace_cpw_deembedding.md) notebook. That notebook uses a different airbox for each
  port type and says that a physical comparison of the two needs airbox and mesh convergence.
- [Inductor Port Comparison](nbs/palace_inductor_port_comparison.md) compares two *lumped* ports (a one-port `gap` port and a
  two-port `interlayer` setup), not a lumped with a wave port. It is a good example of how the port definition and the
  reference conductor change a result, but it does not tell you when to prefer a wave port.

## 2. A single-mode wave port

```python
import gdsfactory as gf

from gsim.palace import DrivenSim

gf.gpdk.PDK.activate()
LAYER = gf.gpdk.LAYER


@gf.cell
def gsg_line(length: float = 300, s_width: float = 10, g_width: float = 40, gap_width: float = 6) -> gf.Component:
    """A ground-signal-ground line on metal1 with a port at the centre of each end."""
    c = gf.Component()
    top = c << gf.c.rectangle((length, g_width), centered=True, layer=LAYER.M1)
    top.move((0, (g_width + s_width) / 2 + gap_width))
    c << gf.c.rectangle((length, s_width), centered=True, layer=LAYER.M1)
    bottom = c << gf.c.rectangle((length, g_width), centered=True, layer=LAYER.M1)
    bottom.move((0, -(g_width + s_width) / 2 - gap_width))
    for name, x, angle in (("o1", -length / 2, 180), ("o2", length / 2, 0)):
        c.add_port(name=name, center=(x, 0), width=s_width, orientation=angle, port_type="electrical", layer=LAYER.M1)
    return c


sim = DrivenSim()
sim.set_output_dir("./palace-sim-waveport")
sim.set_geometry(gsg_line())
sim.set_stack(substrate_thickness=2.0)
sim.set_airbox(margin_x=0.0, margin_y=50.0, z_above=100.0, z_below=100.0)
for name in ("o1", "o2"):
    sim.add_wave_port(name, layer="metal1", max_size=True, mode=1)
sim.set_driven(fmin=1e9, fmax=100e9, num_points=40)
print(sim.validate_config())

result = sim.mesh(preset="coarse")
config_path = sim.write_config()
print(result.port_info)  # the face of every port: xmin, xmax, ymin, ymax, zmin, zmax in um
```

`add_wave_port(name, *, layer, mode=1, max_size=False, z_margin=0.0, lateral_margin=0.0, offset=0.0, excited=True, ...)`:

| Argument | What it does |
| --- | --- |
| `name` | The name of a gdsfactory port of the component. Its centre, width and orientation place the face. |
| `layer` | The stack layer whose z range sets the height of the face, before the margins. |
| `max_size` | The face fills the whole side of the domain, whatever the port centre. Overrides the two margins. |
| `z_margin`, `lateral_margin` | Widen a finite face by this many um above and below the layer, and to each side of the port width. |
| `mode` | The Palace mode number, counted from 1, that this port drives and terminates. |
| `offset` | Written as the Palace `Offset` of the port, the de-embedding distance in um (default 0). |
| `excited` | Whether this port is excited. Both ports of a line are excited in separate solves to get all four S-parameters. |
| `eigensolver_*` | Settings of this port's 2D mode solve (`SolverType`, `EigenTol`, `KSPTol`, `MaxSize`, `Verbose`). `None`: Palace's default. |

A port that faces along x gives a face in a plane of constant x, and one that faces along y a face of constant y. The height of a
finite face is cut to the stack and to the domain.

`result.port_info`, with `result = sim.mesh(...)`, lists the faces that were built. For the line above the two faces are at
x = -150 um and x = 150 um, and each spans the whole domain in y and z. With a finite face (no `max_size`) the same line gets a
face of the port width (10 um) by the thickness of `metal1` (0.7 um) at each end. That is a geometry check, not a good port for a
mode solve: widen the face with `lateral_margin` and `z_margin`, or use `max_size=True` as the CPW notebooks do.

With `margin_x=0.0` the port faces are the end faces of the domain. gsim does not refuse a larger `margin_x`: `validate_config()`
passes and the mesh is built with the face at the end of the line, inside the domain. What Palace does with a wave-port face
that is not on the domain boundary was not run for this page; follow the notebooks and use `margin_x=0.0`.

## 3. The outer wall is PEC by default

Palace solves the port mode with perfect-conductor outer edges. If the 3D domain around it were absorbing, the domain and the
port mode would disagree, which shows up as fake loss, reflection and mode conversion. So when a driven simulation has
wave ports, gsim makes the outer wall of the domain PEC unless you say otherwise. This starts with gsim 0.9.0; earlier
releases left the wall absorbing. Saved configurations and notebooks that relied on that should pass
`absorbing_boundary=True`.

What `config.json` gets for the outer wall of the line of section 2:

| Setting | `Boundaries` |
| --- | --- |
| `absorbing_boundary` not set (default), with wave ports | The outer surfaces are in `PEC`; there is no `Absorbing` |
| `absorbing_boundary=True` | `Absorbing`; the outer surfaces are not in `PEC` |
| `absorbing_boundary=False` | Neither. The outer boundary is left unconditioned, which Palace treats as a magnetic (PMC) wall |
| `metallic_boundaries=True` | `PEC`, as the default. With `absorbing_boundary=True` as well, `write_config()` raises `ValueError` |

A driven simulation with lumped ports only stays absorbing by default. The PEC default applies to wave ports added with
`add_wave_port()` and `add_multimode_wave_port()` and to wave ports declared through `configure_wave_port()` component
metadata. If the mesh has no outer-boundary group at all (the tests of this case use an airbox with zero margin), there is
nothing for the PEC wall to attach to: gsim writes no outer-wall entry and logs a note.

```python
# Add the same wave ports to either of these instead of the default simulation of section 2:
sim_absorbing = DrivenSim(absorbing_boundary=True)  # keep the absorbing wall
sim_pmc = DrivenSim(absorbing_boundary=False)  # no outer-wall entry: Palace's own PMC
```

The first `write_config()` of a simulation with the PEC default logs one info line, "Wave ports: the outer wall is PEC".
The API guide keeps a short summary of this change under
[Numeric wave-port boundaries](api/palace.md#numeric-wave-port-boundaries).

## 4. The port eigenproblem: `waveport_boundary`

For every wave-port entry Palace solves its own 2D eigenproblem. `set_driven(waveport_boundary=...)` chooses which of the
3D boundary conditions that solve sees:

- `"pec"` (the default) treats the absorbing wall, the finite-conductivity conductor surfaces and the impedance boundaries as
  perfect conductors **inside the port solve**. gsim lists their attributes in `Boundaries.WavePortPEC`.
- `"inherit"` keeps those conditions in the port solve, including the conductor loss, and writes no `WavePortPEC`.

Only the `WavePortPEC` override changes. The 3D boundaries stay as they are, and lumped ports are not affected. The two
policies differ only when the model has absorbing, conductivity or impedance boundaries.

```python
sim.set_driven(fmin=1e9, fmax=100e9, num_points=40, waveport_boundary="inherit")
config_path = sim.write_config()  # no "WavePortPEC" in Boundaries
```

`set_driven()` replaces the whole driven setting, so repeat the other sweep options in the same call. A change here needs
no new mesh. The policy is a modelling approximation, not a fix for Palace's mode selection; see
[Numeric wave-port boundaries](api/palace.md#numeric-wave-port-boundaries). The terminal-S method needs the 2D solve that
supplies the mode shapes to use the same conductor condition as the port eigenproblem
([Terminal (multi-pin) Wave Ports](terminal_wave_ports.md), section 6).

## 5. Several modes on one face

Palace has no terminal wave port, and it allows several `WavePort` entries on one face only under a rule: they share the face
with different `Mode`, only one of them may be active, and the active one must have the lowest `Index` of that face.
`add_multimode_wave_port` writes exactly that: the active mode keeps the port's own index and terminates the face, every other
mode is an observation-only entry (`"Active": false`) with an index after all the ports, and the map from Palace index to
(port, mode, active, excited) is written to `port_modes.json` next to `config.json`.

```python
from gsim.palace.ports.modes import read_port_modes

sim.add_multimode_wave_port(
    "o1", layer="metal1", modes=(1, 2), active_mode=1, excite="active", max_size=True
)
sim.mesh(preset="coarse")  # the ports are captured at mesh time
config_path = sim.write_config()
for row in read_port_modes(config_path.parent):
    print(row.index, row.port, "mode", row.mode, "active", row.active, "excited", row.excited)
```

For this line (`o1` multimode, `o2` single-mode) the three `WavePort` entries are index 1 = `o1` mode 1 (active), index 2 = `o2`
mode 1, and index 3 = `o1` mode 2 (not active); `port_modes.json` lists the two entries of the multimode port.

- Only the active entry terminates the face. A wave that arrives in another mode is not matched there, so the S-parameters of
  the observed modes carry a reflection error, and the active mode's are exact. To get every mode with the active-mode accuracy,
  run once per mode with that mode as `active_mode`. The reasoning and an estimate of the error are in section 4 of
  [Terminal (multi-pin) Wave Ports](terminal_wave_ports.md).
- A change of `active_mode` needs a new `mesh()` call.
- Two multimode ports on the same face (same centre and orientation, same layer) are refused by `validate_config()`
  ("are on the same face; give a face one wave port and list its modes with modes=..."). Give the face one port and list
  its modes.

## 6. Stacked ports

Two ports at the same centre and with the same orientation, on **different layers**, are two separate faces, and gsim accepts
them, as long as the faces do not overlap in z. Ports that do overlap are refused by `validate_config()` with one of two
messages: "are on the same face" when one of the ports is multimode, and "generate overlapping faces" when both are
single-mode. On the generic stack, where `metal1` spans z = 1.1 to 1.8 um and `metal2` spans z = 2.3 to 3.0 um, the answers
are:

| Ports at one centre and orientation | `validate_config()` |
| --- | --- |
| `metal1` and `metal2`, finite faces, one or both multimode | Valid |
| `metal1` with `z_margin=0.4` (face z = 0.7 to 2.2 um) and `metal2` | Valid |
| `metal1` with `z_margin=0.6` (face z = 0.5 to 2.4 um, which reaches `metal2`) | Refused |
| `max_size=True` on the `metal1` port: the face spans the whole side of the domain | Refused |
| Both ports on `metal1` | Refused |

```python
@gf.cell
def stacked_ends() -> gf.Component:
    """Two ports at one centre and orientation: they differ only in the layer they are used on."""
    c = gf.Component()
    c.add_polygon([(-20, -61), (20, -61), (20, 61), (-20, 61)], layer=LAYER.M1)
    for name in ("lo", "hi"):
        c.add_port(name=name, center=(-40, 0), width=10, orientation=180, port_type="electrical", layer=LAYER.M1)
    return c


def stacked(z_margin: float = 0.0) -> DrivenSim:
    stacked_sim = DrivenSim()
    stacked_sim.set_geometry(stacked_ends())
    stacked_sim.add_multimode_wave_port("lo", layer="metal1", modes=(1, 2), z_margin=z_margin)
    stacked_sim.add_wave_port("hi", layer="metal2")
    return stacked_sim


print(stacked().validate_config().valid)                # True
print(stacked(z_margin=0.4).validate_config().valid)    # True
print(stacked(z_margin=0.6).validate_config().errors)   # "'lo' and 'hi' are on the same face; ..."
```

Meshed with `set_airbox(margin_x=30.0, margin_y=20.0, z_above=40.0, z_below=10.0)`, the stacked case of the first row gets
one surface per port (`P1`, `P2`) and three `WavePort` entries: one for each port and one extra, on the surface of `lo`, for
its second mode. The two faces are 10 um wide and 0.7 um high, which makes them a check of the port bookkeeping and not a
meaningful port for a mode solve in this thin stack. No Palace solve of stacked ports was run for this page. The behaviour
above is that of `tests/palace/test_wave_port_faces.py`.

## 7. Terminal S-parameters

A wave port reports modal S: one port per mode of the face. For several signal conductors on one face (differential pairs,
coupled lines, a bias line as its own terminal) circuit work needs terminal S: one port per conductor against a common
reference, with a reference impedance of your choice. `gsim.palace.terminal` converts one into the other in post-processing:
a 2D `BoundaryModeSim` of the port plane gives the voltage and current each mode puts on each terminal, a 3D `DrivenSim` with a
multimode port gives the modal S, and `build_terminal_s` joins the two. Palace itself is unchanged.

- Method, conventions and limits: [Terminal (multi-pin) Wave Ports](terminal_wave_ports.md), section 1 (why modal is not
  terminal) and section 8 (the workflow).
- Runnable examples: [Terminal Wave Ports in Palace](nbs/palace_terminal_waveport.md) and
  [Coupled Lines with Terminal Wave Ports](nbs/palace_terminal_waveport_gssg.md).
- One signal over one reference (a microstrip, or a CPW whose grounds are tied) has one mode, and you can stay with modal S.

## 8. Check the port mode

Palace picks the port mode only by its index. If the port eigensolve returns a different solution at some frequency, the sweep
still finishes and the S-parameters are wrong. `validate_solution` compares the effective index of every port solve of a
driven log with that of its neighbours in frequency and reports the result in `report.port_mode_checks`:

```python
from pathlib import Path

from gsim.palace.validation import validate_solution

report = validate_solution(Path("run/output/palace/palace.log").read_text(encoding="utf-8"))
for check in report.port_mode_checks:
    print(check.name, check.status)
    print(*check.evidence, sep="\n")
```

`passed`, `warning` and `unknown` mean different things (a warning is not a failure; a single-frequency run is always
`unknown`), and the check cannot find a wrong mode that is smooth in frequency. Both are explained on
[Palace: Checking a Run](palace-validation.md), together with how to compare against an independent `BoundaryModeSim` solve.

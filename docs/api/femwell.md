# femwell guide

`gsim.femwell` solves waveguide and transmission-line modes with [femwell](https://github.com/HelgeGehring/femwell) on the same
native-2D cross-section mesh that the Palace `BoundaryModeSim` generates (`src/gsim/femwell/adapter.py:1-35`). One mesh, two
solvers: the Palace mode solve and the femwell mode solve see the same triangles and the same stack materials, so a difference
between them is a difference between solvers and not between geometries.

Install the `femwell` extra, `pip install "gsim[femwell]"` (`pyproject.toml:41-44`). The permittivity helpers (`epsilon_by_region`,
`elementwise_epsilon`) work without it; only `solve_modes` needs femwell and scikit-fem (`src/gsim/femwell/__init__.py:9-12`).

Every number on this page is asserted by `tests/femwell/test_femwell_docs.py` or quoted from a comment or docstring of another test
(and then not re-run), and the text says which. The tests also run the four code blocks. Nothing here ran on the cloud, and none of
it needs Palace. The coax was solved on two meshes and the optical mode on one, so the orderings are not shown beyond those.

The functions this page covers:

- Permittivity per mesh region from the stack, `epsilon_by_region(mesh, stack, wavelength_um= or frequency_hz=)`, or per element from
  scattered samples, `elementwise_epsilon(mesh, x_um, y_um, eps_values)` (`src/gsim/femwell/adapter.py:107-214`).
- Solve modes: `solve_modes(msh_path, epsilon=, wavelength_um=, num_modes=, order=, metallic_boundaries=, n_guess=)`
  (`src/gsim/femwell/adapter.py:313-441`).
- Is the window big enough: `boundary_field_ratio(mode)` (`src/gsim/femwell/adapter.py:444-493`).
- The index above which a mode lives in the metal: `conductor_index_bound(epsilon)` (`src/gsim/femwell/adapter.py:217-310`).
- Impedance and wall-mode check of a selected mode: `z0_power_current`, `line_reading` (`src/gsim/femwell/adapter.py:596-651, 708-752`).
- Pick the line mode: `select_line_mode`, `propagating_modes` (`src/gsim/common/modes.py:289-438`).

## Solve a mode

Mesh a cross-section with `BoundaryModeSim`, read its permittivity from the stack, solve, and keep the guided mode. The snippet is
the geometry of the Palace-versus-femwell test (`tests/femwell/test_cross_validation.py:62-83`) without the Palace run; it only
meshes, nothing is submitted:

```python
# Mesh a gpdk cross-section and solve it with femwell
import gdsfactory as gf

from gsim.common.modes import select_line_mode
from gsim.femwell import boundary_field_ratio, epsilon_by_region, solve_modes
from gsim.palace import BoundaryModeSim

gf.gpdk.PDK.activate()
component = gf.Component()
component << gf.c.rectangle((10.0, 0.5), centered=True, layer=(1, 0))

sim = BoundaryModeSim()
sim.set_output_dir("./femwell-guide")
sim.set_stack(substrate_thickness=1.0)
sim.set_geometry(component)
sim.set_airbox(margin_x=1.2, margin_y=1.2, z_above=1.0, z_below=0.8, material="sio2")
sim.set_cross_section("x=0")
sim.mesh(preset="coarse", refined_mesh_size=0.04, max_mesh_size=0.4, verbose=False)

wavelength_um = 1.55
epsilon = epsilon_by_region(sim.mesh_path, sim.stack, wavelength_um=wavelength_um)
modes = solve_modes(sim.mesh_path, epsilon=epsilon, wavelength_um=wavelength_um, num_modes=2)
mode = select_line_mode(modes)
n_eff = complex(mode.n_eff)
assert boundary_field_ratio(mode) < 0.01  # the window does not clip the mode
```

This gives `n_eff` of about 2.4449 and a boundary field ratio of the order of 3e-4 (`tests/femwell/test_femwell_docs.py:265-270`, which
checks `n_eff` to 0.2 % and the ratio between 1e-4 and 1e-3).

Things to know about the arguments:

- **Units.** The mesh is in um. `wavelength_um` is the vacuum wavelength; for an RF line pass `c0 / f` in um
  (`src/gsim/femwell/adapter.py:332-333`).
- **Sign convention.** `exp(+i omega t)`, so a lossy medium has `Im(eps) < 0` (`src/gsim/femwell/adapter.py:33-34`). A conductor enters as
  `eps = eps_r (1 - i tan d) - i sigma / (omega eps0)` (`src/gsim/femwell/adapter.py:104`).
- **Every region needs a permittivity.** A mesh region missing from `epsilon`, or a region in `epsilon` that is not on the mesh, raises
  `ValueError` and names the regions (`src/gsim/femwell/adapter.py:401-424`).
- **`n_guess`.** femwell's default eigenvalue guess follows the largest permittivity. For an RF line with a large conductive `|Im(eps)|`
  that can put the search far from the quasi-TEM mode; pass the index you expect, for example the one from the previous frequency
  (`src/gsim/femwell/adapter.py:349-353`, `nbs/tcad_twmzm.ipynb:1007`). `metallic_boundaries=True` puts a perfect conductor on the
  outer boundary (`src/gsim/femwell/adapter.py:348`).

## Element order

**The default is order 2** (`src/gsim/femwell/adapter.py:319`). The docstring gives the reason and the cost: order 2 converges faster
under mesh refinement than order 1, and it costs about 2.5 times more per solve on a mesh of a hundred elements and about 10 times
above a thousand (`src/gsim/femwell/adapter.py:335-347`). Pass `order=1` for a cheaper solve when percent-level accuracy is enough;
one test does that on purpose to keep five solves cheap (`tests/femwell/test_staircase_convergence.py:105-111`).

The same solve at order 1:

```python
# The same solve at element order 1
modes_order1 = solve_modes(
    sim.mesh_path, epsilon=epsilon, wavelength_um=wavelength_um, num_modes=2, order=1
)
n_eff_order1 = complex(select_line_mode(modes_order1).n_eff)
assert abs(n_eff_order1 - n_eff) < 0.02 * abs(n_eff)
```

On that optical mode the two orders differ by 0.9 % in `n_eff`: about 2.4231 at order 1 against 2.4449 at order 2
(`tests/femwell/test_femwell_docs.py:271-273`). The Palace-versus-femwell test comment says Palace's order-2 solve agrees with femwell
at order 2 to 4.1e-5 (optical) and 2.9e-4 (RF materials), and with femwell at order 1 to 9e-3
(`tests/femwell/test_cross_validation.py:40-45`; that test needs the Palace binary and is not part of the default run).

### The coax impedance at both orders

The coax in `tests/femwell/test_z0_extraction.py` has a 1 um copper inner conductor, a 3 um outer radius, `eps_r` = 2.25 and a perfect
outer wall, solved at 100 GHz (`tests/femwell/test_z0_extraction.py:26-33`). The closed form
`Z0 = eta0 ln(b/a) / (2 pi sqrt(eps_r))` is the lossless value, 43.914 ohm. That test accepts 8 % around it
(`tests/femwell/test_z0_extraction.py:97-104`) and says "measured 3.2 % high" (`tests/femwell/test_z0_extraction.py:100`).

A second reference includes the skin effect of the copper: the same line with the inner conductor's internal impedance from the exact
Bessel solution for a round wire, `sqrt(Z'/Y')`, 45.988 ohm, 4.72 % above the lossless value. That reference is my own construction
for this page and is not a published number; the only support for it is that the order-2 solves land on it. Both references are
computed in `tests/femwell/test_femwell_docs.py:325-349`.

`solve_modes` at both orders, on the test's mesh and on one that caps the element size away from the conductor at 0.15 um instead of
0.35 um:

| Max element size (um) | Order | Re Z0 (ohm) | Error vs lossless closed form | Error vs closed form with skin effect |
| --- | --- | --- | --- | --- |
| 0.35 (the test's mesh) | 1 | 45.323 | +3.21 % | -1.45 % |
| 0.35 | 2 | 45.957 | +4.65 % | -0.07 % |
| 0.15 | 1 | 45.825 | +4.35 % | -0.35 % |
| 0.15 | 2 | 45.996 | +4.74 % | +0.02 % |

`tests/femwell/test_femwell_docs.py:431-450` solves the four cases and checks each Re Z0 to within 0.3 ohm and each error to 0.7 percentage
points, so read the last digits as the mesh generator's, not as a reference. It also holds the readings below.

- **Against the lossless closed form, order 1 looks better on the test's mesh, and that is a coincidence.** The closed form leaves out
  the skin effect, which the solve includes (`tests/femwell/test_femwell_docs.py:469-477`).
- **Order 2 is converged on this problem; order 1 is not.** The two order-2 results agree to 0.2 % and sit within 0.3 % of the
  skin-effect reference. Order 1 moves with the mesh by more than 0.4 % and reads below the reference
  (`tests/femwell/test_femwell_docs.py:453-466`).
- **The 3.2 % comment in `test_z0_extraction.py` is the order-1 number.** Its fixture does not pass `order`, so it runs at order 2
  and reads +4.65 %, inside its 8 % limit. The comment predates the default; that test was not edited.

## Domain size and the coarse far mesh

**Window size.** `boundary_field_ratio(mode)` is the peak electric-field magnitude over the elements touching the outer boundary,
divided by the peak over the whole cross-section. A well-contained mode gives a number orders of magnitude below one
(`src/gsim/femwell/adapter.py:444-453`). The TCAD travelling-wave MZM notebook asserts it is below 0.01 for the optical mode
(`nbs/tcad_twmzm.ipynb:766`); the snippet above gives about 3e-4. Two limits of the check:

- It only counts the outer boundary, not the cut-outs where a metal electrode is left out of the mesh (`src/gsim/femwell/adapter.py:455-458`).
- It needs a window whose edges lie on the domain's bounding box, and raises `ValueError` when no facet does
  (`src/gsim/femwell/adapter.py:466-485`). So it suits open windows. Where the wall is part of the structure (a shielded line, a coax),
  a field at the wall is expected and the 0.01 threshold does not apply; that is my reading, not something a test checks.

**Element size away from the structure.** Refining the mesh around the structure does not remove the error that a coarse far mesh
leaves, and order 2 reduces it without removing it (`src/gsim/femwell/adapter.py:337-346`). The docstring says to cap the element
size there as well when the series impedance (`gamma * Z0`) must be accurate to 1e-4. The numbers are measurements in the docstring of
the series-resistance test, an aluminium strip 2 um wide and 0.5 um thick in oxide at 1 GHz, where `R'` must equal the strip's DC
resistance (`tests/femwell/test_series_resistance.py:1-19`; not re-run for this page). With 50 nm at the strip and 2 um in the oxide,
order 1 reads `R'` 8.0 % low and order 2 lands within 5e-4. Refining to 25 nm at the strip leaves 12.6 % low and 3.2e-3 low; capping the
oxide at 0.5 um brings them to 1.0 % low and 7e-5 high. The test holds the default order to 3e-3 on the first mesh and checks that
order 1 stays more than 3 % low (`tests/femwell/test_series_resistance.py:139-147`).

## Choosing the line mode among many

A solve returns several modes. `select_line_mode(modes)` keeps the ones that propagate and returns the slowest (the highest `Re(n_eff)`)
of them (`src/gsim/common/modes.py:402-419`). The default rule, in `propagating_modes`, keeps a mode when
(`src/gsim/common/modes.py:330-335`):

- `Re(n_eff) > min_index` (default 1, the light line);
- `-max_loss_ratio * Re(n_eff) < Im(n_eff) <= MAX_GAIN_RATIO * Re(n_eff)`, that is, it loses no more per unit length than it advances
  in phase (`max_loss_ratio` defaults to 1) and it does not grow (`MAX_GAIN_RATIO` is 1e-3, `src/gsim/common/modes.py:138`);
- `|n_eff| <= max_index`, only when `max_index` is given. **The default is no bound** (`src/gsim/common/modes.py:353`).

It accepts femwell modes, mappings with an `"n_eff"` key (the Palace result rows) or bare complex numbers (`src/gsim/common/modes.py:31-33`).

### More than a few modes: bound `|n_eff|` (issue #498)

Ask for many modes around a lossy conductor and the solve also returns modes confined to the metal, with `|n_eff|` near
`|sqrt(eps_metal)|`. Some of them lose less per unit length than they advance, so the loss bound keeps them, and the slowest-mode rule then
prefers one of them to the line mode (`src/gsim/common/modes.py:379-386`, `src/gsim/femwell/adapter.py:225-230`). On the aluminium strip
of the series-resistance test, a 12-mode solve returned modes with `|n_eff|` of 1e4 to 5e4, and the slowest-mode rule read `R'` about
1e9 off (`tests/femwell/test_series_resistance.py:150-157`).

The fix is a bound on `|n_eff|` that you pass yourself: `select_line_mode(modes, max_index=conductor_index_bound(epsilon))`, with the
same `epsilon` you gave `solve_modes` (`src/gsim/femwell/adapter.py:354-356`). The bound is `fraction` (default 0.1) times the smallest
`|sqrt(eps)|` among the conductors, a region being a conductor when `|Im(eps)| > |Re(eps)|`. A conductor counts only if that bound is
at least `margin` (default 10) times the largest dielectric index; both parameters must be positive
(`src/gsim/femwell/adapter.py:217-222, 260-265, 278-295, 310`). The test that guards the fixed case asserts `R'` within
3e-3 of the DC value (`tests/femwell/test_series_resistance.py:159-165`). That 12-mode solve is too slow to repeat in the checks of this
page, so the snippet below runs the selection on 12 indices instead, rounded to six digits. The first is a line mode; the other eleven
only illustrate modes in the metal, because a second solve gave a different set:

```python
# Many modes around a lossy conductor: the metal modes beat the line mode
import numpy as np
from scipy.constants import epsilon_0

from gsim.common.modes import select_line_mode
from gsim.femwell import conductor_index_bound

omega = 2 * np.pi * 1e9
epsilon = {"metal": 1.0 - 1j * 3.8e7 / (omega * epsilon_0), "oxide": 3.9 + 0j}
n_eff = [
    5.12968 - 4.67966j, 1.04699 + 12491.3j, 3786.73 - 23394.9j, 1.5364 - 24982.7j,
    38.7311 + 24981.7j, 21003.8 - 17303.8j, 26057.6 - 7991.57j, 13594.7 + 24559.1j,
    0.289707 + 27931.6j, 12415.5 + 26401.6j, 28275.8 + 10582.1j, 143.234 + 29667.2j,
]
unbounded = select_line_mode(n_eff)
bound = conductor_index_bound(epsilon)
assert bound is not None
line = select_line_mode(n_eff, max_index=bound)
print(unbounded, round(bound, 1), line)
assert abs(unbounded) > bound and line == n_eff[0]
```

The unbounded call picks `26057.6 - 7991.57j`, a mode in the metal; the bound is 2613.5 and the bounded call picks the line mode
(`tests/femwell/test_femwell_docs.py:276-287`).

Things to watch:

- **Nothing applies the bound for you.** `select_line_mode` defaults to `max_index=None`. The TCAD notebook calls
  `solve_modes(..., num_modes=4, ...)` and `select_line_mode(modes)` without a bound (`nbs/tcad_twmzm.ipynb:1005, 1010`); I did not check
  whether four modes can bring in metal modes there, and the code comments speak of "more than a few" modes.
- **`conductor_index_bound` can return `None`.** That happens when a conductor is weak, such as moderately doped silicon: a bound taken
  from a strong conductor next to it would let a mode inside the weak one through, so the function returns no bound and warns
  (`src/gsim/femwell/adapter.py:297-309`; tests `tests/femwell/test_adapter.py:165-197`). `max_index=None` then means no bound at all,
  so test `bound is None` and select the mode another way, for example with `field_fraction_outside` or an explicit `max_index`
  (`src/gsim/femwell/adapter.py:248-255`; `tests/femwell/test_femwell_docs.py:309-316`).
- **The loss bound is separate.** Tightening `max_loss_ratio` (for example to 0.5) drops spurious modes that sit just inside the default
  (`src/gsim/common/modes.py:17-22`; `tests/common/test_mode_selection.py:115-126`).

### Degenerate candidates and `NoLineModeError`

When two candidates are within `degeneracy_rtol` (default 0.03) of the selected `Re(n_eff)`, the choice between them is not physically
meaningful and `select_line_mode` issues a `UserWarning` that names the indices (`src/gsim/common/modes.py:419-437`). When no mode
survives, it raises `NoLineModeError`, a `ValueError`, that lists every solved index and says what to try: solve more modes, move
`n_guess` toward the expected line index, or pass a `rule=` (`src/gsim/common/modes.py:150, 412-417`):

```python
# Degenerate candidates and no candidate
import warnings

from gsim.common.modes import NoLineModeError, select_line_mode

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    picked = select_line_mode([3.0 - 0.01j, 2.95 - 0.02j, 1.5 + 0j])
assert picked == 3.0 - 0.01j and "degenerate" in str(caught[0].message)

try:
    select_line_mode([0.9 + 0j, 3.0 - 4.0j])
except NoLineModeError as error:
    print(error)
```

### CPW and shielded pairs: pass `rule=`

A CPW or a shielded pair carries a second propagating mode that the index alone does not separate from the line mode: the
slotline-like mode of a CPW, or the wall-return mode of a shielded pair. Which of the two travels slower depends on the geometry, and the
degeneracy warning fires only when they sit close (`src/gsim/common/modes.py:358-365`). Where the order is not known, separate them on
a field quantity and pass `rule=`, a function that receives every solved mode and returns the admissible ones; the slowest of those is
selected (`src/gsim/common/modes.py:372-375`). The CPW mode has its two gap voltages in phase and the slotline-like mode has them in
antiphase; `PalaceTextResults.mode_voltage` reads them (`src/gsim/common/modes.py:365-368`, `src/gsim/palace/results.py:236-238`). For a
shielded pair, `line_reading` reports `wall_mode` from the balance of the two electrode currents (`src/gsim/femwell/adapter.py:744-752`;
used at `nbs/tcad_twmzm.ipynb:1009-1016`). A rule that keeps the modes inside a band you know is
`rule=lambda solved: [m for m in propagating_modes(solved) if m.n_eff.real < 4.0]` (`tests/femwell/test_femwell_docs.py:298-306`).

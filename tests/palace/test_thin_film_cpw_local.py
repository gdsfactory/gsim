"""Palace boundary-mode thin-film CPW against an independent reference.

A zero-thickness CPW on silicon (eps_r 11.45, 500 um) is solved with
``add_sheet_impedance`` on every metal sheet at 6 GHz. The reference is
``L' = L'_geo + L'_k`` with ``L'_k`` from an independent thin-sheet solver and
``C'`` from conformal mapping, giving ``Z0 = sqrt(L'/C')`` and
``n_eff = c sqrt(L' C')``. The cached numbers below are those reference values.

Marked ``palace_local`` because each case launches Palace. Run with::

    pytest tests/palace/test_thin_film_cpw_local.py -m palace_local
"""

from __future__ import annotations

import gdsfactory as gf
import pytest

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.palace import BoundaryModeSim
from gsim.palace.line_impedance import line_impedance_paths
from gsim.palace.runtime import resolve_palace_binary

pytestmark = pytest.mark.palace_local

FREQ = 6e9
EPS_SI = 11.45
SUBSTRATE_UM = 500.0
GROUND_UM = 500.0
HALF_UM = 2000.0
AIR_UM = 1000.0
FILM_UM = 0.004
MESH_UM = 0.25
# The search aims at the geometric n_eff. With kinetic inductance the CPW mode
# is a higher one of the three, so the case picks the mode nearest the reference.
SEARCH_N_EFF = 2.4951
Z0_RTOL = 0.015
N_EFF_RTOL = 0.01

# (strip width um, gap um, sheet inductance pH/sq, reference Z0 ohm, reference n_eff)
CASES = {
    "10_6_geometric": (10.0, 6.0, 0.0, 50.909, 2.4951),
    "10_6_kinetic": (10.0, 6.0, 23.2, 135.199, 6.62623),
    "4_2.5_kinetic": (4.0, 2.5, 23.2, 201.717, 9.77166),
}


def _binary_available() -> bool:
    try:
        return resolve_palace_binary(download_if_missing=False) is not None
    except Exception:
        return False


if not _binary_available():  # pragma: no cover - environment dependent
    pytest.skip("No runnable Palace binary", allow_module_level=True)


def _cpw(width: float, gap: float) -> gf.Component:
    gf.gpdk.PDK.activate()
    component = gf.Component()
    component.add_polygon(
        [(-50, -width / 2), (50, -width / 2), (50, width / 2), (-50, width / 2)],
        layer=(1, 0),
    )
    lo, hi = width / 2 + gap, width / 2 + gap + GROUND_UM
    for a, b in ((lo, hi), (-hi, -lo)):
        component.add_polygon([(-50, a), (50, a), (50, b), (-50, b)], layer=(1, 0))
    return component


def _stack() -> LayerStack:
    stack = LayerStack(pdk_name="cpw", units="um")
    stack.materials = {"si": {"permittivity": EPS_SI}, "film": {"conductivity": 0.0}}
    stack.layers = {
        "film": Layer(
            name="film",
            gds_layer=(1, 0),
            zmin=0.0,
            zmax=FILM_UM,
            thickness=FILM_UM,
            material="film",
            layer_type="conductor",
        )
    }
    stack.dielectrics = [
        {"name": "si", "zmin": -SUBSTRATE_UM, "zmax": 0.0, "material": "si"}
    ]
    return stack


@pytest.mark.parametrize("case", CASES)
def test_thin_film_cpw_matches_the_reference(tmp_path, case):
    width, gap, ls_ph, z0_ref, n_eff_ref = CASES[case]
    extent = width / 2 + gap + GROUND_UM
    sim = BoundaryModeSim()
    sim.set_output_dir(str(tmp_path))
    sim.set_geometry(_cpw(width, gap))
    sim.set_stack(_stack())
    sim.set_airbox(
        margin_x=10.0, margin_y=HALF_UM - extent, z_above=AIR_UM, z_below=0.0
    )
    sim.set_cross_section("x=0")
    sim.set_boundary_mode(freq=FREQ, num_modes=3, target=SEARCH_N_EFF)
    if ls_ph:
        sim.add_sheet_impedance("film", inductance=ls_ph * 1e-12)
    sim.mesh(preset="default", refined_mesh_size=MESH_UM, verbose=False)
    on_film = (0.0, 0.0)
    paths = line_impedance_paths(
        signal=((-width / 2, width / 2), on_film),
        ground=((width / 2 + gap, extent), on_film),
        domain=sim.mesh_extent,
        others=[((-extent, -width / 2 - gap), on_film)],
    )
    sim.add_impedance_path("cpw", voltage=paths.voltage, current=paths.current)

    results = sim.run_local()

    mode = min(
        results.modes,
        key=lambda m: abs(complex(results.modes[m]["n_eff"]).real - n_eff_ref),
    )
    n_eff = complex(results.modes[mode]["n_eff"]).real
    z0 = results.characteristic_impedance(index=1, mode=mode, quantity="Z_PV")
    assert n_eff == pytest.approx(n_eff_ref, rel=N_EFF_RTOL)
    assert z0 == pytest.approx(z0_ref, rel=Z0_RTOL)

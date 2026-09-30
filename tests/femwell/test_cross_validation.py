"""Palace BoundaryMode vs femwell on the identical mesh and materials.

The cross-validation contract of the TCAD TW-MZM workflow: one shared
fixture builds the native-2D mesh and its piecewise-constant materials
once, and both solvers must agree on n_eff to solver tolerance. This is
the gate for trusting the femwell adapter before enabling continuous
eps(x, y) in examples.

Requires both runtimes (Palace binary + femwell); deselected by default,
run with ``pytest -m palace_local``.

Note the RF-materials case exercises conductive (complex) permittivity on
the shared mesh at the optical domain scale: native-2D meshes represent
electrodes as boundary curves, not volumes, so a true metal CPW cannot be
expressed identically in both solvers — the piecewise-constant material
contract is what is validated here.

What it does not cover: both solvers are handed the same mesh and the
same materials, so the agreement is between the two solvers and not
between either of them and the drawn device.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import speed_of_light as C0  # noqa: N812

from gsim.femwell.adapter import epsilon_by_region, solve_modes
from gsim.palace import BoundaryModeSim
from gsim.palace.results import load_text_results

pytest.importorskip("femwell")
pytest.importorskip("skfem")

pytestmark = pytest.mark.palace_local

WL_UM = 1.55
FREQ_HZ = C0 / (WL_UM * 1e-6)
# Same finite-element problem, different discretizations (Palace order-2
# Nedelec vs femwell order-1): agreement to ~1e-2 relative on n_eff.
N_EFF_RTOL = 1e-2
SIGMA_S_PER_M = 5e3


def _palace_available() -> bool:
    from gsim.palace.runtime import resolve_palace_binary

    try:
        return resolve_palace_binary() is not None
    except Exception:
        return False


if not _palace_available():  # pragma: no cover - environment dependent
    pytest.skip("Palace binary not available", allow_module_level=True)


def _build_sim(tmp_path, *, conductive: bool):
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    comp = gf.Component()
    comp << gf.c.rectangle((10.0, 0.5), centered=True, layer=(1, 0))

    sim = BoundaryModeSim()
    sim.set_output_dir(str(tmp_path))
    sim.set_stack(substrate_thickness=1.0)
    sim.set_geometry(comp)
    sim.set_airbox(
        margin_x=1.2, margin_y=1.2, z_above=1.0, z_below=0.8, material="sio2"
    )
    sim.set_cross_section("x=0")
    sim.set_boundary_mode(freq=FREQ_HZ, num_modes=1, target=3.2)
    if conductive:
        # RF-materials contract: doped-silicon-like complex permittivity.
        sim.set_material("si", permittivity=11.9, conductivity=SIGMA_S_PER_M)
    sim.mesh(preset="coarse", refined_mesh_size=0.04, max_mesh_size=0.4, verbose=False)
    sim.write_config()
    return sim


def _solve_both(sim):
    from gsim.palace.runtime import resolve_palace_binary

    results = sim.run_local(palace_executable=resolve_palace_binary(), verbose=False)
    text = results if hasattr(results, "modes") else load_text_results(results)
    n_eff_palace = text.modes[1]["n_eff"]

    eps = epsilon_by_region(sim.mesh_path, sim.stack, wavelength_um=WL_UM)
    modes = solve_modes(
        sim.mesh_path,
        epsilon=eps,
        wavelength_um=WL_UM,
        num_modes=1,
    )
    n_eff_femwell = complex(modes[0].n_eff)
    return n_eff_palace, n_eff_femwell


class TestCrossSolverContract:
    def test_optical_n_eff_agreement(self, tmp_path):
        sim = _build_sim(tmp_path, conductive=False)
        n_eff_palace, n_eff_femwell = _solve_both(sim)
        assert np.isfinite(n_eff_palace.real)
        assert abs(n_eff_palace.real - n_eff_femwell.real) < N_EFF_RTOL * abs(
            n_eff_palace.real
        )
        # Both must see a guided silicon mode, not cladding.
        assert n_eff_palace.real > 2.0
        assert n_eff_femwell.real > 2.0

    def test_rf_materials_complex_n_eff_agreement(self, tmp_path):
        sim = _build_sim(tmp_path, conductive=True)
        n_eff_palace, n_eff_femwell = _solve_both(sim)
        assert abs(n_eff_palace - n_eff_femwell) < N_EFF_RTOL * abs(n_eff_palace)
        # The conductive core makes the mode lossy in both solvers
        # (exp(+i omega t): Im{n_eff} < 0).
        assert n_eff_palace.imag != 0.0
        assert n_eff_femwell.imag != 0.0
        assert np.sign(n_eff_palace.imag) == np.sign(n_eff_femwell.imag)

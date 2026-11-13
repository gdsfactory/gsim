"""Regression test: the metal-only coplanar-strip line of Baehr-Jones 2012 (femwell).

Baehr-Jones et al., Opt. Express 20(11) 12014 (2012),
https://doi.org/10.1364/OE.20.012014. The cross-section is the generic one of
``nbs/palace_bj2012_electrode.ipynb``: two 50 um wide, 2 um thick aluminum strips
with a 5 um gap, in SiO2 up to the top of the strips (no oxide over them), 1.5 um above
a 2 um BOX on a truncated 750 ohm.cm handle wafer, inside a +-200 um wide shielded box.
The paper's measured control line at 10 GHz (``nbs/data/bj2012_reference.json``):
Z = 61 - 5j ohm, L = 457 nH/m, C = 0.12 nF/m, R = 4.9 kohm/m, 3.8 dB/cm.

What the checks mean:

- the line mode is the one the electrodes carry equal and opposite currents on; the
  shielded box has a slower common ("wall") mode that the default selection would
  pick;
- n and Z against stored values of this mesh (+-2 %): a regression guard;
- the perfect-metal Z, L and C against the paper (+-20 %): L of a perfect metal is
  geometric and 13 % below the paper's, whose L includes the metal's internal
  inductance;
- L = Z n / c must not depend on a uniform dielectric (non-magnetic), checked on the
  same mesh filled with eps 1 and 4;
- the loss of the bulk-aluminum solve against the power-loss estimate from the
  lossless mode's fields (an independent route to R), and its ratio to the paper's
  loss (about two: the documented gap);
- the lossy solve with the effective conductivity of the notebook (1e7 S/m, chosen
  so that R is the paper's) against the paper in R, Z, L, C, n and the loss (+-10 %).
  Only R is fitted; the others are predictions, but L and the loss are not independent
  of R (internal inductance, R / 2Z).

The mesh is the coarse one (about 3 % above the converged perfect-metal Z). The
notebook builds the same cross-section with background dielectric boxes instead of
patterned layers (it needs them for the 3D runs).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import gdsfactory as gf
import numpy as np
import pytest
from scipy.constants import mu_0 as MU0  # noqa: N812
from scipy.constants import speed_of_light as C0  # noqa: N812

from gsim.common.modes import Conductor, propagating_modes, select_line_mode
from gsim.common.stack.extractor import Layer, LayerStack
from gsim.common.stack.materials import MaterialProperties
from gsim.common.transmission_line import line_params_from_neff
from gsim.femwell import epsilon_by_region, line_reading, solve_modes
from gsim.femwell.adapter import electrode_current
from gsim.palace import BoundaryModeSim
from tests._helpers import draw_rectangle

pytest.importorskip("femwell")
pytest.importorskip("skfem")
pytest.importorskip("gmsh")

REFERENCE = json.loads(
    (
        Path(__file__).resolve().parents[2] / "nbs" / "data" / "bj2012_reference.json"
    ).read_text(encoding="utf-8")
)
GEOMETRY = REFERENCE["geometry"]
PAPER = REFERENCE["metal_only_line_10GHz"]

FREQ_HZ = 10e9
STRIP_WIDTH = GEOMETRY["electrode_strip_width_um"]
GAP = GEOMETRY["electrode_gap_um"]
METAL_THICKNESS = GEOMETRY["top_metal_thickness_um"]
BOX_THICKNESS = GEOMETRY["box_thickness_um"]
HANDLE_DEPTH = 100.0  # truncated handle wafer (um)
HALF_WIDTH = 200.0  # half the width of the shielded box (um)
AIR_ABOVE = 100.0  # air above the oxide (um)
METAL_BOTTOM = 1.5  # lower face of the strips above the BOX (um)
OXIDE_OVER_METAL = 0.0  # SiO2 above the strips (um): none
SIGMA_AL_BULK = 3.77e7  # S/m, bulk aluminum
SIGMA_AL = 1.0e7  # S/m, effective: R at 10 GHz is the paper's
SIGMA_HANDLE = 100.0 / GEOMETRY["handle_resistivity_ohm_cm"]  # S/m

#: Stored results of this mesh at 10 GHz, from the run that wrote the test.
STORED = {
    "n_eff": 2.0432,
    "wall_n_eff": 2.3616,
    "z0_ohm": 58.16,
    "l_nh_per_m": 396.4,
    "alpha_bulk_db_per_cm": 1.827,
    "alpha_effective_db_per_cm": 3.742,
}

STRIPS = {  # name: (y0, y1) in um
    "strip_g": (-GAP / 2 - STRIP_WIDTH, -GAP / 2),
    "strip_s": (GAP / 2, GAP / 2 + STRIP_WIDTH),
}
STRIP_Z = (METAL_BOTTOM, METAL_BOTTOM + METAL_THICKNESS)
GAP_BOX = (-10.0, 10.0, 0.8, 5.0, 0.5)  # (y0, y1, z0, z1, size) refinement in the gap


def _build(*, lossy_metal: bool, sigma: float = SIGMA_AL):
    """Component and hand-built stack of the CPS cross-section (strips along x)."""
    gf.gpdk.PDK.activate()
    comp = gf.Component()
    layers: dict[str, Layer] = {}

    def add(name, gds_layer, y0, y1, z0, z1, material, layer_type, mesh):
        draw_rectangle(
            comp, gds_layer, width=10.0, height=y1 - y0, x0=0.0, y=(y0 + y1) / 2
        )
        layers[name] = Layer(
            name=name,
            gds_layer=gds_layer,
            zmin=z0,
            zmax=z1,
            thickness=z1 - z0,
            material=material,
            layer_type=layer_type,
            mesh_resolution=mesh,
        )

    z_oxide_top = STRIP_Z[1] + OXIDE_OVER_METAL
    wide = (-HALF_WIDTH, HALF_WIDTH)
    handle_z = (-BOX_THICKNESS - HANDLE_DEPTH, -BOX_THICKNESS)
    add("handle", (2, 0), *wide, *handle_z, "handle_si", "dielectric", 20.0)
    add("box", (3, 0), *wide, -BOX_THICKNESS, 0.0, "sio2_rf", "dielectric", 5.0)
    add("oxide", (5, 0), *wide, 0.0, z_oxide_top, "sio2_rf", "dielectric", 5.0)
    # A conductor layer is meshed as an outline, a perfect-conductor hole; with
    # lossy metal the strip is a dielectric region of aluminum instead.
    strip_type = "dielectric" if lossy_metal else "conductor"
    for i, (name, span) in enumerate(STRIPS.items()):
        add(name, (10, i), *span, *STRIP_Z, "aluminum", strip_type, 1.0)
    stack = LayerStack(
        pdk_name="bj2012_cps",
        layers=layers,
        materials={
            "sio2_rf": MaterialProperties(permittivity=3.9).to_dict(),
            "handle_si": MaterialProperties(
                permittivity=11.9, conductivity=SIGMA_HANDLE
            ).to_dict(),
            "aluminum": MaterialProperties(
                permittivity=1.0, conductivity=sigma
            ).to_dict(),
            "air": MaterialProperties(permittivity=1.0).to_dict(),
        },
    )
    return comp, stack


def _mesh(comp, stack, output_dir):
    sim = BoundaryModeSim()
    sim.set_output_dir(output_dir)
    sim.set_stack(stack)
    sim.set_geometry(comp)
    sim.set_airbox(margin_x=0.0, margin_y=0.0, z_above=AIR_ABOVE, material="air")
    sim.set_cross_section("x=5")
    sim.mesh(
        preset="coarse",
        refined_mesh_size=1.0,
        max_mesh_size=20.0,
        refinement_boxes=[GAP_BOX],
        verbose=False,
    )
    return sim.mesh_path


def _conductors(model):
    signal = Conductor("strip_s", (STRIPS["strip_s"], STRIP_Z), model)
    ground = Conductor("strip_g", (STRIPS["strip_g"], STRIP_Z), model)
    return signal, ground


def _solve(mesh, epsilon, *, order, num_modes=4, n_guess=2.3, min_index=1.0):
    """Every propagating mode of the shielded box."""
    modes = solve_modes(
        mesh,
        epsilon=epsilon,
        wavelength_um=C0 / FREQ_HZ * 1e6,
        num_modes=num_modes,
        order=order,
        metallic_boundaries=True,
        n_guess=n_guess,
    )
    return list(propagating_modes(list(modes), min_index=min_index))


def _readings(mesh, modes, model):
    signal, ground = _conductors(model)
    return [
        (
            mode,
            line_reading(
                mode, frequency_hz=FREQ_HZ, mesh=mesh, signal=signal, return_=ground
            ),
        )
        for mode in modes
    ]


def _line(readings):
    """The line mode: the slowest propagating mode that is not the wall mode."""
    candidates = [(m, r) for m, r in readings if not r.wall_mode]
    return max(candidates, key=lambda mr: mr[1].n_eff.real)


@pytest.fixture(scope="module")
def pec(tmp_path_factory):
    comp, stack = _build(lossy_metal=False)
    mesh = _mesh(comp, stack, tmp_path_factory.mktemp("bj2012-cps-pec"))
    epsilon = epsilon_by_region(mesh, stack, frequency_hz=FREQ_HZ)
    # Order 2: the contour-current impedance is not converged at order 1.
    readings = _readings(mesh, _solve(mesh, epsilon, order=2), "pec")
    return {"mesh": mesh, "stack": stack, "readings": readings, "epsilon": epsilon}


@pytest.fixture(scope="module")
def pec_line(pec):
    mode, reading = _line(pec["readings"])
    params = line_params_from_neff([FREQ_HZ], [reading.n_eff], z0_ohm=[reading.z0_ohm])
    rlgc = {name: float(value[0]) for name, value in params.rlgc.items()}
    return {"mode": mode, "reading": reading, "rlgc": rlgc, "params": params}


class TestLineMode:
    def test_the_shielded_box_has_a_slower_wall_mode(self, pec):
        walls = [r for _m, r in pec["readings"] if r.wall_mode]
        lines = [r for _m, r in pec["readings"] if not r.wall_mode]
        assert len(walls) == 1
        assert len(lines) == 1
        assert walls[0].n_eff.real > lines[0].n_eff.real
        # The wall mode is the one the default rule would hand back.
        slowest = select_line_mode([m for m, _r in pec["readings"]])
        assert complex(slowest.n_eff) == pytest.approx(complex(walls[0].n_eff))

    def test_a_wave_port_numbers_the_wall_mode_first(self, pec):
        """The line mode is port mode 2: Palace numbers port modes by decreasing index.

        The wave port's eigenproblem is this problem (perfect conductors on the strips
        and
        on the outer wall). In a Palace log of a 200 um section of an earlier geometry
        (2 um of oxide over the strips) the port's mode 1 had n = 2.3823 at 10 GHz, the
        wall mode of that geometry's solve to 0.1 %. A driven section whose ports carry
        mode 1 sees no wave of the 3D line (S21 = -3.5 dB for 200 um, the de-embedded
        index 6.4 instead of 2.2), so the notebook takes the port mode number from this
        ranking.
        """
        ranked = sorted(pec["readings"], key=lambda mr: -mr[1].n_eff.real)
        assert [r.wall_mode for _m, r in ranked] == [True, False]
        wall, line = (r.n_eff.real for _m, r in ranked)
        assert wall == pytest.approx(STORED["wall_n_eff"], rel=0.02)
        assert line == pytest.approx(STORED["n_eff"], rel=0.02)
        assert wall > 1.1 * line

    def test_the_line_mode_matches_the_stored_run(self, pec_line):
        reading = pec_line["reading"]
        assert reading.n_eff.real == pytest.approx(STORED["n_eff"], rel=0.02)
        assert reading.z0_ohm.real == pytest.approx(STORED["z0_ohm"], rel=0.02)
        assert pec_line["rlgc"]["L"] * 1e9 == pytest.approx(
            STORED["l_nh_per_m"], rel=0.02
        )

    def test_a_lossless_line_has_a_real_impedance(self, pec_line):
        z0 = pec_line["reading"].z0_ohm
        assert abs(z0.imag) < 0.02 * z0.real


class TestAgainstThePaper:
    def test_impedance_inductance_and_capacitance(self, pec_line):
        z0 = pec_line["reading"].z0_ohm
        rlgc = pec_line["rlgc"]
        assert z0.real == pytest.approx(PAPER["z0_real_ohm"], rel=0.20)
        assert rlgc["L"] * 1e9 == pytest.approx(PAPER["l_nH_per_m"], rel=0.20)
        assert rlgc["C"] * 1e9 == pytest.approx(PAPER["c_nF_per_m"], rel=0.20)

    def test_the_index_is_that_of_a_line_on_oxide_and_silicon(self, pec_line):
        # Between the oxide (sqrt(3.9) = 1.97) and the silicon (3.45) the paper's
        # 2.22 (from L and C) sits; the perfect-metal line is below it, the
        # lossy one (internal inductance) is at it (see TestEffectiveConductivity).
        n = pec_line["reading"].n_eff.real
        assert np.sqrt(3.9) < n < np.sqrt(11.9)


def _pure_line_mode(mesh, modes):
    """The line mode of a degenerate pair: equal and opposite electrode currents.

    In a uniformly filled shielded box the line mode and the wall mode are both TEM
    and have the same index (n = sqrt(eps)), so the eigensolver returns any
    combination of the two: which one depends on its start vector and on the
    linear-algebra library, not on the physics. The line mode is the combination
    whose two electrodes carry currents that cancel. The currents and fields are
    linear in the mode, so that combination is ``I_b E_a - I_a E_b`` with
    ``I = I_signal + I_return`` of each solved mode.
    """
    assert len(modes) == 2, f"expected the degenerate pair, got {len(modes)} modes"
    signal, ground = _conductors("pec")
    first, second = modes
    total = [
        sum(
            electrode_current(m, frequency_hz=FREQ_HZ, conductor=c, mesh=mesh)
            for c in (signal, ground)
        )
        for m in modes
    ]
    return dataclasses.replace(
        first,
        E=total[1] * first.E - total[0] * second.E,
        H=total[1] * first.H - total[0] * second.H,
    )


class TestInductanceIsGeometric:
    def test_l_does_not_depend_on_a_uniform_dielectric(self, pec, pec_line):
        """Non-magnetic materials: L = Z n / c is the same for a box filled with eps 1
        or 4, and is that of the built line.

        A uniform filling is exactly TEM, so n scales with sqrt(eps) and Z against it.
        The line and wall modes are degenerate there, so the line mode is rebuilt from
        the solved pair (``_pure_line_mode``). Reading the "line" mode of the raw pair
        gave 343 and 352 nH/m on one machine and 312 and 426 nH/m in CI: the readings of
        a mixture of the two modes run from 300 to 40000 nH/m.
        """
        inductance = {}
        for eps_r, n_guess in ((1.0, 1.0), (4.0, 2.0)):
            filled = dict.fromkeys(pec["epsilon"], eps_r + 0j)
            modes = [
                m
                for m in _solve(
                    pec["mesh"], filled, order=2, n_guess=n_guess, min_index=0.99
                )
                if abs(m.n_eff.real - np.sqrt(eps_r)) < 1e-3 * np.sqrt(eps_r)
            ]
            mode = _pure_line_mode(pec["mesh"], modes)
            signal, ground = _conductors("pec")
            reading = line_reading(
                mode,
                frequency_hz=FREQ_HZ,
                mesh=pec["mesh"],
                signal=signal,
                return_=ground,
            )
            assert reading.wall_mode is False
            assert reading.n_eff.real == pytest.approx(np.sqrt(eps_r), rel=1e-3)
            inductance[eps_r] = reading.z0_ohm.real * reading.n_eff.real / C0
        assert inductance[4.0] == pytest.approx(inductance[1.0], rel=1e-3)
        # The built line (oxide, silicon, air) has the same L: it is geometric.
        assert inductance[1.0] * 1e9 == pytest.approx(
            pec_line["rlgc"]["L"] * 1e9, rel=0.01
        )


def _power_loss_resistance(mode, mesh, signal, ground, sigma_s_per_m):
    """R of a pec mode by the power-loss method: R_s * (contour |H|^2 dl) / |I|^2."""
    import skfem

    from gsim.femwell.adapter import boundary_facets_within, electrode_current

    surface_resistance = np.sqrt(np.pi * FREQ_HZ * MU0 / sigma_s_per_m)
    current = abs(
        electrode_current(mode, frequency_hz=FREQ_HZ, conductor=signal, mesh=mesh)
    )

    @skfem.Functional(dtype=np.float64)
    def _h_squared(w):
        (h_x, h_y), h_z = w["H"]
        return np.abs(h_x) ** 2 + np.abs(h_y) ** 2 + np.abs(h_z) ** 2

    total = 0.0
    for conductor in (signal, ground):
        h_span, v_span = conductor.extent
        facets = boundary_facets_within(mode.basis.mesh, h_span=h_span, v_span=v_span)
        facet_basis = skfem.FacetBasis(mode.basis.mesh, mode.basis.elem, facets=facets)
        total += _h_squared.assemble(facet_basis, H=facet_basis.interpolate(mode.H))
    # Mesh in um, H in A/m: the um contour integral and the um-scale current give
    # R = R_s * 1e6 * integral / |I|^2 (the same scale electrode_current lands in).
    return surface_resistance * 1e6 * total / current**2


def _lossy_line(tmp_path_factory, sigma, label):
    """The line mode with the strips meshed as lossy aluminum of conductivity
    ``sigma``."""
    comp, stack = _build(lossy_metal=True, sigma=sigma)
    mesh = _mesh(comp, stack, tmp_path_factory.mktemp(f"bj2012-cps-{label}"))
    epsilon = epsilon_by_region(mesh, stack, frequency_hz=FREQ_HZ)
    # Order 2: Z, L and C come from the contour/conduction current.
    modes = _solve(mesh, epsilon, order=2, num_modes=6)
    _mode, reading = _line(_readings(mesh, modes, "volume"))
    params = line_params_from_neff([FREQ_HZ], [reading.n_eff], z0_ohm=[reading.z0_ohm])
    return {
        "reading": reading,
        "rlgc": {k: float(v[0]) for k, v in params.rlgc.items()},
        "alpha_db_per_cm": float(params.alpha_rf_np_m[0]) * 8.686 / 100,
    }


@pytest.fixture(scope="module")
def lossy(tmp_path_factory):
    """Bulk aluminum."""
    return _lossy_line(tmp_path_factory, SIGMA_AL_BULK, "lossy-bulk")


@pytest.fixture(scope="module")
def lossy_effective(tmp_path_factory):
    """The notebook's effective conductivity."""
    return _lossy_line(tmp_path_factory, SIGMA_AL, "lossy-effective")


class TestMetalLoss:
    def test_the_loss_matches_the_stored_run(self, lossy, lossy_effective):
        assert lossy["alpha_db_per_cm"] == pytest.approx(
            STORED["alpha_bulk_db_per_cm"], rel=0.03
        )
        assert lossy_effective["alpha_db_per_cm"] == pytest.approx(
            STORED["alpha_effective_db_per_cm"], rel=0.03
        )

    def test_the_loss_matches_the_power_loss_estimate(self, pec, pec_line, lossy):
        """Wall losses from the lossless mode's fields, plus its own substrate loss
        (bulk Al)."""
        signal, ground = _conductors("pec")
        resistance = _power_loss_resistance(
            pec_line["mode"], pec["mesh"], signal, ground, SIGMA_AL_BULK
        )
        z0 = pec_line["reading"].z0_ohm.real
        alpha_metal = resistance / (2 * z0) * 8.686 / 100  # dB/cm
        alpha_substrate = float(pec_line["params"].alpha_rf_np_m[0]) * 8.686 / 100
        # The perfect-metal estimate misses the current crowding at the sharp corners
        # that
        # a mesh resolves only slowly (it is 8 % low here, about 10 % at finer corners).
        assert alpha_metal + alpha_substrate == pytest.approx(
            lossy["alpha_db_per_cm"], rel=0.15
        )

    def test_the_paper_loss_is_about_twice_that_of_bulk_aluminum(self, lossy):
        """Documented gap: 3.8 dB/cm measured against about 1.8 simulated for bulk
        aluminum.

        The simulation reproduces the skin-effect loss of bulk aluminum (checked above);
        the paper's 4.9 kohm/m needs a metal about four times less conductive. If this
        ratio moves, the geometry or the metal changed and the notebook's comparison
        table needs revisiting.
        """
        ratio = PAPER["alpha_dB_per_cm"] / lossy["alpha_db_per_cm"]
        assert 1.5 < ratio < 2.5


class TestEffectiveConductivity:
    """With sigma = 1e7 S/m (set from R) the other quantities land on the paper's."""

    def test_the_resistance_is_the_papers(self, lossy_effective):
        assert lossy_effective["rlgc"]["R"] * 1e-3 == pytest.approx(
            PAPER["r_kohm_per_m"], rel=0.10
        )

    def test_the_prediction_for_z_l_c_n_and_the_loss(self, lossy_effective):
        z0 = lossy_effective["reading"].z0_ohm
        rlgc = lossy_effective["rlgc"]
        assert z0.real == pytest.approx(PAPER["z0_real_ohm"], rel=0.10)
        assert z0.imag == pytest.approx(PAPER["z0_imag_ohm"], abs=2.0)
        assert rlgc["L"] * 1e9 == pytest.approx(PAPER["l_nH_per_m"], rel=0.10)
        assert rlgc["C"] * 1e9 == pytest.approx(PAPER["c_nF_per_m"], rel=0.10)
        assert lossy_effective["reading"].n_eff.real == pytest.approx(2.22, rel=0.05)
        assert lossy_effective["alpha_db_per_cm"] == pytest.approx(
            PAPER["alpha_dB_per_cm"], rel=0.10
        )

    def test_the_inductance_rises_with_the_resistance(
        self, pec_line, lossy, lossy_effective
    ):
        """L is Im(series impedance) / omega: perfect metal < bulk aluminum < effective
        metal.

        Measured on this mesh: 396 < 408 < 436 nH/m (the rise is the metal's internal
        inductance, less than R / omega because the current also redistributes).
        """
        l_pec = pec_line["rlgc"]["L"]
        l_bulk = lossy["rlgc"]["L"]
        l_effective = lossy_effective["rlgc"]["L"]
        assert l_pec < l_bulk < l_effective
        assert l_effective - l_bulk > 15e-9

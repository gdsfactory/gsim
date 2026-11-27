"""The lossy CPW of benvial/gfem#2 as a gfem / femwell / Palace comparison.

A shielded coplanar line (gpdk-like stack, 10 um signal, 6 um gaps, 2 um
aluminium at 3.77e7 S/m, silicon below, 100 um of air above) is meshed by
``BoundaryModeSim`` with the metal as lossy volume regions, and gfem solves the
line's port mode on that mesh through ``gsim.gfem``.

Three references:

- the frozen femwell and Palace numbers of benvial/gfem#2 (always compared);
- a live femwell solve on the same mesh, when femwell is installed;
- live Palace results, when ``GSIM_GFEM_PALACE_RESULTS`` points at downloaded
  runs (see ``test_gfem_matches_live_palace``). Nothing here submits a job.

The tests that solve with gfem are marked ``gfem_local`` and deselected by
default; run them with ``pytest -m gfem_local``.

Environment knobs, all optional:

- ``GSIM_GFEM_CPW_FULL=1``: the four frozen frequencies on a 0.25 um metal mesh
  instead of 1 and 50 GHz on a 0.5 um one (a gfem port-mode solve is slow);
- ``GSIM_GFEM_CPW_METAL_UM``: element size in the metal (the frozen numbers
  were computed at 0.15);
- ``GSIM_GFEM_PALACE_RESULTS``: folder of downloaded Palace runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import meshio
import numpy as np
import pytest
from scipy.constants import speed_of_light as C0  # noqa: N812

from gsim.common.modes import Conductor
from gsim.gfem import adapter

pytest.importorskip("gmsh")

# Geometry, um: the line runs along x, the cross-section is the (y, z) plane, and
# the mesh coordinates are (h, v) = (y, z).
X_BOX, Z_BOT, Z_TOP = 150.0, -203.0, 105.2
Z_SI, Z_M0, Z_M1 = -3.0, 3.2, 5.2
EDGE, INNER = 5.0, 11.0
LENGTH_UM = 1000.0
SIGMA = 3.77e7  # S/m
SIGNAL, GROUND, DOMAIN_LAYER = "m3_signal", "m3_ground", "domain"
GROUND_MESH_UM = 2.0
REFINED_UM, MAX_UM = 1.0, 30.0
EDGE_ZONE_UM = 20.0

#: Frozen reference numbers of benvial/gfem#2.
#:
#: The same constants as tests/verification/test_lossy_cpw.py on gfem main
#: (commit 2b5a980; all eight reference constants match exactly), see also
#: benvial/gfem#2. Provenance given there: Palace 2D boundary mode, build
#: 295645f, driven by gsim 0.9.0; femwell 0.1.12; both on the same 27 522-triangle
#: cross-section (metal 0.15 um, order 2). n_eff = n' - i n''.
GHZ = (1e9, 10e9, 25e9, 50e9)
PALACE_N = np.array(
    [
        2.133006767 - 0.613861182j,
        1.910114000 - 0.126555648j,
        1.873480977 - 0.078183035j,
        1.851462947 - 0.056388136j,
    ]
)
FEMWELL_N = np.array(
    [
        2.133005966 - 0.613858958j,
        1.910114697 - 0.126551903j,
        1.873482470 - 0.078173871j,
        1.851467452 - 0.056370523j,
    ]
)
#: Palace's power-voltage impedance of the full line, ohm.
PALACE_Z = np.array(
    [
        66.76036 - 19.21304j,
        59.78197 - 3.96087j,
        58.62420 - 2.44640j,
        57.89154 - 1.76294j,
    ]
)
#: femwell's power-current impedance of the full line, ohm.
FEMWELL_Z_PI = np.array(
    [
        66.81729 - 19.23126j,
        59.83637 - 3.96488j,
        58.68211 - 2.44833j,
        57.96759 - 1.76397j,
    ]
)
#: Attenuation of those indices, dB/cm: from n'' at the frequency.
PALACE_ALPHA = (1.1175, 2.3039, 3.5582, 5.1325)
FEMWELL_ALPHA = (1.1175, 2.3038, 3.5577, 5.1309)

# Two profiles. A gfem port-mode solve costs about 40 s per exactly solved
# frequency at 12.5k triangles, so the default one is the cheap check: a 0.5 um
# metal mesh and 1 and 10 GHz (the skin depth is 2.6 and 0.8 um there). 50 GHz
# needs a finer metal mesh: its skin depth is 0.37 um. ``GSIM_GFEM_CPW_FULL=1``
# runs the four frozen frequencies on a 0.25 um metal mesh;
# ``GSIM_GFEM_CPW_METAL_UM`` overrides the size.
FULL = os.environ.get("GSIM_GFEM_CPW_FULL") == "1"
METAL_UM = float(os.environ.get("GSIM_GFEM_CPW_METAL_UM", "0.25" if FULL else "0.5"))
RUN = (0, 1, 2, 3) if FULL else (0, 1)  # indices into the frozen tables
FREQUENCIES = tuple(GHZ[i] for i in RUN)

#: Relative agreement with the frozen numbers. gfem#3 measured at most 5.5e-4
#: (Re n'), 3.4e-4 (alpha) and 4.4e-4 (Z_PV) against Palace and 5.5e-4 / 6.5e-4
#: against femwell on its own 6949-triangle mesh. On the BoundaryModeSim mesh of
#: the default profile (0.5 um metal, 10 950 triangles, 1 and 10 GHz) the largest
#: difference measured is 8.7e-4 (Z_PI against femwell), so gfem#3's 1.5e-3 holds
#: with a margin of 1.7. The full profile (0.25 um, four frequencies) has not been
#: measured on that mesh.
REFERENCE_RTOL = 1.5e-3
#: Z_PI against a live femwell solve on the same mesh (measured 2.0e-3 at most).
LIVE_FEMWELL_Z_RTOL = 3e-3


def _alpha_of_index(n: np.ndarray, frequencies: tuple[float, ...]) -> np.ndarray:
    """Attenuation (dB/cm) of ``n = n' - i n''`` at ``frequencies`` (Hz)."""
    k0 = 2 * np.pi * np.asarray(frequencies) / C0
    return 20 / np.log(10) * k0 * (-np.asarray(n).imag) / 100


def _alpha_db_per_cm(gamma) -> np.ndarray:
    """Attenuation (dB/cm) of a propagation constant ``gamma`` (1/m)."""
    return 20 / np.log(10) * np.asarray(gamma).real / 100


def _triangle_count(mesh_path: Path) -> int:
    mesh = meshio.read(str(mesh_path), file_format="gmsh")
    return sum(len(block.data) for block in mesh.cells if block.type == "triangle")


def _layer_table():
    """One row per gsim layer: name, GDS layer, y spans, z range, material, size."""
    return [
        # a thin oxide slice that gives the domain its width
        (
            DOMAIN_LAYER,
            (99, 0),
            [(-X_BOX, X_BOX)],
            (Z_SI + 1.0, Z_SI + 1.1),
            "sio2_rf",
            REFINED_UM,
        ),
        (SIGNAL, (49, 0), [(-EDGE, EDGE)], (Z_M0, Z_M1), "aluminum", METAL_UM),
        (
            GROUND,
            (49, 1),
            [(-X_BOX, -INNER), (INNER, X_BOX)],
            (Z_M0, Z_M1),
            "aluminum",
            GROUND_MESH_UM,
        ),
    ]


def _cpw_layer_stack():
    """The explicit stack of the frozen runs (variant A, metal as volume regions)."""
    from gsim.common.stack.extractor import Layer, LayerStack
    from gsim.common.stack.materials import MaterialProperties

    layers = {
        name: Layer(
            name=name,
            gds_layer=gds,
            zmin=z[0],
            zmax=z[1],
            thickness=z[1] - z[0],
            material=material,
            layer_type="dielectric",  # metal meshed as a volume (ADR 0003)
            mesh_resolution=size,
        )
        for name, gds, _spans, z, material, size in _layer_table()
    }
    return LayerStack(
        pdk_name="rf_cpw_A",
        layers=layers,
        materials={
            "sio2_rf": MaterialProperties(
                permittivity=4.1, loss_tangent=0.0, conductivity=0.0
            ).to_dict(),
            "si_rf": MaterialProperties(
                permittivity=11.9, loss_tangent=0.0, conductivity=0.0
            ).to_dict(),
            "air_rf": MaterialProperties(
                permittivity=1.0, loss_tangent=0.0, conductivity=0.0
            ).to_dict(),
            "aluminum": MaterialProperties(
                permittivity=1.0, conductivity=SIGMA
            ).to_dict(),
        },
        dielectrics=[
            {"name": "silicon", "zmin": Z_BOT, "zmax": Z_SI, "material": "si_rf"},
            {"name": "oxide", "zmin": Z_SI, "zmax": Z_M1, "material": "sio2_rf"},
        ],
    )


def _build_cpw_sim(out_dir: Path):
    """Mesh the CPW cross-section with ``BoundaryModeSim``, metal as volume regions.

    Follows the recipe behind the frozen numbers (variant A): one gsim layer per
    conductor so the signal and the ground have their own mesh region, a thin
    oxide slice that gives the domain its width, an explicit stack, and a
    refinement box over the signal, the gaps and the inner ground.

    Returns:
        ``(sim, stack)``.
    """
    import gdsfactory as gf

    from gsim.palace import BoundaryModeSim

    gf.gpdk.PDK.activate()
    comp = gf.Component()
    for _name, gds, spans, _z, _material, _size in _layer_table():
        for y0, y1 in spans:
            comp.add_polygon(
                [(0, y0), (LENGTH_UM, y0), (LENGTH_UM, y1), (0, y1)], layer=gds
            )
    stack = _cpw_layer_stack()
    sim = BoundaryModeSim()
    sim.set_output_dir(str(out_dir))
    sim.set_stack(stack)
    sim.set_geometry(comp)
    sim.set_airbox(margin_x=0.0, margin_y=0.0, z_above=Z_TOP - Z_M1, material="air_rf")
    sim.set_cross_section(f"x={LENGTH_UM / 2:g}")
    pad = 0.5 * (Z_M1 - Z_M0)
    sim.mesh(
        preset="coarse",
        refined_mesh_size=REFINED_UM,
        max_mesh_size=MAX_UM,
        refinement_boxes=[
            (-EDGE_ZONE_UM, EDGE_ZONE_UM, Z_M0 - pad, Z_M1 + pad, METAL_UM)
        ],
        verbose=False,
    )
    return sim, stack


def _build_cpw_mesh(out_dir: Path):
    """``(mesh_path, stack)`` of the CPW cross-section."""
    sim, stack = _build_cpw_sim(out_dir)
    return Path(sim.mesh_path), getattr(sim, "stack", None) or stack


SIGNAL_CONDUCTOR = Conductor(SIGNAL, ((-EDGE, EDGE), (Z_M0, Z_M1)), "volume")
GROUND_CONDUCTOR = Conductor(GROUND, ((INNER, X_BOX), (Z_M0, Z_M1)), "volume")
DOMAIN_EXTENT = ((-X_BOX, X_BOX), (Z_BOT, Z_TOP))


@pytest.fixture(scope="module")
def cpw(tmp_path_factory):
    out = tmp_path_factory.mktemp("gfem-cpw")
    mesh_path, stack = _build_cpw_mesh(out)
    return mesh_path, stack


@pytest.fixture(scope="module")
def femwell_runtime():
    return pytest.importorskip("femwell"), pytest.importorskip("skfem")


@pytest.fixture(scope="module")
def gfem_line(request):
    # skip before the mesh is built when gfem is missing
    pytest.importorskip("gfem")
    mesh_path, stack = request.getfixturevalue("cpw")
    voltage, _loop = adapter.impedance_paths_m(
        SIGNAL_CONDUCTOR.extent, GROUND_CONDUCTOR.extent, DOMAIN_EXTENT
    )
    result = adapter.solve_line(
        mesh_path,
        stack,
        frequency_hz=FREQUENCIES,
        voltage_path_m=voltage,
        metallic_boundaries=True,
    )
    print(f"\ngfem CPW section: {_triangle_count(mesh_path)} triangles")
    return result


def test_frozen_alpha_follows_from_frozen_n():
    for n, alpha in ((PALACE_N, PALACE_ALPHA), (FEMWELL_N, FEMWELL_ALPHA)):
        np.testing.assert_allclose(_alpha_of_index(n, GHZ), alpha, rtol=1e-4)


@pytest.mark.gfem_local
def test_gfem_matches_frozen_palace_and_femwell(gfem_line):
    result = gfem_line
    # With more than three frequencies gfem solves a few exactly and follows the
    # mode through the band with a reduced model it checks at each frequency.
    # The 1e-3 bound is loose (a probe on a coarse mesh gave 4e-8 to 3e-6).
    assert np.isfinite(np.asarray(result.residual)).all()
    assert np.asarray(result.residual).max() < 1e-3
    palace_n, femwell_n, palace_z, femwell_z_pi = (
        table[list(RUN)] for table in (PALACE_N, FEMWELL_N, PALACE_Z, FEMWELL_Z_PI)
    )
    n = np.asarray(result.n_eff)
    alpha = _alpha_db_per_cm(result.gamma)
    z = adapter.line_impedances(
        result,
        signal=SIGNAL_CONDUCTOR.extent,
        ground=GROUND_CONDUCTOR.extent,
        domain=DOMAIN_EXTENT,
    )
    for i, frequency in enumerate(FREQUENCIES):
        print(
            f"{frequency / 1e9:g} GHz: n {n[i]:.6f}, alpha {alpha[i]:.4f} dB/cm, "
            f"Z_PV {z['PV'][i]:.3f} (Palace {palace_z[i]:.3f}), "
            f"Z_PI {z['PI'][i]:.3f} (femwell {femwell_z_pi[i]:.3f}), "
            f"Z_VI {z['VI'][i]:.3f}"
        )
    for reference in (palace_n, femwell_n):
        np.testing.assert_allclose(n.real, reference.real, rtol=REFERENCE_RTOL)
        np.testing.assert_allclose(
            alpha, _alpha_of_index(reference, FREQUENCIES), rtol=REFERENCE_RTOL
        )
    np.testing.assert_allclose(z["PV"], palace_z, rtol=REFERENCE_RTOL)
    np.testing.assert_allclose(z["PI"], femwell_z_pi, rtol=REFERENCE_RTOL)
    for index in range(len(FREQUENCIES)):
        reading = adapter.line_reading(
            result,
            signal=SIGNAL_CONDUCTOR,
            return_=GROUND_CONDUCTOR,
            domain=DOMAIN_EXTENT,
            index=index,
        )
        assert reading.wall_mode is False
        assert reading.z0_ohm == pytest.approx(z["PI"][index])


@pytest.mark.gfem_local
@pytest.mark.usefixtures("femwell_runtime")
def test_gfem_and_femwell_line_readings_agree(gfem_line, cpw):
    from gsim.femwell.adapter import epsilon_by_region, line_reading, solve_modes

    mesh_path, stack = cpw
    for index, frequency in enumerate(FREQUENCIES):
        modes = solve_modes(
            mesh_path,
            epsilon=epsilon_by_region(mesh_path, stack, frequency_hz=frequency),
            wavelength_um=C0 / frequency * 1e6,
            num_modes=4,
            metallic_boundaries=True,
            n_guess=2.6,
        )
        gfem_reading = adapter.line_reading(
            gfem_line,
            signal=SIGNAL_CONDUCTOR,
            return_=GROUND_CONDUCTOR,
            domain=DOMAIN_EXTENT,
            index=index,
        )
        mode = min(modes, key=lambda m: abs(complex(m.n_eff) - gfem_reading.n_eff))
        femwell_reading = line_reading(
            mode,
            frequency_hz=frequency,
            mesh=mesh_path,
            signal=SIGNAL_CONDUCTOR,
            return_=GROUND_CONDUCTOR,
        )
        assert femwell_reading.wall_mode is False
        # Same mesh, same media: the indices agree to 2e-7 (1 GHz) and 2e-6
        # (10 GHz) on the default profile. Z_PI, each solver reading its own
        # loop of the finite-element field, differs by 1.8e-3 and 2.0e-3 there,
        # more than either does from the frozen femwell value computed on the
        # 0.15 um mesh (8.7e-4 at most for gfem): the bound is the measured
        # difference plus a margin of 1.5.
        assert gfem_reading.n_eff.real == pytest.approx(
            femwell_reading.n_eff.real, rel=1e-4
        )
        assert gfem_reading.z0_ohm == pytest.approx(
            femwell_reading.z0_ohm, rel=LIVE_FEMWELL_Z_RTOL
        )


@pytest.fixture(scope="module")
def palace_runs() -> dict[float, Path]:
    """Downloaded Palace run folders, one per frequency, from the environment.

    ``GSIM_GFEM_PALACE_RESULTS`` is a folder with a sub-folder per frequency
    named ``<GHz>ghz`` (``1ghz``, ``10ghz``, ``25ghz``, ``50ghz``), each the
    results folder of one ``BoundaryModeSim`` run on this fixture's mesh. Run
    the test with the same ``GSIM_GFEM_CPW_METAL_UM`` the jobs were meshed with.
    """
    root = os.environ.get("GSIM_GFEM_PALACE_RESULTS")
    if not root:
        pytest.skip("GSIM_GFEM_PALACE_RESULTS is not set: no live Palace results")
    base = Path(root)
    runs = {f: base / f"{f / 1e9:g}ghz" for f in FREQUENCIES}
    missing = [str(p) for p in runs.values() if not p.is_dir()]
    if missing:
        pytest.skip(f"Palace run folders missing: {missing}")
    return runs


@pytest.mark.gfem_local
def test_gfem_matches_live_palace(palace_runs, gfem_line):
    from gsim.palace.results import load_text_results

    runs = palace_runs
    n = np.asarray(gfem_line.n_eff)
    z = np.asarray(gfem_line.Z_PV)
    for i, frequency in enumerate(FREQUENCIES):
        text = load_text_results(runs[frequency])
        # the line mode is the one whose index is nearest gfem's
        mode = min(
            text.modes, key=lambda k: abs(complex(text.modes[k]["n_eff"]) - n[i])
        )
        n_palace = complex(text.modes[mode]["n_eff"])
        z_palace = text.complex_characteristic_impedance(
            index=1, mode=mode, quantity="Z_PV"
        )
        assert z_palace is not None, f"no Z_PV for mode {mode} in {runs[frequency]}"
        assert n[i].real == pytest.approx(n_palace.real, rel=REFERENCE_RTOL)
        assert -n[i].imag == pytest.approx(-n_palace.imag, rel=REFERENCE_RTOL)
        assert z[i] == pytest.approx(z_palace, rel=REFERENCE_RTOL)

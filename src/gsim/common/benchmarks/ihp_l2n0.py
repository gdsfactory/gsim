"""IHP SG13G2 octagonal L2n0 benchmark helpers.

The benchmark source files are vendored under ``nbs/data/ihp_validation/source``
from IHP-Open-PDK commit ``943983786b04351feb20e33e8d3d76c0e0d87cce``
(Apache-2.0).  The helpers build the same 2-port inductor model used by the
notebook: IHP's TopMetal2 winding/feeder layout, a SUBGND reference plane, the
authoritative IHP stack, and 30 GHz interlayer ports from SUBGND to TopMetal1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[4]
SOURCE = REPO / "nbs" / "data" / "ihp_validation" / "source"
RUN_ROOT = REPO / "nbs" / "data" / "ihp_validation" / "runs"
LAYOUT = SOURCE / "L_2n0_simplified.gds"
MEASURED = {
    "die_x2y7": SOURCE / "die_x2y7.s2p",
    "die_x5y8": SOURCE / "die_x5y8.s2p",
}
TARGET_FREQ_HZ = 30e9
Z0_OHM = 50.0

#: IHP process coordinate shift used by the published reference stack.  The
#: benchmark's absolute stack is offset so TopMetal1 lands at 190.18-192.18 um.
STACK_SHIFT_UM = 183.95
SUBGND_ZMIN_UM = 180.0 - STACK_SHIFT_UM
SUBGND_ZMAX_UM = 183.75 - STACK_SHIFT_UM


@dataclass(frozen=True)
class BenchmarkRun:
    """Recorded benchmark artefacts."""

    directory: Path
    sparams: Any
    frequencies_ghz: np.ndarray


def load_benchmark_component(
    *, ground_pad_um: float = 50.0, openems_ground: bool = False
) -> Any:
    """Import the IHP benchmark GDS and add the two reference-plane ports."""
    import gdsfactory as gf
    import ihp

    ihp.PDK.activate()
    gf.clear_cache()
    component = gf.import_gds(LAYOUT)
    # SUBGND is an explicit reference plane, not a process layer in the GDS.
    bbox = component.dbbox()
    pad = float(ground_pad_um)
    if openems_ground:
        component.add_polygon(
            [(-240.0, -140.0), (240.0, -140.0), (240.0, 10.0), (-240.0, 10.0)],
            layer=(8, 0),
        )
    else:
        component.add_polygon(
            [
                (bbox.left - pad, bbox.bottom - pad),
                (bbox.right + pad, bbox.bottom - pad),
                (bbox.right + pad, bbox.top + pad),
                (bbox.left - pad, bbox.top + pad),
            ],
            layer=(250, 0),
        )
    component.add_port(
        "P1", center=(-34.2, 0.0), width=12.0, orientation=180.0, layer=(126, 0)
    )
    component.add_port(
        "P2", center=(34.2, 0.0), width=12.0, orientation=0.0, layer=(126, 0)
    )
    return component


def _add_conformal_tm2_shell(component: Any, *, sidewall_um: float = 0.6) -> None:
    """Add a sidewall dielectric ring around every TopMetal2 polygon."""
    import gdsfactory as gf

    polygons = component.get_polygons(layers=[(134, 0)])
    region = gf.kdb.Region(next(iter(polygons.values())))
    ring = region.sized(int(sidewall_um * 1000)) - region
    component.add_polygon(ring, layer=(999, 0))


def build_benchmark_stack(
    component: Any,
    *,
    shell_variant: bool = False,
    substrate_thickness_um: float = 280.0,
) -> Any:
    """Build the authoritative IHP stack plus the benchmark SUBGND/EPI levels."""
    import ihp

    from gsim.common.pdk import build_layer_stack, resolve_layer_stack
    from gsim.common.stack.extractor import Layer
    from gsim.common.stack.materials import get_material_properties

    resolved = resolve_layer_stack(component, pdk=ihp.PDK, wavelength_um=1.0)
    stack = build_layer_stack(resolved, backend="palace")
    stack.layers["subgnd"] = Layer(
        name="subgnd",
        gds_layer=(250, 0),
        zmin=SUBGND_ZMIN_UM,
        zmax=SUBGND_ZMAX_UM,
        thickness=SUBGND_ZMAX_UM - SUBGND_ZMIN_UM,
        material="aluminum",
        layer_type="conductor",
        mesh_order=0,
    )
    for name in ("si", "sio2", "sin", "aluminum", "tungsten"):
        properties = get_material_properties(name)
        if properties is not None:
            stack.materials.setdefault(name, properties.to_dict())

    passivation_layer = stack.layers["passiv"]  # codespell:ignore passiv
    stack.dielectrics = [
        {
            "name": "substrate",
            "material": "si",
            "zmin": 183.75 - STACK_SHIFT_UM - substrate_thickness_um,
            "zmax": 183.75 - STACK_SHIFT_UM,
        },
        {
            "name": "epi",
            "material": "si",
            "zmin": 183.75 - STACK_SHIFT_UM,
            "zmax": 183.75 - STACK_SHIFT_UM + 3.75,
        },
        {
            "name": "ild",
            "material": "sio2",
            "zmin": -0.2,
            "zmax": passivation_layer.zmin if shell_variant else 14.03,
        },
    ]
    if shell_variant:
        _add_conformal_tm2_shell(component)
        topmetal2 = stack.layers["topmetal2"]
        stack.layers["tm2_shell"] = Layer(
            name="tm2_shell",
            gds_layer=(999, 0),
            zmin=topmetal2.zmin,
            zmax=topmetal2.zmax,
            thickness=topmetal2.thickness,
            material="sin",
            layer_type="dielectric",
            mesh_order=topmetal2.mesh_order - 1
            if topmetal2.mesh_order is not None
            else None,
        )
    return stack


def make_simulation(
    output_dir: str | Path,
    *,
    sweep: bool = False,
    shell_variant: bool = False,
    ground_pad_um: float = 50.0,
    pec_boundary: bool = False,
    port_variant: str = "two_port",
    frequency_hz: float = TARGET_FREQ_HZ,
    substrate_thickness_um: float = 280.0,
    openems_ground: bool = False,
) -> Any:
    """Build a Palace simulation variant for the benchmark."""
    from gsim.palace import DrivenSim

    component = load_benchmark_component(
        ground_pad_um=ground_pad_um, openems_ground=openems_ground
    )
    stack = build_benchmark_stack(
        component,
        shell_variant=shell_variant,
        substrate_thickness_um=substrate_thickness_um,
    )
    sim = DrivenSim()
    sim.set_output_dir(output_dir)
    sim.set_geometry(component)
    sim.set_stack(stack)
    sim.absorbing_boundary = not pec_boundary
    sim.set_airbox(margin_x=150.0, margin_y=150.0, z_above=50.0, z_below=10.0)
    if port_variant == "two_port":
        for name in ("P1", "P2"):
            sim.add_port(
                name,
                from_layer="metal1" if openems_ground else "subgnd",
                to_layer="topmetal1",
                geometry="interlayer",
                impedance=Z0_OHM,
            )
    elif port_variant == "gap":
        # Exploratory differential gap across the two TopMetal1 feed ends.
        component.add_port(
            "Pdiff", center=(0.0, 0.0), width=68.4, orientation=0.0, layer=(126, 0)
        )
        sim.add_port("Pdiff", layer="topmetal1", geometry="gap", impedance=Z0_OHM)
    else:
        raise ValueError(f"unknown port_variant {port_variant!r}")
    if sweep:
        sim.set_driven(fmin=1e9, fmax=50e9, num_points=11, reference_impedance=Z0_OHM)
    else:
        sim.set_driven(f=frequency_hz, num_points=1, reference_impedance=Z0_OHM)
    return sim


def run_benchmark(
    output_dir: str | Path | None = None,
    *,
    sweep: bool = False,
    palace_executable: str | Path | None = None,
    num_processes: int = 8,
    shell_variant: bool = False,
    ground_pad_um: float = 50.0,
    pec_boundary: bool = False,
    port_variant: str = "two_port",
    frequency_hz: float = TARGET_FREQ_HZ,
    refined_mesh_size: float = 5.0,
    max_mesh_size: float = 70.0,
    planar_conductors: bool = False,
    substrate_thickness_um: float = 280.0,
    openems_ground: bool = False,
) -> BenchmarkRun:
    """Mesh, run Palace, and load the benchmark S-parameters."""
    from gsim.palace.results import load_sparams

    output = Path(output_dir or (RUN_ROOT / "l2n0_30ghz"))
    output.mkdir(parents=True, exist_ok=True)
    sim = make_simulation(
        output,
        sweep=sweep,
        shell_variant=shell_variant,
        ground_pad_um=ground_pad_um,
        pec_boundary=pec_boundary,
        port_variant=port_variant,
        frequency_hz=frequency_hz,
        substrate_thickness_um=substrate_thickness_um,
        openems_ground=openems_ground,
    )
    sim.mesh(
        preset="coarse",
        refined_mesh_size=refined_mesh_size,
        max_mesh_size=max_mesh_size,
        margin=150.0,
        z_above=50.0,
        z_below=10.0,
        planar_conductors=planar_conductors,
        verbose=False,
    )
    sim.write_config()
    sim.run_local(
        palace_executable=palace_executable,
        num_processes=num_processes,
        verbose=False,
    )
    sparams = load_sparams(output / "output" / "palace")
    return BenchmarkRun(
        directory=output,
        sparams=sparams,
        frequencies_ghz=np.asarray(sparams.freq, dtype=float),
    )


def load_recorded_run(directory: str | Path) -> BenchmarkRun:
    """Load a previously recorded benchmark run."""
    from gsim.palace.results import load_sparams

    output = Path(directory)
    sparams = load_sparams(output / "output" / "palace")
    return BenchmarkRun(
        directory=output,
        sparams=sparams,
        frequencies_ghz=np.asarray(sparams.freq, dtype=float),
    )


def s_matrix(sparams: Any, index: int = 0) -> np.ndarray:
    """Return the 2x2 complex S matrix at one recorded frequency."""
    names = sparams.port_names
    if len(names) != 2:
        raise ValueError(f"expected 2 ports, got {names}")
    return np.array(
        [
            [sparams[names[i], names[j]].complex[index] for j in range(2)]
            for i in range(2)
        ]
    )


def s_to_z(s: np.ndarray, *, z0: float = Z0_OHM) -> np.ndarray:
    """Convert a 2x2 S matrix to a Z matrix."""
    return z0 * (np.eye(2) + s) @ np.linalg.inv(np.eye(2) - s)


def differential_impedance_from_s(s: np.ndarray, *, z0: float = Z0_OHM) -> complex:
    """Return differential impedance from a 2x2 S matrix."""
    z = s_to_z(s, z0=z0)
    return complex(z[0, 0] - z[0, 1] - z[1, 0] + z[1, 1])


def one_port_impedance_from_s(s11: complex, *, z0: float = Z0_OHM) -> complex:
    """Return terminal impedance from a one-port S11."""
    return z0 * (1 + s11) / (1 - s11)


def lq_from_impedance(z: complex, f_hz: float) -> tuple[float, float]:
    """Return (L_nH, Q) from a series impedance."""
    l_nh = z.imag / (2.0 * np.pi * f_hz) * 1e9
    q = z.imag / z.real if z.real != 0 else float("nan")
    return l_nh, q


def lq_from_s_matrix(
    s: np.ndarray, f_hz: float, *, z0: float = Z0_OHM
) -> tuple[float, float, complex]:
    """Return ``(L_nH, Q, Z_series)`` from a 2x2 S matrix."""
    z = 2.0 * z0 * (1.0 - s[1, 0]) / s[1, 0]
    inductance_nh = z.imag / (2.0 * np.pi * f_hz) * 1e9
    q = z.imag / z.real if z.real != 0 else float("nan")
    return inductance_nh, q, z


def series_impedance(sparams: Any, index: int = 0, *, z0: float = Z0_OHM) -> complex:
    """Return the series impedance from S21 for a 2-port series element."""
    s = s_matrix(sparams, index)
    return 2.0 * z0 * (1.0 - s[1, 0]) / s[1, 0]


def lq_at(sparams: Any, index: int = 0) -> tuple[float, float, complex]:
    """Return ``(L_nH, Q, Z_series)`` at the selected recorded frequency."""
    return lq_from_s_matrix(s_matrix(sparams, index), float(sparams.freq[index]) * 1e9)


__all__ = [
    "LAYOUT",
    "MEASURED",
    "RUN_ROOT",
    "SOURCE",
    "STACK_SHIFT_UM",
    "SUBGND_ZMAX_UM",
    "SUBGND_ZMIN_UM",
    "TARGET_FREQ_HZ",
    "Z0_OHM",
    "BenchmarkRun",
    "build_benchmark_stack",
    "load_benchmark_component",
    "load_recorded_run",
    "lq_at",
    "make_simulation",
    "run_benchmark",
    "s_matrix",
    "series_impedance",
]

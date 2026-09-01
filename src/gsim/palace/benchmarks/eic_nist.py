# Copyright 2026 GDSFactory
"""NIST Parylene-C platinum-CPW benchmark definition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from gsim.common.stack import Layer, LayerStack
from gsim.palace import DrivenSim
from gsim.palace.benchmarks.reference import RemoteArtifact, SParameterData

NIST_SIMULATION_DOI = "10.18434/mds2-2817"
NIST_MEASUREMENT_DOI = "10.18434/mds2-2808"
NIST_SIMULATION_ARCHIVE = RemoteArtifact(
    name="Parylene C CPW Simulation resources for MIDAS.zip",
    url=(
        "https://data.nist.gov/od/ds/mds2-2817/"
        "Parylene%20C%20CPW%20Simulation%20resources%20for%20MIDAS.zip"
    ),
    sha256="d57a0131365b6e9790898903ca7dec96782ddac1e8436747ecfd594b89438bdd",
    size_bytes=191_909_838,
)

PLATINUM_LAYER = (1, 0)
SUBSTRATE_LAYER = (2, 0)
PARYLENE_LAYER = (3, 0)
PDMS_LAYER = (4, 0)
AIR_CHANNEL_LAYER = (5, 0)
GAP_PARTITION_SUBSTRATE_LAYER = (6, 0)
BOTTOM_CONDUCTOR_LAYER = (7, 0)
GAP_PARTITION_SUBSTRATE_LAYERS = (
    GAP_PARTITION_SUBSTRATE_LAYER,
    (8, 0),
)

SIGNAL_WIDTH_UM = 50.0
GAP_WIDTH_UM = 2.5
GROUND_WIDTH_UM = 200.0
SUBSTRATE_WIDTH_UM = 1365.0
PARYLENE_THICKNESS_UM = 6.67
PDMS_HEIGHT_UM = 500.0
CHANNEL_WIDTH_UM = 213.0
BOTTOM_CONDUCTOR_THICKNESS_UM = 50.0
NIST_MAX_TETRAHEDRA = 110_000
NIST_CAPPED_MESH_SIZE_UM = 150.0
NIST_BULK_MESH_SIZE_UM = 200.0
NIST_MESH_ALGORITHM_3D = 10
NIST_CPW_PORT_LENGTH_UM = 5.0
NIST_CPW_LONGITUDINAL_MARGIN_UM = 50.0
NIST_CPW_LATERAL_MARGIN_UM = 100.0
SECTION_LENGTHS_UM = {
    "NP": 60.0,
    "YP": 690.0,
    "AirPDMS": 732.4,
    "PDMS": 420.1,
    "AirChannel": 2695.5,
}


@dataclass(frozen=True)
class NistAirReference:
    """Published NIST Q2D cascade and measured air-chip data."""

    simulation: SParameterData
    measurement: SParameterData


def nist_section_edges_um() -> NDArray[np.float64]:
    """Return all ten boundaries of the symmetric nine-section line."""
    ordered_lengths = [
        SECTION_LENGTHS_UM["NP"],
        SECTION_LENGTHS_UM["YP"],
        SECTION_LENGTHS_UM["AirPDMS"],
        SECTION_LENGTHS_UM["PDMS"],
        SECTION_LENGTHS_UM["AirChannel"],
        SECTION_LENGTHS_UM["PDMS"],
        SECTION_LENGTHS_UM["AirPDMS"],
        SECTION_LENGTHS_UM["YP"],
        SECTION_LENGTHS_UM["NP"],
    ]
    total_length = sum(ordered_lengths)
    return np.asarray(
        [-total_length / 2, *np.cumsum(ordered_lengths) - total_length / 2]
    )


def build_nist_component(*, include_gap_partitions: bool = False):
    """Build the complete 6.5005 mm air-filled H2O-chip CPW layout.

    ``include_gap_partitions`` adds same-material dielectric strips inside each
    2.5 um gap. They introduce no material contrast, but force four 0.625 um
    transverse mesh spans at the metal plane without imposing that size along
    the complete 6.5 mm line.
    """
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    component = gf.Component()
    component.info["benchmark"] = "nist_parylene_c_air_cpw"
    edges = nist_section_edges_um()
    xmin, xmax = float(edges[0]), float(edges[-1])
    substrate_half_width = SUBSTRATE_WIDTH_UM / 2

    _add_rectangle(
        component,
        xmin,
        -substrate_half_width,
        xmax,
        substrate_half_width,
        SUBSTRATE_LAYER,
    )
    _add_rectangle(
        component,
        xmin,
        -substrate_half_width,
        xmax,
        substrate_half_width,
        BOTTOM_CONDUCTOR_LAYER,
    )
    _add_rectangle(
        component,
        float(edges[1]),
        -substrate_half_width,
        float(edges[-2]),
        substrate_half_width,
        PARYLENE_LAYER,
    )
    _add_rectangle(
        component,
        float(edges[2]),
        -substrate_half_width,
        float(edges[-3]),
        substrate_half_width,
        PDMS_LAYER,
    )

    channel_half_width = CHANNEL_WIDTH_UM / 2
    for start_index, end_index in ((2, 3), (4, 5), (6, 7)):
        _add_rectangle(
            component,
            float(edges[start_index]),
            -channel_half_width,
            float(edges[end_index]),
            channel_half_width,
            AIR_CHANNEL_LAYER,
        )

    signal_half_width = SIGNAL_WIDTH_UM / 2
    ground_inner_edge = signal_half_width + GAP_WIDTH_UM
    _add_rectangle(
        component,
        xmin,
        -signal_half_width,
        xmax,
        signal_half_width,
        PLATINUM_LAYER,
    )
    _add_rectangle(
        component,
        xmin,
        ground_inner_edge,
        xmax,
        ground_inner_edge + GROUND_WIDTH_UM,
        PLATINUM_LAYER,
    )
    _add_rectangle(
        component,
        xmin,
        -ground_inner_edge - GROUND_WIDTH_UM,
        xmax,
        -ground_inner_edge,
        PLATINUM_LAYER,
    )
    if include_gap_partitions:
        _add_gap_partition_polygons(component, edges)
        component.info["gap_mesh_partitions"] = True
    component.add_port(
        name="left",
        center=(xmin, 0.0),
        width=SIGNAL_WIDTH_UM,
        orientation=180.0,
        layer=PLATINUM_LAYER,
        port_type="electrical",
    )
    component.add_port(
        name="right",
        center=(xmax, 0.0),
        width=SIGNAL_WIDTH_UM,
        orientation=0.0,
        layer=PLATINUM_LAYER,
        port_type="electrical",
    )
    return component


def build_nist_stack(
    *,
    platinum_thickness_um: float = 0.2,
    include_gap_partitions: bool = False,
) -> LayerStack:
    """Build the Q2D-parity material stack with finite Pt conductivity."""
    materials = {
        "fused_silica": {
            "type": "dielectric",
            "permittivity": 3.82,
            "loss_tangent": 0.0,
        },
        "parylene_c": {
            "type": "dielectric",
            "permittivity": 3.0,
            "loss_tangent": 0.002,
        },
        "pdms": {
            "type": "dielectric",
            "permittivity": 2.77,
            "loss_tangent": 0.054,
        },
        "air": {"type": "dielectric", "permittivity": 1.0, "loss_tangent": 0.0},
        "platinum": {"type": "conductor", "conductivity": 9.04e6},
    }
    layers = {
        "substrate": Layer(
            name="substrate",
            gds_layer=SUBSTRATE_LAYER,
            zmin=-500.0,
            zmax=0.0,
            thickness=500.0,
            material="fused_silica",
            layer_type="substrate",
            mesh_resolution=80.0,
        ),
        "parylene_c": Layer(
            name="parylene_c",
            gds_layer=PARYLENE_LAYER,
            zmin=0.0,
            zmax=PARYLENE_THICKNESS_UM,
            thickness=PARYLENE_THICKNESS_UM,
            material="parylene_c",
            layer_type="dielectric",
            mesh_resolution=5.0,
        ),
        "pdms": Layer(
            name="pdms",
            gds_layer=PDMS_LAYER,
            zmin=PARYLENE_THICKNESS_UM,
            zmax=PARYLENE_THICKNESS_UM + PDMS_HEIGHT_UM,
            thickness=PDMS_HEIGHT_UM,
            material="pdms",
            layer_type="dielectric",
            mesh_resolution=40.0,
        ),
        "air_channel": Layer(
            name="air_channel",
            gds_layer=AIR_CHANNEL_LAYER,
            zmin=PARYLENE_THICKNESS_UM,
            zmax=PARYLENE_THICKNESS_UM + CHANNEL_WIDTH_UM,
            thickness=CHANNEL_WIDTH_UM,
            material="air",
            layer_type="dielectric",
            mesh_resolution=20.0,
        ),
        "platinum": Layer(
            name="platinum",
            gds_layer=PLATINUM_LAYER,
            zmin=0.0,
            zmax=platinum_thickness_um,
            thickness=platinum_thickness_um,
            material="platinum",
            layer_type="conductor",
            mesh_resolution=10.0,
        ),
        "bottom_conductor": Layer(
            name="bottom_conductor",
            gds_layer=BOTTOM_CONDUCTOR_LAYER,
            zmin=-500.0 - BOTTOM_CONDUCTOR_THICKNESS_UM,
            zmax=-500.0,
            thickness=BOTTOM_CONDUCTOR_THICKNESS_UM,
            material="platinum",
            layer_type="conductor",
            mesh_resolution=80.0,
        ),
    }
    if include_gap_partitions:
        for partition_index, gds_layer in enumerate(
            GAP_PARTITION_SUBSTRATE_LAYERS, start=1
        ):
            material_name = f"fused_silica_partition_{partition_index}"
            layer_name = f"gap_partition_substrate_{partition_index}"
            materials[material_name] = dict(materials["fused_silica"])
            layers[layer_name] = Layer(
                name=layer_name,
                gds_layer=gds_layer,
                zmin=-5.0,
                zmax=0.0,
                thickness=5.0,
                material=material_name,
                layer_type="dielectric",
                mesh_resolution=NIST_CAPPED_MESH_SIZE_UM,
            )
    dielectrics = [
        {
            "name": "bottom_ambient",
            "zmin": -500.0 - BOTTOM_CONDUCTOR_THICKNESS_UM,
            "zmax": -500.0,
            "material": "air",
        },
        {
            "name": "substrate",
            "zmin": -500.0,
            "zmax": 0.0,
            "material": "fused_silica",
        },
        {
            "name": "ambient",
            "zmin": 0.0,
            "zmax": PARYLENE_THICKNESS_UM + PDMS_HEIGHT_UM,
            "material": "air",
        },
    ]
    return LayerStack(
        pdk_name="NIST-Parylene-C-Q2D",
        layers=layers,
        materials=materials,
        dielectrics=dielectrics,
    )


def make_nist_simulation(
    output_dir: str | Path,
    *,
    platinum_thickness_um: float = 0.2,
    fmin_hz: float = 0.1e9,
    fmax_hz: float = 20e9,
    num_points: int = 81,
    adaptive_tol: float = 1e-3,
    adaptive_max_samples: int = 20,
    numerical_order: int = 2,
    port_type: Literal["cpw_lumped", "wave"] = "cpw_lumped",
    include_gap_partitions: bool = True,
) -> DrivenSim:
    """Create the complete NIST CPW Palace driven simulation.

    The default is a genuine full-wave 3D model with two 50-ohm CPW lumped
    ports. Numeric wave ports remain available for convergence studies, but
    their boundary eigensolve is not used by the capped screening setup.
    """
    if port_type not in {"cpw_lumped", "wave"}:
        raise ValueError("port_type must be 'cpw_lumped' or 'wave'")
    simulation = DrivenSim()
    simulation.set_output_dir(output_dir)
    simulation.set_geometry(
        build_nist_component(include_gap_partitions=include_gap_partitions)
    )
    simulation.set_stack(
        build_nist_stack(
            platinum_thickness_um=platinum_thickness_um,
            include_gap_partitions=include_gap_partitions,
        )
    )
    simulation.set_airbox(
        margin_x=(
            NIST_CPW_LONGITUDINAL_MARGIN_UM if port_type == "cpw_lumped" else 0.0
        ),
        margin_y=NIST_CPW_LATERAL_MARGIN_UM,
        z_above=250.0,
        z_below=0.0,
    )
    for port_name in ("left", "right"):
        if port_type == "cpw_lumped":
            simulation.add_cpw_port(
                port_name,
                layer="platinum",
                s_width=SIGNAL_WIDTH_UM,
                gap_width=GAP_WIDTH_UM,
                length=NIST_CPW_PORT_LENGTH_UM,
                impedance=50.0,
                excited=True,
            )
        else:
            simulation.add_wave_port(
                port_name,
                layer="platinum",
                max_size=True,
                mode=1,
                excited=True,
            )
    simulation.set_driven(
        fmin=fmin_hz,
        fmax=fmax_hz,
        num_points=num_points,
        adaptive_tol=adaptive_tol,
        adaptive_max_samples=adaptive_max_samples,
        reference_impedance=50.0,
    )
    simulation.absorbing_order = 1
    simulation.set_numerical(order=numerical_order, tolerance=1e-8)
    return simulation


def require_nist_tetrahedron_budget(
    mesh_result: Any,
    *,
    maximum: int = NIST_MAX_TETRAHEDRA,
) -> int:
    """Return the tetrahedron count or reject a NIST solver submission.

    This gate is intentionally evaluated after writing the finalized mesh and
    before either a local Palace run or a cloud submission.
    """
    tetrahedra = (getattr(mesh_result, "mesh_stats", None) or {}).get("tetrahedra")
    if not isinstance(tetrahedra, int) or tetrahedra < 1:
        raise ValueError("Finalized NIST mesh has no positive tetrahedron count")
    if tetrahedra > maximum:
        raise RuntimeError(
            f"NIST mesh has {tetrahedra:,} tetrahedra; hard limit is "
            f"{maximum:,}. Palace execution is blocked."
        )
    return tetrahedra


def load_nist_air_reference(path: str | Path) -> NistAirReference:
    """Load the curated Q2D cascade and measurement arrays."""
    with np.load(path) as arrays:
        simulation = SParameterData(
            frequency_hz=arrays["simulation_frequency_hz"],
            s=arrays["simulation_s"],
            source=f"{path}: NIST Q2D cascade",
        )
        measurement = SParameterData(
            frequency_hz=arrays["measurement_frequency_hz"],
            s=arrays["measurement_s"],
            source=f"{path}: NIST measured air chip",
        )
    return NistAirReference(simulation=simulation, measurement=measurement)


def cascade_nist_rlcg(data_directory: str | Path) -> SParameterData:
    """Reproduce the NIST ``constructSfromSim.m`` air-channel cascade."""
    data_path = Path(data_directory)
    file_specs = {
        "NP": ("1_NP_25umgap.csv", 1, 1.0),
        "YP": ("2_YP_25umgap.csv", 1, 1e-3),
        "AirPDMS": ("3_AirPDMS_25umgap.csv", 1, 1e-3),
        "PDMS": ("4_PDMS_25umgap.csv", 1, 1e-3),
        "AirChannel": ("5b_AirChannel.csv", 2, 1e-3),
    }
    frequency_hz: NDArray[np.float64] | None = None
    impedances: dict[str, NDArray[np.complex128]] = {}
    propagations: dict[str, NDArray[np.complex128]] = {}

    for name, (filename, frequency_column, conductance_scale) in file_specs.items():
        current_frequency, rlcg = _load_rlcg(
            data_path / filename,
            frequency_column=frequency_column,
            conductance_scale=conductance_scale,
        )
        if frequency_hz is None:
            frequency_hz = current_frequency
        elif not np.array_equal(frequency_hz, current_frequency):
            raise ValueError(f"Frequency grid in {filename} does not match NP data")
        series_impedance = rlcg[:, 0] + 1j * 2 * np.pi * current_frequency * rlcg[:, 1]
        shunt_admittance = rlcg[:, 3] + 1j * 2 * np.pi * current_frequency * rlcg[:, 2]
        impedances[name] = np.sqrt(series_impedance) / np.sqrt(shunt_admittance)
        propagations[name] = np.sqrt(series_impedance) * np.sqrt(shunt_admittance)

    if frequency_hz is None:
        raise ValueError("No RLCG data files were configured")
    section_names = [
        "NP",
        "YP",
        "AirPDMS",
        "PDMS",
        "AirChannel",
        "PDMS",
        "AirPDMS",
        "YP",
        "NP",
    ]
    network = _transformer(np.full(frequency_hz.size, 50.0), impedances["NP"])
    previous_name = section_names[0]
    for section_index, section_name in enumerate(section_names):
        if section_index:
            network = network @ _transformer(
                impedances[previous_name], impedances[section_name]
            )
        network = network @ _transmission_line(
            propagations[section_name], SECTION_LENGTHS_UM[section_name] * 1e-6
        )
        previous_name = section_name
    network = network @ _transformer(
        impedances[previous_name], np.full(frequency_hz.size, 50.0)
    )
    return SParameterData(
        frequency_hz=frequency_hz,
        s=_t_to_s(np.asarray(network, dtype=np.complex128)),
        source=f"NIST Q2D RLCG cascade from {data_path}",
    )


def _add_rectangle(
    component, xmin: float, ymin: float, xmax: float, ymax: float, layer
) -> None:
    """Add an axis-aligned rectangle polygon without introducing child cells."""
    component.add_polygon(
        [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)], layer=layer
    )


def _add_gap_partition_polygons(component, edges: NDArray[np.float64]) -> None:
    """Add same-material strips that divide each CPW gap into four spans."""
    signal_half_width = SIGNAL_WIDTH_UM / 2
    xmin, xmax = float(edges[0]), float(edges[-1])
    span_width = GAP_WIDTH_UM / 4
    for span_index, gds_layer in zip(
        (0, 2), GAP_PARTITION_SUBSTRATE_LAYERS, strict=True
    ):
        inner_edge = signal_half_width + span_index * span_width
        outer_edge = inner_edge + span_width
        for ymin, ymax in (
            (-outer_edge, -inner_edge),
            (inner_edge, outer_edge),
        ):
            _add_rectangle(component, xmin, ymin, xmax, ymax, gds_layer)


def _load_rlcg(
    path: Path, *, frequency_column: int, conductance_scale: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load one published ANSYS Q2D RLCG table in SI units."""
    values = np.asarray(np.loadtxt(path, delimiter=",", skiprows=1), dtype=np.float64)
    frequency_hz = np.asarray(values[:, frequency_column] * 1e9, dtype=np.float64)
    raw_rlcg = values[:, frequency_column + 1 : frequency_column + 5]
    scales = np.asarray([1e3, 1e-9, 1e-12, conductance_scale], dtype=np.float64)
    return frequency_hz, np.asarray(raw_rlcg * scales, dtype=np.float64)


def _transformer(
    impedance_left: NDArray[np.complex128],
    impedance_right: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Construct the row-wise impedance-transformer T matrices."""
    scale = 1 / np.sqrt(4 * impedance_left * impedance_right)
    diagonal = scale * (impedance_left + impedance_right)
    off_diagonal = scale * (impedance_right - impedance_left)
    result = np.empty((impedance_left.size, 2, 2), dtype=complex)
    result[:, 0, 0] = diagonal
    result[:, 0, 1] = off_diagonal
    result[:, 1, 0] = off_diagonal
    result[:, 1, 1] = diagonal
    return result


def _transmission_line(
    propagation: NDArray[np.complex128], length_m: float
) -> NDArray[np.complex128]:
    """Construct the row-wise matched-line T matrices."""
    result = np.zeros((propagation.size, 2, 2), dtype=complex)
    result[:, 0, 0] = np.exp(-propagation * length_m)
    result[:, 1, 1] = np.exp(propagation * length_m)
    return result


def _t_to_s(transmission: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Convert row-wise T matrices to S matrices using NIST's convention."""
    denominator = transmission[:, 1, 1]
    determinant = np.linalg.det(transmission)
    scattering = np.empty_like(transmission)
    scattering[:, 0, 0] = transmission[:, 0, 1] / denominator
    scattering[:, 0, 1] = 1 / denominator
    scattering[:, 1, 0] = determinant / denominator
    scattering[:, 1, 1] = -transmission[:, 1, 0] / denominator
    return scattering

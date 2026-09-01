# Copyright 2026 GDSFactory
"""IHP SG13G2 spiral-inductor benchmark definition."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from gsim.common.stack import Layer, LayerStack
from gsim.palace import DrivenSim
from gsim.palace.benchmarks.reference import RemoteArtifact

IHP_SOURCE_COMMIT = "a98509eb57989f60e7965bee88214d081d53c27e"
_IHP_RAW_ROOT = (
    "https://raw.githubusercontent.com/SkillSurf/frac-n-pll-vco-smacd_2026/"
    f"{IHP_SOURCE_COMMIT}/HFSS%20Inductor%20Files"
)

IHP_GDS_ARTIFACT = RemoteArtifact(
    name="4nH_2port_test1.gds",
    url=(
        f"{_IHP_RAW_ROOT}/Results/Results%20for%20the%202port%20inductor/"
        "4nH_2port_test1.gds"
    ),
    sha256="6e0a9573bcba530103fea9bd4da0df753848d10a6268a2c05a6494b0cfe8a97b",
    size_bytes=1804,
)
IHP_HFSS_REFERENCE_ARTIFACT = RemoteArtifact(
    name="realimaginary.s2p",
    url=(
        f"{_IHP_RAW_ROOT}/Results/Results%20for%20the%202port%20inductor/"
        "realimaginary.s2p"
    ),
    sha256="1db51374a3980bf6f24c99b84a4f6f7937b077483bac6575d18234d2ff2b50d3",
    size_bytes=19755,
)
IHP_NATIVE_MODEL_ARTIFACT = RemoteArtifact(
    name="Feb_01st.aedt",
    url=f"{_IHP_RAW_ROOT}/HFSS_Parametric_Model/Feb_01st.aedt",
    sha256="3a3c0980c55631250063b922c2681fb451add3c199275c2759193e0175331454",
)
IHP_STACK_ARTIFACT = RemoteArtifact(
    name="IHP PDK xml file.xml",
    url=f"{_IHP_RAW_ROOT}/IHP%20PDK%20xml%20file.xml",
    sha256="c610fd87231debde837e8ab3c66842b407e4c274387d5fb594b32171bd695cd0",
)

IHP_EXPECTED_BBOX_UM = (-226.581, -273.289, 229.419, 222.927)


class IHPPortSpec(TypedDict):
    """Location and orientation of one HFSS reference-plane port."""

    center: tuple[float, float]
    orientation: float


IHP_PORT_SPECS: dict[str, IHPPortSpec] = {
    "port1": {"center": (-33.581, -263.073), "orientation": 270.0},
    "port2": {"center": (36.419, -263.073), "orientation": 270.0},
}


def build_ihp_stack() -> LayerStack:
    """Build the finite-conductivity stack used by HFSS ``EMDesign2``."""
    materials = {
        "silicon_substrate": {
            "type": "dielectric",
            "permittivity": 11.9,
            "conductivity": 2.0,
        },
        "silicon_epi": {
            "type": "dielectric",
            "permittivity": 11.9,
            "conductivity": 5.0,
        },
        "sio2": {"type": "dielectric", "permittivity": 4.1, "loss_tangent": 0.0},
        "passive": {
            "type": "dielectric",
            "permittivity": 6.6,
            "loss_tangent": 0.0,
        },
        "air": {"type": "dielectric", "permittivity": 1.0, "loss_tangent": 0.0},
        "metal1": {"type": "conductor", "conductivity": 2.164e7},
        "metal5": {"type": "conductor", "conductivity": 2.319e7},
        "topvia1": {"type": "conductor", "conductivity": 2.191e6},
        "topmetal1": {"type": "conductor", "conductivity": 2.78e7},
    }
    layers = {
        "Metal1": Layer(
            name="Metal1",
            gds_layer=(62, 0),
            zmin=1.04,
            zmax=1.46,
            thickness=0.42,
            material="metal1",
            layer_type="conductor",
            mesh_resolution=4.0,
        ),
        "Metal5": Layer(
            name="Metal5",
            gds_layer=(123, 0),
            zmin=5.09,
            zmax=5.58,
            thickness=0.49,
            material="metal5",
            layer_type="conductor",
            mesh_resolution=3.0,
        ),
        "TopVia1": Layer(
            name="TopVia1",
            gds_layer=(125, 0),
            zmin=5.58,
            zmax=6.4303,
            thickness=0.8503,
            material="topvia1",
            layer_type="via",
            mesh_resolution=2.0,
        ),
        "TopMetal1": Layer(
            name="TopMetal1",
            gds_layer=(126, 0),
            zmin=6.4303,
            zmax=8.4303,
            thickness=2.0,
            material="topmetal1",
            layer_type="conductor",
            mesh_resolution=4.0,
        ),
    }
    dielectrics = [
        {
            "name": "substrate",
            "zmin": -283.75,
            "zmax": -3.75,
            "material": "silicon_substrate",
        },
        {"name": "epi", "zmin": -3.75, "zmax": 0.0, "material": "silicon_epi"},
        {"name": "oxide", "zmin": 0.0, "zmax": 15.7303, "material": "sio2"},
        {
            "name": "passive",
            "zmin": 15.7303,
            "zmax": 16.1303,
            "material": "passive",
        },
    ]
    return LayerStack(
        pdk_name="IHP-SG13G2-HFSS-EMDesign2",
        layers=layers,
        materials=materials,
        dielectrics=dielectrics,
    )


def load_ihp_component(gds_path: str | Path):
    """Import the pinned GDS and attach the two HFSS gap-port locations."""
    return _load_ihp_component_cached(str(Path(gds_path).resolve()))


@lru_cache(maxsize=4)
def _load_ihp_component_cached(gds_path: str):
    """Import each immutable GDS path once per process to avoid cell conflicts."""
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    component = gf.import_gds(gds_path, rename_duplicated_cells=True)
    component.info["benchmark"] = "ihp_sg13g2_4nh_inductor"
    for port_name, spec in IHP_PORT_SPECS.items():
        component.add_port(
            name=port_name,
            center=spec["center"],
            width=30.0,
            orientation=spec["orientation"],
            layer=(126, 0),
            port_type="electrical",
        )
    return component


def make_ihp_simulation(
    gds_path: str | Path,
    output_dir: str | Path,
    *,
    fmin_hz: float = 1.5e9,
    fmax_hz: float = 3.5e9,
    num_points: int = 41,
    adaptive_tol: float = 1e-3,
    adaptive_max_samples: int = 20,
) -> DrivenSim:
    """Create the full two-excitation IHP Palace driven simulation."""
    simulation = DrivenSim()
    simulation.set_output_dir(output_dir)
    simulation.set_geometry(load_ihp_component(gds_path))
    simulation.set_stack(build_ihp_stack())
    simulation.set_airbox(
        margin_x=150.0,
        margin_y=150.0,
        z_above=150.0,
        z_below=150.0,
    )
    for port_name in IHP_PORT_SPECS:
        simulation.add_port(
            port_name,
            from_layer="Metal1",
            to_layer="TopMetal1",
            impedance=50.0,
            excited=True,
            geometry="via",
        )
    simulation.set_driven(
        fmin=fmin_hz,
        fmax=fmax_hz,
        num_points=num_points,
        adaptive_tol=adaptive_tol,
        adaptive_max_samples=adaptive_max_samples,
        reference_impedance=50.0,
    )
    return simulation

"""Tests for exposing Palace's advanced wave-port eigensolver controls.

Palace's numeric wave ports support per-port SolverType/EigenTol/KSPTol/
MaxSize/Verbose on the 2D port-mode eigenproblem (see the WavePort object
in Palace's own JSON schema), but WavePortConfig previously had nowhere to
put them - gsim always emitted a bare Index/Mode/Offset/Excitation/Attributes
entry. These fields are per-port on the emitted config (matching Palace's
own schema), independent of the max_size flag on WavePortConfig/PalacePort,
which sizes port geometry and predates this feature.
"""

from __future__ import annotations

import json

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.palace.mesh.config_generator import generate_palace_config
from gsim.palace.models.ports import WavePortConfig
from gsim.palace.ports.config import PalacePort, PortType


def _stack_with_metal() -> LayerStack:
    stack = LayerStack()
    stack.layers["metal"] = Layer(
        name="metal",
        gds_layer=(1, 0),
        zmin=0.0,
        zmax=3.0,
        thickness=3.0,
        material="aluminum",
        layer_type="conductor",
    )
    stack.materials = {"aluminum": {"conductivity": 3.77e7}}
    return stack


def _groups() -> dict:
    return {
        "volumes": {"airbox": {"phys_group": 1}},
        "conductor_surfaces": {
            "metal_xy": {"phys_group": 4},
            "metal_z": {"phys_group": 5},
        },
        "pec_surfaces": {},
        "port_surfaces": {"P1": {"phys_group": 6}, "P2": {"phys_group": 7}},
        "boundary_surfaces": {"absorbing": {"phys_group": [8, 12]}},
    }


def _generate(tmp_path, ports):
    config_path = generate_palace_config(
        groups=_groups(),
        ports=ports,
        port_info=[],
        stack=_stack_with_metal(),
        output_path=tmp_path,
        model_name="palace",
        fmax=100e9,
        simulation_type="driven",
        absorbing_boundary=True,
        hints=None,
    )
    return json.loads(config_path.read_text())


class TestWavePortConfigEigensolverFields:
    def test_defaults_are_none(self) -> None:
        wp = WavePortConfig(name="o1", layer="metal")
        assert wp.eigensolver_type is None
        assert wp.eigensolver_tol is None
        assert wp.eigensolver_ksp_tol is None
        assert wp.eigensolver_max_size is None
        assert wp.eigensolver_verbose is None

    def test_max_size_bool_and_eigensolver_max_size_int_coexist(self) -> None:
        """The pre-existing `max_size` (geometry, bool) and the new
        `eigensolver_max_size` (Palace MaxSize, int) are independent
        fields on the same model - setting one must not affect the other.
        """
        wp = WavePortConfig(
            name="o1", layer="metal", max_size=True, eigensolver_max_size=40
        )
        assert wp.max_size is True
        assert wp.eigensolver_max_size == 40


class TestWavePortEigensolverControlsInConfig:
    def test_all_five_fields_emitted_when_set(self, tmp_path):
        ports = [
            PalacePort(
                name="o1",
                port_type=PortType.WAVEPORT,
                layer="metal",
                eigensolver_type="SLEPc",
                eigensolver_tol=1e-6,
                eigensolver_ksp_tol=1e-8,
                eigensolver_max_size=40,
                eigensolver_verbose=2,
            ),
        ]
        config = _generate(tmp_path, ports)
        entry = config["Boundaries"]["WavePort"][0]
        assert entry["SolverType"] == "SLEPc"
        assert entry["EigenTol"] == 1e-6
        assert entry["KSPTol"] == 1e-8
        assert entry["MaxSize"] == 40
        assert entry["Verbose"] == 2

    def test_none_fields_omitted_not_emitted_as_null(self, tmp_path):
        """Unset controls must be absent from the emitted dict entirely -
        matching the existing R/L/C 'only if not None' pattern in this
        same function - not present with a JSON null value, which Palace
        would have to specifically treat as 'unset' rather than simply
        not seeing the key.
        """
        ports = [
            PalacePort(name="o1", port_type=PortType.WAVEPORT, layer="metal"),
        ]
        config = _generate(tmp_path, ports)
        entry = config["Boundaries"]["WavePort"][0]
        for key in ("SolverType", "EigenTol", "KSPTol", "MaxSize", "Verbose"):
            assert key not in entry

    def test_per_port_independence(self, tmp_path):
        """Two wave ports with different eigensolver settings must not
        leak into each other's emitted entry."""
        ports = [
            PalacePort(
                name="o1",
                port_type=PortType.WAVEPORT,
                layer="metal",
                eigensolver_type="SLEPc",
            ),
            PalacePort(
                name="o2",
                port_type=PortType.WAVEPORT,
                layer="metal",
                excited=False,
                eigensolver_type="ARPACK",
                eigensolver_max_size=64,
            ),
        ]
        config = _generate(tmp_path, ports)
        entries = config["Boundaries"]["WavePort"]
        by_index = {e["Index"]: e for e in entries}
        assert by_index[1]["SolverType"] == "SLEPc"
        assert "MaxSize" not in by_index[1]
        assert by_index[2]["SolverType"] == "ARPACK"
        assert by_index[2]["MaxSize"] == 64

"""Palace test configuration."""

from __future__ import annotations

from pathlib import Path

import gdsfactory as gf
import gmsh
import pytest
import yaml

from gsim.common.stack import Layer, LayerStack

_MESH_COUNT_KEYS = ("elements", "nodes", "tetrahedra")
_MESH_COUNT_RTOL = 0.01


@pytest.fixture
def port_stack():
    """Two metal levels with a one-micrometer interlayer gap."""
    gf.gpdk.PDK.activate()
    return LayerStack(
        layers={
            name: Layer(
                name=name,
                gds_layer=(index, 0),
                zmin=bottom,
                zmax=top,
                thickness=top - bottom,
                material="aluminum",
                layer_type="conductor",
            )
            for index, (name, bottom, top) in enumerate(
                [("m1", 1, 2), ("m2", 3, 5)], start=1
            )
        }
    )


@pytest.fixture
def kernel():
    """Provide a real OpenCASCADE model and release it after each test."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("primitives")
    try:
        yield gmsh.model.occ
    finally:
        gmsh.finalize()


class MeshRegressionFixture:
    """Single-YAML regression fixture: exact groups, 1% tolerance on counts."""

    def __init__(self, original_datadir: Path, request: pytest.FixtureRequest) -> None:
        """Store the reference directory and pytest request."""
        self._original_datadir = original_datadir
        self._request = request

    def check(self, snapshot: dict) -> None:
        """Assert snapshot matches reference; regenerate with --force-regen."""
        force_regen = self._request.config.getoption("force_regen", default=False)
        ref_path = self._original_datadir / f"{self._request.node.name}.yml"

        if force_regen or not ref_path.exists():
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            ref_path.write_text(yaml.dump(snapshot, default_flow_style=False))
            pytest.fail(
                f"Reference file written: {ref_path}. Re-run without --force-regen."
            )

        ref = yaml.safe_load(ref_path.read_text())

        assert snapshot["groups"] == ref["groups"], (
            "Mesh group names do not match reference."
        )

        assert snapshot["mesh"]["invalid_elements"] == 0, "Mesh has invalid elements."
        for key in _MESH_COUNT_KEYS:
            obtained = snapshot["mesh"][key]
            expected = ref["mesh"][key]
            assert obtained == pytest.approx(expected, rel=_MESH_COUNT_RTOL), (
                f"Mesh count '{key}' = {obtained} differs from reference {expected} "
                f"by more than {_MESH_COUNT_RTOL:.0%}."
            )


@pytest.fixture
def mesh_regression(original_datadir, request):
    """Fixture for mesh snapshot regression with 1% tolerance on element counts."""
    return MeshRegressionFixture(original_datadir, request)

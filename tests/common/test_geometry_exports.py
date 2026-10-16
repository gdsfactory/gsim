"""Verify exported geometry and display fallback without a GUI."""

from unittest.mock import Mock

import numpy as np
import pytest
import pyvista as pv

from gsim.common.geometry_model import GeometryModel, Prism
from gsim.common.viz import render3d_pyvista as renderer


@pytest.fixture
def model():
    prism = Prism(np.array([[0, 0], [2, 0], [2, 3], [0, 3]]), 1, 5)
    return GeometryModel({"core": [prism], "clad": [prism]}, ((0, 0, 1), (2, 3, 5)))


def test_multiblock_export_preserves_layers_and_volume(model, tmp_path):
    path = tmp_path / "geometry.vtm"
    renderer.export_3d_mesh(model, str(path), fmt="vtk")
    exported = pv.read(path)
    assert isinstance(exported, pv.MultiBlock)
    assert exported.keys() == ["core_0", "clad_0"]
    for mesh in exported:
        assert isinstance(mesh, pv.PolyData)
        assert mesh.volume == pytest.approx(24)
    with pytest.raises(ValueError, match="Unsupported format"):
        renderer.export_3d_mesh(model, str(tmp_path / "geometry.unknown"))
    with pytest.raises(ValueError, match="glTF export failed"):
        renderer.export_3d_mesh(model, str(tmp_path / "geometry.gltf"))


@pytest.mark.parametrize("camera", [((1, 2, 3), (0, 0, 0), (0, 0, 1)), "unknown"])
def test_notebook_backend_fallback_and_web_export(model, monkeypatch, tmp_path, camera):
    plotter = Mock()
    plotter.add_legend.side_effect = RuntimeError("no legend backend")
    monkeypatch.setattr(renderer.pv, "Plotter", Mock(return_value=plotter))
    backend = Mock(side_effect=[ImportError("trame missing"), None])
    monkeypatch.setattr(renderer.pv, "set_jupyter_backend", backend)
    renderer.plot_prisms_3d(
        model, notebook=True, camera_position=camera, show_simulation_box=False
    )
    assert backend.call_args_list[1].args == ("static",)
    assert plotter.camera_position == (camera if isinstance(camera, tuple) else "iso")
    destination = str(tmp_path / "geometry.html")
    assert renderer.create_web_export(model, destination) == destination
    plotter.export_html.assert_called_once_with(destination)


def test_missing_export_dependency(model, tmp_path, monkeypatch):
    monkeypatch.setattr(renderer, "PYVISTA_AVAILABLE", False)
    with pytest.raises(ImportError, match="PyVista is required"):
        renderer.plot_prisms_3d(model)
    with pytest.raises(ImportError, match="PyVista is required"):
        renderer.export_3d_mesh(model, str(tmp_path / "mesh.vtk"))
    with pytest.raises(ImportError, match="PyVista is required"):
        renderer.create_web_export(model, str(tmp_path / "mesh.html"))

"""Check real VTK mesh construction before the display boundary."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import pyvista as pv
from shapely.geometry import Polygon, box

from gsim.common.geometry_model import GeometryModel, Prism, extract_geometry_model
from gsim.common.viz import render3d_open3d as open_renderer
from gsim.common.viz import render3d_pyvista as renderer
from gsim.common.viz.render2d_interactive import plot_prism_slices_interactive
from gsim.meep.overlay import DielectricOverlay, FiberOverlay, PortOverlay, SimOverlay


@pytest.fixture
def _open3d_containers(monkeypatch):
    # Open3D owns the containers and normal calculation; gsim builds the faces.
    def mesh_container():
        return SimpleNamespace(
            compute_vertex_normals=lambda: None,
            paint_uniform_color=Mock(),
            vertex_colors=[],
        )

    monkeypatch.setattr(open_renderer, "OPEN3D_AVAILABLE", True)
    monkeypatch.setattr(
        open_renderer,
        "o3d",
        SimpleNamespace(
            geometry=SimpleNamespace(
                TriangleMesh=mesh_container,
                LineSet=SimpleNamespace(
                    create_from_triangle_mesh=lambda mesh: SimpleNamespace(
                        points=mesh.vertices, lines=np.array([[0, 1]])
                    )
                ),
            ),
            utility=SimpleNamespace(
                Vector3dVector=np.asarray, Vector3iVector=np.asarray
            ),
        ),
        raising=False,
    )


@pytest.fixture(params=["pyvista", "open3d"])
def prism_meshes(request, _open3d_containers):
    if request.param == "pyvista":
        return renderer._convert_prisms_to_meshes

    def convert(model):
        return {
            layer: [
                pv.PolyData(
                    mesh.vertices,
                    np.column_stack((np.full(len(mesh.triangles), 3), mesh.triangles)),
                )
                for mesh in meshes
            ]
            for layer, meshes in open_renderer._convert_prisms_to_open3d(model).items()
        }

    return convert


@pytest.mark.usefixtures("_open3d_containers")
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("color_by_layer", [False, True])
def test_open3d_plot_preserves_mesh_coordinates_opacity_and_camera(
    monkeypatch, empty, color_by_layer
):
    displayed = []
    monkeypatch.setattr(
        "plotly.graph_objects.Figure.show", lambda self, **_kw: displayed.append(self)
    )
    prism = Prism(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), 0, 1)
    model = GeometryModel(
        {"core": [] if empty else [prism], "clad": [] if empty else [prism]},
        ((0, 0, 0), (1, 1, 1)),
    )
    open_renderer.plot_prisms_3d_open3d(
        model, color_by_layer=color_by_layer, show_edges=True
    )
    figure = displayed[0]
    assert figure.data[-1].name == "Simulation Box"
    assert figure.layout.scene.xaxis.range == pytest.approx(
        (-6, 6) if empty else (-0.1, 1.1)
    )
    assert [button.label for button in figure.layout.updatemenus[0].buttons] == [
        "Iso",
        "Top",
        "Front",
        "Right",
    ]
    if not empty:
        assert [mesh.opacity for mesh in figure.data if mesh.type == "mesh3d"] == [
            1.0,
            0.2,
        ]
        np.testing.assert_allclose(figure.data[0].x, [0, 1, 1, 0, 0, 1, 1, 0])
        assert tuple(figure.data[1].x) == (0, 1, None)
        if not color_by_layer:
            assert figure.data[0].color == "rgb(178,178,178)"


@pytest.mark.parametrize(
    "shape",
    [
        box(0, 0, 2, 2),
        Polygon(
            [(0, 0), (2, 0), (2, 2), (0, 2)],
            holes=[[(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]],
        ),
        Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]),
    ],
)
def test_prism_mesh_is_closed_and_preserves_cross_section_boundary(shape, prism_meshes):
    prism = Prism(np.array(shape.exterior.coords[:-1]), 0, 2, original_polygon=shape)
    model = GeometryModel({"core": [prism]}, ((0, 0, 0), (2, 2, 2)))
    mesh = prism_meshes(model)["core"][0]
    assert mesh.n_open_edges == 0
    np.testing.assert_allclose(mesh.bounds, [0, 2, 0, 2, 0, 2])
    section = mesh.slice(normal="z", origin=(0, 0, 1))
    assert section.compute_cell_sizes(length=True)["Length"].sum() == pytest.approx(
        shape.length
    )


def test_many_triangular_prisms_merge_without_losing_geometry(prism_meshes):
    triangles = [
        Prism(np.array([[2 * i, 0], [2 * i + 1, 0], [2 * i, 1]]), 0, 1)
        for i in range(101)
    ]
    rectangle = Prism(np.array([[0, 2], [1, 2], [1, 3], [0, 3]]), 0, 1)
    model = GeometryModel({"core": [*triangles, rectangle]}, ((0, 0, 0), (201, 3, 1)))
    meshes = prism_meshes(model)["core"]
    assert len(meshes) == 2
    assert sum(mesh.triangulate().volume for mesh in meshes) == pytest.approx(51.5)
    assert all(mesh.n_open_edges == 0 for mesh in meshes)
    assert renderer._merge_triangular_prisms([]) is None


@pytest.mark.parametrize(
    ("theme", "camera", "notebook"),
    [
        ("default", "isometric", False),
        ("dark", "xy", True),
        ("document", "xz", False),
        ("default", "yz", False),
    ],
)
def test_plotter_receives_real_meshes_and_layer_rendering_options(
    monkeypatch, theme, camera, notebook
):
    prism = Prism(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), 0, 1)
    model = GeometryModel({"core": [prism], "clad": [prism]}, ((0, 0, 0), (1, 1, 1)))
    plotter = Mock()
    monkeypatch.setattr(renderer.pv, "Plotter", Mock(return_value=plotter))
    monkeypatch.setattr(renderer.pv, "set_plot_theme", Mock())
    monkeypatch.setattr(renderer.pv, "set_jupyter_backend", Mock())
    renderer.plot_prisms_3d(
        model, notebook=notebook, theme=theme, camera_position=camera
    )
    assert plotter.add_mesh.call_count == 3
    assert [call.kwargs["opacity"] for call in plotter.add_mesh.call_args_list[:2]] == [
        1,
        0.2,
    ]
    assert plotter.add_mesh.call_args.kwargs["style"] == "wireframe"
    plotter.reset_camera.assert_called_once()
    plotter.show.assert_called_once()


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_interactive_slices_preserve_geometry_ports_and_pml(axis):
    shape = box(0, 0, 2, 2)
    prism = Prism(np.array(shape.exterior.coords[:-1]), 0, 1, original_polygon=shape)
    model = GeometryModel({"core": [prism]}, ((0, 0, 0), (2, 2, 1)))
    overlay = SimOverlay(
        (-1, -1, -1),
        (3, 3, 2),
        0.25,
        ports=[
            PortOverlay(
                f"p{normal}{source}", (1, 1, 0.5), 0.5, normal, "+", source, 0.5
            )
            for normal in [0, 1]
            for source in [False, True]
        ],
        dielectrics=[DielectricOverlay("oxide", "sio2", -1, 2)],
        fiber=FiberOverlay(1, 0.5, 15, 1, "TE"),
    )
    figure = plot_prism_slices_interactive(
        model, **{axis: 0.5 if axis == "z" else 1}, overlay=overlay
    )
    names = [trace.name for trace in figure.data]
    assert {"core", "oxide", "Source", "Monitor", "Sim cell", "PML"} <= set(names)
    assert names.count("PML") == 4
    core = next(trace for trace in figure.data if trace.name == "core")
    assert (min(core.x), max(core.x)) == (0, 2)
    assert (min(core.y), max(core.y)) == ((0, 2) if axis == "z" else (0, 1))


@pytest.mark.parametrize("hole_count", [0, 1, 2])
def test_geometry_extraction_preserves_material_area_and_holes(hole_count):
    holes = [
        list(box(0.5, 0.5, 1, 1).exterior.coords),
        list(box(2, 2, 2.5, 2.5).exterior.coords),
    ]
    shape = Polygon(box(0, 0, 3, 3).exterior.coords, holes=holes[:hole_count])
    bounds = ((0, 0, -1), (3, 3, 1))
    source = SimpleNamespace(
        geometry_layers={
            "core": SimpleNamespace(mesh_order=2, material="si", sidewall_angle=0)
        },
        polygons={"core": shape},
        bbox=bounds,
        get_layer_bbox=lambda _name: bounds,
    )
    model = extract_geometry_model(source)
    assert sum(
        Polygon(prism.vertices).area for prism in model.all_prisms
    ) == pytest.approx(shape.area)
    assert all(
        prism.material == "si" and prism.height == 2 for prism in model.all_prisms
    )
    assert model.get_layer_center("core") == (1.5, 1.5, 0)

"""Three.js / FastAPI-based 3D rendering for GeometryModel.

Provides browser-based interactive 3D visualisation via a local FastAPI server
that serves a Three.js HTML page.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from gsim.common.geometry_model import GeometryModel
from gsim.common.viz._colors import generate_layer_colors_with_opacity
from gsim.common.viz._mesh_helpers import simulation_box_corners
from gsim.common.viz.render3d_open3d import _convert_prisms_to_open3d

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def serve_threejs_visualization(
    geometry_model: GeometryModel,
    *,
    show_edges: bool = False,
    color_by_layer: bool = True,
    show_simulation_box: bool = True,
    layer_opacity: dict[str, float] | None = None,
    port: int = 8000,
    auto_open: bool = True,
    show_stats: bool = False,
    **kwargs: Any,
) -> str:
    """Start a FastAPI server with a Three.js 3D viewer.

    Args:
        geometry_model: GeometryModel containing prisms and bbox.
        show_edges: Show wireframe edges.
        color_by_layer: Colour each layer differently.
        show_simulation_box: Draw the simulation box.
        layer_opacity: Per-layer opacity override.
        port: Port to serve on (default 8000).
        auto_open: Open the browser automatically.
        show_stats: Show FPS counter.
        **kwargs: Extra Three.js options.

    Returns:
        URL of the running server.
    """
    try:
        import threading
        import time
        import webbrowser

        import uvicorn
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
    except ImportError as err:
        raise ImportError(
            "FastAPI and uvicorn required. Install with: pip install fastapi uvicorn"
        ) from err

    app = FastAPI(title="3D Geometry Viewer")

    layer_meshes = _convert_prisms_to_open3d(geometry_model)
    colors, opacity_dict = generate_layer_colors_with_opacity(
        list(layer_meshes.keys()), layer_opacity
    )

    logger.info("Converting geometry: %d layers found", len(layer_meshes))
    for layer_name, meshes in layer_meshes.items():
        logger.info("  Layer '%s': %d meshes", layer_name, len(meshes))

    threejs_data = _convert_to_threejs_data(
        layer_meshes, colors, opacity_dict, color_by_layer
    )

    if show_simulation_box:
        threejs_data["simulation_box"] = _create_simulation_box_threejs(geometry_model)

    total_vertices = 0
    total_faces = 0
    for layer in threejs_data.get("layers", []):
        for mesh in layer.get("meshes", []):
            total_vertices += len(mesh.get("vertices", [])) // 3
            total_faces += len(mesh.get("faces", [])) // 3
    logger.info(
        "Three.js data prepared: %d vertices, %d faces",
        total_vertices,
        total_faces,
    )

    @app.get("/", response_class=HTMLResponse)
    def get_visualization():
        """Return the Three.js viewer HTML page."""
        return _generate_threejs_html(
            threejs_data,
            show_edges=show_edges,
            show_stats=show_stats,
            **kwargs,
        )

    @app.get("/api/geometry")
    def get_geometry_data():
        """Return geometry mesh data as JSON."""
        return threejs_data

    @app.get("/api/info")
    def get_info():
        """Return summary info about the geometry."""
        return {
            "layers": len(threejs_data.get("layers", [])),
            "total_meshes": sum(
                len(layer.get("meshes", [])) for layer in threejs_data.get("layers", [])
            ),
            "has_simulation_box": "simulation_box" in threejs_data,
            "layer_names": [
                layer.get("name") for layer in threejs_data.get("layers", [])
            ],
        }

    server_url = f"http://localhost:{port}"

    def run_server():
        try:
            uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
        except Exception as e:
            logger.info("Server error: %s", e)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(1)

    if auto_open:
        webbrowser.open(server_url)

    logger.info("Server started at: %s", server_url)
    logger.info("Geometry API available at: %s/api/geometry", server_url)
    logger.info("Server running in background. Keep Python session alive to view.")

    return server_url


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _convert_to_threejs_data(
    layer_meshes: dict[str, list[Any]],
    colors: dict[str, list[float]],
    opacity_dict: dict[str, float],
    color_by_layer: bool,
) -> dict[str, Any]:
    """Convert Open3D meshes to Three.js-compatible JSON data."""
    threejs_data: dict[str, Any] = {"layers": []}

    for layer_name, meshes in layer_meshes.items():
        layer_color = colors[layer_name] if color_by_layer else [0.7, 0.7, 0.7]
        layer_opacity = opacity_dict.get(layer_name, 0.8)

        layer_data: dict[str, Any] = {
            "name": layer_name,
            "color": [int(c * 255) for c in layer_color[:3]],
            "opacity": layer_opacity,
            "meshes": [],
        }

        for i, mesh in enumerate(meshes):
            vertices = np.asarray(mesh.vertices).flatten().tolist()
            faces = np.asarray(mesh.triangles).flatten().tolist()

            layer_data["meshes"].append(
                {
                    "vertices": vertices,
                    "faces": faces,
                    "id": f"{layer_name}_{i}",
                }
            )

        threejs_data["layers"].append(layer_data)

    return threejs_data


def _create_simulation_box_threejs(
    geometry_model: GeometryModel,
) -> dict[str, Any]:
    """Create simulation box data for Three.js."""
    points, lines = simulation_box_corners(geometry_model)
    return {
        "vertices": points.flatten().tolist(),
        "lines": [idx for pair in lines for idx in pair],
        "color": [0, 0, 0],
    }


def _generate_threejs_html(
    threejs_data: dict[str, Any],
    *,
    show_edges: bool = False,
    show_stats: bool = False,
    **kwargs: Any,
) -> str:
    """Generate HTML using the viewer.html template."""
    template_path = Path(__file__).parent / "templates" / "viewer.html"

    with open(template_path) as f:
        template = f.read()

    width = kwargs.get("width", "100vw")
    height = kwargs.get("height", "100vh")
    background_color = kwargs.get("background_color", "#f0f0f0")
    show_wireframe = str(show_edges).lower()
    stats_display = "block" if show_stats else "none"

    data_json = json.dumps(threejs_data, indent=2)

    return template.format(
        geometry_data=data_json,
        show_wireframe=show_wireframe,
        show_stats=str(show_stats).lower(),
        width=width,
        height=height,
        background_color=background_color,
        stats_display=stats_display,
    )

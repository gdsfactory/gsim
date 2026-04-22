"""MEEP runner script generation.

Generates a self-contained Python script that runs on the cloud MEEP instance.
The script reads a JSON config + GDS file, builds the MEEP simulation, runs it,
and outputs S-parameters as CSV.
"""

from __future__ import annotations

_MEEP_RUNNER_TEMPLATE = '''\
#!/usr/bin/env python3
"""Auto-generated MEEP runner script.

Reads simulation config from JSON + layout from GDS, builds geometry,
runs FDTD, and extracts S-parameters via mode decomposition.

Cloud dependencies: meep, gdsfactory, numpy, scipy, shapely
"""

import csv
import cmath
import json
import logging
import math
import os
import sys
import time

import meep as mp
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("meep_runner")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path):
    """Load simulation config from JSON file."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

def build_materials(config):
    """Build MEEP material objects from config."""
    materials = {}
    for name, props in config["materials"].items():
        n = props["refractive_index"]
        k = props["extinction_coeff"]
        if k > 0:
            materials[name] = mp.Medium(index=n, D_conductivity=2 * cmath.pi * k / n)
        else:
            materials[name] = mp.Medium(index=n)
    return materials


# ---------------------------------------------------------------------------
# Geometry: GDS + layer_stack -> mp.Prism objects
# ---------------------------------------------------------------------------

def load_gds_component(gds_filename):
    """Load GDS file as a gdsfactory Component."""
    import gdsfactory as gf
    gf.gpdk.PDK.activate()
    return gf.import_gds(gds_filename)


def _klayout_poly_to_coords(poly_obj, dbu):
    """Convert a KLayout polygon object to a list of (x, y) coordinate tuples.

    Args:
        poly_obj: KLayout PolygonWithProperties or similar object
        dbu: Database unit scaling factor (converts integer coords to microns)

    Returns:
        list of (float, float) coordinate tuples
    """
    coords = []
    for pt in poly_obj.each_point_hull():
        coords.append((pt.x * dbu, pt.y * dbu))
    return coords


def extract_layer_polygons(component, gds_layer, simplify_tol=0.0):
    """Extract merged Shapely polygons for a GDS layer from a component.

    Uses gdsfactory / KLayout to get polygons for the given (layer, datatype)
    tuple, then converts to Shapely with hole support.

    Handles both old gdsfactory (dict keys are (layer, datatype) tuples,
    values are numpy arrays) and new gdsfactory 9.x (dict keys are integer
    KLayout layer indices, values are PolygonWithProperties objects).

    Args:
        component: gdsfactory Component
        gds_layer: [layer_number, datatype] list
        simplify_tol: Shapely simplification tolerance in um (0=off)

    Returns:
        list of Shapely Polygon objects
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    layer_tuple = tuple(gds_layer)
    raw = component.get_polygons(layers=(layer_tuple,), merge=True)

    # Collect all polygon objects from the dict, regardless of key type.
    # Old gdsfactory: keys are (layer, datatype) tuples, values are ndarray lists
    # New gdsfactory 9.x: keys are integer layer indices, values are
    # PolygonWithProperties lists
    all_objects = []
    if isinstance(raw, dict):
        for v in raw.values():
            if isinstance(v, list):
                all_objects.extend(v)
            else:
                all_objects.append(v)
    elif raw is not None:
        all_objects = list(raw) if hasattr(raw, '__iter__') else [raw]

    dbu = getattr(getattr(component, 'kcl', None), 'dbu', 0.001)

    shapely_polygons = []
    for obj in all_objects:
        try:
            # KLayout PolygonWithProperties — convert via each_point_hull
            if hasattr(obj, 'each_point_hull'):
                coords = _klayout_poly_to_coords(obj, dbu)
            # Numpy array — legacy path
            elif hasattr(obj, 'shape') and len(obj) >= 3:
                coords = [(float(p[0]), float(p[1])) for p in obj]
            else:
                continue

            if len(coords) < 3:
                continue
            poly = Polygon(coords)
            if poly.is_valid and not poly.is_empty:
                shapely_polygons.append(poly)
        except Exception:
            continue

    if not shapely_polygons:
        return []

    merged = unary_union(shapely_polygons)
    if simplify_tol > 0:
        merged = merged.simplify(simplify_tol, preserve_topology=True)
    if hasattr(merged, "geoms"):
        return list(merged.geoms)
    return [merged] if not merged.is_empty else []


def triangulate_polygon_with_holes(polygon):
    """Triangulate a polygon with holes using Delaunay triangulation.

    Returns list of triangles as coordinate lists, each triangle being
    [[x1,y1], [x2,y2], [x3,y3]].
    """
    from scipy.spatial import Delaunay
    from shapely.geometry import Point

    all_points = list(polygon.exterior.coords[:-1])
    for interior in polygon.interiors:
        all_points.extend(list(interior.coords[:-1]))

    if len(all_points) < 3:
        return [list(polygon.exterior.coords[:-1])]

    points_2d = np.array(all_points)
    tri = Delaunay(points_2d)

    triangles = []
    for simplex in tri.simplices:
        triangle_pts = points_2d[simplex]
        centroid = np.mean(triangle_pts, axis=0)
        if polygon.contains(Point(centroid)):
            triangles.append(triangle_pts.tolist())

    return triangles if triangles else [list(polygon.exterior.coords[:-1])]


def extract_xz_rectangles_runner(component, layer_stack, y_cut, eps=1e-9):
    """Inlined XZ cross-section cutter (mirrors gsim.common.cross_section)."""
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    dbu = getattr(getattr(component, "kcl", None), "dbu", 0.001)

    rects = []
    for layer_entry in layer_stack:
        gds_layer_tuple = tuple(layer_entry["gds_layer"])
        shapely_polys = _xz_runner_layer_polys(component, gds_layer_tuple, dbu)
        if not shapely_polys:
            continue

        merged = unary_union(shapely_polys)
        if merged.is_empty:
            continue

        minx, miny, maxx, maxy = merged.bounds
        if y_cut < miny - eps or y_cut > maxy + eps:
            continue

        cut_line = LineString([(minx - 1.0, y_cut), (maxx + 1.0, y_cut)])
        intersection = merged.intersection(cut_line)
        intervals = _xz_runner_line_intervals(intersection)

        for x0, x1 in intervals:
            if x1 - x0 <= eps:
                continue
            rects.append({
                "x0": x0,
                "x1": x1,
                "zmin": layer_entry["zmin"],
                "zmax": layer_entry["zmax"],
                "layer_name": layer_entry["layer_name"],
                "material": layer_entry["material"],
            })
    return rects


def _xz_runner_layer_polys(component, gds_layer_tuple, dbu):
    """Return shapely Polygons (with holes) for one GDS layer of component."""
    from shapely.geometry import Polygon

    raw = component.get_polygons(layers=(gds_layer_tuple,), merge=True)
    if not isinstance(raw, dict) or not raw:
        return []

    polys = []
    for value in raw.values():
        items = list(value) if isinstance(value, list) else [value]
        for obj in items:
            exterior, holes = _xz_runner_poly_to_coords(obj, dbu)
            if exterior is None or len(exterior) < 3:
                continue
            try:
                poly = Polygon(exterior, holes=holes)
            except (ValueError, TypeError):
                continue
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            if hasattr(poly, "geoms"):
                polys.extend(poly.geoms)
            else:
                polys.append(poly)
    return polys


def _xz_runner_poly_to_coords(obj, dbu):
    """Convert a polygon-like object to (exterior, list_of_holes)."""
    if hasattr(obj, "each_point_hull"):
        exterior = [(pt.x * dbu, pt.y * dbu) for pt in obj.each_point_hull()]
        holes = []
        try:
            n_holes = obj.holes()
        except AttributeError:
            n_holes = 0
        for i in range(n_holes):
            try:
                holes.append(
                    [(pt.x * dbu, pt.y * dbu) for pt in obj.each_point_hole(i)]
                )
            except (AttributeError, IndexError):
                continue
        return exterior, holes
    if hasattr(obj, "__iter__"):
        try:
            return [(float(p[0]), float(p[1])) for p in obj], []
        except (TypeError, IndexError):
            return None, []
    return None, []


def _xz_runner_line_intervals(intersection):
    """Extract sorted (x0, x1) intervals from a shapely line intersection."""
    from shapely.geometry import LineString, MultiLineString

    if intersection.is_empty:
        return []
    lines = []
    if isinstance(intersection, LineString):
        lines = [intersection]
    elif isinstance(intersection, MultiLineString):
        lines = list(intersection.geoms)
    else:
        for geom in getattr(intersection, "geoms", []):
            if isinstance(geom, LineString):
                lines.append(geom)

    intervals = []
    for line in lines:
        xs = [c[0] for c in line.coords]
        intervals.append((min(xs), max(xs)))
    intervals.sort()
    merged = []
    for x0, x1 in intervals:
        if merged and x0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])
    return [(a, b) for a, b in merged]


def _build_geometry_xz(config, materials, component):
    """Build meep geometry from XZ cross-section rectangles."""
    y_cut = config.get("y_cut")
    if y_cut is None:
        y_cut = 0.0

    rects = extract_xz_rectangles_runner(
        component, config["layer_stack"], y_cut
    )

    geometry = []
    for r in rects:
        mat = materials.get(r["material"], mp.Medium())
        width_x = r["x1"] - r["x0"]
        thickness_z = r["zmax"] - r["zmin"]
        if width_x <= 0 or thickness_z <= 0:
            continue
        center_x = (r["x0"] + r["x1"]) / 2.0
        center_z = (r["zmin"] + r["zmax"]) / 2.0
        block = mp.Block(
            size=mp.Vector3(width_x, mp.inf, thickness_z),
            center=mp.Vector3(center_x, 0.0, center_z),
            material=mat,
        )
        geometry.append(block)

    logger.info("XZ: %d rectangles extracted at y=%.4f", len(geometry), y_cut)
    return geometry


def build_background_slabs(config, materials):
    """Build background mp.Block slabs from dielectric entries.

    These infinite-XY slabs fill the simulation cell at each z-range with
    the correct cladding/substrate material.  They must come FIRST in the
    geometry list so that patterned prisms (added later) take precedence.

    XY 2D (``plane='xy'``) skips slabs entirely — the z-dimension is
    collapsed.  XZ 2D (``plane='xz'``) DOES include slabs because they
    form the vertical stack.  3D always includes slabs.
    """
    is_3d = config.get("is_3d", True)
    plane = config.get("plane", "xy")
    if not is_3d and plane == "xy":
        return []

    slabs = []
    for diel in sorted(config["dielectrics"], key=lambda d: d["zmin"]):
        mat = materials.get(diel["material"])
        if mat is None:
            continue
        zmin = diel["zmin"]
        zmax = diel["zmax"]
        thickness = zmax - zmin
        if thickness <= 0:
            continue
        block = mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, thickness),
            center=mp.Vector3(0, 0, (zmin + zmax) / 2),
            material=mat,
        )
        slabs.append(block)
    return slabs


def build_symmetries(config):
    """Build MEEP symmetry objects from config."""
    direction_map = {"X": mp.X, "Y": mp.Y, "Z": mp.Z}
    symmetries = []
    for sym in config["symmetries"]:
        symmetries.append(
            mp.Mirror(direction_map[sym["direction"]], phase=sym["phase"])
        )
    return symmetries


# Map string component names to MEEP field components
_COMPONENT_MAP = {
    "Ex": mp.Ex, "Ey": mp.Ey, "Ez": mp.Ez,
    "Hx": mp.Hx, "Hy": mp.Hy, "Hz": mp.Hz,
}


def _make_or_condition(*fns):
    """Combine stopping conditions with OR logic.

    meep treats lists passed to ``until_after_sources`` as AND (all must
    be True).  This helper wraps multiple callables into a single function
    that returns True when ANY condition fires — the intended behaviour
    for "decay check OR time-cap safety net".
    """
    def _check(sim_obj):
        return any(fn(sim_obj) for fn in fns)
    return _check


def _make_time_cap(cap):
    """Return a callable that fires once *cap* sim-time units have elapsed."""
    _t0 = [None]
    def _check(sim_obj):
        if _t0[0] is None:
            _t0[0] = sim_obj.meep_time()
        return (sim_obj.meep_time() - _t0[0]) >= cap
    return _check


def _make_wall_time_cap(wall_seconds):
    """Return a callable that fires once *wall_seconds* of real time have elapsed."""
    _deadline = [None]
    def _check(sim_obj):
        if _deadline[0] is None:
            _deadline[0] = time.time() + wall_seconds
        return time.time() >= _deadline[0]
    return _check


def resolve_decay_monitor_point(config):
    """Return center mp.Vector3 for decay monitoring.

    Uses the port named by stopping.decay_monitor_port if set,
    otherwise picks the first non-source port.  Falls back to
    the first port if all are sources.
    """
    stopping = config["stopping"]
    target_name = stopping.get("decay_monitor_port")
    ports = config["ports"]

    if target_name:
        for p in ports:
            if p["name"] == target_name:
                return mp.Vector3(*p["center"])

    # Fallback: first non-source port, or first port
    for p in ports:
        if not p["is_source"]:
            return mp.Vector3(*p["center"])
    if ports:
        return mp.Vector3(*ports[0]["center"])

    return mp.Vector3(0, 0, 0)


def build_geometry(config, materials):
    """Build MEEP geometry from GDS file + layer stack config.

    For each layer in layer_stack:
      1. Extract polygons from the GDS for that layer's gds_layer
      2. Extrude to 3D as mp.Prism with correct z-range and material
      3. Handle polygon holes via Delaunay triangulation

    In XY 2D mode (is_3d=False, plane="xy"), all prisms are placed at z=0
    and sidewall angles are ignored, matching gplugins behaviour.

    In XZ 2D mode (is_3d=False, plane="xz"), the geometry is built from
    axis-aligned rectangles sliced at y=y_cut (see _build_geometry_xz).
    """
    gds_filename = config["gds_filename"]
    component = load_gds_component(gds_filename)

    is_3d = config.get("is_3d", True)
    plane = config.get("plane", "xy")
    if not is_3d and plane == "xz":
        return _build_geometry_xz(config, materials, component), component

    accuracy = config["accuracy"]
    simplify_tol = accuracy["simplify_tol"]

    geometry = []
    total_vertices = 0

    for layer_entry in config["layer_stack"]:
        material_name = layer_entry["material"]
        mat = materials.get(material_name, mp.Medium())
        zmin = layer_entry["zmin"] if is_3d else 0
        zmax = layer_entry["zmax"]
        height = zmax - zmin if is_3d else (layer_entry["zmax"] - layer_entry["zmin"])
        gds_layer = layer_entry["gds_layer"]
        sidewall_angle_deg = layer_entry["sidewall_angle"]
        if is_3d and sidewall_angle_deg:
            sw_rad = math.radians(sidewall_angle_deg)
        else:
            sw_rad = 0

        if height <= 0:
            continue

        polygons = extract_layer_polygons(
            component, gds_layer, simplify_tol=simplify_tol,
        )

        for polygon in polygons:
            if polygon.is_empty or not polygon.is_valid:
                continue

            if hasattr(polygon, "interiors") and polygon.interiors:
                # Polygon has holes — triangulate
                triangles = triangulate_polygon_with_holes(polygon)
                for tri_coords in triangles:
                    vertices = [mp.Vector3(p[0], p[1], zmin) for p in tri_coords]
                    total_vertices += len(vertices)
                    prism = mp.Prism(
                        vertices=vertices,
                        height=height,
                        material=mat,
                        sidewall_angle=sw_rad,
                    )
                    geometry.append(prism)
            else:
                # Simple polygon — direct extrusion
                coords = list(polygon.exterior.coords[:-1])
                vertices = [mp.Vector3(p[0], p[1], zmin) for p in coords]
                total_vertices += len(vertices)
                prism = mp.Prism(
                    vertices=vertices,
                    height=height,
                    material=mat,
                    sidewall_angle=sw_rad,
                )
                geometry.append(prism)

    logger.info("Total vertices across all prisms: %d", total_vertices)
    return geometry, component


# ---------------------------------------------------------------------------
# Sources and monitors
# ---------------------------------------------------------------------------

def get_port_z_span(config):
    """Get z-span for waveguide port mode monitors.

    Prefers the precomputed ``monitor_z_span`` (sized around the core
    layer: ``core_thickness + 2*port_margin``) so monitors capture the
    guided mode without spanning the full cell.

    In XY 2D mode returns a large arbitrary value (20 um) since the
    z-dimension is collapsed and the size doesn't affect the simulation.
    """
    is_3d = config.get("is_3d", True)
    plane = config.get("plane", "xy")
    if not is_3d and plane == "xy":
        return 20
    mz = config.get("monitor_z_span")
    if mz is not None:
        return mz
    zmin = min(l["zmin"] for l in config["layer_stack"])
    zmax = max(l["zmax"] for l in config["layer_stack"])
    return zmax - zmin


def _build_fiber_source(config, fiber):
    """Construct a mp.GaussianBeamSource for XZ 2D fiber coupling.

    Polarization (PIC convention):
      - TE -> E along waveguide width (Ey, out of XZ plane)
      - TM -> E in the XZ plane (Ex)
    """
    fdtd = config["fdtd"]
    fcen = fdtd["fcen"]
    fwidth = config["source"]["fwidth"]

    k_dir = mp.Vector3(*fiber["k_direction"])
    if fiber["polarization"] == "TE":
        e_dir = mp.Vector3(0, 1, 0)
    else:
        e_dir = mp.Vector3(1, 0, 0)

    center = mp.Vector3(fiber["x"], 0.0, fiber["z"])

    src_x_size = _estimate_source_x_size(config)

    # beam_x0 is the focus offset *relative to* the source center, not an
    # absolute position (MEEP docs). Focus sits on the source line -> zero.
    src = mp.GaussianBeamSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth, is_integrated=True),
        center=center,
        size=mp.Vector3(src_x_size, 0, 0),
        beam_x0=mp.Vector3(0, 0, 0),
        beam_kdir=k_dir,
        beam_w0=fiber["waist"],
        beam_E0=e_dir,
    )
    return [src]


def _estimate_source_x_size(config):
    """Source-line X length for the fiber Gaussian beam.

    The line is centered on ``fiber.x`` (not the cell center). To avoid
    spilling into PML — which corrupts the launched beam when the fiber
    sits off-center in the cell — cap the half-size to the distance from
    ``fiber.x`` to the nearest cell-interior edge (minus a small safety).

    Target length is ``6 * waist`` so the Gaussian envelope is captured
    well. If the cell is too narrow for that, cap and log a warning —
    the user should increase ``domain.margin_xy`` or move the fiber.
    """
    bbox = config.get("component_bbox")
    domain = config["domain"]
    dpml = domain["dpml"]
    margin_xy = domain["margin_xy"]
    fiber = config.get("fiber_source")

    if bbox is not None:
        width = bbox[2] - bbox[0]
        bbox_left, bbox_right = bbox[0], bbox[2]
    else:
        width = 20.0
        bbox_left, bbox_right = -10.0, 10.0

    interior_width = width + 2 * margin_xy
    if fiber is None:
        return max(interior_width, 2.0)

    interior_left = bbox_left - margin_xy
    interior_right = bbox_right + margin_xy
    fx = fiber["x"]
    waist = fiber.get("waist", 0.0)

    safety = max(dpml * 0.1, 0.1)
    max_half = min(fx - interior_left, interior_right - fx) - safety
    if max_half <= 0.0:
        logger.warning(
            "Fiber x=%.2f is outside or at the cell interior edge "
            "(interior [%.2f, %.2f]); using minimum source size. "
            "Increase domain.margin_xy or move the fiber.",
            fx, interior_left, interior_right,
        )
        return 2.0

    target = max(6.0 * waist, 2.0)
    if target > 2.0 * max_half:
        logger.warning(
            "Fiber source line capped at %.2f um (target %.2f = 6*waist); "
            "cell is too narrow on one side of fiber x=%.2f. "
            "Increase domain.margin_xy so the Gaussian envelope fits.",
            2.0 * max_half, target, fx,
        )
        return 2.0 * max_half
    return target


def build_sources(config):
    """Build MEEP source from config port data.

    The source is offset from the port center by ``source_port_offset``
    along the propagation direction (into the device).  This separates
    the soft source from the port monitor so eigenmode coefficients
    measure the true incident amplitude rather than half of it.

    In 2D mode (is_3d=False), enforces transverse-electric parity
    (EVEN_Y + ODD_Z) to match gplugins 2D convention.

    If the config has a fiber_source entry (XZ 2D grating-coupler sim),
    returns a single GaussianBeamSource instead of port-based EigenModeSources.
    """
    fiber = config.get("fiber_source")
    if fiber is not None:
        return _build_fiber_source(config, fiber)

    fdtd = config["fdtd"]
    fcen = fdtd["fcen"]
    df = fdtd["df"]
    fwidth = config["source"]["fwidth"]
    z_span = get_port_z_span(config)
    port_margin = config["domain"]["port_margin"]
    source_port_offset = config["domain"].get("source_port_offset", 0.1)
    is_3d = config.get("is_3d", True)
    plane = config.get("plane", "xy")
    if is_3d:
        eig_parity = mp.NO_PARITY
    elif plane == "xz":
        # XZ 2D: cell_y=0, invariant axis is Y. Use ODD_Y for TE-like modes.
        eig_parity = mp.ODD_Y
    else:
        eig_parity = mp.EVEN_Y + mp.ODD_Z

    sources = []
    for port in config["ports"]:
        if not port["is_source"]:
            continue

        # Offset source center into the device along propagation direction
        center_list = list(port["center"])
        normal_axis = port["normal_axis"]
        direction = port["direction"]
        if direction == "+":
            center_list[normal_axis] += source_port_offset
        else:
            center_list[normal_axis] -= source_port_offset
        center = mp.Vector3(*center_list)

        size = [0, 0, 0]
        transverse_axis = 1 - normal_axis
        size[transverse_axis] = port["width"] + 2 * port_margin
        size[2] = z_span

        # Propagation axis
        prop_axis = mp.X if normal_axis == 0 else mp.Y

        # eig_kpoint points INTO the device (direction of incoming mode)
        if normal_axis == 0:
            kpoint = mp.Vector3(x=1) if direction == "+" else mp.Vector3(x=-1)
        else:
            kpoint = mp.Vector3(y=1) if direction == "+" else mp.Vector3(y=-1)

        eig_src = mp.EigenModeSource(
            src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
            center=center,
            size=mp.Vector3(*size),
            eig_band=1,
            direction=prop_axis,
            eig_kpoint=kpoint,
            eig_match_freq=True,
            eig_parity=eig_parity,
        )
        sources.append(eig_src)

    return sources


FIBER_FLUX_OFFSET = 0.3  # µm below fiber source center


def build_monitors(config, sim):
    """Build mode monitors at all ports and return flux regions.

    The source-port monitor is offset further into the device (past
    the source) by ``source_port_offset + distance_source_to_monitors``
    so the forward-going mode from the source passes through it at
    full amplitude.  This matches the gplugins approach.

    When ``config["fiber_source"]`` is set, also builds a flux monitor
    just in front of (below) the Gaussian beam to measure the launched
    power, used as normalization reference for fiber->waveguide S-params.

    Returns:
        (port_monitors, fiber_flux) where fiber_flux is None unless a
        fiber source is configured.
    """
    fdtd = config["fdtd"]
    fcen = fdtd["fcen"]
    df = fdtd["df"]
    nfreq = fdtd["num_freqs"]
    z_span = get_port_z_span(config)
    port_margin = config["domain"]["port_margin"]
    source_port_offset = config["domain"].get("source_port_offset", 0.1)
    distance_source_to_monitors = config["domain"].get(
        "distance_source_to_monitors", 0.2
    )

    monitors = {}
    for port in config["ports"]:
        center_list = list(port["center"])
        normal_axis = port["normal_axis"]
        direction = port["direction"]

        # All monitors shift inward from port center (matches gplugins):
        #   source-port monitor: source_port_offset + distance_source_to_monitors
        #   non-source monitors: source_port_offset
        if port["is_source"]:
            offset = source_port_offset + distance_source_to_monitors
        else:
            offset = source_port_offset

        if offset > 0:
            if direction == "+":
                center_list[normal_axis] += offset
            else:
                center_list[normal_axis] -= offset

        center = mp.Vector3(*center_list)

        size = [0, 0, 0]
        transverse_axis = 1 - normal_axis
        size[transverse_axis] = port["width"] + 2 * port_margin
        size[2] = z_span

        flux = sim.add_mode_monitor(
            fcen, df, nfreq,
            mp.ModeRegion(center=center, size=mp.Vector3(*size)),
        )
        monitors[port["name"]] = flux

    fiber_flux = None
    fiber = config.get("fiber_source")
    if fiber is not None:
        src_x_size = _estimate_source_x_size(config)
        z_monitor = fiber["z"] - FIBER_FLUX_OFFSET
        fiber_flux = sim.add_flux(
            fcen, df, nfreq,
            mp.FluxRegion(
                center=mp.Vector3(fiber["x"], 0.0, z_monitor),
                size=mp.Vector3(src_x_size, 0, 0),
                direction=mp.Z,
            ),
        )

    return monitors, fiber_flux


# ---------------------------------------------------------------------------
# S-parameter extraction
# ---------------------------------------------------------------------------

def _port_kpoint(port):
    """Return positive-axis kpoint vector for eigenmode decomposition.

    Always points along +normal_axis so that:
      alpha[band, freq, 0] = forward (+axis) coefficient
      alpha[band, freq, 1] = backward (-axis) coefficient
    """
    if port["normal_axis"] == 0:
        return mp.Vector3(x=1)
    return mp.Vector3(y=1)


def extract_s_params(config, sim, monitors, fiber_flux=None):
    """Extract S-parameters via mode decomposition.

    Uses MEEP eigenmode coefficients with explicit kpoint_func to anchor
    the forward/backward convention:
      alpha[band, freq, 0] = forward (+normal_axis) coefficient
      alpha[band, freq, 1] = backward (-normal_axis) coefficient

    In 2D mode, uses transverse-electric parity (EVEN_Y + ODD_Z) for
    eigenmode decomposition.

    Port "direction" field = direction of incoming mode along normal_axis:
      "+" -> incoming goes +normal, outgoing (reflected/transmitted) goes -normal
      "-" -> incoming goes -normal, outgoing (reflected/transmitted) goes +normal

    When ``config["fiber_source"]`` is set, the reference is the launched
    power through ``fiber_flux`` rather than a port eigenmode coefficient;
    S-params are named ``S{i+1}0`` (fiber indexed as port 0).

    Returns:
        (s_params, debug_data) tuple where debug_data contains eigenmode
        diagnostics for post-run analysis.
    """
    if config.get("fiber_source") is not None:
        return _extract_s_params_fiber(config, sim, monitors, fiber_flux)

    port_names = [p["name"] for p in config["ports"]]
    ports = {p["name"]: p for p in config["ports"]}
    source_port = None
    for p in config["ports"]:
        if p["is_source"]:
            source_port = p["name"]
            break

    if not source_port:
        return {}, {}

    fdtd = config["fdtd"]
    fcen = fdtd["fcen"]
    nfreq = fdtd["num_freqs"]
    df = fdtd["df"]
    freqs = np.linspace(fcen - df / 2, fcen + df / 2, nfreq)

    debug_data = {
        "eigenmode_info": {},
        "raw_coefficients": {},
        "incident_coefficients": {},
    }

    def _incoming_idx(direction):
        """Alpha index for the incoming (incident) mode at a port."""
        return 0 if direction == "+" else 1

    def _outgoing_idx(direction):
        """Alpha index for the outgoing (reflected/transmitted) mode at a port."""
        return 1 if direction == "+" else 0

    def _collect_eigenmode_debug(port_name, ob, freqs):
        """Collect eigenmode diagnostics from coefficients object."""
        info = {"band": 1}
        try:
            kdom = ob.kdom
            kdom_list = [[float(kdom[i].x), float(kdom[i].y), float(kdom[i].z)]
                         for i in range(len(kdom))]
            info["kdom"] = kdom_list
            info["n_eff"] = [float(np.linalg.norm(kdom_list[i])) / float(freqs[i])
                             if float(freqs[i]) > 0 else 0.0
                             for i in range(min(len(kdom_list), len(freqs)))]
        except Exception:
            info["kdom"] = []
            info["n_eff"] = []
        try:
            cg = ob.cg
            info["group_velocity"] = [float(cg[i]) for i in range(len(cg))]
        except Exception:
            info["group_velocity"] = []
        return info

    # Get incident coefficient at source port for normalization
    is_3d = config.get("is_3d", True)
    plane = config.get("plane", "xy")
    if is_3d:
        eig_parity = mp.NO_PARITY
    elif plane == "xz":
        # XZ 2D: cell_y=0, invariant axis is Y. Use ODD_Y for TE-like modes.
        eig_parity = mp.ODD_Y
    else:
        eig_parity = mp.EVEN_Y + mp.ODD_Z

    src_dir = ports[source_port]["direction"]
    src_kp = _port_kpoint(ports[source_port])
    src_ob = sim.get_eigenmode_coefficients(
        monitors[source_port], [1], eig_parity=eig_parity,
        kpoint_func=lambda f, n, kp=src_kp: kp,
    )
    incident_coeffs = src_ob.alpha[0, :, _incoming_idx(src_dir)]

    debug_data["eigenmode_info"][source_port] = _collect_eigenmode_debug(
        source_port, src_ob, freqs)
    debug_data["incident_coefficients"] = {
        "port": source_port,
        "magnitudes": [float(abs(c)) for c in incident_coeffs],
    }
    debug_data["raw_coefficients"][source_port] = {
        "forward_mag": [float(abs(src_ob.alpha[0, i, 0])) for i in range(len(freqs))],
        "backward_mag": [float(abs(src_ob.alpha[0, i, 1])) for i in range(len(freqs))],
    }

    s_params = {}

    for i, port_i in enumerate(port_names):
        for j, port_j in enumerate(port_names):
            if port_j != source_port:
                continue

            s_name = f"S{i+1}{j+1}"

            port_kp = _port_kpoint(ports[port_i])
            ob = sim.get_eigenmode_coefficients(
                monitors[port_i], [1], eig_parity=eig_parity,
                kpoint_func=lambda f, n, kp=port_kp: kp,
            )

            # Collect debug info (skip source port — already collected)
            if port_i != source_port:
                debug_data["eigenmode_info"][port_i] = _collect_eigenmode_debug(
                    port_i, ob, freqs)
                nf = len(freqs)
                debug_data["raw_coefficients"][port_i] = {
                    "forward_mag": [
                        float(abs(ob.alpha[0, k, 0])) for k in range(nf)
                    ],
                    "backward_mag": [
                        float(abs(ob.alpha[0, k, 1])) for k in range(nf)
                    ],
                }

            # Outgoing direction: reflected at source port, transmitted at output ports
            port_dir = ports[port_i]["direction"]
            alpha = ob.alpha[0, :, _outgoing_idx(port_dir)]

            s_params[s_name] = alpha / incident_coeffs

    # Power conservation: sum |Sij|^2 per frequency
    power_conservation = np.zeros(len(freqs))
    for s_name, s_vals in s_params.items():
        power_conservation += np.abs(s_vals) ** 2
    debug_data["power_conservation"] = [float(p) for p in power_conservation]

    return s_params, debug_data


def _extract_s_params_fiber(config, sim, monitors, fiber_flux):
    """Extract fiber->waveguide S-parameters for a Gaussian-beam source.

    Normalization reference is the net flux through ``fiber_flux`` (placed
    just in front of the Gaussian beam): P_fiber(f). For each port i,
    S{i+1}0(f) = alpha_out(f) / sqrt(P_fiber(f)), where alpha_out is the
    outgoing mode coefficient (away from the GC, into the waveguide).
    """
    if fiber_flux is None:
        logger.warning("fiber_source set but no fiber_flux monitor; skipping S-params")
        return {}, {}

    port_names = [p["name"] for p in config["ports"]]
    ports = {p["name"]: p for p in config["ports"]}

    fdtd = config["fdtd"]
    fcen = fdtd["fcen"]
    nfreq = fdtd["num_freqs"]
    df = fdtd["df"]
    freqs = np.linspace(fcen - df / 2, fcen + df / 2, nfreq)

    # Net power crossing the fiber flux plane (beam propagates toward -Z,
    # so raw flux is negative; take |·| to get launched power into the chip).
    p_fiber = np.abs(np.array(mp.get_fluxes(fiber_flux)))

    # 2D XZ: use ODD_Y parity to pick the TE-like waveguide mode.
    is_3d = config.get("is_3d", True)
    plane = config.get("plane", "xy")
    if is_3d:
        eig_parity = mp.NO_PARITY
    elif plane == "xz":
        eig_parity = mp.ODD_Y
    else:
        eig_parity = mp.EVEN_Y + mp.ODD_Z

    debug_data = {
        "eigenmode_info": {},
        "raw_coefficients": {},
        "incident_coefficients": {
            "port": "fiber",
            "power_flux": [float(p) for p in p_fiber],
        },
    }

    s_params = {}
    sqrt_p_fiber = np.sqrt(np.where(p_fiber > 0, p_fiber, np.nan))

    for i, port_i in enumerate(port_names):
        port = ports[port_i]
        port_kp = _port_kpoint(port)
        ob = sim.get_eigenmode_coefficients(
            monitors[port_i], [1], eig_parity=eig_parity,
            kpoint_func=lambda f, n, kp=port_kp: kp,
        )

        debug_data["eigenmode_info"][port_i] = _collect_eigenmode_debug_basic(
            ob, freqs
        )
        nf = len(freqs)
        debug_data["raw_coefficients"][port_i] = {
            "forward_mag": [float(abs(ob.alpha[0, k, 0])) for k in range(nf)],
            "backward_mag": [float(abs(ob.alpha[0, k, 1])) for k in range(nf)],
        }

        outgoing_idx = 1 if port["direction"] == "+" else 0
        alpha_out = ob.alpha[0, :, outgoing_idx]
        s_params[f"S{i+1}0"] = alpha_out / sqrt_p_fiber

    # Coupling efficiency per frequency: sum |S|² (fraction of launched power
    # reaching each port mode).
    coupling = np.zeros(len(freqs))
    for s_vals in s_params.values():
        coupling += np.abs(np.nan_to_num(s_vals)) ** 2
    debug_data["power_conservation"] = [float(c) for c in coupling]

    return s_params, debug_data


def _collect_eigenmode_debug_basic(ob, freqs):
    """Collect kdom / n_eff / group velocity from an eigenmode coeffs object."""
    info = {"band": 1}
    try:
        kdom = ob.kdom
        kdom_list = [
            [float(kdom[i].x), float(kdom[i].y), float(kdom[i].z)]
            for i in range(len(kdom))
        ]
        info["kdom"] = kdom_list
        info["n_eff"] = [
            float(np.linalg.norm(kdom_list[i])) / float(freqs[i])
            if float(freqs[i]) > 0 else 0.0
            for i in range(min(len(kdom_list), len(freqs)))
        ]
    except Exception:
        info["kdom"] = []
        info["n_eff"] = []
    try:
        cg = ob.cg
        info["group_velocity"] = [float(cg[i]) for i in range(len(cg))]
    except Exception:
        info["group_velocity"] = []
    return info


def save_results(config, s_params, output_path="s_parameters.csv"):
    """Save S-parameters to CSV file.  Only rank 0 writes."""
    if not mp.am_master():
        return
    fdtd = config["fdtd"]
    fcen = fdtd["fcen"]
    df = fdtd["df"]
    nfreq = fdtd["num_freqs"]

    freqs = np.linspace(fcen - df / 2, fcen + df / 2, nfreq)
    wavelengths = 1.0 / freqs

    fieldnames = ["wavelength"]
    for name in sorted(s_params.keys()):
        fieldnames.extend([f"{name}_mag", f"{name}_phase"])

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, wl in enumerate(wavelengths):
            row = {"wavelength": f"{wl:.6f}"}
            for name in sorted(s_params.keys()):
                val = s_params[name][i]
                row[f"{name}_mag"] = f"{abs(val):.6f}"
                row[f"{name}_phase"] = f"{cmath.phase(val) * 180 / cmath.pi:.4f}"
            writer.writerow(row)

    logger.info("S-parameters saved to %s", output_path)


def save_debug_log(config, s_params, debug_data, wall_seconds=0.0,
                   output_path="meep_debug.json"):
    """Save eigenmode diagnostics as JSON for post-run analysis.  Only rank 0 writes."""
    if not mp.am_master():
        return
    fdtd = config["fdtd"]
    domain = config["domain"]
    resolution = config["resolution"]["pixels_per_um"]
    stopping = config["stopping"]

    meep_time = 0.0
    timesteps = 0
    try:
        # These are set by caller if available
        meep_time = debug_data.get("_meep_time", 0.0)
        timesteps = debug_data.get("_timesteps", 0)
    except Exception:
        pass

    log = {
        "metadata": {
            "resolution": resolution,
            "cell_size": debug_data.get("_cell_size", []),
            "meep_time": meep_time,
            "timesteps": timesteps,
            "wall_seconds": wall_seconds,
            "stopping_mode": stopping["mode"],
        },
        "eigenmode_info": debug_data.get("eigenmode_info", {}),
        "incident_coefficients": debug_data.get("incident_coefficients", {}),
        "raw_coefficients": debug_data.get("raw_coefficients", {}),
        "power_conservation": debug_data.get("power_conservation", []),
    }

    with open(output_path, "w") as f:
        json.dump(log, f, indent=2)
    logger.info("Debug log saved to %s", output_path)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def save_geometry_diagnostics(sim, config, cell_center):
    """Save geometry cross-section plots showing epsilon, sources, monitors, PML."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping geometry diagnostics")
        return
    if not mp.am_master():
        # plot2D is collective — all ranks call it, only master saves
        pass

    is_3d = config.get("is_3d", True)
    plane = config.get("plane", "xy")
    is_xz = plane == "xz"

    if is_xz:
        # XZ 2D: cell is invariant in Y. Plot the XZ cross-section.
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            sim.plot2D(ax=ax)
            y_cut = config.get("y_cut") or 0.0
            ax.set_title(f"XZ cross-section at y={y_cut:.3f} um")
            ax.set_xlabel("x (um)")
            ax.set_ylabel("z (um)")
            fig.tight_layout()
            if mp.am_master():
                fig.savefig("meep_geometry_xz.png", dpi=150)
                logger.info("Saved meep_geometry_xz.png")
            plt.close(fig)
        except Exception as e:
            logger.warning("XZ geometry plot failed: %s", e)
        return

    if is_3d:
        z_min = min(l["zmin"] for l in config["layer_stack"])
        z_max = max(l["zmax"] for l in config["layer_stack"])
        z_core = (z_min + z_max) / 2
    else:
        z_core = 0

    # XY cross-section at z=core center
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        if is_3d:
            xy_plane = mp.Volume(
                center=mp.Vector3(cell_center.x, cell_center.y, z_core),
                size=mp.Vector3(sim.cell_size.x, sim.cell_size.y, 0),
            )
            sim.plot2D(ax=ax, output_plane=xy_plane)
        else:
            sim.plot2D(ax=ax)
        ax.set_title(f"XY cross-section at z={z_core:.3f} um (core center)")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        fig.tight_layout()
        if mp.am_master():
            fig.savefig("meep_geometry_xy.png", dpi=150)
            logger.info("Saved meep_geometry_xy.png")
        plt.close(fig)
    except Exception as e:
        logger.warning("XY geometry plot failed: %s", e)

    # For 3D sims: XZ and YZ cross-sections
    if sim.cell_size.z > 0:
        # XZ at y=center
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            xz_plane = mp.Volume(
                center=mp.Vector3(cell_center.x, cell_center.y, cell_center.z),
                size=mp.Vector3(sim.cell_size.x, 0, sim.cell_size.z),
            )
            sim.plot2D(ax=ax, output_plane=xz_plane)
            ax.set_title(f"XZ cross-section at y={cell_center.y:.3f} um")
            ax.set_xlabel("x (um)")
            ax.set_ylabel("z (um)")
            fig.tight_layout()
            if mp.am_master():
                fig.savefig("meep_geometry_xz.png", dpi=150)
                logger.info("Saved meep_geometry_xz.png")
            plt.close(fig)
        except Exception as e:
            logger.warning("XZ geometry plot failed: %s", e)

        # YZ at x=center
        try:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            yz_plane = mp.Volume(
                center=mp.Vector3(cell_center.x, cell_center.y, cell_center.z),
                size=mp.Vector3(0, sim.cell_size.y, sim.cell_size.z),
            )
            sim.plot2D(ax=ax, output_plane=yz_plane)
            ax.set_title(f"YZ cross-section at x={cell_center.x:.3f} um")
            ax.set_xlabel("y (um)")
            ax.set_ylabel("z (um)")
            fig.tight_layout()
            if mp.am_master():
                fig.savefig("meep_geometry_yz.png", dpi=150)
                logger.info("Saved meep_geometry_yz.png")
            plt.close(fig)
        except Exception as e:
            logger.warning("YZ geometry plot failed: %s", e)


def save_field_snapshot(sim, config, cell_center):
    """Save post-run field snapshot (Ey overlaid on epsilon)."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping field snapshot")
        return

    is_3d = config.get("is_3d", True)
    plane = config.get("plane", "xy")
    is_xz = plane == "xz"

    if is_xz:
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            sim.plot2D(ax=ax, fields=mp.Ey)
            y_cut = config.get("y_cut") or 0.0
            ax.set_title(f"Ey field at y={y_cut:.3f} um (post-run)")
            ax.set_xlabel("x (um)")
            ax.set_ylabel("z (um)")
            fig.tight_layout()
            if mp.am_master():
                fig.savefig("meep_fields_xz.png", dpi=150)
                logger.info("Saved meep_fields_xz.png")
            plt.close(fig)
        except Exception as e:
            logger.warning("Field snapshot (XZ) failed: %s", e)
        return

    if is_3d:
        z_min = min(l["zmin"] for l in config["layer_stack"])
        z_max = max(l["zmax"] for l in config["layer_stack"])
        z_core = (z_min + z_max) / 2
    else:
        z_core = 0

    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        if is_3d:
            xy_plane = mp.Volume(
                center=mp.Vector3(cell_center.x, cell_center.y, z_core),
                size=mp.Vector3(sim.cell_size.x, sim.cell_size.y, 0),
            )
            sim.plot2D(ax=ax, output_plane=xy_plane, fields=mp.Ey)
        else:
            sim.plot2D(ax=ax, fields=mp.Ey)
        ax.set_title(f"Ey field at z={z_core:.3f} um (post-run)")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        fig.tight_layout()
        if mp.am_master():
            fig.savefig("meep_fields_xy.png", dpi=150)
            logger.info("Saved meep_fields_xy.png")
        plt.close(fig)
    except Exception as e:
        logger.warning("Field snapshot failed: %s", e)


def save_animation_field(sim, xy_plane, frame_counter):
    """Save raw 2D field data during time-stepping (no plotting).

    Saves a compressed .npz with the Ey field array and timestamp.
    Rendering with a globally fixed colorbar happens after sim.run().

    Returns:
        Incremented frame counter.
    """
    field_data = np.real(sim.get_array(vol=xy_plane, component=mp.Ey))
    t = sim.meep_time()
    if mp.am_master():
        os.makedirs("frames", exist_ok=True)
        np.savez_compressed(
            f"frames/meep_field_{frame_counter:04d}.npz",
            field=field_data,
            time=t,
        )
    return frame_counter + 1


def render_animation_frames(eps_data, extent, axes=("x", "y")):
    """Render saved field .npz files into PNGs with fixed global colorbar.

    Two-pass: first finds the global field maximum across all frames,
    then renders every frame with the same vmin/vmax so field decay is
    clearly visible.

    ``axes`` labels the horizontal/vertical axes on the plot (``("x", "y")``
    for XY slices, ``("x", "z")`` for XZ 2D slices).

    Call only on master rank after sim.run().
    """
    import glob

    if not HAS_MATPLOTLIB:
        logger.warning(
            "matplotlib not available, .npz field files kept but not rendered"
        )
        return

    npz_files = sorted(glob.glob("frames/meep_field_*.npz"))
    if not npz_files:
        logger.warning("No field data files found to render")
        return

    # Pass 1 — global max
    global_max = 0.0
    for path in npz_files:
        d = np.load(path)
        global_max = max(global_max, float(np.max(np.abs(d["field"]))))
    if global_max == 0:
        global_max = 1.0

    logger.info(
        "Rendering %d frames (Ey global max = %.4g) ...",
        len(npz_files), global_max,
    )

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Pass 2 — render each frame
    for i, path in enumerate(npz_files):
        d = np.load(path)
        field = d["field"]
        t = float(d["time"])

        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        # Epsilon background (grayscale)
        ax.imshow(
            eps_data.T, origin="lower", extent=extent,
            cmap="binary", interpolation="none",
        )
        # Field overlay with fixed colorbar
        im = ax.imshow(
            field.T, origin="lower", extent=extent,
            cmap="RdBu", interpolation="spline36", alpha=0.8,
            vmin=-global_max, vmax=global_max,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.06)
        fig.colorbar(im, cax=cax, label="Ey")
        ax.set_title(f"Ey  t={t:.2f}")
        ax.set_xlabel(f"{axes[0]} (um)")
        ax.set_ylabel(f"{axes[1]} (um)")
        fig.tight_layout()
        fig.savefig(f"frames/meep_frame_{i:04d}.png", dpi=150)
        plt.close(fig)

    # Clean up .npz intermediates
    for path in npz_files:
        os.remove(path)

    logger.info("Rendered %d frames with fixed colorbar", len(npz_files))


def compile_animation_mp4(fps=15):
    """Stitch meep_frame_*.png into meep_animation.mp4 via ffmpeg.

    Falls back gracefully if ffmpeg is not available — frame PNGs
    are still kept as individual files.
    """
    import glob
    import subprocess

    frames = sorted(glob.glob("frames/meep_frame_*.png"))
    if not frames:
        logger.warning("No animation frames found to compile")
        return

    logger.info("Compiling %d frames into meep_animation.mp4 ...", len(frames))
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", "frames/meep_frame_%04d.png",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "meep_animation.mp4",
            ],
            check=True,
            capture_output=True,
        )
        logger.info("Saved meep_animation.mp4")
    except FileNotFoundError:
        logger.warning("ffmpeg not found — frame PNGs saved but MP4 not created")
    except subprocess.CalledProcessError as e:
        logger.warning("ffmpeg failed: %s", e.stderr.decode()[:500])
        logger.info("Frame PNGs are still available in frames/")


def save_epsilon_raw(sim, config, cell_center):
    """Save raw epsilon array as .npy for XY slice at core center."""
    is_3d = config.get("is_3d", True)
    if is_3d:
        z_min = min(l["zmin"] for l in config["layer_stack"])
        z_max = max(l["zmax"] for l in config["layer_stack"])
        z_core = (z_min + z_max) / 2
    else:
        z_core = 0

    try:
        if is_3d:
            xy_plane = mp.Volume(
                center=mp.Vector3(cell_center.x, cell_center.y, z_core),
                size=mp.Vector3(sim.cell_size.x, sim.cell_size.y, 0),
            )
        else:
            xy_plane = mp.Volume(
                center=cell_center,
                size=mp.Vector3(sim.cell_size.x, sim.cell_size.y, 0),
            )
        eps_data = sim.get_array(vol=xy_plane, component=mp.Dielectric)
        if mp.am_master():
            np.save("meep_epsilon_xy.npy", eps_data)
            logger.info("Saved meep_epsilon_xy.npy (shape=%s)", eps_data.shape)
    except Exception as e:
        logger.warning("Epsilon raw save failed: %s", e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run MEEP simulation."""
    config_path = "%%CONFIG_FILENAME%%"
    config = load_config(config_path)

    logger.info("Building materials...")
    materials = build_materials(config)

    logger.info("Building background dielectric slabs...")
    background_slabs = build_background_slabs(config, materials)
    logger.info("Created %d background slabs", len(background_slabs))

    logger.info("Building geometry from GDS + layer stack...")
    geometry, component = build_geometry(config, materials)
    logger.info(
        "Created %d prisms from %d layers",
        len(geometry), len(config['layer_stack']),
    )

    # Background slabs first, then patterned prisms (later objects take precedence)
    geometry = background_slabs + geometry

    logger.info("Building sources...")
    sources = build_sources(config)

    if not sources:
        logger.error("No source port found in config")
        sys.exit(1)

    resolution = config["resolution"]["pixels_per_um"]
    fdtd = config["fdtd"]
    is_3d = config.get("is_3d", True)
    plane = config.get("plane", "xy")
    is_xz = plane == "xz"

    # Compute simulation cell from component bounds + layer z-range
    # Use original component bbox if available (port extension changes GDS bbox)
    component_bbox = config["component_bbox"]
    if component_bbox is not None:
        bbox_left, bbox_bottom, bbox_right, bbox_top = component_bbox
    else:
        bbox = component.dbbox()
        bbox_left, bbox_right = bbox.left, bbox.right
        bbox_bottom, bbox_top = bbox.bottom, bbox.top

    domain = config["domain"]
    dpml = domain["dpml"]
    margin_xy = domain["margin_xy"]

    # XY: margin_xy is gap between geometry bbox and PML
    cell_x = (bbox_right - bbox_left) + 2 * (margin_xy + dpml)
    cell_y = (bbox_top - bbox_bottom) + 2 * (margin_xy + dpml)

    # Z range for 3D and XZ 2D. Include both layers and dielectrics
    # so PDKs without explicit box/clad layers still have headroom.
    z_vals = [l["zmin"] for l in config["layer_stack"]] + [
        l["zmax"] for l in config["layer_stack"]
    ]
    for d in config.get("dielectrics", []):
        z_vals.extend((d["zmin"], d["zmax"]))
    if z_vals:
        z_min = min(z_vals)
        z_max = max(z_vals)
    else:
        z_min, z_max = 0.0, 0.0

    margin_z_above = domain.get("margin_z_above", 0.0)
    margin_z_below = domain.get("margin_z_below", 0.0)

    if is_3d:
        # 3D: z-margins are already baked via z_crop; just add dpml.
        cell_z = (z_max - z_min) + 2 * dpml
        cell_center = mp.Vector3(
            (bbox_right + bbox_left) / 2,
            (bbox_top + bbox_bottom) / 2,
            (z_max + z_min) / 2,
        )
    elif is_xz:
        # XZ 2D: cell_y collapsed; cell_z spans the full stack with
        # z-margins and PML added (there is no z_crop in 2D).
        cell_y = 0.0
        z_lo = z_min - margin_z_below
        z_hi = z_max + margin_z_above
        cell_z = (z_hi - z_lo) + 2 * dpml
        cell_center = mp.Vector3(
            (bbox_right + bbox_left) / 2,
            0.0,
            (z_hi + z_lo) / 2,
        )
    else:
        # XY 2D: collapse z-dimension entirely.
        cell_z = 0
        cell_center = mp.Vector3(
            (bbox_right + bbox_left) / 2,
            (bbox_top + bbox_bottom) / 2,
            0,
        )

    logger.info(
        "Cell size: %.2f x %.2f x %.2f um (%s)",
        cell_x, cell_y, cell_z, "3D" if is_3d else "2D",
    )
    logger.info("PML: %.2f um, margin_xy: %.2f", dpml, margin_xy)
    logger.info("Resolution: %s pixels/um", resolution)

    accuracy = config["accuracy"]
    diagnostics = config["diagnostics"]
    preview_only = diagnostics["preview_only"]

    # In preview mode, skip expensive subpixel averaging
    eps_avg = False if preview_only else accuracy["eps_averaging"]

    # Symmetries are NOT used for S-parameter extraction runs.
    # MEEP's get_eigenmode_coefficients with add_mode_monitor produces
    # incorrect normalization when the source monitor straddles a mirror
    # symmetry plane (coefficients underestimated by ~2x).  This is
    # consistent with gplugins which also never uses mp.Mirror for
    # S-parameter extraction.  Symmetries are only applied in preview-only
    # mode (geometry validation, no FDTD/S-params).
    cfg_symmetries = config["symmetries"]
    if cfg_symmetries and not preview_only:
        logger.info("Symmetries present in config but IGNORED for S-parameter "
                     "extraction (causes incorrect eigenmode normalization). "
                     "Symmetries are only used in preview-only mode.")
    use_symmetries = build_symmetries(config) if preview_only else []

    if is_xz:
        pml_layers = [
            mp.PML(thickness=dpml, direction=mp.X),
            mp.PML(thickness=dpml, direction=mp.Z),
        ]
    elif not is_3d:
        pml_layers = [
            mp.PML(thickness=dpml, direction=mp.X),
            mp.PML(thickness=dpml, direction=mp.Y),
        ]
    else:
        pml_layers = [mp.PML(dpml)]

    sim_kwargs = dict(
        cell_size=mp.Vector3(cell_x, cell_y, cell_z),
        geometry_center=cell_center,
        geometry=geometry,
        sources=sources,
        resolution=resolution,
        boundary_layers=pml_layers,
        symmetries=use_symmetries,
        split_chunks_evenly=config["split_chunks_evenly"],
        eps_averaging=eps_avg,
    )
    spx_maxeval = accuracy["subpixel_maxeval"]
    if spx_maxeval > 0:
        sim_kwargs["subpixel_maxeval"] = spx_maxeval
    spx_tol = accuracy["subpixel_tol"]
    if spx_tol != 1e-4:
        sim_kwargs["subpixel_tol"] = spx_tol
    sim = mp.Simulation(**sim_kwargs)

    # --- Diagnostics & preview mode ---
    diag_geometry = diagnostics["save_geometry"]
    diag_fields = diagnostics["save_fields"]
    diag_epsilon = diagnostics["save_epsilon_raw"]

    if diag_geometry or diag_epsilon or preview_only:
        logger.info("Initializing simulation for diagnostics...")
        sim.init_sim()
        if diag_geometry or preview_only:
            save_geometry_diagnostics(sim, config, cell_center)
        if diag_epsilon:
            save_epsilon_raw(sim, config, cell_center)

    if preview_only:
        logger.info("MEEP_PREVIEW_ONLY=1 — skipping simulation run.")
        save_debug_log(config, {}, {"_meep_time": 0, "_timesteps": 0,
                                    "_cell_size": [cell_x, cell_y, cell_z]})
        logger.info("Preview complete.")
        sys.exit(0)

    logger.info("Building monitors...")
    monitors, fiber_flux = build_monitors(config, sim)

    stopping = config["stopping"]
    run_after = stopping["run_after_sources"]

    # Build verbose step functions
    step_funcs = []
    verbose_interval = config["verbose_interval"]
    if verbose_interval > 0:
        _wall_start = time.time()
        _cell_vol = [None]
        def _verbose_print(sim_obj):
            elapsed = time.time() - _wall_start
            if _cell_vol[0] is None:
                _cell_vol[0] = mp.Volume(
                    center=sim_obj.geometry_center,
                    size=sim_obj.cell_size,
                )
            energy = sim_obj.field_energy_in_box(box=_cell_vol[0])
            logger.info(
                "t=%.2f | wall=%.1fs | energy=%.6e",
                sim_obj.meep_time(), elapsed, energy,
            )
        step_funcs.append(mp.at_every(verbose_interval, _verbose_print))

    # Animation field capture step function (raw data, no plotting yet)
    diag_animation = diagnostics["save_animation"]
    animation_interval = diagnostics["animation_interval"]
    _frame_counter = [0]  # mutable container for closure
    _anim_plane = None
    _anim_axes = ("x", "y")

    if diag_animation:
        if is_xz:
            # XZ 2D: slice is the whole simulation plane (cell_y is already 0).
            _anim_plane = mp.Volume(
                center=cell_center,
                size=mp.Vector3(sim.cell_size.x, 0, sim.cell_size.z),
            )
            _anim_axes = ("x", "z")
        else:
            if is_3d:
                z_min_anim = min(l["zmin"] for l in config["layer_stack"])
                z_max_anim = max(l["zmax"] for l in config["layer_stack"])
                z_core_anim = (z_min_anim + z_max_anim) / 2
            else:
                z_core_anim = 0
            _anim_plane = mp.Volume(
                center=mp.Vector3(cell_center.x, cell_center.y, z_core_anim),
                size=mp.Vector3(sim.cell_size.x, sim.cell_size.y, 0),
            )
            _anim_axes = ("x", "y")

        def _capture_frame(sim_obj):
            _frame_counter[0] = save_animation_field(
                sim_obj, _anim_plane, _frame_counter[0]
            )

        step_funcs.append(mp.at_every(animation_interval, _capture_frame))
        logger.info(
            "Animation: saving field data every %s time units",
            animation_interval,
        )

    stop_mode = stopping["mode"]
    wall_time_max = stopping.get("wall_time_max", 0)
    if wall_time_max > 0:
        logger.info("Wall-clock safety net: %.0f seconds", wall_time_max)

    wall_start = time.time()

    if stop_mode == "dft_decay":
        decay_by = stopping["decay_by"]
        min_time = stopping["dft_min_run_time"]
        logger.info(
            "Running simulation (dft_decay mode: tol=%s, "
            "min=%.1f, max=%.1f)...",
            decay_by, min_time, run_after,
        )
        dft_fn = mp.stop_when_dft_decayed(
            tol=decay_by,
            minimum_run_time=min_time,
            maximum_run_time=run_after,
        )
        cond = dft_fn
        if wall_time_max > 0:
            cond = _make_or_condition(dft_fn, _make_wall_time_cap(wall_time_max))
        sim.run(*step_funcs, until_after_sources=cond)
    elif stop_mode == "energy_decay":
        dt = stopping["decay_dt"]
        decay_by = stopping["decay_by"]
        logger.info(
            "Running simulation (energy_decay mode: dt=%s, "
            "decay_by=%s, cap=%.1f)...",
            dt, decay_by, run_after,
        )
        energy_fn = mp.stop_when_energy_decayed(dt=dt, decay_by=decay_by)
        parts = [energy_fn, _make_time_cap(run_after)]
        if wall_time_max > 0:
            parts.append(_make_wall_time_cap(wall_time_max))
        sim.run(*step_funcs, until_after_sources=_make_or_condition(*parts))
    elif stop_mode == "field_decay":
        dt = stopping["decay_dt"]
        comp_name = stopping["decay_component"]
        comp = _COMPONENT_MAP.get(comp_name, mp.Ey)
        decay_by = stopping["decay_by"]
        monitor_pt = resolve_decay_monitor_point(config)
        logger.info("Running simulation (field_decay mode: component=%s, dt=%s, "
                     "decay_by=%s, cap=%.1f)...", comp_name, dt, decay_by, run_after)

        decay_fn = mp.stop_when_fields_decayed(dt, comp, monitor_pt, decay_by)
        parts = [decay_fn, _make_time_cap(run_after)]
        if wall_time_max > 0:
            parts.append(_make_wall_time_cap(wall_time_max))
        sim.run(*step_funcs, until_after_sources=_make_or_condition(*parts))
    else:
        logger.info("Running simulation (until_after_sources=%.1f)...", run_after)
        if wall_time_max > 0:
            cond = _make_or_condition(
                _make_time_cap(run_after),
                _make_wall_time_cap(wall_time_max),
            )
            sim.run(*step_funcs, until_after_sources=cond)
        else:
            sim.run(*step_funcs, until_after_sources=run_after)

    wall_seconds = time.time() - wall_start
    if wall_time_max > 0 and wall_seconds >= wall_time_max * 0.95:
        logger.warning(
            "Wall-clock limit likely triggered (%.1fs elapsed, limit %.0fs). "
            "Results may be incomplete — consider increasing wall_time_max or "
            "using a coarser resolution.",
            wall_seconds, wall_time_max,
        )

    if diag_fields:
        save_field_snapshot(sim, config, cell_center)

    if diag_animation and _frame_counter[0] > 0 and _anim_plane is not None:
        # get_array is collective — all ranks must call
        eps_data = sim.get_array(vol=_anim_plane, component=mp.Dielectric)
        if mp.am_master():
            _ctr = _anim_plane.center
            _sz = _anim_plane.size
            if _anim_axes == ("x", "z"):
                _extent = [
                    _ctr.x - _sz.x / 2, _ctr.x + _sz.x / 2,
                    _ctr.z - _sz.z / 2, _ctr.z + _sz.z / 2,
                ]
            else:
                _extent = [
                    _ctr.x - _sz.x / 2, _ctr.x + _sz.x / 2,
                    _ctr.y - _sz.y / 2, _ctr.y + _sz.y / 2,
                ]
            render_animation_frames(eps_data, _extent, axes=_anim_axes)
            compile_animation_mp4()

    logger.info("Extracting S-parameters...")
    s_params, debug_data = extract_s_params(config, sim, monitors, fiber_flux)

    # Attach simulation metadata to debug_data
    debug_data["_meep_time"] = sim.meep_time()
    debug_data["_timesteps"] = sim.timestep()
    debug_data["_cell_size"] = [cell_x, cell_y, cell_z]

    save_results(config, s_params)
    save_debug_log(config, s_params, debug_data, wall_seconds=wall_seconds)
    logger.info("Done!")


if __name__ == "__main__":
    main()
'''


def generate_meep_script(config_filename: str = "sim_config.json") -> str:
    """Generate the Python script that runs on the cloud MEEP instance.

    The script is self-contained and reads a JSON config + GDS file to:
    1. Load GDS layout via gdsfactory
    2. Extract polygons per layer, handling holes via triangulation
    3. Create mp.Prism objects with correct materials and sidewall angles
    4. Create EigenModeSource at the source port
    5. Create mode monitors at all ports
    6. Build mp.Simulation with PML boundaries
    7. Run and extract S-parameters
    8. Save results as CSV

    Args:
        config_filename: Name of the JSON config file (default: sim_config.json)

    Returns:
        Python script as a string
    """
    return _MEEP_RUNNER_TEMPLATE.replace("%%CONFIG_FILENAME%%", config_filename)

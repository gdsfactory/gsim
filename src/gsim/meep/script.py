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
import math
import sys
import time

import meep as mp
import numpy as np


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
        k = props.get("extinction_coeff", 0.0)
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


def build_background_slabs(config, materials):
    """Build background mp.Block slabs from dielectric entries.

    These infinite-XY slabs fill the simulation cell at each z-range with
    the correct cladding/substrate material.  They must come FIRST in the
    geometry list so that patterned prisms (added later) take precedence.
    """
    slabs = []
    for diel in sorted(config.get("dielectrics", []), key=lambda d: d["zmin"]):
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
    for sym in config.get("symmetries", []):
        symmetries.append(
            mp.Mirror(direction_map[sym["direction"]], phase=sym.get("phase", 1))
        )
    return symmetries


# Map string component names to MEEP field components
_COMPONENT_MAP = {
    "Ex": mp.Ex, "Ey": mp.Ey, "Ez": mp.Ez,
    "Hx": mp.Hx, "Hy": mp.Hy, "Hz": mp.Hz,
}


def resolve_decay_monitor_point(config):
    """Return center mp.Vector3 for decay monitoring.

    Uses the port named by stopping.decay_monitor_port if set,
    otherwise picks the first non-source port.  Falls back to
    the first port if all are sources.
    """
    stopping = config.get("fdtd", {}).get("stopping", {})
    target_name = stopping.get("decay_monitor_port")
    ports = config.get("ports", [])

    if target_name:
        for p in ports:
            if p["name"] == target_name:
                return mp.Vector3(*p["center"])

    # Fallback: first non-source port, or first port
    for p in ports:
        if not p.get("is_source", False):
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
    """
    gds_filename = config.get("gds_filename", "layout.gds")
    component = load_gds_component(gds_filename)

    accuracy = config.get("accuracy", {})
    simplify_tol = accuracy.get("simplify_tol", 0.0)

    geometry = []
    total_vertices = 0

    for layer_entry in config["layer_stack"]:
        material_name = layer_entry["material"]
        mat = materials.get(material_name, mp.Medium())
        zmin = layer_entry["zmin"]
        zmax = layer_entry["zmax"]
        height = zmax - zmin
        gds_layer = layer_entry["gds_layer"]
        sidewall_angle_deg = layer_entry.get("sidewall_angle", 0.0)
        sw_rad = math.radians(sidewall_angle_deg) if sidewall_angle_deg else 0

        if height <= 0:
            continue

        polygons = extract_layer_polygons(component, gds_layer, simplify_tol=simplify_tol)

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

    print(f"  Total vertices across all prisms: {total_vertices}")
    return geometry, component


# ---------------------------------------------------------------------------
# Sources and monitors
# ---------------------------------------------------------------------------

def get_port_z_span(config):
    """Get z-span for ports from layer stack."""
    zmin = min(l["zmin"] for l in config["layer_stack"])
    zmax = max(l["zmax"] for l in config["layer_stack"])
    return zmax - zmin


def build_sources(config):
    """Build MEEP source from config port data."""
    fdtd = config["fdtd"]
    fcen = fdtd["fcen"]
    df = fdtd["df"]
    z_span = get_port_z_span(config)
    port_margin = config.get("domain", {}).get("port_margin", 2.0)

    sources = []
    for port in config["ports"]:
        if not port["is_source"]:
            continue

        center = mp.Vector3(*port["center"])
        normal_axis = port["normal_axis"]
        direction = port["direction"]

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
            src=mp.GaussianSource(frequency=fcen, fwidth=df),
            center=center,
            size=mp.Vector3(*size),
            eig_band=1,
            direction=prop_axis,
            eig_kpoint=kpoint,
            eig_match_freq=True,
        )
        sources.append(eig_src)

    return sources


def build_monitors(config, sim):
    """Build mode monitors at all ports and return flux regions."""
    fdtd = config["fdtd"]
    fcen = fdtd["fcen"]
    df = fdtd["df"]
    nfreq = fdtd["num_freqs"]
    z_span = get_port_z_span(config)
    port_margin = config.get("domain", {}).get("port_margin", 2.0)

    monitors = {}
    for port in config["ports"]:
        center = mp.Vector3(*port["center"])
        normal_axis = port["normal_axis"]

        size = [0, 0, 0]
        transverse_axis = 1 - normal_axis
        size[transverse_axis] = port["width"] + 2 * port_margin
        size[2] = z_span

        flux = sim.add_mode_monitor(
            fcen, df, nfreq,
            mp.ModeRegion(center=center, size=mp.Vector3(*size)),
        )
        monitors[port["name"]] = flux

    return monitors


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


def extract_s_params(config, sim, monitors):
    """Extract S-parameters via mode decomposition.

    Uses MEEP eigenmode coefficients with explicit kpoint_func to anchor
    the forward/backward convention:
      alpha[band, freq, 0] = forward (+normal_axis) coefficient
      alpha[band, freq, 1] = backward (-normal_axis) coefficient

    Port "direction" field = direction of incoming mode along normal_axis:
      "+" -> incoming goes +normal, outgoing (reflected/transmitted) goes -normal
      "-" -> incoming goes -normal, outgoing (reflected/transmitted) goes +normal
    """
    port_names = [p["name"] for p in config["ports"]]
    ports = {p["name"]: p for p in config["ports"]}
    source_port = None
    for p in config["ports"]:
        if p["is_source"]:
            source_port = p["name"]
            break

    if not source_port:
        return {}

    def _incoming_idx(direction):
        """Alpha index for the incoming (incident) mode at a port."""
        return 0 if direction == "+" else 1

    def _outgoing_idx(direction):
        """Alpha index for the outgoing (reflected/transmitted) mode at a port."""
        return 1 if direction == "+" else 0

    # Get incident coefficient at source port for normalization
    src_dir = ports[source_port]["direction"]
    src_kp = _port_kpoint(ports[source_port])
    src_ob = sim.get_eigenmode_coefficients(
        monitors[source_port], [1], eig_parity=mp.NO_PARITY,
        kpoint_func=lambda f, n, kp=src_kp: kp,
    )
    incident_coeffs = src_ob.alpha[0, :, _incoming_idx(src_dir)]

    s_params = {}

    for i, port_i in enumerate(port_names):
        for j, port_j in enumerate(port_names):
            if port_j != source_port:
                continue

            s_name = f"S{i+1}{j+1}"

            port_kp = _port_kpoint(ports[port_i])
            ob = sim.get_eigenmode_coefficients(
                monitors[port_i], [1], eig_parity=mp.NO_PARITY,
                kpoint_func=lambda f, n, kp=port_kp: kp,
            )

            # Outgoing direction: reflected at source port, transmitted at output ports
            port_dir = ports[port_i]["direction"]
            alpha = ob.alpha[0, :, _outgoing_idx(port_dir)]

            s_params[s_name] = alpha / incident_coeffs

    return s_params


def save_results(config, s_params, output_path="s_parameters.csv"):
    """Save S-parameters to CSV file."""
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

    print(f"S-parameters saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run MEEP simulation."""
    config_path = "%%CONFIG_FILENAME%%"
    config = load_config(config_path)

    print("Building materials...")
    materials = build_materials(config)

    print("Building background dielectric slabs...")
    background_slabs = build_background_slabs(config, materials)
    print(f"  Created {len(background_slabs)} background slabs")

    print("Building geometry from GDS + layer stack...")
    geometry, component = build_geometry(config, materials)
    print(f"  Created {len(geometry)} prisms from {len(config['layer_stack'])} layers")

    # Background slabs first, then patterned prisms (later objects take precedence)
    geometry = background_slabs + geometry

    print("Building sources...")
    sources = build_sources(config)

    if not sources:
        print("ERROR: No source port found in config", file=sys.stderr)
        sys.exit(1)

    resolution = config["resolution"]["pixels_per_um"]
    fdtd = config["fdtd"]

    # Compute simulation cell from component bounds + layer z-range
    bbox = component.dbbox()
    z_min = min(l["zmin"] for l in config["layer_stack"])
    z_max = max(l["zmax"] for l in config["layer_stack"])

    domain = config.get("domain", {})
    dpml = domain.get("dpml", 1.0)
    margin_xy = domain.get("margin_xy", 0.5)

    # XY: margin_xy is gap between geometry bbox and PML
    cell_x = (bbox.right - bbox.left) + 2 * (margin_xy + dpml)
    cell_y = (bbox.top - bbox.bottom) + 2 * (margin_xy + dpml)
    # Z: margin_z_above/below is already baked into layer_stack via set_z_crop(),
    #    so only add dpml beyond the stack extent
    cell_z = (z_max - z_min) + 2 * dpml
    cell_center = mp.Vector3(
        (bbox.right + bbox.left) / 2,
        (bbox.top + bbox.bottom) / 2,
        (z_max + z_min) / 2,
    )

    print(f"Cell size: {cell_x:.2f} x {cell_y:.2f} x {cell_z:.2f} um")
    print(f"PML: {dpml:.2f} um, margin_xy: {margin_xy:.2f}")
    print(f"Resolution: {resolution} pixels/um")

    accuracy = config.get("accuracy", {})
    sim_kwargs = dict(
        cell_size=mp.Vector3(cell_x, cell_y, cell_z),
        geometry_center=cell_center,
        geometry=geometry,
        sources=sources,
        resolution=resolution,
        boundary_layers=[mp.PML(dpml)],
        symmetries=build_symmetries(config),
        split_chunks_evenly=config.get("split_chunks_evenly", False),
        eps_averaging=accuracy.get("eps_averaging", True),
    )
    spx_maxeval = accuracy.get("subpixel_maxeval", 0)
    if spx_maxeval > 0:
        sim_kwargs["subpixel_maxeval"] = spx_maxeval
    spx_tol = accuracy.get("subpixel_tol", 1e-4)
    if spx_tol != 1e-4:
        sim_kwargs["subpixel_tol"] = spx_tol
    sim = mp.Simulation(**sim_kwargs)

    print("Building monitors...")
    monitors = build_monitors(config, sim)

    stopping = fdtd.get("stopping", {})
    run_after = fdtd["run_after_sources"]

    # Build verbose step functions
    step_funcs = []
    verbose_interval = config.get("verbose_interval", 0)
    if verbose_interval > 0:
        _wall_start = time.time()
        def _verbose_print(sim_obj):
            elapsed = time.time() - _wall_start
            print(f"  t={sim_obj.meep_time():.2f} | wall={elapsed:.1f}s", flush=True)
        step_funcs.append(mp.at_every(verbose_interval, _verbose_print))

    if stopping.get("mode") == "decay":
        dt = stopping.get("decay_dt", 50.0)
        comp_name = stopping.get("decay_component", "Ey")
        comp = _COMPONENT_MAP.get(comp_name, mp.Ey)
        decay_by = stopping.get("decay_by", 1e-3)
        monitor_pt = resolve_decay_monitor_point(config)
        print(f"Running simulation (decay mode: component={comp_name}, "
              f"dt={dt}, decay_by={decay_by}, cap={run_after:.1f})...")

        # Build capped decay condition: stop when decayed OR max time reached
        decay_fn = mp.stop_when_fields_decayed(dt, comp, monitor_pt, decay_by)
        time_fn = mp.stop_when_fields_decayed(run_after, comp, monitor_pt, 1.0)

        def capped_condition(sim_obj):
            return decay_fn(sim_obj) or time_fn(sim_obj)

        sim.run(*step_funcs, until_after_sources=capped_condition)
    else:
        print(f"Running simulation (until_after_sources={run_after:.1f})...")
        sim.run(*step_funcs, until_after_sources=run_after)

    print("Extracting S-parameters...")
    s_params = extract_s_params(config, sim, monitors)

    save_results(config, s_params)
    print("Done!")


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

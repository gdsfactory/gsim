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
    return gf.import_gds(gds_filename)


def extract_layer_polygons(component, gds_layer):
    """Extract merged Shapely polygons for a GDS layer from a component.

    Uses gdsfactory / KLayout to get polygons for the given (layer, datatype)
    tuple, then converts to Shapely with hole support.

    Args:
        component: gdsfactory Component
        gds_layer: [layer_number, datatype] list

    Returns:
        list of Shapely Polygon objects
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    layer_tuple = tuple(gds_layer)
    raw = component.get_polygons(layers=(layer_tuple,), merge=True)

    # get_polygons returns dict[tuple, list[ndarray]]
    if isinstance(raw, dict):
        arrays = raw.get(layer_tuple, [])
    else:
        arrays = raw if raw else []

    shapely_polygons = []
    for arr in arrays:
        if len(arr) >= 3:
            coords = [(float(p[0]), float(p[1])) for p in arr]
            try:
                poly = Polygon(coords)
                if poly.is_valid and not poly.is_empty:
                    shapely_polygons.append(poly)
            except Exception:
                continue

    if not shapely_polygons:
        return []

    merged = unary_union(shapely_polygons)
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


def build_geometry(config, materials):
    """Build MEEP geometry from GDS file + layer stack config.

    For each layer in layer_stack:
      1. Extract polygons from the GDS for that layer's gds_layer
      2. Extrude to 3D as mp.Prism with correct z-range and material
      3. Handle polygon holes via Delaunay triangulation
    """
    gds_filename = config.get("gds_filename", "layout.gds")
    component = load_gds_component(gds_filename)

    geometry = []

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

        polygons = extract_layer_polygons(component, gds_layer)

        for polygon in polygons:
            if polygon.is_empty or not polygon.is_valid:
                continue

            if hasattr(polygon, "interiors") and polygon.interiors:
                # Polygon has holes — triangulate
                triangles = triangulate_polygon_with_holes(polygon)
                for tri_coords in triangles:
                    vertices = [mp.Vector3(p[0], p[1], zmin) for p in tri_coords]
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
                prism = mp.Prism(
                    vertices=vertices,
                    height=height,
                    material=mat,
                    sidewall_angle=sw_rad,
                )
                geometry.append(prism)

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

    sources = []
    for port in config["ports"]:
        if not port["is_source"]:
            continue

        center = mp.Vector3(*port["center"])
        normal_axis = port["normal_axis"]

        size = [0, 0, 0]
        transverse_axis = 1 - normal_axis
        size[transverse_axis] = port["width"]
        size[2] = z_span

        eig_src = mp.EigenModeSource(
            src=mp.GaussianSource(frequency=fcen, fwidth=df),
            center=center,
            size=mp.Vector3(*size),
            eig_band=1,
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

    monitors = {}
    for port in config["ports"]:
        center = mp.Vector3(*port["center"])
        normal_axis = port["normal_axis"]

        size = [0, 0, 0]
        transverse_axis = 1 - normal_axis
        size[transverse_axis] = port["width"]
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

def extract_s_params(config, sim, monitors):
    """Extract S-parameters via mode decomposition."""
    port_names = [p["name"] for p in config["ports"]]
    source_port = None
    for p in config["ports"]:
        if p["is_source"]:
            source_port = p["name"]
            break

    s_params = {}

    for i, port_i in enumerate(port_names):
        for j, port_j in enumerate(port_names):
            if port_j != source_port:
                continue

            s_name = f"S{i+1}{j+1}"
            monitor = monitors[port_i]

            ob = sim.get_eigenmode_coefficients(
                monitor, [1], eig_parity=mp.NO_PARITY,
            )

            alpha = ob.alpha[0, :, 0]
            s_params[s_name] = alpha

    # Normalize by source port forward coefficient
    if source_port:
        src_idx = port_names.index(source_port)
        norm_key = f"S{src_idx+1}{src_idx+1}"
        norm_coeffs = s_params.get(norm_key)
        if norm_coeffs is not None:
            for key in s_params:
                s_params[key] = s_params[key] / norm_coeffs

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

    print("Building geometry from GDS + layer stack...")
    geometry, component = build_geometry(config, materials)
    print(f"  Created {len(geometry)} prisms from {len(config['layer_stack'])} layers")

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

    padding = 1.0  # um PML padding
    cell_x = (bbox.right - bbox.left) + 2 * padding
    cell_y = (bbox.top - bbox.bottom) + 2 * padding
    cell_z = (z_max - z_min) + 2 * padding
    cell_center = mp.Vector3(
        (bbox.right + bbox.left) / 2,
        (bbox.top + bbox.bottom) / 2,
        (z_max + z_min) / 2,
    )

    print(f"Cell size: {cell_x:.2f} x {cell_y:.2f} x {cell_z:.2f} um")
    print(f"Resolution: {resolution} pixels/um")

    sim = mp.Simulation(
        cell_size=mp.Vector3(cell_x, cell_y, cell_z),
        geometry_center=cell_center,
        geometry=geometry,
        sources=sources,
        resolution=resolution,
        boundary_layers=[mp.PML(padding)],
    )

    print("Building monitors...")
    monitors = build_monitors(config, sim)

    run_time = fdtd["run_time_factor"] / fdtd["df"]
    print(f"Running simulation for {run_time:.1f} time units...")

    sim.run(until=run_time)

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

"""Mesh validation utilities for Palace simulations.

This module provides:
- Full mesh/config validation for a simulation object (`validate_mesh`)
- Lumped-port geometry checks mirroring Palace's internal checks
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from gsim.palace.models.results import ValidationResult

__all__ = ["PortGeometryError", "check_lumped_port_geometry", "validate_mesh"]


class PortGeometryError(ValueError):
    """Raised when a lumped port fails Palace's geometry sanity check."""


_DIR_MAP: dict[str, np.ndarray] = {
    "x": np.array([1.0, 0.0, 0.0]),
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}


def _parse_direction(direction: str) -> np.ndarray:
    """Parse a direction string (e.g. '+X', '-Y', 'Z') into a unit vector."""
    key = direction.strip().lower()
    if key not in _DIR_MAP:
        raise ValueError(f"Unknown port direction '{direction}'")
    v = _DIR_MAP[key].copy()
    return v / np.linalg.norm(v)


def _perp_dist(v: np.ndarray, normals: list[np.ndarray], origin: np.ndarray) -> float:
    """Return distance from ``v`` to subspace spanned by ``normals`` at ``origin``."""
    d = v - origin
    for n in normals:
        d = d - np.dot(d, n) * n
    return float(np.linalg.norm(d))


def _palace_obb(pts: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], bool]:
    """Compute the Palace-style oriented bounding box approximation from points."""
    n = len(pts)
    if n < 4:
        raise ValueError(f"OBB requires at least 4 vertices, got {n}")

    rel_tol = 1.0e-6

    lex_idx = sorted(range(n), key=lambda i: tuple(pts[i]))[0]
    p_000 = pts[lex_idx]

    dists = np.linalg.norm(pts - p_000, axis=1)
    p_111 = pts[int(np.argmax(dists))]

    dists2 = np.linalg.norm(pts - p_111, axis=1)
    p_000 = pts[int(np.argmax(dists2))]

    origin = p_000
    diag_vec = p_111 - p_000
    diag_len = float(np.linalg.norm(diag_vec))
    if diag_len == 0.0:
        raise ValueError("OBB degenerate: p_000 == p_111")
    n_1 = diag_vec / diag_len

    perp1 = np.array([_perp_dist(p, [n_1], origin) for p in pts])
    t_0 = pts[int(np.argmax(perp1))]

    t0_vec = t_0 - origin
    n_2_raw = t0_vec - np.dot(t0_vec, n_1) * n_1
    n2_len = float(np.linalg.norm(n_2_raw))
    if n2_len < rel_tol * diag_len:
        raise ValueError("OBB degenerate: t_0 collinear with diagonal")
    n_2 = n_2_raw / n2_len

    max_out = max(_perp_dist(p, [n_1, n_2], origin) for p in pts)
    planar = max_out < rel_tol * diag_len

    if planar:
        t_1 = t_0
    else:
        eps = rel_tol * max_out
        candidates = [
            p for p in pts if abs(_perp_dist(p, [n_1, n_2], origin) - max_out) < eps
        ]
        t_1 = min(candidates, key=lambda p: float(np.linalg.norm(p - origin)))

    t0_dist = float(np.linalg.norm(t_0 - origin))
    t1_dist = float(np.linalg.norm(t_1 - origin))
    t0_gt_t1 = t0_dist > t1_dist

    v_001 = t_1 if t0_gt_t1 else t_0
    v_011 = p_111 if planar else (t_0 if t0_gt_t1 else t_1)

    extremals = [p_000, v_001, v_011, p_111]

    e_0 = np.array([1.0, 0.0, 0.0])
    e_1 = np.array([0.0, 1.0, 0.0])
    dot_min = np.inf
    combos = list(itertools.combinations(range(4), 2))
    for (i0, j0), (i1, j1) in itertools.product(combos, repeat=2):
        if i0 == i1 and j0 == j1:
            continue
        a, b = extremals[i0], extremals[j0]
        c, d_ = extremals[i1], extremals[j1]
        if np.allclose(a, b) or np.allclose(c, d_):
            continue
        e_ij_0 = (b - a) / np.linalg.norm(b - a)
        e_ij_1 = (d_ - c) / np.linalg.norm(d_ - c)
        dot = abs(float(np.dot(e_ij_0, e_ij_1)))
        if dot < dot_min:
            dot_min = dot
            e_0, e_1 = e_ij_0.copy(), e_ij_1.copy()
            if dot_min < rel_tol:
                break

    center = 0.5 * (p_000 + p_111)

    axes: list[np.ndarray] = [e_0.copy(), e_1.copy()]
    if planar:
        axes.append(np.zeros(3))
    else:
        axes.append(np.cross(e_0, e_1))

    half_extents = [0.0, 0.0, 0.0]
    for v in extremals:
        v0 = v - center
        for i, ax in enumerate(axes):
            half_extents[i] = max(half_extents[i], abs(float(np.dot(v0, ax))))

    axes = [ax * half_extents[i] for i, ax in enumerate(axes)]

    return center, axes, planar


def _get_port_vertices(mesh_path: Path, phys_group_tag: int) -> np.ndarray:
    """Load boundary vertices for a surface physical group from a Gmsh mesh."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.open(str(mesh_path))
        result = gmsh.model.mesh.getNodesForPhysicalGroup(2, phys_group_tag)
        if len(result) == 3:
            _, coords, _ = result
        elif len(result) == 2:
            _, coords = result
        else:
            raise ValueError(
                "Unexpected return value from gmsh.model.mesh.getNodesForPhysicalGroup"
            )
        return np.array(coords).reshape(-1, 3)
    finally:
        gmsh.finalize()


def _check_surface_geometry(
    *,
    mesh_path: Path,
    phys_tag: int,
    direction_str: str,
    context: str,
    rel_tol: float,
) -> list[str]:
    """Validate one lumped-port surface against Palace's rectangularity check."""
    errors: list[str] = []

    try:
        dir_vec = _parse_direction(direction_str)
    except ValueError:
        errors.append(f"{context}: unknown direction '{direction_str}'.")
        return errors

    try:
        pts = _get_port_vertices(mesh_path, phys_tag)
    except Exception as exc:
        errors.append(
            f"{context}: could not read mesh vertices for physical group "
            f"{phys_tag}: {exc}"
        )
        return errors

    if len(pts) < 4:
        errors.append(
            f"{context}: boundary has only {len(pts)} vertices "
            f"(need >= 4 for a valid lumped port surface)."
        )
        return errors

    projections = pts @ dir_vec
    projected_length = float(projections.max() - projections.min())

    try:
        _center, axes, _planar = _palace_obb(pts)
    except ValueError as exc:
        errors.append(f"{context}: OBB computation failed: {exc}")
        return errors

    lengths = [2.0 * float(np.linalg.norm(ax)) for ax in axes]
    normals = [ax / np.linalg.norm(ax) if np.linalg.norm(ax) > 0 else ax for ax in axes]

    deviations = []
    for n in normals:
        nn = float(np.linalg.norm(n))
        dot = float(abs(np.dot(dir_vec, n))) if nn > 0 else 0.0
        dot = min(1.0, dot)
        deviations.append(float(np.degrees(np.arccos(dot))))

    l_component = int(np.argmin(deviations))
    bbox_length = lengths[l_component]

    if bbox_length == 0.0:
        errors.append(
            f"{context}: OBB has zero length along the most-aligned axis "
            f"(axis {l_component}). The surface may be degenerate or the "
            f"direction '{direction_str}' is perpendicular to all OBB axes."
        )
        return errors

    discrepancy = abs(bbox_length - projected_length)
    if discrepancy >= rel_tol * bbox_length:
        dev_deg = deviations[l_component]
        errors.append(
            f"{context}, direction '{direction_str}': Palace geometry check FAILED.\n"
            f"  Bounding-box length  = {bbox_length:.6e} m "
            f"(OBB axis {l_component}, deviation {dev_deg:.3f}°)\n"
            f"  Projected length     = {projected_length:.6e} m\n"
            f"  Discrepancy          = {discrepancy:.3e} m "
            f"(tolerance = {rel_tol * bbox_length:.3e} m)\n"
            f"\n"
            f"  This surface is not rectangular/aligned for a Palace lumped port.\n"
            f"  Check the port polygon, element direction, and generated mesh "
            f"boundary for this exact port element."
        )

    return errors


def check_lumped_port_geometry(
    mesh_path: Path,
    port_surfaces: dict,
    palace_ports: list,
    *,
    rel_tol: float = 1.0e-6,
) -> list[str]:
    """Check every lumped-port surface for Palace's geometry constraint."""
    from gsim.palace.ports.config import PortGeometry, PortType

    errors: list[str] = []

    for port_idx, port in enumerate(palace_ports, start=1):
        if port.port_type != PortType.LUMPED:
            continue

        port_key = f"P{port_idx}"
        port_group = port_surfaces.get(port_key)
        if port_group is None:
            continue

        if port.multi_element and port_group.get("type") == "cpw":
            for element_idx, element in enumerate(
                port_group.get("elements", []), start=1
            ):
                phys_tag = element.get("phys_group")
                direction_str = str(element.get("direction", "")).upper()
                context = (
                    f"Port {port_idx} ('{port.name}') element {element_idx} "
                    f"(physical group {phys_tag})"
                )
                if phys_tag is None:
                    errors.append(
                        f"{context}: missing physical group id in mesh groups."
                    )
                    continue
                errors.extend(
                    _check_surface_geometry(
                        mesh_path=mesh_path,
                        phys_tag=int(phys_tag),
                        direction_str=direction_str,
                        context=context,
                        rel_tol=rel_tol,
                    )
                )
        else:
            phys_tag = port_group.get("phys_group")
            if phys_tag is None:
                errors.append(
                    f"Port {port_idx} ('{port.name}')"
                    ": missing physical group id in mesh groups."
                )
                continue
            direction_str = (
                "+Z" if port.geometry == PortGeometry.VIA else port.direction.upper()
            )
            context = f"Port {port_idx} ('{port.name}') (physical group {phys_tag})"
            errors.extend(
                _check_surface_geometry(
                    mesh_path=mesh_path,
                    phys_tag=int(phys_tag),
                    direction_str=direction_str,
                    context=context,
                    rel_tol=rel_tol,
                )
            )

    return errors


def validate_mesh(sim) -> ValidationResult:
    """Validate generated mesh and config for a Palace simulation object."""
    errors: list[str] = []
    warnings_list: list[str] = []

    mesh_result = getattr(sim, "_mesh_result", None) or getattr(
        sim, "_last_mesh_result", None
    )
    if mesh_result is None:
        errors.append("No mesh generated. Call mesh() first.")
        return ValidationResult(valid=False, errors=errors, warnings=warnings_list)

    groups = mesh_result.groups

    if not groups.get("volumes"):
        errors.append("No dielectric volumes in mesh.")
    else:
        warnings_list.append(f"Volumes: {list(groups['volumes'].keys())}")

    has_conductors = bool(groups.get("conductor_surfaces"))
    has_pec = bool(groups.get("pec_surfaces"))
    if not has_conductors and not has_pec:
        errors.append(
            "No conductor surfaces in mesh. "
            "Check that conductor layers have polygons and correct layer_type."
        )
    else:
        if has_conductors:
            warnings_list.append(
                f"Conductor surfaces: {list(groups['conductor_surfaces'].keys())}"
            )
        if has_pec:
            warnings_list.append(f"PEC surfaces: {list(groups['pec_surfaces'].keys())}")

    port_surfaces = groups.get("port_surfaces", {})
    if not port_surfaces and sim.simulation_type == "driven":
        errors.append("No port surfaces in mesh.")
    else:
        for port_name, port_info in port_surfaces.items():
            if port_info.get("type") == "cpw":
                n_elems = len(port_info.get("elements", []))
                if n_elems < 2:
                    errors.append(
                        f"CPW port '{port_name}' has {n_elems} elements "
                        "(expected >= 2)."
                    )

    if not groups.get("boundary_surfaces", {}).get("absorbing"):
        warnings_list.append(
            "No absorbing boundary found. This is expected if airbox_margin=0."
        )

    last_ports = getattr(sim, "_last_ports", None)
    if last_ports and groups.get("port_surfaces") and mesh_result.mesh_path.exists():
        errors.extend(
            check_lumped_port_geometry(
                mesh_path=mesh_result.mesh_path,
                port_surfaces=groups["port_surfaces"],
                palace_ports=last_ports,
            )
        )

    output_dir = getattr(sim, "_output_dir", None)
    if output_dir is not None:
        import json

        config_path = output_dir / "config.json"
        if not config_path.exists():
            try:
                sim.write_config(validate_mesh=False)
            except Exception as e:
                errors.append(
                    f"Could not auto-generate config.json during validate_mesh: {e}"
                )

        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                boundaries = config.get("Boundaries", {})
                if not boundaries.get("Conductivity") and not boundaries.get("PEC"):
                    errors.append("config.json has no Conductivity or PEC boundaries.")
                if (
                    not boundaries.get("LumpedPort")
                    and not boundaries.get("WavePort")
                    and (
                        sim.simulation_type == "driven"
                        or sim.simulation_type == "waveport"
                    )
                ):
                    errors.append("config.json has no LumpedPort nor Waveport entries.")
            except json.JSONDecodeError as e:
                errors.append(f"config.json is invalid JSON: {e}")

    return ValidationResult(
        valid=len(errors) == 0, errors=errors, warnings=warnings_list
    )

"""Mesh generator for Palace EM simulation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import gmsh

from . import gmsh_utils
from .config_generator import (
    collect_mesh_stats,
    generate_palace_config,
    write_config,
)
from .geometry import (
    GeometryData,
    add_dielectrics,
    add_metals,
    add_patterned_dielectrics,
    add_pec_blocks,
    add_ports,
    build_entities,
    extract_geometry,
)
from .groups import assign_physical_groups

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack
    from gsim.palace.models import DrivenConfig, EigenmodeConfig
    from gsim.palace.models.pec import PECBlockConfig
    from gsim.palace.ports.config import PalacePort

logger = logging.getLogger(__name__)


@dataclass
class MeshResult:
    """Result from mesh generation."""

    mesh_path: Path
    config_path: Path | None = None
    port_info: list = field(default_factory=list)
    mesh_stats: dict = field(default_factory=dict)
    # Data needed for deferred config generation
    groups: dict = field(default_factory=dict)
    output_dir: Path | None = None
    model_name: str = "palace"
    fmax: float = 100e9


def _setup_mesh_fields(
    kernel,
    groups: dict,
    geometry: GeometryData,
    stack: LayerStack,
    refined_cellsize: float,
    max_cellsize: float,
) -> None:
    """Set up mesh refinement fields.

    Args:
        kernel: gmsh OCC kernel
        groups: Physical group information
        geometry: Extracted geometry data
        stack: LayerStack with material properties
        refined_cellsize: Fine mesh size near conductors (um)
        max_cellsize: Coarse mesh size in air/dielectric (um)
    """
    boundary_lines: list[int] = []
    conductor_line_count = 0
    port_line_count = 0
    pec_line_count = 0
    dielectric_line_count = 0

    # Conductor-surface edges are always refined — the refined_cellsize only
    # takes effect where boundary curves drive the Threshold field, and metal
    # edges are the dominant field-concentration sites.
    for surface_info in groups["conductor_surfaces"].values():
        for tag in surface_info["tags"]:
            lines = gmsh_utils.get_boundary_lines(tag, kernel)
            boundary_lines.extend(lines)
            conductor_line_count += len(lines)

    # User-defined PEC conductor surfaces are always refined — they mark
    # narrow vertical stitches between ground planes at port boundaries,
    # which are field-concentration sites by construction.
    for surface_info in groups["pec_surfaces"].values():
        for tag in surface_info["tags"]:
            lines = gmsh_utils.get_boundary_lines(tag, kernel)
            boundary_lines.extend(lines)
            pec_line_count += len(lines)

    # Port boundaries are always refined — drives S-parameter / impedance
    # accuracy regardless of preset.
    for surface_info in groups["port_surfaces"].values():
        if surface_info.get("type") == "cpw":
            for elem in surface_info["elements"]:
                for tag in elem["tags"]:
                    lines = gmsh_utils.get_boundary_lines(tag, kernel)
                    boundary_lines.extend(lines)
                    port_line_count += len(lines)
        else:
            for tag in surface_info["tags"]:
                lines = gmsh_utils.get_boundary_lines(tag, kernel)
                boundary_lines.extend(lines)
                port_line_count += len(lines)

    # Dielectric interfaces are also field-concentration regions for
    # photonic structures; refine high-permittivity dielectric boundaries
    # (for example, core/air interfaces) when they exist as volume groups.
    for volume_name, volume_info in groups.get("volumes", {}).items():
        if volume_info.get("is_via", False):
            continue

        layer = stack.layers.get(volume_name)
        material_name = layer.material if layer is not None else volume_name
        material_props = stack.materials.get(material_name, {})
        permittivity = material_props.get("permittivity", 1.0)
        if isinstance(permittivity, list):
            try:
                permittivity = max(float(v) for v in permittivity)
            except (TypeError, ValueError):
                permittivity = 1.0
        if not isinstance(permittivity, int | float) or permittivity <= 1.0:
            continue

        for vtag in volume_info.get("tags", []):
            try:
                boundaries = gmsh.model.getBoundary(
                    [(3, vtag)],
                    combined=False,
                    oriented=False,
                    recursive=False,
                )
            except Exception:
                logger.debug(
                    "Skipping dielectric boundary refinement for stale volume tag %s",
                    vtag,
                )
                continue

            for dim, stag in boundaries:
                if dim != 2:
                    continue
                lines = gmsh_utils.get_boundary_lines(stag, kernel)
                boundary_lines.extend(lines)
                dielectric_line_count += len(lines)

    boundary_lines = sorted(set(boundary_lines))

    logger.info(
        "Mesh refinement: %d boundary lines "
        "(conductor=%d, port=%d, pec=%d, dielectric=%d)",
        len(boundary_lines),
        conductor_line_count,
        port_line_count,
        pec_line_count,
        dielectric_line_count,
    )

    # Setup main refinement field
    field_ids = []
    if boundary_lines:
        field_id = gmsh_utils.setup_mesh_refinement(
            boundary_lines, refined_cellsize, max_cellsize
        )
        field_ids.append(field_id)

    # Add box refinement for dielectrics based on permittivity
    xmin, ymin, xmax, ymax = geometry.bbox
    field_counter = 10

    for dielectric in stack.dielectrics:
        material_name = dielectric["material"]
        material_props = stack.materials.get(material_name, {})
        permittivity = material_props.get("permittivity", 1.0)

        if permittivity > 1:
            local_max = max_cellsize / math.sqrt(permittivity)
            gmsh_utils.setup_box_refinement(
                field_counter,
                xmin,
                ymin,
                dielectric["zmin"],
                xmax,
                ymax,
                dielectric["zmax"],
                local_max,
                max_cellsize,
            )
            field_ids.append(field_counter)
            field_counter += 1

    if field_ids:
        gmsh_utils.finalize_mesh_fields(field_ids)


def generate_mesh(
    component,
    stack: LayerStack,
    ports: list[PalacePort],
    output_dir: str | Path,
    model_name: str = "palace",
    refined_mesh_size: float = 5.0,
    max_mesh_size: float = 300.0,
    margin_x: float = 50.0,
    margin_y: float = 50.0,
    air_margin: float = 50.0,
    fmax: float = 100e9,
    show_gui: bool = False,
    simulation_type: str = "driven",
    driven_config: DrivenConfig | None = None,
    eigenmode_config: EigenmodeConfig | None = None,
    write_config: bool = True,
    planar_conductors: bool = False,
    pec_blocks: list[PECBlockConfig] | None = None,
    absorbing_boundary: bool = True,
    merge_via_distance: float = 2.0,
    curve_fit_mode: str = "line",
    curve_fit_layers: list[str] | None = None,
    curve_fit_tolerance_um: float = 0.0,
    curve_fit_min_points: int = 8,
    high_order_elements: bool = False,
    high_order_order: int = 2,
    high_order_optimize: bool = True,
    verbosity: int = 3,
) -> MeshResult:
    """Generate mesh for Palace EM simulation.

    Args:
        component: gdsfactory Component
        stack: LayerStack from palace-api
        ports: List of PalacePort objects (single and multi-element)
        output_dir: Directory for output files
        model_name: Base name for output files
        refined_mesh_size: Mesh size near conductors (um)
        max_mesh_size: Max mesh size in air/dielectric (um)
        margin_x: X-axis margin around design (um)
        margin_y: Y-axis margin around design (um)
        air_margin: Air box margin (um)
        fmax: Max frequency for config (Hz)
        show_gui: Show gmsh GUI during meshing
        simulation_type: Type of simulation (driven, eigenmode or electrostatics)
        driven_config: Optional DrivenConfig for frequency sweep settings
        eigenmode_config: Optional EigenmodeConfig for eigenmode problems
        write_config: Whether to write config.json (default True)
        pec_blocks: PEC configuration
        planar_conductors: If True, treat conductors as 2D PEC surfaces
        absorbing_boundary: If True, use absorbing boundary conditions on outer surfaces
        merge_via_distance: Max gap between vias to merge (um)
        curve_fit_mode: Patterned dielectric boundary mode: line/spline/bspline
        curve_fit_layers: Layer names where curve fitting is applied
        curve_fit_tolerance_um: Point merge tolerance before curve fitting
        curve_fit_min_points: Min contour points required for curve fitting
        high_order_elements: Enable high-order geometric mesh elements
        high_order_order: Polynomial order for high-order elements
        high_order_optimize: Run gmsh high-order optimization after meshing
        verbosity: Sets gmsh verbosity level

    Returns:
        MeshResult with paths and metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    msh_path = output_dir / f"{model_name}.msh"

    # Extract geometry
    logger.info("Extracting geometry...")
    geometry = extract_geometry(component, stack)
    logger.info("  Polygons: %s", len(geometry.polygons))
    logger.info("  Bbox: %s", geometry.bbox)

    # Initialize gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", verbosity)

    if "palace_mesh" in gmsh.model.list():
        gmsh.model.setCurrent("palace_mesh")
        gmsh.model.remove()
    gmsh.model.add("palace_mesh")

    kernel = gmsh.model.occ
    config_path: Path | None = None
    port_info: list = []

    try:
        # Add geometry
        logger.info("Adding metals...")
        metal_tags = add_metals(
            kernel, geometry, stack, planar_conductors, merge_via_distance
        )

        # Add PEC blocks if configured
        pec_block_tags: dict = {}
        if pec_blocks:
            logger.info("Adding PEC blocks...")
            pec_block_tags = add_pec_blocks(kernel, component, pec_blocks, stack)

        logger.info("Adding ports...")
        domain_bbox = (
            geometry.bbox[0] - margin_x,
            geometry.bbox[1] - margin_y,
            geometry.bbox[2] + margin_x,
            geometry.bbox[3] + margin_y,
        )
        port_tags, port_info = add_ports(kernel, ports, stack, domain_bbox=domain_bbox)

        logger.info("Adding dielectrics...")
        dielectric_tags = add_dielectrics(
            kernel, geometry, stack, margin_x, margin_y, air_margin
        )

        logger.info("Adding patterned dielectric layers...")
        patterned_dielectric_tags = add_patterned_dielectrics(
            kernel,
            geometry,
            stack,
            curve_fit_mode=curve_fit_mode,
            curve_fit_layers=curve_fit_layers,
            curve_fit_tolerance_um=curve_fit_tolerance_um,
            curve_fit_min_points=curve_fit_min_points,
        )

        all_dielectric_tags = {
            name: list(tags) for name, tags in dielectric_tags.items()
        }
        for layer_name, vol_tags in patterned_dielectric_tags.items():
            all_dielectric_tags.setdefault(layer_name, []).extend(vol_tags)

        # Build entities and run boolean pipeline
        logger.info("Running boolean pipeline...")
        entities = build_entities(
            metal_tags,
            dielectric_tags,
            patterned_dielectric_tags,
            port_tags,
            port_info,
            pec_block_tags=pec_block_tags or None,
            stack=stack,
        )
        pg_map = gmsh_utils.run_boolean_pipeline(entities)

        # Assign physical groups
        logger.info("Assigning physical groups...")
        groups = assign_physical_groups(
            kernel,
            metal_tags,
            all_dielectric_tags,
            port_tags,
            port_info,
            entities,
            pg_map,
            stack,
            pec_block_tags=pec_block_tags or None,
        )

        # Setup mesh fields
        logger.info("Setting up mesh refinement...")
        _setup_mesh_fields(
            kernel,
            groups,
            geometry,
            stack,
            refined_mesh_size,
            max_mesh_size,
        )

        # Show GUI if requested
        if show_gui:
            gmsh.fltk.run()

        if high_order_elements:
            logger.info(
                "Enabling high-order elements (order=%d, optimize=%s)",
                high_order_order,
                high_order_optimize,
            )
            gmsh.option.setNumber("Mesh.ElementOrder", high_order_order)
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
            gmsh.option.setNumber(
                "Mesh.HighOrderOptimize", 1 if high_order_optimize else 0
            )

        logger.info("Generating mesh...")
        gmsh.model.mesh.generate(3)

        if high_order_elements:
            gmsh.model.mesh.setOrder(high_order_order)
            if high_order_optimize:
                try:
                    gmsh.model.mesh.optimize("HighOrder")
                except Exception as exc:
                    logger.warning(
                        "High-order optimization failed, using unoptimized high-order "
                        "elements: %s",
                        exc,
                    )

        # Collect mesh statistics
        mesh_stats = collect_mesh_stats()

        # Save mesh
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(msh_path))

        logger.info("Mesh saved: %s", msh_path)

        # Generate config if requested
        config_path = None
        if write_config:
            logger.info("Generating Palace config...")
            config_path = generate_palace_config(
                groups,
                ports,
                port_info,
                stack,
                output_dir,
                model_name,
                fmax,
                simulation_type,
                driven_config,
                eigenmode_config,
                absorbing_boundary,
            )

    finally:
        gmsh.clear()
        gmsh.finalize()

    # Build result (store groups for deferred config generation)
    result = MeshResult(
        mesh_path=msh_path,
        config_path=config_path,
        port_info=port_info,
        mesh_stats=mesh_stats,
        groups=groups,
        output_dir=output_dir,
        model_name=model_name,
        fmax=fmax,
    )

    return result


# Re-export write_config from config_generator for backward compatibility
__all__ = ["GeometryData", "MeshResult", "generate_mesh", "write_config"]

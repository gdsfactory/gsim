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
    refine_from_curves: bool = False,
) -> None:
    """Set up mesh refinement fields.

    Args:
        kernel: gmsh OCC kernel
        groups: Physical group information
        geometry: Extracted geometry data
        stack: LayerStack with material properties
        refined_cellsize: Fine mesh size near conductors (um)
        max_cellsize: Coarse mesh size in air/dielectric (um)
        refine_from_curves: Refine mesh based on distance to conductor edges
    """
    # Collect boundary lines from conductor surfaces
    boundary_lines = []
    for surface_info in groups["conductor_surfaces"].values():
        for tag in surface_info["tags"]:
            lines = gmsh_utils.get_boundary_lines(tag, kernel)
            boundary_lines.extend(lines)

    # Add PEC surface edges when refine_from_curves is enabled
    if refine_from_curves:
        for surface_info in groups["pec_surfaces"].values():
            for tag in surface_info["tags"]:
                lines = gmsh_utils.get_boundary_lines(tag, kernel)
                boundary_lines.extend(lines)

    # Add port boundaries
    for surface_info in groups["port_surfaces"].values():
        if surface_info.get("type") == "cpw":
            # CPW port: get tags from each element
            for elem in surface_info["elements"]:
                for tag in elem["tags"]:
                    lines = gmsh_utils.get_boundary_lines(tag, kernel)
                    boundary_lines.extend(lines)
        else:
            # Regular port
            for tag in surface_info["tags"]:
                lines = gmsh_utils.get_boundary_lines(tag, kernel)
                boundary_lines.extend(lines)

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
    margin: float = 50.0,
    air_margin: float = 50.0,
    fmax: float = 100e9,
    show_gui: bool = False,
    simulation_type: str = "driven",
    driven_config: DrivenConfig | None = None,
    eigenmode_config: EigenmodeConfig | None = None,
    write_config: bool = True,
    planar_conductors: bool = False,
    refine_from_curves: bool = False,
    pec_blocks: list[PECBlockConfig] | None = None,
    merge_via_distance: float = 2.0,
    verbosity: int = 3,
    absorbing_boundary: bool = True,
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
        margin: XY margin around design (um)
        air_margin: Air box margin (um)
        fmax: Max frequency for config (Hz)
        show_gui: Show gmsh GUI during meshing
        simulation_type: Type of simulation (driven, eigenmode or electrostatics)
        driven_config: Optional DrivenConfig for frequency sweep settings
        eigenmode_config: Optional EigenmodeConfig for eigenmode problems
        write_config: Whether to write config.json (default True)
        planar_conductors: If True, treat conductors as 2D PEC surfaces
        refine_from_curves: Refine mesh based on distance to conductor edges
        merge_via_distance: Max gap between vias to merge (um)

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
        port_tags, port_info = add_ports(kernel, ports, stack)

        logger.info("Adding dielectrics...")
        dielectric_tags = add_dielectrics(kernel, geometry, stack, margin, air_margin)

        # Build entities and run boolean pipeline
        logger.info("Running boolean pipeline...")
        entities = build_entities(
            metal_tags,
            dielectric_tags,
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
            dielectric_tags,
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
            refine_from_curves,
        )

        # Show GUI if requested
        if show_gui:
            gmsh.fltk.run()

        logger.info("Generating mesh...")
        gmsh.model.mesh.generate(3)

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

"""Mesh configuration models for Palace simulations.

This module contains Pydantic models for mesh generation configuration.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

GradingSource = Literal["conductors", "junctions", "fine_regions"]


class MeshGrading(BaseModel):
    """Element size that grows linearly with the distance from chosen features.

    Native-2D cross-section meshes only. At a distance ``d`` (um) from the
    nearest graded feature the size is ``size + growth_per_um * d``, capped
    at ``max_mesh_size``, where ``size`` is the feature's own size. It
    replaces the sizing that applies to every internal curve without
    grading (half of ``refined_mesh_size`` on the curve, reaching
    ``max_mesh_size`` at ``max_mesh_size / 2``, a growth of about 2 per um):
    only the chosen features refine the mesh. The conductors and junctions
    get ``refined_mesh_size`` itself, twice the size the ungraded mesh puts
    on them; halve ``refined_mesh_size`` to keep that size.

    Attributes:
        growth_per_um: Increase of the element size per um of distance.
            0.1 to 0.25 keeps the elements around small features well
            shaped; larger values give fewer nodes.
        sources: Features the size grows from:

            - ``"conductors"``: the edges of the conductors, at
              ``refined_mesh_size``.
            - ``"junctions"``: the curves of the declared interfaces
              between two regions (a TCAD Interface, such as a
              metallurgical junction), at ``refined_mesh_size``.
            - ``"fine_regions"``: each refinement box, whose own size is
              held inside it, and the internal edges of each layer with a
              numeric ``mesh_resolution`` below ``max_mesh_size``, at that
              resolution. Without grading only layers finer than
              ``refined_mesh_size`` get their own size, since every
              internal curve is refined anyway.

            A feature size above ``max_mesh_size`` is held at
            ``max_mesh_size``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    growth_per_um: float = Field(gt=0)
    sources: tuple[GradingSource, ...] = Field(
        default=("conductors", "junctions", "fine_regions"), min_length=1
    )

    @field_validator("sources")
    @classmethod
    def unique_sources(
        cls, sources: tuple[GradingSource, ...]
    ) -> tuple[GradingSource, ...]:
        """Each source is named once."""
        if len(set(sources)) != len(sources):
            raise ValueError(f"Mesh grading sources repeat: {sources}.")
        return sources


class MeshConfig(BaseModel):
    """Configuration for mesh generation with quality presets.

    Attributes:
        refined_mesh_size: Mesh size near conductors (um)
        max_mesh_size: Maximum mesh size in air/dielectric (um)
        cells_per_wavelength: Number of mesh cells per wavelength
        margin: XY margin around design (um)
        margin_x: X-axis margin override (um). Falls back to margin.
        margin_y: Y-axis margin override (um). Falls back to margin.
        airbox_margin: Extra airbox around stack (um); 0 = disabled
        fmax: Maximum frequency for mesh sizing (Hz)
        boundary_conditions: List of boundary conditions for each face
        planar_conductors: Treat conductors as 2D PEC surfaces instead of volumes
        merge_via_distance: Vias of one layer closer than this (um) are merged
            into one volume. Merging fills the gaps between the vias, so an
            array of small cuts becomes a solid slab with several times the
            drawn area (a 2 x 10 array of 0.4 um cuts at 1.5 um pitch grows
            8.25 times). A warning is logged when a via layer grows by more
            than 5%. 0 keeps every via as drawn.
        clip_merged_vias: Clip each merged via polygon to the overlap of the
            conductor layers directly below and above it, so merging adds no
            via area outside the metal. The drawn cuts are always kept, and
            contacts with no metal on one side stay as drawn. A slab that lies
            entirely between two continuous metals is left unchanged; only
            ``merge_via_distance=0`` keeps its cuts.
        refinement_boxes: Boxes of a native-2D cross-section mesh held to an
            element size, each ``(h_min, h_max, z_min, z_max, size)`` in um.
            Refinement lines size the elements on them only, and the size
            grows at once with the distance; a box holds it over an area
        mesh_grading: Native-2D cross-section meshes only: a
            :class:`MeshGrading` whose size grows from chosen features at a
            set rate, in place of the fixed growth from every internal
            curve. ``None`` (default) keeps that fixed growth.
        algorithm_3d: Gmsh 3D meshing algorithm, "delaunay" or "hxt". In the
            800 um CPW tests behind gsim#283 HXT gave a different mesh and was
            slower than Delaunay at every thread count.
        threads: Threads for 3D meshing. With Delaunay and ``surface_threads=1``
            the mesh does not depend on it, and it did not speed meshing up in
            those tests either. With HXT the mesh depends on the thread count.
        surface_threads: Threads for 1D and 2D (surface) meshing. Values above 1
            are faster but gave a different mesh on every run, so keep 1 when
            the mesh must be reproducible.
        show_gui: Show gmsh GUI during meshing
        preview_only: Generate preview only, don't save mesh
    """

    model_config = ConfigDict(validate_assignment=True)

    refined_mesh_size: float = Field(default=5.0, gt=0)
    max_mesh_size: float = Field(default=300.0, gt=0)
    cells_per_wavelength: int = Field(default=10, ge=1)
    margin: float = Field(default=0.0, ge=0)
    margin_x: float | None = Field(
        default=None,
        ge=0,
        description="X margin override (um). Falls back to margin.",
    )
    margin_y: float | None = Field(
        default=None,
        ge=0,
        description="Y margin override (um). Falls back to margin.",
    )
    airbox_margin: float = Field(default=0.0, ge=0)
    fmax: float = Field(default=100e9, gt=0)
    boundary_conditions: list[str] | None = None
    planar_conductors: bool = False
    merge_via_distance: float = Field(default=2.0, ge=0)
    clip_merged_vias: bool = False
    curve_fit_mode: Literal["line", "spline", "bspline"] = "line"
    curve_fit_layers: list[str] = Field(default_factory=lambda: ["core", "core2"])
    curve_fit_tolerance_um: float = Field(default=0.0, ge=0)
    curve_fit_min_points: int = Field(default=8, ge=3)
    curve_fit_corner_angle_deg: float = Field(default=45.0, gt=0, lt=180)
    high_order_elements: bool = False
    high_order_order: int = Field(default=2, ge=2, le=6)
    high_order_optimize: bool = True
    refinement_boxes: list[tuple[float, float, float, float, float]] = Field(
        default_factory=list,
        description=(
            "Native-2D meshes only: (h_min, h_max, z_min, z_max, size) boxes "
            "on the cross-section (um) inside which elements keep to size."
        ),
    )
    mesh_grading: MeshGrading | None = None
    algorithm_3d: Literal["delaunay", "hxt"] = "delaunay"
    threads: int = Field(default=1, ge=1)
    surface_threads: int = Field(default=1, ge=1)
    show_gui: bool = False
    preview_only: bool = False

    @property
    def effective_margin_x(self) -> float:
        """Resolved X margin (margin_x if set, else margin)."""
        return self.margin_x if self.margin_x is not None else self.margin

    @property
    def effective_margin_y(self) -> float:
        """Resolved Y margin (margin_y if set, else margin)."""
        return self.margin_y if self.margin_y is not None else self.margin

    @field_validator("refinement_boxes")
    @classmethod
    def validate_refinement_boxes(
        cls, boxes: list[tuple[float, float, float, float, float]]
    ) -> list[tuple[float, float, float, float, float]]:
        """Every box spans an area and has a size to hold it to."""
        for box in boxes:
            h_min, h_max, z_min, z_max, size = box
            if size <= 0.0 or h_max <= h_min or z_max <= z_min:
                raise ValueError(
                    "A refinement box needs an ascending extent and a positive "
                    f"size, got {box}."
                )
        return boxes

    @model_validator(mode="after")
    def set_default_boundary_conditions(self) -> Self:
        """Set default boundary conditions if not provided."""
        if self.boundary_conditions is None:
            self.boundary_conditions = ["ABC", "ABC", "ABC", "ABC", "ABC", "ABC"]
        return self

    @classmethod
    def coarse(cls, **kwargs: Any) -> Self:
        """Fast mesh for quick iteration (~2.5 elements per wavelength).

        This preset is suitable for initial debugging and quick checks.
        Not recommended for accurate results.
        """
        defaults: dict[str, Any] = {
            "refined_mesh_size": 10.0,
            "max_mesh_size": 600.0,
            "cells_per_wavelength": 5,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def default(cls, **kwargs: Any) -> Self:
        """Balanced mesh (~5 elements per wavelength).

        This preset provides a good balance between accuracy and computation time.
        Suitable for most simulations.
        """
        defaults: dict[str, Any] = {
            "refined_mesh_size": 5.0,
            "max_mesh_size": 300.0,
            "cells_per_wavelength": 10,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def fine(cls, **kwargs: Any) -> Self:
        """High accuracy mesh (~10 elements per wavelength).

        This preset provides higher accuracy at the cost of increased
        computation time. Use for final production simulations.
        """
        defaults: dict[str, Any] = {
            "refined_mesh_size": 2.0,
            "max_mesh_size": 70.0,
            "cells_per_wavelength": 20,
        }
        defaults.update(kwargs)
        return cls(**defaults)


__all__ = [
    "MeshConfig",
]

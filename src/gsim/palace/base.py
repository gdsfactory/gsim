"""Base mixin for Palace simulation classes.

Provides common visualization methods shared across all simulation types.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class PalaceSimMixin:
    """Mixin providing common methods for all Palace simulation classes.

    Requires the class to have:
        - _output_dir: Path | None (private attribute)
    """

    _output_dir: Path | None

    def plot_mesh(
        self,
        output: str | Path | None = None,
        show_groups: list[str] | None = None,
        interactive: bool = True,
    ) -> None:
        """Plot the mesh wireframe using PyVista.

        Requires mesh() to be called first.

        Args:
            output: Output PNG path (only used if interactive=False)
            show_groups: List of group name patterns to show (None = all).
                Example: ["metal", "P"] to show metal layers and ports.
            interactive: If True, open interactive 3D viewer.
                If False, save static PNG to output path.

        Raises:
            ValueError: If output_dir not set or mesh file doesn't exist

        Example:
            >>> sim.mesh(preset="default")
            >>> sim.plot_mesh(show_groups=["metal", "P"])
        """
        from gsim.viz import plot_mesh as _plot_mesh

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")

        mesh_path = self._output_dir / "palace.msh"
        if not mesh_path.exists():
            raise ValueError(
                f"Mesh file not found: {mesh_path}. Call mesh() first."
            )

        # Default output path if not interactive
        if output is None and not interactive:
            output = self._output_dir / "mesh.png"

        _plot_mesh(
            msh_path=mesh_path,
            output=output,
            show_groups=show_groups,
            interactive=interactive,
        )

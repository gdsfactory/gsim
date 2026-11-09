"""IPython kernel configuration for executing the docs notebooks.

The docs workflow points ``IPYTHONDIR`` at this directory so matplotlib figures
are exported as native SVG instead of low-resolution PNG. Matplotlib already
draws SVG text as paths by default, so no rc overrides are needed.
"""

c = get_config()  # ty: ignore[unresolved-reference]

# Vector output for matplotlib figures. nbconvert prefers SVG over PNG anyway,
# so emitting only SVG keeps the notebooks smaller.
c.InlineBackend.figure_formats = ["svg"]

c.InlineBackend.print_figure_kwargs = {
    # Trim excess whitespace around figures.
    "bbox_inches": "tight",
    # Resolution of artists rasterized inside the SVG (dense meshes). The
    # inline backend otherwise passes the figure dpi (100), which is no
    # sharper than the old PNGs.
    "dpi": 200,
}

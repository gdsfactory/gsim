"""IPython kernel configuration for executing the docs notebooks.

The docs workflow points ``IPYTHONDIR`` at this directory so matplotlib figures
are exported as SVG instead of low-resolution PNG.
"""

c = get_config()  # ty: ignore[unresolved-reference]

# Vector output for matplotlib figures. nbconvert prefers SVG over PNG anyway,
# so emitting only SVG keeps the notebooks smaller.
c.InlineBackend.figure_formats = ["svg"]

# Trim excess whitespace around figures.
c.InlineBackend.print_figure_kwargs = {"bbox_inches": "tight"}

c.InlineBackend.rc = {
    # Draw text as paths so figures look the same in every browser.
    "svg.fonttype": "path",
    # Resolution for artists that are rasterized inside an SVG
    # (images and rasterized meshes).
    "savefig.dpi": 200,
}

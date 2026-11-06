"""Check that the docs kernel exports matplotlib figures as SVG.

Runs a one-cell notebook on the given kernel and fails if the figure output is
not ``image/svg+xml``, for example because ``IPYTHONDIR`` does not point at
``scripts/ipython``.

Usage: ``python scripts/check_inline_svg.py [kernel_name]``
"""

from __future__ import annotations

import sys

import nbformat
from nbclient import NotebookClient


def main(kernel_name: str = "gsim") -> int:
    """Execute a probe notebook and report the figure output formats."""
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_code_cell(
            "import matplotlib.pyplot as plt\n"
            "print(plt.get_backend())\n"
            "plt.plot([0, 1], [0, 1])\n"
            "plt.show()"
        )
    ]
    NotebookClient(notebook, kernel_name=kernel_name, timeout=120).execute()
    formats = {
        mime
        for output in notebook.cells[0].outputs
        if output.get("output_type") == "display_data"
        for mime in output.get("data", {})
    }
    backend = "".join(
        output.get("text", "")
        for output in notebook.cells[0].outputs
        if output.get("output_type") == "stream"
    ).strip()
    if "image/svg+xml" not in formats or "image/png" in formats:
        print(f"Expected SVG-only matplotlib output, got: {sorted(formats)}")
        if not formats:
            print(
                f"No figure was displayed; matplotlib backend is {backend!r}. "
                "Is MPLBACKEND set to a non-inline backend?"
            )
        return 1
    print("matplotlib figures are exported as SVG")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))

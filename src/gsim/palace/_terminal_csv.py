"""Read the terminal and domain CSV tables of a Palace run.

Electrostatic and magnetostatic runs write their matrices as small CSV tables
(``terminal-*.csv``, ``domain-E.csv``). This module finds those files in any
form a run's results take and reads them as UTF-8, since some headers are not
ASCII (the inverse inductance matrix header has a superscript minus one).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from gsim.palace.results import PalaceTextResults, _find_file

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _result_file(
    source: str | Path | dict | PalaceTextResults, name: str
) -> Path | None:
    """Path of one output file of a run, None if the run has no such file.

    Args:
        source: The results dict that ``run()`` returns, the
            ``PalaceTextResults`` that ``run_local()`` returns, or the
            simulation or Palace output directory.
        name: File name, e.g. ``terminal-C.csv``.

    Returns:
        The path of an existing file, or None.
    """
    if isinstance(source, PalaceTextResults):
        source = source.files
    if isinstance(source, dict):
        value = source.get(name)
        path = Path(value) if value is not None else None
    else:
        path = _find_file(Path(source), name)
    return path if path is not None and path.exists() else None


def _read_table(path: Path) -> NDArray[np.float64]:
    """Read a Palace CSV table; the header line is skipped."""
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2, encoding="utf-8")


def _read_column(path: Path, column: str) -> NDArray[np.float64] | None:
    """Read one named column of a Palace CSV table, None if it has no such column."""
    with path.open(encoding="utf-8") as file:
        header = [name.strip() for name in file.readline().split(",")]
    if column not in header:
        return None
    return _read_table(path)[:, header.index(column)]

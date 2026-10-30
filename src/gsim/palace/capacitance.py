"""Capacitance matrices from an electrostatic Palace run.

For a problem with N terminals, Palace writes the Maxwell capacitance matrix
(``terminal-C.csv``), the mutual capacitance matrix (``terminal-Cm.csv``), its
inverse, the excitation voltages and the stored energies. This module reads
them, labels them with the terminal names and checks that they hold together.

The conventions, in Palace's own outputs:

- Maxwell matrix ``C``, with ``Q = C V``: ``C[i][i] > 0`` and ``C[i][j] <= 0``.
  Its row sum is the capacitance of terminal ``i`` to ground.
- Mutual matrix ``C_m``: ``C_m[i][j] = -C[i][j]`` for ``i != j``, and
  ``C_m[i][i]`` is the capacitance of terminal ``i`` to ground.
- For the excitation of terminal ``i``, the stored electric energy is
  ``E_elec = 1/2 C[i][i] V_inc[i]**2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from gsim.palace._terminal_csv import _read_column, _read_table, _result_file

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from numpy.typing import NDArray

    from gsim.palace.results import PalaceTextResults


@dataclass(frozen=True, eq=False)
class CapacitanceMatrices:
    """The capacitance matrices of an electrostatic run, labelled by terminal.

    Attributes:
        terminals: Terminal names, in Palace's index order.
        maxwell: Maxwell capacitance matrix, in F.
        mutual: Mutual capacitance matrix, in F, as Palace writes it.
        inverse: Inverse of the Maxwell matrix, in 1/F, when Palace wrote it.
        excitation_voltage: Voltage applied in each excitation, in V.
        stored_energy: Electric energy of each excitation, in J.
    """

    terminals: tuple[str, ...]
    maxwell: NDArray[np.float64]
    mutual: NDArray[np.float64]
    inverse: NDArray[np.float64] | None = None
    excitation_voltage: NDArray[np.float64] | None = None
    stored_energy: NDArray[np.float64] | None = None

    def _index(self, name: str) -> int:
        """Position of a terminal in the matrices, KeyError if there is none."""
        try:
            return self.terminals.index(name)
        except ValueError:
            msg = f"unknown terminal {name!r}; the terminals are {list(self.terminals)}"
            raise KeyError(msg) from None

    def between(self, first: str, second: str) -> float:
        """Capacitance between two electrodes, in F (their mutual capacitance)."""
        if first == second:
            msg = "between() needs two different terminals; use to_ground() for one"
            raise ValueError(msg)
        return float(self.mutual[self._index(first), self._index(second)])

    def to_ground(self, name: str) -> float:
        """Capacitance of an electrode to ground, in F."""
        index = self._index(name)
        return float(self.mutual[index, index])

    def pairs(self) -> pd.DataFrame:
        """Every capacitance, electrode to electrode and electrode to ground.

        One row per pair of terminals (:meth:`between`), then one row per
        terminal against ``"ground"`` (:meth:`to_ground`), so the two kinds
        are reported apart.

        Returns:
            A table with the columns ``first``, ``second`` and
            ``capacitance_F``.
        """
        import pandas as pd

        rows = [
            (first, second, self.between(first, second))
            for index, first in enumerate(self.terminals)
            for second in self.terminals[index + 1 :]
        ]
        rows.extend((name, "ground", self.to_ground(name)) for name in self.terminals)
        return pd.DataFrame(rows, columns=["first", "second", "capacitance_F"])

    def eliminate(self, names: Sequence[str]) -> CapacitanceMatrices:
        """Remove floating terminals with a Schur complement of the Maxwell matrix.

        A floating conductor holds no net charge. With ``C`` split into the
        terminals that stay (``t``) and the floating ones (``f``), the matrix
        the others see is ``C_tt - C_tf C_ff^-1 C_ft``, which is exact for an
        electrostatic problem. The mutual matrix is rebuilt from it (off
        diagonal ``-C_ij``, diagonal the row sum), the inverse is the block of
        the full inverse that belongs to the remaining terminals, and the
        voltages and energies, which belong to the excitations of the removed
        terminals too, are dropped.

        Args:
            names: Terminals to remove.

        Returns:
            The matrices of the remaining terminals; ``self`` when ``names``
            is empty.

        Raises:
            KeyError: If a name is not a terminal.
            ValueError: If a name is repeated, no terminal would remain, or
                the floating terminals have no capacitance to anything (a
                singular block).
        """
        drop = [self._index(name) for name in names]
        if not drop:
            return self
        if len(set(drop)) != len(drop):
            msg = f"the terminals to eliminate must be different: {list(names)}"
            raise ValueError(msg)
        keep = [i for i in range(len(self.terminals)) if i not in drop]
        if not keep:
            msg = "cannot eliminate every terminal"
            raise ValueError(msg)
        c = self.maxwell
        try:
            solved = np.linalg.solve(c[np.ix_(drop, drop)], c[np.ix_(drop, keep)])
        except np.linalg.LinAlgError:
            msg = (
                "cannot eliminate "
                f"{list(names)}: their block of the Maxwell matrix is singular"
            )
            raise ValueError(msg) from None
        reduced = c[np.ix_(keep, keep)] - c[np.ix_(keep, drop)] @ solved
        ground = reduced.sum(axis=1)
        mutual = -(reduced - np.diag(np.diag(reduced))) + np.diag(ground)
        return CapacitanceMatrices(
            terminals=tuple(self.terminals[i] for i in keep),
            maxwell=np.asarray(reduced, dtype=np.float64),
            mutual=mutual,
            inverse=None if self.inverse is None else self.inverse[np.ix_(keep, keep)],
        )

    def maxwell_frame(self) -> pd.DataFrame:
        """The Maxwell matrix as a DataFrame indexed by terminal name."""
        import pandas as pd

        return pd.DataFrame(self.maxwell, index=self.terminals, columns=self.terminals)

    def mutual_frame(self) -> pd.DataFrame:
        """The mutual matrix as a DataFrame indexed by terminal name."""
        import pandas as pd

        return pd.DataFrame(self.mutual, index=self.terminals, columns=self.terminals)

    def problems(self, rtol: float = 1e-3) -> list[str]:
        """List what is inconsistent in the matrices; empty when they hold together.

        Checks that every value is finite, that the Maxwell matrix is symmetric
        and positive semi-definite, which is what a non-negative stored energy
        for every set of voltages means, that each terminal has a positive self
        capacitance, that its off-diagonal entries are not positive and that no
        terminal has a negative capacitance to ground, that the mutual matrix
        follows from it, and, when Palace wrote them, that the inverse inverts
        it and that each excitation's stored energy is ``1/2 C[i][i] V**2``.
        Only the first problem is reported when a value is not finite, since
        every other check is meaningless then.

        Args:
            rtol: Tolerance, relative to the largest Maxwell entry. The default
                lets through the small errors of a discretization, such as a
                weakly coupled pair whose coefficient comes out slightly
                positive. The inconsistencies these checks look for, a wrong
                sign or the wrong matrix, are as large as the matrix itself.
        """
        maxwell, mutual = self.maxwell, self.mutual
        values = {
            "the Maxwell matrix": maxwell,
            "the mutual matrix": mutual,
            "the inverse": self.inverse,
            "the excitation voltages": self.excitation_voltage,
            "the stored energies": self.stored_energy,
        }
        bad = [
            name
            for name, v in values.items()
            if v is not None and not np.isfinite(v).all()
        ]
        if bad:
            return [f"these are not finite (NaN or infinity): {', '.join(bad)}"]

        scale = float(np.abs(maxwell).max())
        tol = rtol * scale
        found: list[str] = []

        asymmetry = float(np.abs(maxwell - maxwell.T).max())
        if asymmetry > tol:
            found.append(
                f"the Maxwell matrix is not symmetric: max |C_ij - C_ji| = "
                f"{asymmetry:.3g} F ({asymmetry / scale:.1%} of its largest entry)"
            )

        smallest = float(np.linalg.eigvalsh((maxwell + maxwell.T) / 2).min())
        if smallest < -tol:
            found.append(
                f"the stored energy is negative for some voltages: the Maxwell "
                f"matrix has an eigenvalue of {smallest:.3g} F"
            )

        if np.diag(maxwell).min() <= tol:
            found.append(
                f"a terminal has no self capacitance: the smallest diagonal entry "
                f"of the Maxwell matrix is {np.diag(maxwell).min():.3g} F"
            )

        off_diagonal = maxwell - np.diag(np.diag(maxwell))
        if off_diagonal.max() > tol:
            found.append(
                f"a Maxwell off-diagonal entry is positive "
                f"({off_diagonal.max():.3g} F); the coefficients between two "
                f"conductors should not be"
            )

        ground = maxwell.sum(axis=1)
        if ground.min() < -tol:
            found.append(
                f"a terminal has a negative capacitance to ground "
                f"({ground.min():.3g} F): a row of the Maxwell matrix sums below zero"
            )

        expected_mutual = -off_diagonal + np.diag(ground)
        mismatch = float(np.abs(mutual - expected_mutual).max())
        if mismatch > tol:
            found.append(
                f"the mutual matrix does not follow the Maxwell one (off-diagonal "
                f"= -C_ij, diagonal = row sum): largest difference {mismatch:.3g} F"
            )

        if self.inverse is not None:
            error = float(np.abs(maxwell @ self.inverse - np.eye(len(maxwell))).max())
            if error > rtol:
                found.append(
                    f"the inverse does not invert the Maxwell matrix: "
                    f"max |C C^-1 - I| = {error:.3g}"
                )

        if self.excitation_voltage is not None and self.stored_energy is not None:
            expected_energy = 0.5 * np.diag(maxwell) * self.excitation_voltage**2
            error = float(np.abs(self.stored_energy - expected_energy).max())
            if error > rtol * float(np.abs(expected_energy).max()):
                found.append(
                    f"the stored energy does not match 1/2 C_ii V_i^2: largest "
                    f"difference {error:.3g} J"
                )
        return found


def load_capacitance(
    source: str | Path | dict | PalaceTextResults,
    *,
    terminal_names: Sequence[str] | None = None,
) -> CapacitanceMatrices:
    """Load the capacitance matrices of an electrostatic Palace run.

    Args:
        source: The results dict that ``run()`` returns, the
            ``PalaceTextResults`` that ``run_local()`` returns, or the
            simulation or Palace output directory.
        terminal_names: Names for the terminals, in Palace's index order, which
            is the order they were added to the simulation. Defaults to
            ``T1``, ``T2``, ...

    Returns:
        The matrices, labelled by terminal.

    Raises:
        FileNotFoundError: If ``terminal-C.csv`` or ``terminal-Cm.csv`` is missing.
        ValueError: If the terminal names do not match the matrices.
    """

    def require(name: str) -> Path:
        path = _result_file(source, name)
        if path is None:
            where = str(source) if isinstance(source, (str, Path)) else "the results"
            msg = f"{name} not found in {where}"
            raise FileNotFoundError(msg)
        return path

    maxwell_path = require("terminal-C.csv")
    mutual_path = require("terminal-Cm.csv")
    maxwell = _read_table(maxwell_path)[:, 1:]
    mutual = _read_table(mutual_path)[:, 1:]
    count = len(maxwell)
    if maxwell.shape != (count, count) or mutual.shape != (count, count):
        msg = (
            f"the capacitance matrices are not square: {maxwell.shape}, {mutual.shape}"
        )
        raise ValueError(msg)

    names = (
        tuple(f"T{index}" for index in range(1, count + 1))
        if terminal_names is None
        else tuple(terminal_names)
    )
    if len(names) != count:
        msg = f"{len(names)} names given for {count} terminals"
        raise ValueError(msg)
    if len(set(names)) != count:
        msg = f"the terminal names must be different: {list(names)}"
        raise ValueError(msg)

    inverse_path = _result_file(source, "terminal-Cinv.csv")
    voltage_path = _result_file(source, "terminal-V.csv")
    energy_path = _result_file(source, "domain-E.csv")
    return CapacitanceMatrices(
        terminals=names,
        maxwell=maxwell,
        mutual=mutual,
        inverse=None if inverse_path is None else _read_table(inverse_path)[:, 1:],
        excitation_voltage=(
            None if voltage_path is None else _read_table(voltage_path)[:, 1]
        ),
        stored_energy=(
            None if energy_path is None else _read_column(energy_path, "E_elec (J)")
        ),
    )


__all__ = ["CapacitanceMatrices", "load_capacitance"]

"""Modal to terminal S-parameters for multi-conductor (multi-pin) ports.

A field solver returns *modal* S-parameters: one port per propagating mode.
Circuit work (differential/common-mode analysis, bias lines, SPICE export)
needs *terminal* S-parameters: one port per conductor, referenced to a common
ground and a chosen reference impedance. This module converts one into the
other using, for every physical port, the matrix of terminal voltages
``T_V`` and terminal currents ``T_I`` of its modes.

Conventions
-----------

Time convention exp(+j w t), as in the rest of gsim. A physical port ``p`` has
``n_p`` terminals and ``n_p`` modes kept.

* Column ``k`` of ``T_V[p]`` holds the terminal voltages (terminal minus the
  reference conductor) of mode ``k`` for a unit forward modal wave. Column
  ``k`` of ``T_I[p]`` holds the terminal currents flowing *into* the network
  for that wave.
* Modal waves: ``V = T_V (a + b)`` and ``I = T_I (a - b)``.
* Terminal power waves for a real reference impedance ``Zr``:
  ``a_t = F (V + Zr I)`` and ``b_t = F (V - Zr I)`` with
  ``F = diag(1 / (2 sqrt(Zr)))``. For real ``Zr`` these equal pseudo waves.
* With block-diagonal ``T = diag(T_V[p])`` and ``W = diag(T_I[p])``, and
  ``A = F (T + Zr W)``, ``B = F (T - Zr W)``::

      S_t = (B + A S_m) (A + B S_m)^-1
      S_m = (A - S_t B)^-1 (S_t A - B)

* Port ordering: physical port by physical port, terminals (or modes) in
  column order inside each port. A two-end line is ordered (A1, B1, A2, B2)
  and its modal S is (end 1 mode 1, end 1 mode 2, end 2 mode 1, end 2 mode 2).

Properties
----------

* **Joint per-mode scaling.** Scaling column ``k`` of ``T_V`` and ``T_I``
  together by any complex constant leaves ``S_t`` unchanged when ``S_m`` has
  no cross-mode terms. The solver's modal normalisation therefore cancels;
  only the voltage/current *shape* of each mode matters.
* **Degenerate modes.** If several modes share the same propagation constant,
  any basis of their subspace gives the same ``S_t``. Mode identification can
  be unstable there; the terminal S is not.
* **Currents.** For a reciprocal line ``T_I^T T_V`` is diagonal with entries
  ``r_k = V_k^T I_k`` (the unconjugated reaction), so
  ``T_I = T_V^{-T} diag(r)``, see :func:`currents_from_reaction`. Replacing
  ``r_k`` by the real power ``2 P_k`` is only an approximation when the line
  is lossy.
* **Uniform line.** The modal S of a uniform line is ``[[0, E], [E, 0]]`` with
  ``E = diag(exp(-gamma_k * length))``, see :func:`uniform_line_modal_s`.

Example::

    import numpy as np
    from gsim.palace.terminal import modal_to_terminal_s, uniform_line_modal_s

    # t_v, t_i: (nf, 2, 2) for each end; gamma: (nf, 2) in 1/m
    s_modal = uniform_line_modal_s(gamma, length=700e-6)
    s_term = modal_to_terminal_s(s_modal, [t_v, t_v], [t_i, t_i], z_ref=50.0)
    # s_term is ordered (A1, B1, A2, B2)

Limits: real positive reference impedances only; cross-mode terms in the modal
S must come from the solver (a multi-mode port solve).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "currents_from_reaction",
    "degenerate_mode_groups",
    "modal_to_terminal_s",
    "terminal_to_modal_s",
    "uniform_line_modal_s",
]

_MAX_COND = 1e12


def _check_finite(arr: NDArray, name: str) -> None:
    """Raise ``ValueError`` if ``arr`` has NaN or infinite entries."""
    if not np.all(np.isfinite(arr)):
        msg = f"{name} contains non-finite values"
        raise ValueError(msg)


def _port_blocks(
    blocks: Sequence[ArrayLike], nf: int, name: str
) -> list[NDArray[np.complex128]]:
    """Return one ``(nf, n, n)`` complex array per physical port."""
    out: list[NDArray[np.complex128]] = []
    for p, block in enumerate(blocks):
        arr = np.asarray(block, dtype=complex)
        label = f"{name}[{p}]"
        if arr.ndim == 2:
            arr = np.broadcast_to(arr, (nf, *arr.shape))
        if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
            msg = f"{label} must be square (n, n) or (nf, n, n), got shape {arr.shape}"
            raise ValueError(msg)
        if arr.shape[0] != nf:
            msg = f"{label} has {arr.shape[0]} frequencies, S has {nf}"
            raise ValueError(msg)
        _check_finite(arr, label)
        cond = np.linalg.cond(arr)
        if np.any(cond > _MAX_COND):
            msg = (
                f"{label} is ill-conditioned (cond = {cond.max():.1e}): "
                "the modes do not span the terminals"
            )
            raise ValueError(msg)
        out.append(np.array(arr))
    return out


def _block_diag(
    blocks: list[NDArray[np.complex128]], nf: int
) -> NDArray[np.complex128]:
    """Stack per-port ``(nf, n, n)`` blocks into one block-diagonal array."""
    n_tot = sum(b.shape[1] for b in blocks)
    out = np.zeros((nf, n_tot, n_tot), dtype=complex)
    i0 = 0
    for b in blocks:
        n = b.shape[1]
        out[:, i0 : i0 + n, i0 : i0 + n] = b
        i0 += n
    return out


def _reference_impedance(z_ref: ArrayLike, nf: int, nt: int) -> NDArray[np.float64]:
    """Return ``(nf, nt)`` real positive reference impedances."""
    arr = np.asarray(z_ref)
    if np.iscomplexobj(arr):
        if np.any(arr.imag != 0):
            msg = "complex z_ref is not supported: only real, positive z_ref"
            raise ValueError(msg)
        arr = arr.real
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 0:
        arr = np.full((nf, nt), float(arr))
    elif arr.ndim == 1:
        if arr.shape[0] != nt:
            msg = f"z_ref has {arr.shape[0]} entries for {nt} terminals"
            raise ValueError(msg)
        arr = np.broadcast_to(arr, (nf, nt))
    elif arr.ndim == 2:
        if arr.shape != (nf, nt):
            msg = f"z_ref has shape {arr.shape}, expected ({nf}, {nt})"
            raise ValueError(msg)
    else:
        msg = f"z_ref must be a scalar, (nt,) or (nf, nt), got ndim={arr.ndim}"
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        msg = "z_ref must be finite and > 0"
        raise ValueError(msg)
    return np.array(arr)


def _wave_matrices(
    n_dim: int,
    nf: int,
    t_v: Sequence[ArrayLike],
    t_i: Sequence[ArrayLike],
    z_ref: ArrayLike,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Return ``A = F (T + Zr W)`` and ``B = F (T - Zr W)``, each ``(nf, N, N)``."""
    if len(t_v) != len(t_i):
        msg = f"t_v has {len(t_v)} ports, t_i has {len(t_i)}"
        raise ValueError(msg)
    if len(t_v) == 0:
        msg = "at least one port is required"
        raise ValueError(msg)
    tv = _port_blocks(t_v, nf, "t_v")
    ti = _port_blocks(t_i, nf, "t_i")
    for p, (a, b) in enumerate(zip(tv, ti, strict=True)):
        if a.shape != b.shape:
            msg = f"t_v[{p}] and t_i[{p}] shapes differ: {a.shape} vs {b.shape}"
            raise ValueError(msg)
    nt = sum(b.shape[1] for b in tv)
    if nt != n_dim:
        msg = f"S has {n_dim} ports but the T blocks describe {nt} terminals/modes"
        raise ValueError(msg)
    zr = _reference_impedance(z_ref, nf, nt)
    t_mat = _block_diag(tv, nf)
    w_mat = _block_diag(ti, nf)
    f_vec = (1.0 / (2.0 * np.sqrt(zr)))[:, :, None]
    zw = zr[:, :, None] * w_mat
    return f_vec * (t_mat + zw), f_vec * (t_mat - zw)


def _check_square_s(s: ArrayLike, name: str) -> NDArray[np.complex128]:
    """Return ``s`` as a finite complex ``(nf, N, N)`` array or raise ``ValueError``."""
    arr = np.asarray(s, dtype=complex)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        msg = f"{name} must have shape (nf, N, N), got {arr.shape}"
        raise ValueError(msg)
    _check_finite(arr, name)
    return arr


def modal_to_terminal_s(
    s_modal: ArrayLike,
    t_v: Sequence[ArrayLike],
    t_i: Sequence[ArrayLike],
    *,
    z_ref: ArrayLike = 50.0,
) -> NDArray[np.complex128]:
    """Convert modal S-parameters to terminal (power-wave) S-parameters.

    Args:
        s_modal: Modal S-parameters, shape ``(nf, M, M)``.
        t_v: One terminal-voltage matrix per physical port, each ``(n_p, n_p)``
            or ``(nf, n_p, n_p)``. Column ``k`` is mode ``k``.
        t_i: Terminal currents flowing into the network, same layout as ``t_v``.
        z_ref: Real positive reference impedance in ohm: a scalar, one value
            per terminal ``(Nt,)`` or one per frequency and terminal
            ``(nf, Nt)``.

    Returns:
        Terminal S-parameters, shape ``(nf, Nt, Nt)`` with ``Nt = M``. For real
        ``z_ref`` power waves and pseudo waves coincide.

    Raises:
        ValueError: On shape or port-count mismatch, non-square or
            ill-conditioned ``t_v``/``t_i`` blocks (``cond > 1e12``, the modes
            do not span the terminals), non-positive or complex ``z_ref``, or
            non-finite input.
    """
    s_m = _check_square_s(s_modal, "s_modal")
    nf, m, _ = s_m.shape
    a_mat, b_mat = _wave_matrices(m, nf, t_v, t_i, z_ref)
    num = b_mat + a_mat @ s_m
    den = a_mat + b_mat @ s_m
    # S_t = num den^-1  <=>  S_t^T = solve(den^T, num^T)
    return np.linalg.solve(den.transpose(0, 2, 1), num.transpose(0, 2, 1)).transpose(
        0, 2, 1
    )


def terminal_to_modal_s(
    s_terminal: ArrayLike,
    t_v: Sequence[ArrayLike],
    t_i: Sequence[ArrayLike],
    *,
    z_ref: ArrayLike = 50.0,
) -> NDArray[np.complex128]:
    """Convert terminal S-parameters back to modal S-parameters.

    Inverse of :func:`modal_to_terminal_s`; same arguments, conventions and
    errors, with ``s_terminal`` of shape ``(nf, Nt, Nt)``.
    """
    s_t = _check_square_s(s_terminal, "s_terminal")
    nf, nt, _ = s_t.shape
    a_mat, b_mat = _wave_matrices(nt, nf, t_v, t_i, z_ref)
    return np.linalg.solve(a_mat - s_t @ b_mat, s_t @ a_mat - b_mat)


def currents_from_reaction(
    t_v: ArrayLike, reaction: ArrayLike
) -> NDArray[np.complex128]:
    """Terminal currents from terminal voltages and the modal reaction.

    For a reciprocal line ``T_I^T T_V`` is diagonal with entries
    ``r_k = V_k^T I_k`` (the unconjugated reaction of mode ``k``), so
    ``T_I = T_V^{-T} diag(r)``. Using the real power ``2 P_k`` for ``r_k`` is
    exact only without loss.

    Args:
        t_v: ``(n, n)`` or ``(nf, n, n)`` terminal voltages (column = mode).
        reaction: ``(n,)`` or ``(nf, n)`` reaction of each mode.

    Returns:
        ``T_I`` with the shape of ``t_v``.

    Raises:
        ValueError: On shape mismatch, ill-conditioned ``t_v`` or non-finite
            input.
    """
    v = np.asarray(t_v, dtype=complex)
    r = np.asarray(reaction, dtype=complex)
    if v.ndim not in (2, 3) or v.shape[-1] != v.shape[-2]:
        msg = f"t_v must be (n, n) or (nf, n, n), got {v.shape}"
        raise ValueError(msg)
    n = v.shape[-1]
    bad_shape = (
        r.shape[-1] != n
        or r.ndim > v.ndim - 1
        or (r.ndim == 2 and v.ndim == 3 and r.shape[0] != v.shape[0])
    )
    if bad_shape:
        msg = f"reaction shape {r.shape} does not match t_v shape {v.shape}"
        raise ValueError(msg)
    _check_finite(v, "t_v")
    _check_finite(r, "reaction")
    if np.any(np.linalg.cond(v) > _MAX_COND):
        msg = "t_v is ill-conditioned: the modes do not span the terminals"
        raise ValueError(msg)
    diag = r[..., :, None] * np.eye(n)
    return np.linalg.solve(np.swapaxes(v, -1, -2), diag)


def uniform_line_modal_s(gamma: ArrayLike, length: float) -> NDArray[np.complex128]:
    """Modal S-parameters of a uniform multi-mode line, no cross-mode terms.

    ``S = [[0, E], [E, 0]]`` with ``E = diag(exp(-gamma_k * length))``.

    Args:
        gamma: Propagation constants in 1/m, shape ``(nf, n)`` (or ``(n,)``).
        length: Line length in m.

    Returns:
        ``(nf, 2n, 2n)`` ordered (end 1 modes, end 2 modes).

    Raises:
        ValueError: If ``Re(gamma) < 0`` beyond round-off (a mode that grows
            along the line), if ``gamma`` is not finite, or if ``length`` is
            negative or not finite.
    """
    g = np.atleast_2d(np.asarray(gamma, dtype=complex))
    if g.ndim != 2:
        msg = f"gamma must be (nf, n) or (n,), got shape {g.shape}"
        raise ValueError(msg)
    _check_finite(g, "gamma")
    if not np.isfinite(length) or length < 0:
        msg = "length must be finite and >= 0"
        raise ValueError(msg)
    if np.any(g.real < -1e-9 * np.abs(g)):
        msg = "Re(gamma) < 0: the mode grows along +z (check the sign convention)"
        raise ValueError(msg)
    nf, n = g.shape
    e = np.exp(-g * length)
    s = np.zeros((nf, 2 * n, 2 * n), dtype=complex)
    idx = np.arange(n)
    s[:, idx, n + idx] = e
    s[:, n + idx, idx] = e
    return s


def degenerate_mode_groups(
    n_eff: ArrayLike, *, rtol: float = 1e-6
) -> list[tuple[int, ...]]:
    """Group modes whose propagation constants agree.

    Modes ``i`` and ``j`` are linked when
    ``|n_i - n_j| <= rtol * max(|n_i|, |n_j|)``; groups are the connected
    components (single linkage). Within a group the terminal S does not depend
    on the basis chosen for the modes.

    Args:
        n_eff: Effective indices (or propagation constants) at one frequency,
            shape ``(n,)``.
        rtol: Relative tolerance.

    Returns:
        Groups of mode indices, each sorted, ordered by smallest index.

    Raises:
        ValueError: If ``n_eff`` is not 1-D or not finite.
    """
    n = np.asarray(n_eff, dtype=complex)
    if n.ndim != 1:
        msg = f"n_eff must be 1-D, got shape {n.shape}"
        raise ValueError(msg)
    _check_finite(n, "n_eff")
    count = n.shape[0]
    parent = list(range(count))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(count):
        for j in range(i + 1, count):
            scale = max(abs(n[i]), abs(n[j]))
            if abs(n[i] - n[j]) <= rtol * scale:
                parent[find(j)] = find(i)
    groups: dict[int, list[int]] = {}
    for i in range(count):
        groups.setdefault(find(i), []).append(i)
    return sorted((tuple(g) for g in groups.values()), key=lambda g: g[0])

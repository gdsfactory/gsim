"""Helpers on the terminal network: shorting terminals and mixed-mode pairs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray
    from skrf import Network


def _named_ports(
    net: Network, names: Sequence[str], what: str
) -> tuple[list[str], list[int]]:
    """The port names of a network and the positions of the named ports.

    Raises:
        ValueError: If the network has no port names, a name is unknown or a
            name is repeated.
    """
    known = [str(n) for n in (net.port_names or [])]
    if len(known) != net.nports:
        raise ValueError(f"{what} needs a network with one name per port.")
    if len(set(names)) != len(list(names)):
        raise ValueError(f"{what}: a port name is repeated in {list(names)}.")
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"{what}: unknown ports {unknown}; the network has {known}.")
    return known, [known.index(n) for n in names]


def _real_z0(net: Network, what: str) -> NDArray[np.float64]:
    """The reference impedances of a network, which must be real."""
    z0 = np.asarray(net.z0)
    if np.any(z0.imag != 0.0):
        raise ValueError(f"{what}: complex port impedances are not supported.")
    return np.asarray(z0.real, dtype=float)


def short_terminals(net: Network, names: Sequence[str]) -> Network:
    """Short the named terminals to the reference: ``V = 0`` on them.

    The result is the network seen from the remaining terminals when the named
    ones are tied to the reference conductor. It equals the sub-block of the
    admittance matrix of the remaining terminals, computed here directly on
    the S-parameters (a short is the reflection ``-1`` for real reference
    impedances): ``S' = S_kk - S_kn (I + S_nn)^-1 S_nk`` with ``n`` the shorted
    and ``k`` the kept ports. That also works when ``Y`` does not exist.

    Shorting terminals at the ports of a line ties them at its two ends only,
    not along its length; the two agree for an electrically short line.

    Args:
        net: A network with named ports and real reference impedances.
        names: The terminals to short, at least one and not all.

    Returns:
        A new network with the other ports, in their order, named as before.

    Raises:
        ValueError: For unknown, repeated or all port names, complex reference
            impedances, or a set of shorted ports whose ``I + S_nn`` is
            singular.
    """
    import skrf as rf

    known, short = _named_ports(net, names, "short_terminals")
    if not short:
        raise ValueError("short_terminals needs at least one port name.")
    keep = [i for i in range(net.nports) if i not in short]
    if not keep:
        raise ValueError("short_terminals cannot short every port.")
    z0 = _real_z0(net, "short_terminals")
    s = np.asarray(net.s, dtype=np.complex128)
    s_kk = s[:, keep][:, :, keep]
    s_kn = s[:, keep][:, :, short]
    s_nk = s[:, short][:, :, keep]
    s_nn = s[:, short][:, :, short]
    try:
        solved = np.linalg.solve(np.eye(len(short)) + s_nn, s_nk)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "The shorted ports cannot be tied to the reference: I + S_nn is singular."
        ) from error
    reduced = s_kk - s_kn @ solved
    out = rf.Network(frequency=net.frequency, s=reduced, z0=z0[:, keep])
    out.s_def = net.s_def
    out.port_names = [known[i] for i in keep]
    return out


def mixed_mode(
    net: Network,
    pairs: Sequence[tuple[str, str]],
    *,
    z0_mm: ArrayLike | None = None,
) -> Network:
    """Mixed-mode network of terminal pairs, by name.

    Wraps ``Network.se2gmm``, which works in place on ports in a fixed order and
    returns ``None``: the ports are taken by name, put in the order scikit-rf
    wants (pair 1, pair 2, ..., the others), converted on a copy, and returned.
    The result has the ``p`` differential ports first, the ``p`` common-mode
    ports next and the unpaired terminals last; scikit-rf names the mixed ports
    ``d(<first>,<second>)`` and ``c(<first>,<second>)``. The differential
    voltage is ``first - second`` (swap a pair to flip the sign of ``S_dc``).

    By default the reference impedances are twice (differential) and half
    (common) the single-ended one of the pair.

    Both terminals of a pair must have the same reference impedance (at every
    frequency). The transform written here, ``d = (a - b) / sqrt(2)`` and
    ``c = (a + b) / sqrt(2)`` on the waves, is exact only then; for two
    different impedances it ignores the mismatch between them (a matched
    network with ``z0 = (50, 75)`` and ``S = 0`` would come out as ``S = 0``
    where ``Network.se2gmm`` gives ``S_dd = S_cc = -0.0101``), so such a pair is
    refused rather than converted wrongly. Renormalise the network to one
    impedance per pair first (``Network.renormalize``), or choose ``z_ref``
    equal on the two terminals of a pair in :func:`build_terminal_s`.

    Args:
        net: A network with named ports.
        pairs: ``(first, second)`` port names per differential port.
        z0_mm: Optional mixed-mode reference impedances, as in ``se2gmm``.

    Returns:
        A new mixed-mode network; ``net`` is unchanged.

    Raises:
        ValueError: For no pairs, a pair that is not two names, unknown or
            repeated names, complex reference impedances, or a pair whose two
            terminals have different reference impedances.
    """
    import skrf as rf

    pair_list = [tuple(pair) for pair in pairs]
    if not pair_list or any(len(pair) != 2 for pair in pair_list):
        raise ValueError("mixed_mode needs one or more (first, second) pairs of names.")
    flat = [name for pair in pair_list for name in pair]
    known, idx = _named_ports(net, flat, "mixed_mode")
    rest = [i for i in range(net.nports) if i not in idx]
    z0 = _real_z0(net, "mixed_mode")
    unequal = []
    for ia, ib in zip(idx[0::2], idx[1::2], strict=True):
        differ = np.flatnonzero(~np.isclose(z0[:, ia], z0[:, ib], rtol=1e-9, atol=0.0))
        if differ.size:
            first = int(differ[0])  # the first frequency at which they differ
            unequal.append(
                f"({known[ia]}, {known[ib]}): "
                f"{z0[first, ia]:g} and {z0[first, ib]:g} ohm"
            )
    if unequal:
        raise ValueError(
            "mixed_mode: the two terminals of a pair must have the same reference "
            f"impedance, but they differ for {'; '.join(unequal)}. The transform "
            "is exact only for equal impedances; renormalize the network to one "
            "impedance per pair first (Network.renormalize)."
        )
    # The transform is written out instead of calling Network.se2gmm, whose
    # pairing and port naming differ between scikit-rf versions. For a pair
    # (a, b) the waves are d = (a - b)/sqrt(2) and c = (a + b)/sqrt(2); with
    # the reference impedances 2 z0 and z0/2 this is exact when a and b share
    # z0, which is checked above.
    n_pairs, n = len(pair_list), net.nports
    m = np.zeros((n, n))
    z0_out = np.zeros((z0.shape[0], n))
    names = []
    for k, (ia, ib) in enumerate(zip(idx[0::2], idx[1::2], strict=True)):
        m[k, ia], m[k, ib] = 1 / np.sqrt(2), -1 / np.sqrt(2)
        m[n_pairs + k, ia], m[n_pairs + k, ib] = 1 / np.sqrt(2), 1 / np.sqrt(2)
        mean = z0[:, ia]
        z0_out[:, k], z0_out[:, n_pairs + k] = 2 * mean, mean / 2
        names.append((k, f"d({known[ia]},{known[ib]})"))
        names.append((n_pairs + k, f"c({known[ia]},{known[ib]})"))
    for j, i in enumerate(rest):
        m[2 * n_pairs + j, i] = 1.0
        z0_out[:, 2 * n_pairs + j] = z0[:, i]
        names.append((2 * n_pairs + j, known[i]))
    s_mm = np.einsum("ij,fjk,lk->fil", m, np.asarray(net.s), m)
    out = rf.Network(frequency=net.frequency, s=s_mm, z0=z0_out)
    out.s_def = net.s_def
    out.port_names = [name for _, name in sorted(names)]
    if getattr(out, "port_modes", None) is not None:  # scikit-rf 2.x only
        out.port_modes = np.array(["D"] * n_pairs + ["C"] * n_pairs + ["S"] * len(rest))
    if z0_mm is not None:
        out.renormalize(np.asarray(z0_mm, dtype=complex))
    return out

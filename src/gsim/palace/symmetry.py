"""Post-processing helpers for symmetry-plane (half-model) results.

A half model's S-parameters are the full structure's modal S-parameters as
they are; no scaling is applied. A PMC plane gives the even/common mode, a PEC
plane the odd/differential mode. See ``SYMMETRY.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gsim.palace.results import SParam, SParams

if TYPE_CHECKING:
    from collections.abc import Mapping


def full_model_impedance(z_half: float, kind: str) -> float:
    """Convert a half-model impedance to the full-model mode impedance.

    Args:
        z_half: Per-line characteristic impedance of the half model in ohm.
        kind: ``"pec"`` (odd mode) or ``"pmc"`` (even mode).

    Returns:
        ``2 * z_half`` (differential impedance) for ``"pec"`` and
        ``z_half / 2`` (common-mode impedance) for ``"pmc"``.

    Raises:
        ValueError: If *kind* is not ``"pec"`` or ``"pmc"``.
    """
    if kind == "pec":
        return 2.0 * z_half
    if kind == "pmc":
        return z_half / 2.0
    raise ValueError(f"kind must be 'pec' or 'pmc', got {kind!r}")


def _check_halves(even: SParams, odd: SParams) -> None:
    """Raise ``ValueError`` unless *even*/*odd* are matching PMC/PEC half models."""
    if even.symmetry is None or odd.symmetry is None:
        raise ValueError(
            "Both results must come from half models: no symmetry record found "
            "in port_information.json."
        )
    if even.symmetry["kind"] == odd.symmetry["kind"]:
        raise ValueError(
            "The two half models must have opposite plane kinds (one PMC, one PEC)."
        )
    if even.symmetry["kind"] != "pmc":
        raise ValueError(
            "The first argument must be the PMC (even) half and the second the "
            "PEC (odd) half."
        )
    for key in ("axis", "position", "keep"):
        if even.symmetry.get(key) != odd.symmetry.get(key):
            raise ValueError(
                f"The half models use different planes ({key}: "
                f"{even.symmetry.get(key)!r} vs {odd.symmetry.get(key)!r})."
            )
    if even.port_names != odd.port_names:
        raise ValueError(
            f"The half models have different ports: {even.port_names} vs "
            f"{odd.port_names}."
        )
    if len(even.freq) != len(odd.freq) or not np.allclose(even.freq, odd.freq):
        raise ValueError("The half models have different frequencies.")
    for name in even.port_names:
        type_even = even.port_meta.get(name, {}).get("type")
        type_odd = odd.port_meta.get(name, {}).get("type")
        if type_even != type_odd:
            raise ValueError(
                f"Port '{name}' has different port types in the half models "
                f"({type_even} vs {type_odd})."
            )
    for name in even.port_names:
        z_even = even.port_meta.get(name, {}).get("Z0")
        z_odd = odd.port_meta.get(name, {}).get("Z0")
        if z_even != z_odd:
            raise ValueError(
                f"Port '{name}' has different reference impedance in the half "
                f"models ({z_even} vs {z_odd})."
            )


def mixed_mode_from_halves(even: SParams, odd: SParams) -> dict:
    """Pair an even (PMC) and an odd (PEC) half model as mixed-mode results.

    Args:
        even: Result of the PMC half model (common mode).
        odd: Result of the PEC half model (differential mode).

    Returns:
        ``{"cc": even, "dd": odd, "z_ref_cc": ..., "z_ref_dd": ...}``. For
        lumped ports the references are per-port dicts: ``R/2`` for the
        common mode and ``2R`` for the differential mode, with R the port
        impedance of the half model. Wave-port references are ``"modal"``
        (the string, if all ports are wave ports).

    Raises:
        ValueError: If the results are not matching PMC/PEC half models.
    """
    _check_halves(even, odd)

    z_cc: dict[str, object] = {}
    z_dd: dict[str, object] = {}
    for name in even.port_names:
        meta = even.port_meta.get(name, {})
        z0 = meta.get("Z0")
        if meta.get("type") == "waveport":
            z_cc[name] = z_dd[name] = "modal"
        else:
            z_cc[name] = None if z0 is None else z0 / 2.0
            z_dd[name] = None if z0 is None else 2.0 * z0

    if z_cc and all(v == "modal" for v in z_cc.values()):
        return {"cc": even, "dd": odd, "z_ref_cc": "modal", "z_ref_dd": "modal"}
    return {"cc": even, "dd": odd, "z_ref_cc": z_cc, "z_ref_dd": z_dd}


def _to_sparam(values: np.ndarray) -> SParam:
    """Build an :class:`SParam` from complex values."""
    magnitude = np.maximum(np.abs(values), np.finfo(float).tiny)
    return SParam(db=20 * np.log10(magnitude), deg=np.rad2deg(np.angle(values)))


def combine_even_odd(
    even: SParams,
    odd: SParams,
    mirror_names: Mapping[str, str] | None = None,
) -> SParams:
    """Combine PMC and PEC half models into the single-ended 2N-port result.

    With half-model ports i, j, their mirror images i', j' and ``cc``/``dd``
    the even/odd results::

        S_ij   = S_i'j' = (cc_ij + dd_ij) / 2
        S_ij'  = S_i'j  = (cc_ij - dd_ij) / 2

    Only the S-parameters present in both inputs (the excited columns) are
    combined. Conversion between the modes is zero by construction.

    Args:
        even: Result of the PMC half model.
        odd: Result of the PEC half model.
        mirror_names: Map from port name to its mirror image's name. Defaults
            to ``f"{name}_mirror"``.

    Returns:
        :class:`SParams` with the original ports followed by their mirrors.

    Raises:
        ValueError: For mismatched half models (including different port
            types), wave ports (a full model with wave ports is modal anyway),
            unknown ``mirror_names`` keys or clashing mirror names.
    """
    _check_halves(even, odd)
    names = even.port_names
    if any(even.port_meta.get(n, {}).get("type") == "waveport" for n in names):
        raise ValueError(
            "combine_even_odd supports lumped ports only; a full model with wave "
            "ports is modal, run it with max_size wave ports instead."
        )

    mirrors = {n: f"{n}_mirror" for n in names}
    if mirror_names:
        unknown = [k for k in mirror_names if k not in mirrors]
        if unknown:
            raise ValueError(
                f"mirror_names has keys that are not ports of the half models: "
                f"{unknown}; ports are {names}."
            )
        mirrors.update(mirror_names)
    new_names = list(mirrors.values())
    if len(set(new_names)) != len(new_names) or set(new_names) & set(names):
        raise ValueError(
            f"Invalid mirror port names {new_names}: they must be unique and "
            f"differ from the existing ports {names}."
        )

    data: dict[tuple[str, str], SParam] = {}
    even_pairs = even.keys()  # SParams.keys() is a method, not a dict view
    odd_pairs = set(odd.keys())
    for to, frm in even_pairs:
        if (to, frm) not in odd_pairs:
            continue
        cc = even[to, frm].complex
        dd = odd[to, frm].complex
        same = _to_sparam((cc + dd) / 2)
        cross = _to_sparam((cc - dd) / 2)
        data[(to, frm)] = same
        data[(mirrors[to], mirrors[frm])] = same
        data[(to, mirrors[frm])] = cross
        data[(mirrors[to], frm)] = cross

    port_meta = dict(even.port_meta)
    port_meta.update(
        {mirrors[n]: dict(even.port_meta[n]) for n in names if n in even.port_meta}
    )
    return SParams(
        freq=even.freq,
        data=data,
        port_names=[*names, *new_names],
        port_meta=port_meta,
    )


__all__ = ["combine_even_odd", "full_model_impedance", "mixed_mode_from_halves"]

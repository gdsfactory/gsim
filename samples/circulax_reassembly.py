"""Reassemble gsim's modulator exports with nothing but numpy.

This is the reference for the consuming side of gsim's compact-model
handoff (circulax, or any circuit tool): it reads the two files a Study
exports and wires them together with plain network math — no gsim, no
circulax, no sax imported anywhere. Copy it as a starting point.

The two artifacts, produced on the gsim side by::

    study.line.export_touchstone()          # -> electrode.s2p
    study.charge.export_junction_model()    # -> junction.json

are meant to be wired as follows:

- ``electrode.s2p`` is the Traveling-wave electrode at the solved bias
  as a two-port (Touchstone v1, ``# Hz S RI R <z_ref>``). For the RF
  drive path, place it between the generator and the termination; the
  driven response below is exactly that wiring.
- ``junction.json`` is the phase shifter's junction as a series-RC
  shunt branch per meter of electrode, tabulated over the bias sweep
  (see ``gsim.common.circuit.write_junction_model`` for the format).
  Its per-meter shunt admittance at frequency ``f`` and bias ``V`` is
  ``Y_j(f, V) = j*w*C_j / (1 + j*w*R_s*C_j)`` with ``w = 2*pi*f`` —
  the branch a circuit tool inserts per unit length to re-bias the
  line, or lumps into ``C_j*L`` for driver design.

Run it against a Study's output directory::

    python circulax_reassembly.py electrode.s2p junction.json \
        --z-gen 50 --z-load 50
"""

import argparse
import json
from pathlib import Path

import numpy as np


def read_touchstone(path):
    """Read a two-port ``# Hz S RI R <z_ref>`` Touchstone v1 file.

    Args:
        path: The ``.s2p`` file gsim wrote.

    Returns:
        ``(freq_hz, s, z_ref_ohm)`` with ``s`` shaped ``(len(freq), 2, 2)``.
    """
    z_ref = 50.0
    rows = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        if line.startswith("#"):
            tokens = line[1:].upper().split()
            if tokens[:3] != ["HZ", "S", "RI"]:
                raise ValueError(f"unsupported Touchstone flavor: {line}")
            if len(tokens) > 4 and tokens[3] == "R":
                z_ref = float(tokens[4])
            continue
        rows.append([float(token) for token in line.split()])
    data = np.asarray(rows)
    s = np.empty((data.shape[0], 2, 2), dtype=complex)
    # Touchstone two-port order: S11, S21, S12, S22.
    s[:, 0, 0] = data[:, 1] + 1j * data[:, 2]
    s[:, 1, 0] = data[:, 3] + 1j * data[:, 4]
    s[:, 0, 1] = data[:, 5] + 1j * data[:, 6]
    s[:, 1, 1] = data[:, 7] + 1j * data[:, 8]
    return data[:, 0], s, z_ref


def read_junction_model(path):
    """Read a ``gsim-junction-model`` JSON file.

    Args:
        path: The ``junction.json`` file gsim wrote.

    Returns:
        The parsed payload with the three columns as numpy arrays.
    """
    payload = json.loads(Path(path).read_text())
    if payload.get("format") != "gsim-junction-model":
        raise ValueError(f"{path} is not a gsim-junction-model file")
    for column in ("bias_v", "r_s_ohm_m", "c_j_f_per_m"):
        payload[column] = np.asarray(payload[column])
    return payload


def driven_response(s, z_ref_ohm, *, z_gen_ohm=50.0, z_load_ohm=50.0):
    """``V_load / V_gen``: ideal generator, the two-port, a load.

    Args:
        s: S-matrix per frequency, shaped ``(n, 2, 2)``.
        z_ref_ohm: Real reference impedance of the S-parameters.
        z_gen_ohm: Generator impedance.
        z_load_ohm: Load impedance.

    Returns:
        The complex voltage transfer per frequency.
    """
    s11, s12 = s[:, 0, 0], s[:, 0, 1]
    s21, s22 = s[:, 1, 0], s[:, 1, 1]
    # S -> ABCD at a real reference, then close the ports.
    a = ((1 + s11) * (1 - s22) + s12 * s21) / (2 * s21)
    b = z_ref_ohm * ((1 + s11) * (1 + s22) - s12 * s21) / (2 * s21)
    c = ((1 - s11) * (1 - s22) - s12 * s21) / (z_ref_ohm * 2 * s21)
    d = ((1 - s11) * (1 + s22) + s12 * s21) / (2 * s21)
    return z_load_ohm / (a * z_load_ohm + b + z_gen_ohm * (c * z_load_ohm + d))


def junction_shunt_admittance(freq_hz, r_s_ohm_m, c_j_f_per_m):
    """The junction's shunt admittance per meter of electrode (S/m).

    Args:
        freq_hz: Frequency (Hz), scalar or array.
        r_s_ohm_m: Series resistance from the model file (ohm*m).
        c_j_f_per_m: Junction capacitance from the model file (F/m).

    Returns:
        ``Y_j = j*w*C_j / (1 + j*w*R_s*C_j)``.
    """
    omega = 2 * np.pi * np.asarray(freq_hz)
    return 1j * omega * c_j_f_per_m / (1 + 1j * omega * r_s_ohm_m * c_j_f_per_m)


def main(argv=None):
    """Print the reassembled driven response and the junction table.

    Args:
        argv: Command-line arguments; ``sys.argv`` when omitted.

    Returns:
        Exit code 0.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("touchstone", help="electrode.s2p from the line stage")
    parser.add_argument("junction", help="junction.json from the charge stage")
    parser.add_argument("--z-gen", type=float, default=50.0)
    parser.add_argument("--z-load", type=float, default=50.0)
    args = parser.parse_args(argv)

    freq_hz, s, z_ref = read_touchstone(args.touchstone)
    response = driven_response(s, z_ref, z_gen_ohm=args.z_gen, z_load_ohm=args.z_load)
    print(f"Driven response, Z_gen = {args.z_gen} ohm, Z_load = {args.z_load} ohm:")
    print(f"{'f (GHz)':>10} {'|V_load/V_gen|':>16} {'phase (deg)':>12}")
    for f, h in zip(freq_hz, response, strict=True):
        print(f"{f / 1e9:>10.3f} {abs(h):>16.6e} {np.degrees(np.angle(h)):>12.2f}")

    model = read_junction_model(args.junction)
    print(
        f"\nJunction branch on contact {model['contact']!r} "
        f"(fitted at {model['freq_hz'] / 1e9:g} GHz):"
    )
    print(f"{'bias (V)':>10} {'R_s (ohm*m)':>14} {'C_j (F/m)':>14}")
    for v, r, c in zip(
        model["bias_v"], model["r_s_ohm_m"], model["c_j_f_per_m"], strict=True
    ):
        print(f"{v:>10.3f} {r:>14.6e} {c:>14.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Palace verification against an exactly solvable TEM line.

The fixture is an ideal homogeneous box with PEC plates and PMC sidewalls, so
the uniform-line impedance, delay, and complex S-parameters are known in closed
form.  It checks the solver and the S-parameter/TDR chain independently of the
IHP device.

These tests are marked ``palace_local`` because each selected case launches
Palace.  They are deselected by default; run them with::

    pytest tests/palace/test_tem_line.py -m palace_local

Run a single profile by node id if the full 15-case sweep is too large.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c as light_speed
from scipy.constants import epsilon_0, mu_0

from gsim.palace.signal_integrity import line_network, tdr
from gsim.palace.validation import validate_solution

TEM_EPS = 4.1
TEM_ZC = np.sqrt(mu_0 / (epsilon_0 * TEM_EPS)) * 0.5


@pytest.fixture(scope="module")
def palace_binary() -> Path:
    """Return a runnable Palace binary, or skip the test module."""
    from gsim.palace.runtime import resolve_palace_binary

    binary = resolve_palace_binary(download_if_missing=False)
    if binary is None:
        pytest.skip("No runnable Palace binary; set PALACE_EXECUTABLE or PALACE_BIN")
    return Path(binary)


def _tem_write_box(
    path: Path,
    kind: str,
    hx: float,
    L: float = 1000.0,
    W: float = 20.0,
    H: float = 10.0,
) -> int:
    """Write a structured Gmsh 2.2 box with one of three element types."""
    if kind not in {"tet", "prism", "hex"} or min(hx, L, W, H) <= 0:
        raise ValueError("Choose a supported element and positive dimensions")
    xs = np.linspace(0, L, int(np.ceil(L / hx)) + 1)
    ys = np.linspace(-W / 2, W / 2, 3)
    zs = np.linspace(0, H, 3)
    nx = len(xs) - 1
    grid = np.arange(1, (nx + 1) * 9 + 1).reshape(nx + 1, 3, 3)
    xyz = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    cells: list[tuple[int, list[int]]] = []
    for i in range(nx):
        for j in range(2):
            for k in range(2):
                n = [
                    grid[i, j, k],
                    grid[i + 1, j, k],
                    grid[i + 1, j + 1, k],
                    grid[i, j + 1, k],
                    grid[i, j, k + 1],
                    grid[i + 1, j, k + 1],
                    grid[i + 1, j + 1, k + 1],
                    grid[i, j + 1, k + 1],
                ]
                if kind == "hex":
                    cells.append((5, n))
                elif kind == "tet":
                    for ids in [
                        (0, 1, 2, 6),
                        (0, 2, 3, 6),
                        (0, 3, 7, 6),
                        (0, 7, 4, 6),
                        (0, 4, 5, 6),
                        (0, 5, 1, 6),
                    ]:
                        cells.append((4, [n[q] for q in ids]))  # noqa: PERF401
                else:
                    # Split the y-z cross-section along its positive diagonal.
                    a = [
                        grid[i, j, k],
                        grid[i, j + 1, k],
                        grid[i, j + 1, k + 1],
                        grid[i, j, k + 1],
                    ]
                    b = [
                        grid[i + 1, j, k],
                        grid[i + 1, j + 1, k],
                        grid[i + 1, j + 1, k + 1],
                        grid[i + 1, j, k + 1],
                    ]
                    for ids in [(0, 1, 2), (0, 2, 3)]:
                        cells.append(  # noqa: PERF401
                            (6, [a[q] for q in ids] + [b[q] for q in ids])
                        )

    faces: dict[tuple[int, ...], list[int] | None] = {}
    templates = {
        5: [
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ],
        4: [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)],
        6: [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],
    }
    for typ, n in cells:
        for template in templates[typ]:
            face = [n[q] for q in template]
            key = tuple(sorted(face))
            faces[key] = None if key in faces else face

    boundaries: list[tuple[int, int, list[int]]] = []
    for face in faces.values():
        if face is None:
            continue
        points = xyz[np.asarray(face) - 1]
        if np.allclose(points[:, 0], 0):
            tag = 4
        elif np.allclose(points[:, 0], L):
            tag = 5
        elif np.allclose(points[:, 2], 0) or np.allclose(points[:, 2], H):
            tag = 2
        else:
            tag = 3
        boundaries.append((2 if len(face) == 3 else 3, tag, face))

    with path.open("w") as file:
        file.write(
            "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n5\n"
            '3 1 "dielectric"\n2 2 "plates_PEC"\n2 3 "sides_PMC"\n'
            '2 4 "near"\n2 5 "far"\n$EndPhysicalNames\n'
        )
        file.write(f"$Nodes\n{len(xyz)}\n")
        for node_id, (x, y, z) in enumerate(xyz, 1):
            file.write(f"{node_id} {x:.12g} {y:.12g} {z:.12g}\n")
        file.write(f"$EndNodes\n$Elements\n{len(cells) + len(boundaries)}\n")
        element_id = 0
        for typ, tag, vertices in boundaries:
            element_id += 1
            file.write(
                f"{element_id} {typ} 2 {tag} {tag} "
                + " ".join(map(str, vertices))
                + "\n"
            )
        for typ, vertices in cells:
            element_id += 1
            file.write(
                f"{element_id} {typ} 2 1 1 " + " ".join(map(str, vertices)) + "\n"
            )
        file.write("$EndElements\n")
    return len(cells)


def _tem_config(order: int, zref: float, *, dense: bool = False) -> dict:
    driven: dict[str, object] = {
        "Samples": [{"Type": "Point", "Freq": [25.0, 112.5, 200.0]}]
    }
    if dense:
        driven = {
            "Samples": [
                {
                    "Type": "Linear",
                    "MinFreq": 0.25,
                    "MaxFreq": 200.0,
                    "FreqStep": 0.25,
                    "SaveStep": 0,
                }
            ],
            "AdaptiveTol": 1e-7,
            "AdaptiveMaxSamples": 80,
        }
    return {
        "Problem": {"Type": "Driven", "Verbose": 1, "Output": "output/palace"},
        "Model": {"Mesh": "palace.msh", "L0": 1e-6, "Refinement": {"MaxIts": 0}},
        "Domains": {
            "Materials": [{"Attributes": [1], "Permittivity": TEM_EPS, "LossTan": 0.0}]
        },
        "Boundaries": {
            "PEC": {"Attributes": [2]},
            "LumpedPort": [
                {
                    "Index": index,
                    "Direction": "Z",
                    "Excitation": index,
                    "Attributes": [index + 3],
                    "R": zref,
                }
                for index in (1, 2)
            ],
        },
        "Solver": {
            "Order": order,
            "Linear": {
                "Type": "MUMPS",
                "KSPType": "GMRES",
                "Tol": 1e-11,
                "MaxIts": 4,
                "MGMaxLevels": 1,
                "PCMatReal": False,
                "ComplexCoarseSolve": True,
                "EstimatorMaxIts": 0,
                "DivFreeMaxIts": 0,
            },
            "Driven": driven,
        },
    }


def _tem_launch(directory: Path, config: dict, executable: Path) -> None:
    """Write the config and run one Palace process in ``directory``."""
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    env = os.environ.copy()
    runtime_lib = executable.resolve().parent.parent / "lib"
    if runtime_lib.is_dir():
        env["LD_LIBRARY_PATH"] = (
            str(runtime_lib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        )
    env["OPENBLAS_NUM_THREADS"] = "1"
    with (directory / "solver.log").open("w") as log:
        result = subprocess.run(  # noqa: S603 -- explicitly selected Palace binary
            [str(executable), "-np", "1", "-nt", "1", "config.json"],
            cwd=directory,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"Palace exited {result.returncode}; inspect {directory}")
    _tem_validate(directory, adaptive="AdaptiveTol" in config["Solver"]["Driven"])


def _tem_validate(directory: Path, *, adaptive: bool = False) -> None:
    report = validate_solution(
        (directory / "solver.log").read_text(),
        estimator_disabled=True,
        exit_code=0,
    )
    assert report.core.status == "passed", report.core
    if adaptive:
        assert len(report.adaptive) == 1
        assert report.adaptive[0].check.status == "passed"


def _tem_read_columns(
    directory: Path, sources: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    # Read the Palace CSV through pandas so the column names are stripped
    # exactly like the Palace writer emitted them.
    import pandas as pd

    table = pd.read_csv(directory / "output/palace/port-S.csv")
    table.columns = table.columns.str.strip()
    columns = np.empty((len(table), 2, len(sources)), complex)
    for j, source in enumerate(sources):
        for response in (1, 2):
            amplitude = 10 ** (table[f"|S[{response}][{source}]| (dB)"].to_numpy() / 20)
            phase = np.deg2rad(table[f"arg(S[{response}][{source}]) (deg.)"].to_numpy())
            columns[:, response - 1, j] = amplitude * np.exp(1j * phase)
    assert np.isfinite(columns).all()
    return table.iloc[:, 0].to_numpy() * 1e9, columns


def _tem_short_case(
    directory: Path, kind: str, order: int, hx: float, executable: Path
) -> dict[str, float | int | str]:
    directory.mkdir(parents=True)
    cells = _tem_write_box(directory / "palace.msh", kind, hx)
    _tem_launch(directory, _tem_config(order, TEM_ZC), executable)
    f, s = _tem_read_columns(directory, (1, 2))
    assert s.shape == (3, 2, 2)
    np.testing.assert_allclose(f, np.array([25, 112.5, 200]) * 1e9)
    exact = line_network(f, TEM_ZC, 1e-3 * np.sqrt(TEM_EPS) / light_speed, TEM_ZC).s
    stats = json.loads((directory / "output/palace/palace.json").read_text())["Problem"]
    return {
        "element": kind,
        "order": order,
        "hx_um": hx,
        "cells": cells,
        "unknowns": stats["DegreesOfFreedom"],
        "max complex S error": float(np.max(np.abs(s - exact))),
    }


def test_tem_config_and_analytic_reference() -> None:
    """The fixture config is stable and the analytic reference is well formed."""
    config = _tem_config(2, TEM_ZC)
    assert config["Boundaries"]["PEC"] == {"Attributes": [2]}
    assert [port["Attributes"] for port in config["Boundaries"]["LumpedPort"]] == [
        [4],
        [5],
    ]
    assert config["Solver"]["Order"] == 2

    f = np.array([25e9, 112.5e9, 200e9])
    exact = line_network(f, TEM_ZC, 1e-3 * np.sqrt(TEM_EPS) / light_speed, TEM_ZC)
    assert exact.s.shape == (3, 2, 2)
    np.testing.assert_allclose(exact.s[:, 0, 0], exact.s[:, 1, 1])
    np.testing.assert_allclose(exact.s[:, 0, 1], exact.s[:, 1, 0])


@pytest.mark.palace_local
@pytest.mark.parametrize(
    ("kind", "order"),
    [
        ("tet", 1),
        ("tet", 2),
        ("prism", 1),
        ("prism", 2),
        ("hex", 1),
        ("hex", 2),
    ],
)
def test_tem_short_refinement(
    tmp_path: Path,
    palace_binary: Path,
    kind: str,
    order: int,
) -> None:
    """Each element type/order converges against the exact line."""
    spacings = [50.0, 25.0, 12.5] if order == 1 else [50.0, 25.0]
    rows = [
        _tem_short_case(
            tmp_path / f"{kind}-p{order}-hx{hx:g}",
            kind,
            order,
            hx,
            palace_binary,
        )
        for hx in spacings
    ]
    errors = [float(row["max complex S error"]) for row in rows]
    assert all(np.isfinite(errors))
    assert errors[-1] <= errors[0], (kind, order, errors)


@pytest.mark.palace_local
def test_tem_dense_tdr(
    tmp_path: Path,
    palace_binary: Path,
) -> None:
    """The dense line recovers its known impedance and delay through TDR."""
    directory = tmp_path / "dense"
    directory.mkdir()
    _tem_write_box(directory / "palace.msh", "prism", 25.0, L=6000.0)

    columns = []
    for source in (1, 2):
        chunk = directory / f"source{source}"
        chunk.mkdir()
        shutil.copyfile(directory / "palace.msh", chunk / "palace.msh")
        config = _tem_config(2, 50.0, dense=True)
        for port in config["Boundaries"]["LumpedPort"]:
            if port["Index"] != source:
                port.pop("Excitation")
        _tem_launch(chunk, config, palace_binary)
        f, column = _tem_read_columns(chunk, (source,))
        columns.append((f, column))

    np.testing.assert_allclose(columns[0][0], columns[1][0], rtol=0, atol=1)
    f = columns[0][0]
    s = np.concatenate([column for _, column in columns], axis=2)
    assert s.shape == (800, 2, 2)
    np.testing.assert_allclose(f, np.arange(1, 801) * 0.25e9)

    delay = 0.006 * np.sqrt(TEM_EPS) / light_speed
    exact = line_network(f, TEM_ZC, delay, 50.0).s
    t, _, impedance = tdr(f, s[:, 0, 0], 50.0)
    resolution = 2 / f[-1]
    plateau = (t > 2.5 * resolution) & (t < 2 * delay - 2.5 * resolution)
    assert plateau.any()
    impedance_estimate = np.median(impedance[plateau])

    # For this ideal PEC line S21(0)=1 exactly; reflection uses the general DC rule.
    import skrf as rf

    through = rf.Network(
        f=np.r_[0.0, f],
        s=np.r_[1.0, s[:, 1, 0]][:, None, None],
        z0=50.0,
        f_unit="Hz",
    )
    t_impulse, impulse = through.impulse_response(window="hamming", bandpass=False)
    delay_estimate = t_impulse[np.argmax(np.abs(impulse))]

    assert np.max(np.abs(s - exact)) < 1e-3
    assert abs(impedance_estimate / TEM_ZC - 1) < 0.02
    assert abs(delay_estimate - delay) < resolution

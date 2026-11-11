"""Validate T-coil circuit closure and winner artifacts without cloud jobs."""

# ruff: noqa: S102
# Copyright (c) 2026 gdsfactory

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import least_squares

NOTEBOOK = Path(__file__).resolve().parents[2] / "nbs/palace_tcoil.ipynb"


@pytest.fixture
def circuit():
    namespace = {"np": np, "least_squares": least_squares, "R_T": 75.0}
    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        nodes = [
            node
            for node in tree.body
            if (
                isinstance(node, ast.FunctionDef)
                and node.name
                in {
                    "razavi_targets",
                    "tcoil_y",
                    "y_to_s",
                    "terminated_response",
                    "bandwidth",
                    "low_freq_lmk",
                    "fit_tcoil",
                }
            )
            or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id in ("_NAMES", "_SCALE")
                    for t in node.targets
                )
            )
        ]
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"),
            namespace,
        )
    return namespace


@pytest.mark.parametrize("resistance", [25.0, 50.0, 75.0, 100.0])
def test_ideal_match_and_analytic_transfer(circuit, resistance):
    f = np.linspace(1e9, 150e9, 301)
    capacitance = 100e-15
    target = circuit["razavi_targets"](resistance, capacitance)
    target.pop("M")
    s = circuit["y_to_s"](circuit["tcoil_y"](f, **target), z0=resistance)
    reflection, transfer = circuit["terminated_response"](
        s, f, capacitance, z0=resistance
    )
    # Razavi pole-zero cancellation gives a second-order Butterworth response.
    scaled_s = 2j * np.pi * f * resistance * capacitance
    expected = 1 / (1 + scaled_s / 2 + scaled_s**2 / 8)
    np.testing.assert_allclose(reflection, 0, atol=2e-13)
    np.testing.assert_allclose(transfer, expected, atol=2e-13)


def test_intrinsic_fit_before_external_capacitor(circuit):
    f = np.linspace(1e9, 100e9, 150)
    params = dict(
        L1=110e-12,
        L2=95e-12,
        k=0.4,
        C_B=25e-15,
        R1=1.2,
        R2=1.5,
        C_A=8e-15,
        C_Bg=6e-15,
        C_T=10e-15,
    )
    fit = circuit["fit_tcoil"](f, circuit["tcoil_y"](f, **params), fmax=100e9)
    for name, expected in params.items():
        assert fit[name] == pytest.approx(expected, rel=1e-4)
    assert fit["rms_S"] < 1e-7


def test_notebook_passes_reference_impedance():
    for cell in json.loads(NOTEBOOK.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("y_to_s", "terminated_response")
            ):
                keyword = next(k for k in node.keywords if k.arg == "z0")
                assert isinstance(keyword.value, ast.Name)
                assert keyword.value.id == "R_T"


def test_drc_exports_simulated_frame(tmp_path):
    gf = pytest.importorskip("gdsfactory")
    ihp = pytest.importorskip("ihp")
    from gdsfactory.typings import LayerSpec

    ihp.PDK.activate()
    namespace = {"gf": gf, "Component": gf.Component, "LayerSpec": LayerSpec}
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    for cell in cells:
        if cell["cell_type"] == "code" and "def add_ground_frame(" in "".join(
            cell["source"]
        ):
            tree = ast.parse("".join(cell["source"]))
            exec(
                compile(
                    ast.Module(body=[tree.body[0]], type_ignores=[]),
                    str(NOTEBOOK),
                    "exec",
                ),
                namespace,
            )
    coil = gf.Component()
    coil.add_polygon([(0, 0), (0, 10), (10, 10), (10, 0)], layer="TopMetal2drawing")
    coil.add_port(
        name="P1", center=(5, 0), width=2, orientation=270, layer="TopMetal2drawing"
    )
    framed = namespace["add_ground_frame"](coil)
    gds = tmp_path / "winner.gds"
    namespace.update(winner_layout=framed, gds=gds)
    source = next(
        "".join(c["source"]) for c in cells if "DRC_DIR =" in "".join(c["source"])
    )
    tree = ast.parse(source)
    export = next(
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "write_gds"
    )
    exec(
        compile(ast.Module(body=[export], type_ignores=[]), str(NOTEBOOK), "exec"),
        namespace,
    )
    loaded = gf.import_gds(gds)
    metal1 = gf.get_layer_tuple("Metal1drawing")
    assert loaded.get_polygons(by="tuple")[metal1]
    assert loaded.dbbox() == framed.dbbox()

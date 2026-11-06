"""Exercise the inductor tutorial's broadband fit without cloud simulations."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gsim.palace.circuit import CircuitSynthesis, VectorFit, differential_impedance


@pytest.fixture
def notebook_sources():
    """Read the published tutorial's fitting and validation code."""
    path = Path(__file__).resolve().parents[2] / "nbs/palace_inductor.ipynb"
    notebook = json.loads(path.read_text(encoding="utf8"))
    return ["".join(cell["source"]) for cell in notebook["cells"]]


@pytest.fixture
def notebook_workflow(notebook_sources):
    """Run the actual notebook fit call and validation function in isolation."""
    fit_source = next(
        source for source in notebook_sources if source.startswith("# Vector-fitting")
    )
    fit_assignment = next(
        node
        for node in ast.parse(fit_source).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "fit_vf"
            for target in node.targets
        )
    )
    validation_source = next(
        source for source in notebook_sources if "def validate_broadband_fit(" in source
    )
    validation_function = next(
        node
        for node in ast.parse(validation_source).body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_broadband_fit"
    )
    namespace = {"np": np, "differential_impedance": differential_impedance}
    exec(  # noqa: S102 -- Execute repository-owned notebook code.
        compile(
            ast.Module(body=[validation_function], type_ignores=[]),
            "validation_cell",
            "exec",
        ),
        namespace,
    )

    def fit(circuit, frequencies):
        fit_namespace = {"circuit": circuit, "cs_f": frequencies}
        exec(  # noqa: S102 -- Exercise the notebook's pole order and enforcement flag.
            compile(
                ast.Module(body=[fit_assignment], type_ignores=[]), "fit_cell", "exec"
            ),
            fit_namespace,
        )
        return fit_namespace["fit_vf"]

    return fit, namespace["validate_broadband_fit"]


@pytest.fixture
def passive_two_port():
    """Three parallel series-RLC branches with a grounded RC shunt per port.

    All component values are positive, so the reference is passive. Its nodal
    circuit and closed-form differential impedance are independent references.
    The six-pole approximation needs passivity enforcement outside the band.
    """
    frequencies = np.linspace(10e9, 200e9, 201)
    resistances = np.array([4.4, 1.6, 0.48])
    inductances = np.array([158, 204, 220]) * 1e-12
    capacitances = np.array([3.26, 6.53, 1.47]) * 1e-15
    laplace = 2j * np.pi * frequencies
    branch_admittance = np.sum(
        1
        / (
            resistances[:, None]
            + laplace * inductances[:, None]
            + 1 / (laplace * capacitances[:, None])
        ),
        axis=0,
    )
    reference_z_diff = 2 / (2 * branch_admittance + 1e-4 + laplace * 1e-15)

    inverse_inductance = np.zeros((8, 8))
    conductance = np.zeros((8, 8))
    capacitance = np.zeros((8, 8))
    for branch in range(3):
        resistor_node, inductor_node = 2 + 2 * branch, 3 + 2 * branch
        for matrix, value, left, right in (
            (conductance, 1 / resistances[branch], 0, resistor_node),
            (inverse_inductance, 1 / inductances[branch], resistor_node, inductor_node),
            (capacitance, capacitances[branch], inductor_node, 1),
        ):
            matrix[left, left] += value
            matrix[right, right] += value
            matrix[left, right] -= value
            matrix[right, left] -= value
    conductance[:2, :2] += 1e-4 * np.eye(2)
    capacitance[:2, :2] += 1e-15 * np.eye(2)
    circuit = CircuitSynthesis(
        nodes=["port_1_re", "port_2_re"] + [f"sample_{index}_re" for index in range(6)],
        L_inv=inverse_inductance,
        R_inv=conductance,
        C=capacitance,
    )
    np.testing.assert_allclose(
        differential_impedance(circuit.port_impedance(frequencies)),
        reference_z_diff,
        rtol=1e-10,
    )
    return circuit, frequencies, reference_z_diff


def test_notebook_enforces_passivity_and_preserves_analytic_impedance(
    notebook_workflow, passive_two_port, monkeypatch
):
    """The tutorial's exact 2+2 fit stays passive and within its 5% error limit."""
    fit, validate = notebook_workflow
    circuit, frequencies, reference_z_diff = passive_two_port
    enforced_limits = []
    original_enforce = VectorFit.passivity_enforce

    def record_enforcement(self, **kwargs):
        enforced_limits.append(kwargs.get("f_max"))
        return original_enforce(self, **kwargs)

    monkeypatch.setattr(VectorFit, "passivity_enforce", record_enforcement)
    fitted = fit(circuit, frequencies)
    assert enforced_limits == [float(frequencies.max())]
    relative_error, max_singular_value = validate(fitted, frequencies, reference_z_diff)
    assert relative_error < 0.04
    assert max_singular_value <= 1 + 1e-6


@pytest.mark.parametrize("failure", ["disabled", "no_op"])
def test_notebook_rejects_missing_enforcement(
    notebook_workflow, passive_two_port, monkeypatch, failure
):
    """Ignoring the flag or making enforcement a no-op must fail validation."""
    fit, validate = notebook_workflow
    circuit, frequencies, reference_z_diff = passive_two_port
    if failure == "disabled":
        original_fit = circuit.fit_rlc

        def without_enforcement(*args, **kwargs):
            kwargs["enforce_passivity"] = False
            return original_fit(*args, **kwargs)

        monkeypatch.setattr(circuit, "fit_rlc", without_enforcement)
    else:
        monkeypatch.setattr(VectorFit, "passivity_enforce", lambda self, **kwargs: self)
    fitted = fit(circuit, frequencies)
    with pytest.raises(AssertionError, match="passivity test"):
        validate(fitted, frequencies, reference_z_diff)


@pytest.mark.parametrize("order", [(4, 0), (1, 1)])
def test_notebook_rejects_inaccurate_lower_order_fit(
    notebook_workflow, passive_two_port, order
):
    """Orders used before the fix do not meet the independent impedance limit."""
    _, validate = notebook_workflow
    circuit, frequencies, reference_z_diff = passive_two_port
    fitted = circuit.fit_rlc(
        frequencies,
        model="vector_fit",
        n_poles_real=order[0],
        n_poles_cmplx=order[1],
        enforce_passivity=True,
    )
    with pytest.raises(AssertionError, match="Z_diff error"):
        validate(fitted, frequencies, reference_z_diff)


@pytest.mark.parametrize("gain", [1.001, 8.5e10])
def test_notebook_checks_sampled_passivity_independently(notebook_workflow, gain):
    """A false-positive rational passivity result cannot hide in-band gain."""
    _, validate = notebook_workflow
    fitted = SimpleNamespace(
        is_stable=True,
        is_passive=lambda: True,
        s=lambda frequencies: np.broadcast_to(
            gain * np.eye(2), (len(frequencies), 2, 2)
        ),
    )
    with pytest.raises(AssertionError, match="nonpassive in band"):
        validate(fitted, np.linspace(10e9, 200e9, 201), np.ones(201))

"""Deprecation warnings preserve caller attribution and explicit-setting intent."""

import os
import subprocess
import sys
import warnings

import pytest

from gsim import fdtd


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_domain_schema_keeps_public_fields(mode):
    schema = fdtd.Domain.model_json_schema(mode=mode)
    assert schema["properties"]["clip_geometry"]["default"] is False
    assert {"padding", "pml_cells", "x_bounds", "y_bounds", "z_bounds"} <= set(
        schema["properties"]
    )


@pytest.mark.parametrize(
    "statement",
    [
        "Domain(clip_geometry=True)",
        "domain = Domain(); domain(clip_geometry=False)",
        "domain = Domain(); domain.clip_geometry = True",
        'Simulation(domain={"clip_geometry": False})',
    ],
)
def test_explicit_clipping_warns_at_the_caller_under_default_filters(statement):
    environment = os.environ.copy()
    environment.pop("PYTHONWARNINGS", None)
    result = subprocess.run(  # noqa: S603 - fixed snippets from the parametrized test
        [
            sys.executable,
            "-c",
            "from gsim.fdtd import Domain, Simulation\n" + statement,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert "<string>:2: DeprecationWarning: clip_geometry" in result.stderr
    assert "0.12.0 or later" in result.stderr
    assert "12 months" in result.stderr
    assert "pydantic" not in result.stderr


def test_unset_switch_round_trips_without_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        domain = fdtd.Domain()
        domain(padding_y=0.7)
        domain.padding_z = 0.5
        assert "clip_geometry" not in domain.model_fields_set
        exported = domain.model_dump()
        assert "clip_geometry" not in exported
        assert "clip_geometry" not in domain.model_dump_json()
        copies = [
            fdtd.Domain.model_validate(exported),
            fdtd.Domain.model_validate_json(domain.model_dump_json()),
            fdtd.Simulation(domain=domain).domain,
            fdtd.Simulation(domain=exported).domain,
        ]
    assert all(copy == domain for copy in copies)
    assert not [
        warning for warning in caught if "clip_geometry" in str(warning.message)
    ]


@pytest.mark.parametrize("clip", [False, True])
def test_explicit_switch_survives_unrelated_updates(clip):
    with pytest.warns(DeprecationWarning, match="clip_geometry"):
        domain = fdtd.Domain(clip_geometry=clip)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        domain(padding_y=0.7)
        domain.padding_z = 0.5
    assert domain.model_dump()["clip_geometry"] is clip
    assert "clip_geometry" in domain.model_fields_set
    assert not [
        warning for warning in caught if "clip_geometry" in str(warning.message)
    ]


@pytest.mark.parametrize("explicit_false", [False, True])
def test_add_monitors_defers_no_clipping_error_until_write(
    tmp_path,
    fdtd_pdk_module,
    explicit_false,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        simulation = fdtd.Simulation(
            pdk=fdtd_pdk_module,
            domain={"clip_geometry": False} if explicit_false else {},
        )
    simulation.domain(z_bounds=(0, 0.1))
    simulation.source(port="o1")
    simulation.geometry("straight")
    simulation.add_monitors(xy_name="field_xy")
    assert simulation.monitors["field_xy"].center_um[2] == pytest.approx(0.11)
    with pytest.raises(fdtd.FDTDGeometryError, match="must contain"):
        simulation.write(tmp_path)

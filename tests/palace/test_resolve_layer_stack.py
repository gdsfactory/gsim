"""Tests for the pure layer-stack resolver used by mesh reuse."""

from __future__ import annotations

import copy
from unittest.mock import Mock

import pytest

from gsim.common.stack import LayerStack
from gsim.palace import DrivenSim
from gsim.palace.base import _resolve_layer_stack
from gsim.palace.models import MaterialConfig


def _prebuilt_stack() -> LayerStack:
    return LayerStack(
        dielectrics=[{"name": "oxide", "zmin": -2.0, "zmax": 5.2, "material": "sio2"}],
        materials={
            "sio2": {"type": "dielectric", "permittivity": 3.9},
            "aluminum": {"type": "conductor", "conductivity": 3.77e7},
        },
    )


@pytest.fixture
def overrides() -> dict[str, MaterialConfig]:
    return {
        "sio2": MaterialConfig.dielectric(permittivity=4.1, loss_tangent=1e-3),
        "gold": MaterialConfig.conductor(conductivity=4.1e7),
    }


@pytest.fixture
def fake_get_stack(monkeypatch):
    """Patch the PDK/YAML stack builder; record the arguments it receives."""
    calls: list[dict] = []

    def _get_stack(yaml_path=None, **kwargs):
        calls.append({"yaml_path": yaml_path, **kwargs})
        return LayerStack(
            dielectrics=[
                {"name": "oxide", "zmin": -2.0, "zmax": 5.2, "material": "sio2"},
                {"name": "air_box", "zmin": 5.2, "zmax": 50.0, "material": "Air"},
            ],
            materials={"sio2": {"type": "dielectric", "permittivity": 3.9}},
        )

    monkeypatch.setattr("gsim.common.stack.get_stack", _get_stack)
    return calls


def test_prebuilt_stack_gets_material_overrides_in_place(overrides):
    """F4: a prebuilt stack comes back as the same object, with overrides written in."""
    stack = _prebuilt_stack()

    resolved = _resolve_layer_stack(stack, {"_prebuilt": True}, overrides)

    assert resolved is stack
    assert stack.materials == {
        "sio2": {"type": "dielectric", "permittivity": 4.1, "loss_tangent": 1e-3},
        "aluminum": {"type": "conductor", "conductivity": 3.77e7},
        "gold": {"type": "conductor", "conductivity": 4.1e7},
    }
    assert stack.dielectrics == [
        {"name": "oxide", "zmin": -2.0, "zmax": 5.2, "material": "sio2"}
    ]


def test_sim_writes_overrides_into_prebuilt_stack():
    """F4: ``_resolve_stack`` keeps the prebuilt stack and writes overrides in."""
    stack = _prebuilt_stack()
    sim = DrivenSim()
    sim.set_stack(stack)
    sim.set_material("gold", material_type="conductor", conductivity=4.1e7)

    resolved = sim._resolve_stack()

    assert resolved is stack
    assert sim.stack is stack
    assert sim._stack_kwargs == {"_prebuilt": True}
    assert stack.materials["gold"] == {"type": "conductor", "conductivity": 4.1e7}
    assert stack.materials["sio2"] == {"type": "dielectric", "permittivity": 3.9}


def test_pdk_stack_drops_air_and_leaves_kwargs_untouched(fake_get_stack, overrides):
    """The resolver builds from kwargs without mutating them and drops air regions."""
    stack_kwargs = {
        "yaml_path": "custom.yaml",
        "include_substrate": False,
        "add_oxide_dielectric": True,
    }

    resolved = _resolve_layer_stack(None, stack_kwargs, overrides)

    assert stack_kwargs == {
        "yaml_path": "custom.yaml",
        "include_substrate": False,
        "add_oxide_dielectric": True,
    }
    assert fake_get_stack == [
        {
            "yaml_path": "custom.yaml",
            "include_substrate": False,
            "add_oxide_dielectric": True,
        }
    ]
    assert [d["name"] for d in resolved.dielectrics] == ["oxide"]
    assert resolved.materials["sio2"] == {
        "type": "dielectric",
        "permittivity": 4.1,
        "loss_tangent": 1e-3,
    }
    assert resolved.materials["gold"] == {"type": "conductor", "conductivity": 4.1e7}


def test_pdk_stack_without_yaml_path_leaves_kwargs_untouched(fake_get_stack):
    """Kwargs without ``yaml_path`` gain no key and reach the builder as None."""
    stack_kwargs: dict = {}

    _resolve_layer_stack(None, stack_kwargs, {})

    assert stack_kwargs == {}
    assert fake_get_stack == [{"yaml_path": None}]


def test_sim_assigns_resolved_pdk_stack(fake_get_stack):
    """``_resolve_stack`` stores the PDK/YAML stack on the sim and returns it."""
    sim = DrivenSim()
    sim.set_stack(yaml_path="custom.yaml")

    resolved = sim._resolve_stack()

    assert sim.stack is resolved
    assert [d["name"] for d in resolved.dielectrics] == ["oxide"]
    assert fake_get_stack[0]["yaml_path"] == "custom.yaml"


# ----- PDK-authoritative stacks ------------------------------------------------------


@pytest.fixture
def fake_pdk(monkeypatch):
    """Patch the PDK-authoritative chain; record what the resolver receives."""
    calls: list[dict] = []

    def _resolve(component=None, **options):
        calls.append({"component": component, **options})
        return options

    def _build(resolved):
        dielectrics = resolved["dielectrics"]
        if dielectrics is None:
            dielectrics = [
                {"name": "ild", "zmin": -1.0, "zmax": 5.0, "material": "sio2"},
                {"name": "air_box", "zmin": 5.0, "zmax": 50.0, "material": "Air"},
            ]
        return LayerStack(
            dielectrics=list(dielectrics),
            materials={
                "Al": {"type": "conductor", "conductivity": 3.77e7},
                "sio2": {"type": "dielectric", "permittivity": 3.9},
            },
        )

    monkeypatch.setattr("gsim.common.pdk.resolve_layer_stack", _resolve)
    monkeypatch.setattr("gsim.palace.stack.build_palace_layer_stack", _build)
    return calls


def test_authoritative_stack_reads_its_options_and_leaves_kwargs_untouched(fake_pdk):
    """The options reach the PDK resolver, and nothing is written back."""
    component = object()
    regions = [{"name": "ild", "zmin": -1.0, "zmax": 5.0, "material": "sio2"}]
    stack_kwargs = {
        "pdk_authoritative": True,
        "dielectrics": regions,
        "role_overrides": {"via1": "via"},
        "use_pdk_overlay": False,
    }
    before = copy.deepcopy(stack_kwargs)

    resolved = _resolve_layer_stack(None, stack_kwargs, {}, component=component)

    assert stack_kwargs == before
    assert fake_pdk == [
        {
            "component": component,
            "dielectrics": regions,
            "role_overrides": {"via1": "via"},
            "use_pdk_overlay": False,
        }
    ]
    assert resolved.dielectrics == regions


@pytest.mark.usefixtures("fake_pdk")
def test_authoritative_stack_drops_air_and_overrides_alias_tokens():
    """A canonical material name also overrides the PDK token that aliases it."""
    aluminum = MaterialConfig.conductor(conductivity=2.5e7)

    resolved = _resolve_layer_stack(
        None, {"pdk_authoritative": True}, {"aluminum": aluminum}
    )

    assert [d["name"] for d in resolved.dielectrics] == ["ild"]
    assert resolved.materials["Al"] == aluminum.to_dict()
    assert resolved.materials["aluminum"] == aluminum.to_dict()
    assert resolved.materials["sio2"]["permittivity"] == 3.9


@pytest.mark.usefixtures("fake_pdk")
def test_authoritative_stack_needs_a_dielectric_or_an_airbox():
    """The refusal reads the airbox argument, not any simulation state."""
    stack_kwargs = {"pdk_authoritative": True, "dielectrics": []}

    with pytest.raises(ValueError, match="No dielectric volumes"):
        _resolve_layer_stack(None, stack_kwargs, {})

    resolved = _resolve_layer_stack(
        None, stack_kwargs, {}, airbox_config={"z_above": 1.0}
    )
    assert resolved.dielectrics == []


def test_sim_retries_an_authoritative_stack_after_a_refusal(fake_pdk):
    """Following the error's advice stays on the authoritative path."""
    sim = DrivenSim()
    sim.set_stack(pdk_authoritative=True, dielectrics=[])
    with pytest.raises(ValueError, match="No dielectric volumes"):
        sim._resolve_stack()

    sim.set_airbox(margin_x=0, margin_y=0, z_above=1.0, z_below=1.0)

    assert sim._resolve_stack() is sim.stack
    assert sim._stack_kwargs["pdk_authoritative"] is True
    assert len(fake_pdk) == 2


def test_authoritative_options_never_reach_the_inferred_extractor(fake_get_stack):
    """The pure resolver filters them, so a copy of unresolved kwargs is safe too."""
    stack_kwargs = {
        "yaml_path": None,
        "include_substrate": False,
        "pdk_authoritative": False,
        "dielectrics": [{"name": "ild", "zmin": 0.0, "zmax": 1.0, "material": "sio2"}],
        "role_overrides": {"via1": "via"},
        "use_pdk_overlay": True,
    }
    before = copy.deepcopy(stack_kwargs)

    _resolve_layer_stack(None, stack_kwargs, {})

    assert stack_kwargs == before
    assert fake_get_stack == [{"yaml_path": None, "include_substrate": False}]


@pytest.mark.usefixtures("fake_get_stack")
def test_ignored_authoritative_options_warn_once(monkeypatch):
    """Resolving again, as each fingerprint check does, does not warn again."""
    warning = Mock()
    monkeypatch.setattr("gsim.palace.base.logger.warning", warning)
    sim = DrivenSim()
    sim.set_stack(role_overrides={"via1": "via"})

    sim._resolve_stack()
    sim._resolve_stack()

    messages = [call.args[0] for call in warning.call_args_list]
    assert sum("only applies with" in message for message in messages) == 1

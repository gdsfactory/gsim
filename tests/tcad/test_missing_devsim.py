"""Import-time degradation: gsim.tcad works without DEVSIM installed."""

from __future__ import annotations

import sys

import pytest


def test_package_imports_without_devsim():
    import gsim.tcad  # noqa: F401


def test_require_devsim_names_the_extra(monkeypatch):
    from gsim.tcad.runtime import require_devsim

    # Block the import even when devsim happens to be installed.
    monkeypatch.setitem(sys.modules, "devsim", None)
    with pytest.raises(ImportError, match=r"gsim\[tcad\]"):
        require_devsim()


def test_import_simple_physics_names_the_extra(monkeypatch):
    from gsim.tcad.runtime import import_simple_physics

    monkeypatch.setitem(sys.modules, "devsim", None)
    monkeypatch.setitem(sys.modules, "devsim.python_packages", None)
    monkeypatch.setitem(sys.modules, "devsim.python_packages.simple_physics", None)
    with pytest.raises(ImportError, match=r"gsim\[tcad\]"):
        import_simple_physics()


def test_tcad_extra_declared_in_packaging():
    from importlib import metadata

    try:
        requires = metadata.requires("gsim") or []
    except metadata.PackageNotFoundError:
        pytest.skip("gsim not installed as a distribution")
    extras = {
        req.split("extra == ")[1].strip("\"'") for req in requires if "extra == " in req
    }
    assert "tcad" in extras
    assert any("devsim" in req for req in requires)

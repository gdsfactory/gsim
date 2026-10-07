"""Import-time degradation: gsim.femwell works without femwell installed."""

from __future__ import annotations

import sys

import pytest


def test_package_imports_without_femwell():
    import gsim.femwell  # noqa: F401


def test_require_femwell_names_the_extra(monkeypatch):
    from gsim.femwell.runtime import require_femwell

    monkeypatch.setitem(sys.modules, "femwell", None)
    with pytest.raises(ImportError, match=r"gsim\[femwell\]"):
        require_femwell()


def test_require_skfem_names_the_extra(monkeypatch):
    from gsim.femwell.runtime import require_skfem

    monkeypatch.setitem(sys.modules, "skfem", None)
    with pytest.raises(ImportError, match=r"gsim\[femwell\]"):
        require_skfem()


def test_femwell_extra_declared_in_packaging():
    from importlib import metadata

    try:
        requires = metadata.requires("gsim") or []
    except metadata.PackageNotFoundError:
        pytest.skip("gsim not installed as a distribution")
    extras = {
        req.split("extra == ")[1].strip("\"'") for req in requires if "extra == " in req
    }
    assert "femwell" in extras
    assert any("femwell" in req and "extra" in req for req in requires)

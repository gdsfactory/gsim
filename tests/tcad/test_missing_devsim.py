"""Import-time degradation: gsim.tcad works without DEVSIM installed."""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

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
    tcad = [req for req in requires if "extra == " in req and "tcad" in req]
    assert any(req.startswith("devsim>") for req in tcad)
    assert any(req.startswith("devsim-openblas") for req in tcad)


@pytest.fixture
def fresh_devsim(monkeypatch):
    """A process where DEVSIM has not been imported and no library is chosen.

    Returns the list ``devsim_openblas.configure`` appends to when called.
    """
    monkeypatch.delitem(sys.modules, "devsim", raising=False)
    monkeypatch.delenv("DEVSIM_MATH_LIBS", raising=False)
    configured: list[bool] = []
    fake = SimpleNamespace(configure=lambda: configured.append(True))
    monkeypatch.setitem(sys.modules, "devsim_openblas", fake)
    return configured


def test_first_import_is_pointed_at_devsim_openblas(fresh_devsim):
    from gsim.tcad.runtime import _select_math_libraries

    _select_math_libraries()
    assert fresh_devsim == [True]


def test_a_chosen_math_library_is_left_alone(fresh_devsim, monkeypatch):
    from gsim.tcad.runtime import _select_math_libraries

    monkeypatch.setenv("DEVSIM_MATH_LIBS", "libmkl_rt.so")
    _select_math_libraries()
    assert fresh_devsim == []


def test_an_imported_devsim_is_left_alone(fresh_devsim, monkeypatch):
    """DEVSIM has read DEVSIM_MATH_LIBS already; setting it now changes nothing."""
    from gsim.tcad.runtime import _select_math_libraries

    monkeypatch.setitem(sys.modules, "devsim", SimpleNamespace())
    _select_math_libraries()
    assert fresh_devsim == []


def test_without_devsim_openblas_devsim_finds_its_own(fresh_devsim, monkeypatch):
    from gsim.tcad.runtime import _select_math_libraries

    monkeypatch.setitem(sys.modules, "devsim_openblas", None)
    _select_math_libraries()
    assert fresh_devsim == []


def test_devsim_without_math_libraries_names_the_remedy(monkeypatch):
    """DEVSIM's initialiser raises RuntimeError when it finds no BLAS/LAPACK."""
    from gsim.tcad import runtime

    def _initialiser_fails(*_, **__):
        raise RuntimeError("Issues initializing DEVSIM.")

    monkeypatch.setattr(runtime, "_select_math_libraries", lambda: None)
    monkeypatch.setattr(runtime, "require_module", _initialiser_fails)
    with pytest.raises(ImportError, match="devsim-openblas") as info:
        runtime.require_devsim()
    assert "DEVSIM_MATH_LIBS" in str(info.value)
    assert isinstance(info.value.__cause__, RuntimeError)


def test_reset_device_does_not_reimport_devsim(monkeypatch):
    """A DEVSIM whose import fails must not be imported again to release.

    DEVSIM declares its default derivatives in a one-shot C initialiser.
    An install without the math libraries raises ``RuntimeError`` partway
    through that initialiser and leaves no ``sys.modules`` entry, so a
    second import redeclares what the first declared and raises again —
    out of a release path that only means to clean up.
    """
    from gsim.tcad import ChargeTransportSim

    monkeypatch.delitem(sys.modules, "devsim", raising=False)
    real_import = builtins.__import__

    def _devsim_fails_to_initialise(name, *args, **kwargs):
        if name == "devsim" or name.startswith("devsim."):
            raise RuntimeError("Issues initializing DEVSIM.")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _devsim_fails_to_initialise)

    sim = ChargeTransportSim()
    sim._device = "gsim_tcad_device_0"
    sim.reset_device()
    assert sim._device is None

"""Import-time degradation: gsim.tcad works without DEVSIM installed."""

from __future__ import annotations

import builtins
import os
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
    assert any(req.startswith("scipy-openblas32") for req in tcad)


_MATH_ENV = ("DEVSIM_MATH_LIBS", "DEVSIM_BLAS_PREFIX", "DEVSIM_BLAS_SUFFIX")


@pytest.fixture
def fresh_devsim(monkeypatch, tmp_path):
    """A process where DEVSIM has not been imported and no library is chosen.

    scipy-openblas32 is replaced by one whose library sits in *tmp_path*;
    returns that library's path.
    """
    from gsim.tcad import runtime

    monkeypatch.delitem(sys.modules, "devsim", raising=False)
    # _select_math_libraries writes os.environ itself, which monkeypatch.delenv
    # does not undo for a variable that was unset: give it a throwaway copy.
    environ = {k: v for k, v in os.environ.items() if k not in _MATH_ENV}
    monkeypatch.setattr(os, "environ", environ)
    library = tmp_path / f"libscipy_openblas{runtime._SHARED_LIBRARY_SUFFIX}"
    library.touch()
    fake = SimpleNamespace(get_lib_dir=lambda: str(tmp_path))
    monkeypatch.setitem(sys.modules, "scipy_openblas32", fake)
    return library


def test_first_import_is_pointed_at_scipy_openblas32(fresh_devsim):
    from gsim.tcad.runtime import _select_math_libraries

    _select_math_libraries()
    assert os.environ["DEVSIM_MATH_LIBS"] == str(fresh_devsim)
    assert os.environ["DEVSIM_BLAS_PREFIX"] == "scipy_"
    assert os.environ["DEVSIM_BLAS_SUFFIX"] == "_"


@pytest.mark.usefixtures("fresh_devsim")
def test_a_chosen_math_library_is_left_alone(monkeypatch):
    from gsim.tcad.runtime import _select_math_libraries

    monkeypatch.setenv("DEVSIM_MATH_LIBS", "libmkl_rt.so")
    _select_math_libraries()
    assert os.environ["DEVSIM_MATH_LIBS"] == "libmkl_rt.so"
    assert "DEVSIM_BLAS_PREFIX" not in os.environ


@pytest.mark.usefixtures("fresh_devsim")
def test_an_imported_devsim_is_left_alone(monkeypatch):
    """DEVSIM has read DEVSIM_MATH_LIBS already; setting it now changes nothing."""
    from gsim.tcad.runtime import _select_math_libraries

    monkeypatch.setitem(sys.modules, "devsim", SimpleNamespace())
    _select_math_libraries()
    assert not any(name in os.environ for name in _MATH_ENV)


@pytest.mark.usefixtures("fresh_devsim")
def test_without_scipy_openblas32_devsim_finds_its_own(monkeypatch):
    from gsim.tcad.runtime import _select_math_libraries

    monkeypatch.setitem(sys.modules, "scipy_openblas32", None)
    _select_math_libraries()
    assert not any(name in os.environ for name in _MATH_ENV)


def test_a_scipy_openblas32_without_its_library_is_skipped(fresh_devsim):
    from gsim.tcad.runtime import _select_math_libraries

    fresh_devsim.unlink()
    _select_math_libraries()
    assert not any(name in os.environ for name in _MATH_ENV)


def test_devsim_without_math_libraries_names_the_remedy(monkeypatch):
    """DEVSIM's initialiser raises RuntimeError when it finds no BLAS/LAPACK."""
    from gsim.tcad import runtime

    def _initialiser_fails(*_, **__):
        raise RuntimeError("Issues initializing DEVSIM.")

    monkeypatch.setattr(runtime, "_select_math_libraries", lambda: None)
    monkeypatch.setattr(runtime, "require_module", _initialiser_fails)
    with pytest.raises(ImportError, match="scipy-openblas32") as info:
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
    from gsim.tcad.device import _LIVE_DEVICES, Device

    monkeypatch.delitem(sys.modules, "devsim", raising=False)
    real_import = builtins.__import__

    def _devsim_fails_to_initialise(name, *args, **kwargs):
        if name == "devsim" or name.startswith("devsim."):
            raise RuntimeError("Issues initializing DEVSIM.")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _devsim_fails_to_initialise)

    device = Device.__new__(Device)
    device.name = "gsim_tcad_device_0"
    device.mesh_name = None
    _LIVE_DEVICES.add(device.name)
    device.release()
    assert device.name is None
    assert "gsim_tcad_device_0" not in _LIVE_DEVICES

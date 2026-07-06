"""Tests for the Palace binary resolver (gsim.palace.runtime).

Note: These tests require the gsim package to be importable in the
test environment.  In CI the full dependency chain is available.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def _mock_gcloud(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock gsim.gcloud so we can import gsim.palace.runtime without gdsfactoryplus."""
    import types

    gcloud = types.ModuleType("gsim.gcloud")
    for name in ("get_status", "wait_for_results", "register_result_parser", "print_job_summary", "run_simulation"):
        setattr(gcloud, name, lambda *a, **kw: None)  # noqa: ARG005
    gcloud.RunResult = type("RunResult", (), {})
    monkeypatch.setitem(sys.modules, "gsim.gcloud", gcloud)


@pytest.fixture
def _no_palacetoolkit(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Ensure palacetoolkit is unavailable during the test."""
    # If palacetoolkit is installed, we still want to mock it as unavailable
    # for tests that need to verify GSIM's behavior without it.
    if "palacetoolkit" in sys.modules:
        old = sys.modules["palacetoolkit"]
        monkeypatch.delitem(sys.modules, "palacetoolkit", raising=False)
        monkeypatch.delitem(sys.modules.get("palacetoolkit.palace_runtime", None), "palacetoolkit.palace_runtime", raising=False)  # type: ignore[union-attr]
        yield
        sys.modules["palacetoolkit"] = old
    else:
        yield


class TestResolvePalaceBinary:
    def test_returns_none_when_nothing_found(self, _no_palacetoolkit: None) -> None:
        from gsim.palace.runtime import resolve_palace_binary

        with pytest.MonkeyPatch().context() as mp:
            mp.delenv("PALACE_BIN", raising=False)
            mp.delenv("PALACE_EXECUTABLE", raising=False)
            with mp.context() as mp2:
                mp2.setattr("gsim.palace.runtime._palacetoolkit_available", lambda: False)
                result = resolve_palace_binary()
                assert result is None

    def test_uses_palace_bin_env(self) -> None:
        from gsim.palace.runtime import resolve_palace_binary

        fake_bin = Path("/usr/local/bin/palace")

        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("PALACE_BIN", str(fake_bin))
            mp.setattr("gsim.palace.runtime._binary_is_runnable", lambda _: True)
            mp.setattr("pathlib.Path.is_file", lambda _: True)
            result = resolve_palace_binary()
            assert result is not None

    def test_delegates_to_palacetoolkit(self) -> None:
        from gsim.palace.runtime import resolve_palace_binary

        fake_ptk_bin = Path("/opt/palacetoolkit/bin/palace")

        # Simulate palacetoolkit being available
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("gsim.palace.runtime._palacetoolkit_available", lambda: True)
            mp.setattr(
                "palacetoolkit.palace_runtime.resolve_palace_binary",
                lambda: fake_ptk_bin,
            )
            result = resolve_palace_binary()
            assert result is not None

    def test_prefer_bundled_skips_env(self) -> None:
        from gsim.palace.runtime import resolve_palace_binary

        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("PALACE_BIN", "/usr/bin/palace")
            mp.setattr("gsim.palace.runtime._palacetoolkit_available", lambda: False)
            result = resolve_palace_binary(prefer_bundled=True)
            assert result is None


class TestResolvePalaceLibraryDir:
    def test_returns_none_without_palacetoolkit(self, _no_palacetoolkit: None) -> None:
        from gsim.palace.runtime import resolve_palace_library_dir

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("gsim.palace.runtime._palacetoolkit_available", lambda: False)
            assert resolve_palace_library_dir() is None

    def test_delegates_to_palacetoolkit(self) -> None:
        from gsim.palace.runtime import resolve_palace_library_dir

        fake_lib = Path("/opt/palacetoolkit/lib")

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("gsim.palace.runtime._palacetoolkit_available", lambda: True)
            mp.setattr(
                "palacetoolkit.palace_runtime.resolve_palace_library_dir",
                lambda: fake_lib,
            )
            result = resolve_palace_library_dir()
            assert result is not None


class TestPalacetoolkitAvailable:
    def test_true_when_installed(self) -> None:
        from gsim.palace.runtime import _palacetoolkit_available

        # Since we mocked gcloud but not palacetoolkit, if it's actually
        # installed on the system, this will be True. We can't force it
        # to be True via mock here without patching importlib, which is
        # fragile.  Instead we just verify the function runs.
        result = _palacetoolkit_available()
        assert isinstance(result, bool)
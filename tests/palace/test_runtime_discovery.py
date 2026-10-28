"""Validate runtime discovery at the network and process boundaries."""

import io
import json
import os
import subprocess
from unittest.mock import Mock
from zipfile import ZipFile

import pytest

from gsim.palace import runtime


def test_invalid_environment_binary_falls_back_to_path(tmp_path, monkeypatch):
    binary = tmp_path / "palace"
    binary.touch()
    monkeypatch.setenv("PALACE_BIN", str(tmp_path / "absent"))
    monkeypatch.setenv("PALACE_EXECUTABLE", "palace-custom")
    lookup = Mock(return_value=str(binary))
    monkeypatch.setattr(runtime.shutil, "which", lookup)
    assert runtime.resolve_palace_binary(prefer_bundled=False) == binary
    lookup.assert_called_once_with("palace-custom")


@pytest.mark.parametrize(
    ("assets", "expected"),
    [
        ([], None),
        (
            [
                {"name": "other.whl"},
                {"name": "palacetoolkit_palace_cpu-1-linux_x86_64.whl"},
            ],
            None,
        ),
        (
            [
                {
                    "name": "palacetoolkit_palace_cpu-1-linux_x86_64.whl",
                    "browser_download_url": "https://example.org/runtime.whl",
                }
            ],
            "https://example.org/runtime.whl",
        ),
    ],
)
def test_release_discovery(monkeypatch, assets, expected):
    request = Mock(return_value=io.BytesIO(json.dumps({"assets": assets}).encode()))
    monkeypatch.setattr(runtime, "urlopen", request)
    assert runtime._binary_wheel_url_from_release("1", 3) == expected
    assert request.call_args.kwargs["timeout"] == 3


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"other/file": b""}, "does not contain"),
        ({"palacetoolkit_palace_cpu/bin/palace": b""}, "missing expected"),
        (
            {
                "palacetoolkit_palace_cpu/bin/other": b"",
                "palacetoolkit_palace_cpu/lib/library.so": b"",
                "palacetoolkit_palace_cpu/share/openmpi/help.txt": b"MPI help",
            },
            "did not produce",
        ),
        (
            {
                "palacetoolkit_palace_cpu/bin/palace": b"launcher",
                "palacetoolkit_palace_cpu/lib/library.so": b"library",
            },
            "missing expected",
        ),
        (
            {
                "palacetoolkit_palace_cpu/bin/palace": b"launcher",
                "palacetoolkit_palace_cpu/lib/library.so": b"library",
                "palacetoolkit_palace_cpu/share/openmpi/help.txt": b"MPI help",
            },
            None,
        ),
    ],
)
@pytest.mark.parametrize("release_fails", [False, True])
def test_downloaded_runtime_layout(
    tmp_path, monkeypatch, payload, error, release_fails
):
    archive = io.BytesIO()
    with ZipFile(archive, "w") as wheel:
        for path, content in payload.items():
            wheel.writestr(path, content)
    monkeypatch.setenv("PALACETOOLKIT_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(runtime, "_is_linux_x86_64", lambda: True)
    monkeypatch.setattr(
        runtime,
        "_binary_wheel_url_from_release",
        Mock(side_effect=OSError("offline"))
        if release_fails
        else Mock(return_value="https://example.org/runtime.whl"),
    )
    monkeypatch.setattr(
        runtime, "urlopen", lambda *_args, **_kwargs: io.BytesIO(archive.getvalue())
    )
    if error:
        with pytest.raises(RuntimeError, match=error):
            runtime.install_palace_runtime(force=True)
    else:
        binary = runtime.install_palace_runtime(force=True)
        assert binary.read_bytes() == b"launcher"
        assert runtime._cached_binary() == binary
        if os.name != "nt":
            assert binary.stat().st_mode & 0o111 == 0o111


def test_cached_runtime_without_share_is_not_resolved(tmp_path, monkeypatch):
    monkeypatch.setenv("PALACETOOLKIT_RUNTIME_DIR", str(tmp_path))
    prefix = runtime._cached_runtime_prefix()
    (prefix / "bin").mkdir(parents=True)
    (prefix / "lib").mkdir()
    launcher = prefix / "bin" / "palace"
    launcher.write_text("launcher")
    launcher.chmod(0o755)
    monkeypatch.setattr(runtime, "_palace_cpu_available", lambda: False)
    monkeypatch.setattr(runtime, "_palace_toolkit_available", lambda: False)

    assert runtime._cached_binary() is None
    assert (
        runtime.resolve_palace_binary(prefer_bundled=True, download_if_missing=False)
        is None
    )


@pytest.mark.parametrize(
    "status", ["missing", "not_executable", "success", "help", "failure", "timeout"]
)
@pytest.mark.parametrize("prior", ["", "/other/lib"])
def test_binary_smoke_test(tmp_path, monkeypatch, status, prior):
    binary = tmp_path / "palace"
    if status != "missing":
        binary.write_text("launcher")
        binary.chmod(0o644 if status == "not_executable" else 0o755)
    library = tmp_path / "lib"
    library.mkdir()
    monkeypatch.setenv("LD_LIBRARY_PATH", prior)
    monkeypatch.setattr(runtime.os, "access", lambda *_args: status != "not_executable")
    process = (
        Mock(side_effect=subprocess.TimeoutExpired("palace", 5))
        if status == "timeout"
        else Mock(
            side_effect=[
                subprocess.CompletedProcess([], 0 if status == "success" else 1),
                subprocess.CompletedProcess([], 0 if status == "help" else 1),
            ]
        )
    )
    monkeypatch.setattr(runtime.subprocess, "run", process)
    assert runtime._binary_is_runnable(binary, library) == (
        status not in {"missing", "not_executable"}
    )
    if process.called:
        assert process.call_args.kwargs["env"]["LD_LIBRARY_PATH"] == str(library) + (
            f":{prior}" if prior else ""
        )


@pytest.mark.parametrize("enabled", ["1", "false", "NO", "off", "0"])
def test_runtime_requirement(tmp_path, monkeypatch, enabled):
    monkeypatch.setenv("PALACETOOLKIT_AUTO_DOWNLOAD_BINARY", enabled)
    assert runtime._auto_download_enabled() == (enabled == "1")
    monkeypatch.setattr(runtime, "resolve_palace_binary", lambda: None)
    with pytest.raises(RuntimeError, match="custom guidance"):
        runtime.require_palace_binary(hint="custom guidance")
    binary = tmp_path / "palace"
    monkeypatch.setattr(runtime, "resolve_palace_binary", lambda: binary)
    assert runtime.require_palace_binary() == binary


@pytest.mark.parametrize("result", ["directory", "missing", "none", "failure"])
def test_optional_toolkit_library_resolution(tmp_path, monkeypatch, result):
    import sys
    from types import SimpleNamespace

    library = tmp_path / "lib"
    if result == "directory":
        library.mkdir()
    resolver = (
        Mock(side_effect=RuntimeError("toolkit unavailable"))
        if result == "failure"
        else Mock(return_value=None if result == "none" else library)
    )
    monkeypatch.setattr(runtime, "_palace_cpu_available", lambda: False)
    monkeypatch.setattr(runtime, "_palace_toolkit_available", lambda: True)
    monkeypatch.setattr(runtime, "_cached_library_dir", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "palacetoolkit.palace_runtime",
        SimpleNamespace(resolve_palace_library_dir=resolver),
    )
    assert runtime.resolve_palace_library_dir() == (
        library if result == "directory" else None
    )

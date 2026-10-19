"""Executable permissions for installed and previously cached Palace runtimes."""

import io
import os
import stat
from unittest.mock import Mock
from zipfile import ZipFile, ZipInfo

import pytest

from gsim.palace import runtime

_BIN_NAMES = (
    "palace",
    "palace-x86_64.bin",
    "mpirun",
    "orted",
    "orterun",
    "palace-launcher",
    "tools/mpi-helper",
)


def test_missing_runtime_does_not_restore_permissions_or_download(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PALACETOOLKIT_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(runtime, "_palace_cpu_available", lambda: False)
    monkeypatch.setattr(runtime, "_palace_toolkit_available", lambda: False)
    restore_permissions = Mock(side_effect=AssertionError("Runtime is absent"))
    install_runtime = Mock(side_effect=AssertionError("Downloads are disabled"))
    monkeypatch.setattr(runtime, "_restore_bin_permissions", restore_permissions)
    monkeypatch.setattr(runtime, "install_palace_runtime", install_runtime)

    assert (
        runtime.resolve_palace_binary(prefer_bundled=True, download_if_missing=False)
        is None
    )

    restore_permissions.assert_not_called()
    install_runtime.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_installed_runtime_restores_all_launcher_permissions(tmp_path, monkeypatch):
    archive = io.BytesIO()
    with ZipFile(archive, "w") as wheel:
        for name in _BIN_NAMES:
            entry = ZipInfo(f"palacetoolkit_palace_cpu/bin/{name}")
            entry.create_system = 3
            entry.external_attr = (stat.S_IFREG | 0o755) << 16
            wheel.writestr(entry, b"launcher")
        wheel.writestr("palacetoolkit_palace_cpu/lib/library.so", b"library")
    monkeypatch.setenv("PALACETOOLKIT_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(runtime, "_is_linux_x86_64", lambda: True)
    monkeypatch.setattr(
        runtime, "_binary_wheel_url_from_release", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        runtime, "urlopen", lambda *_a, **_k: io.BytesIO(archive.getvalue())
    )

    binary = runtime.install_palace_runtime()

    for name in _BIN_NAMES:
        launcher = binary.parent / name
        assert launcher.read_bytes() == b"launcher"
        if os.name != "nt":
            assert launcher.stat().st_mode & 0o111 == 0o111
    library = binary.parent.parent / "lib/library.so"
    assert library.read_bytes() == b"library"
    if os.name != "nt":
        assert library.stat().st_mode & 0o111 == 0


@pytest.mark.skipif(os.name == "nt", reason="Unix executable permissions and symlinks")
@pytest.mark.parametrize("lookup", ["install", "resolve"])
@pytest.mark.parametrize("mode", [0o644, 0o755])
def test_cached_runtime_repairs_launchers_without_download(
    tmp_path, monkeypatch, lookup, mode
):
    monkeypatch.setenv("PALACETOOLKIT_RUNTIME_DIR", str(tmp_path))
    prefix = runtime._cached_runtime_prefix()
    for name in _BIN_NAMES:
        path = prefix / "bin" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("launcher")
        path.chmod(mode)
    library = prefix / "lib" / "library.so"
    library.parent.mkdir()
    library.touch(mode=0o644)
    tools_dir = prefix / "bin" / "tools"
    tools_dir.chmod(0o750)

    external_binary = tmp_path / "external-binary"
    external_binary.touch(mode=0o644)
    (prefix / "bin" / "linked-binary").symlink_to(external_binary)
    (prefix / "bin" / "linked-directory").symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.setattr(runtime, "_is_linux_x86_64", lambda: False)
    monkeypatch.setattr(runtime, "_palace_cpu_available", lambda: False)
    monkeypatch.setattr(runtime, "_palace_toolkit_available", lambda: False)
    monkeypatch.setattr(
        runtime,
        "_binary_is_runnable",
        lambda binary, _library: os.access(binary, os.X_OK),
    )
    download = Mock(side_effect=AssertionError("Cached runtime must not download"))
    monkeypatch.setattr(runtime, "urlopen", download)
    if mode == 0o755:
        monkeypatch.setattr(
            runtime.Path, "chmod", Mock(side_effect=PermissionError("Read-only cache"))
        )

    binary = (
        runtime.install_palace_runtime()
        if lookup == "install"
        else runtime.resolve_palace_binary(
            prefer_bundled=True, download_if_missing=False
        )
    )

    assert binary == prefix / "bin" / "palace"
    for name in _BIN_NAMES:
        assert (prefix / "bin" / name).stat().st_mode & 0o111 == 0o111
    assert stat.S_IMODE(library.stat().st_mode) == 0o644
    assert stat.S_IMODE(external_binary.stat().st_mode) == 0o644
    assert stat.S_IMODE(tools_dir.stat().st_mode) == 0o750
    download.assert_not_called()

"""The upload manifest covers exact archive bytes, independently of cache keys."""

from __future__ import annotations

import hashlib
import tempfile
import zipfile
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from gsim import hashing


def test_manifest_covers_nested_hidden_and_diagnostic_files(tmp_path):
    (tmp_path / "config.json").write_bytes(b"abc")
    (tmp_path / ".DS_Store").write_bytes(b"")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "mesh.pyc").write_bytes(b"abc")
    (tmp_path / "metadata.json").write_bytes(b"")

    manifest = hashing.compute_input_manifest(tmp_path)

    assert manifest == {
        ".DS_Store": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "__pycache__/mesh.pyc": (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        ),
        "config.json": (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        ),
        "metadata.json": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
    }


def test_manifest_matches_sdk_archive_entries_and_bytes(tmp_path, monkeypatch):
    from gdsfactoryplus import sim

    uploader = getattr(sim, "web", sim)
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "mesh.msh").write_bytes(b"mesh\x00bytes")
    (tmp_path / ".hidden").write_bytes(b"included")
    archive_bytes = []

    @contextmanager
    def closed_temporary_archive(*_args, **kwargs):
        # Older SDK releases keep NamedTemporaryFile open during ZipFile writes,
        # which Windows forbids. This test checks archive entries/bytes, so isolate
        # that unrelated temporary-file lifecycle while retaining the SDK builder.
        with tempfile.TemporaryDirectory() as directory:
            name = Path(directory) / f"archive{kwargs.get('suffix', '')}"
            yield SimpleNamespace(name=str(name))

    class UploadClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def put(self, _url, *, content, headers):
            assert headers["Content-Type"] == "application/zip"
            archive_bytes.append(content.read())
            return SimpleNamespace(raise_for_status=lambda: None, headers={})

    monkeypatch.setattr(uploader.httpx, "Client", UploadClient)
    monkeypatch.setattr(
        uploader.tempfile, "NamedTemporaryFile", closed_temporary_archive
    )
    uploader._upload_file("https://example.com/input.zip", tmp_path)
    with zipfile.ZipFile(BytesIO(archive_bytes[0])) as archive:
        uploaded = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
        }

    assert hashing.compute_input_manifest(tmp_path) == uploaded


def test_manifest_rejects_symbolic_link_inputs(tmp_path):
    (tmp_path / "config.json").write_bytes(b"abc")
    (tmp_path / "link.json").symlink_to(tmp_path / "config.json")

    with pytest.raises(ValueError, match="symbolic link"):
        hashing.compute_input_manifest(tmp_path)


def test_manifest_changes_with_bytes_without_changing_physics_cache(tmp_path):
    (tmp_path / "config.json").write_bytes(b"abc")
    original_hashes = {
        solver: hashing.compute_input_hash(tmp_path, solver)
        for solver in ("palace", "fdtd")
    }
    (tmp_path / "metadata.json").write_bytes(b"diagnostics")

    for solver, original_hash in original_hashes.items():
        assert hashing.compute_input_hash(tmp_path, solver) == original_hash
    before = hashing.compute_input_manifest(tmp_path)
    (tmp_path / "metadata.json").write_bytes(b"new diagnostics")
    assert (
        hashing.compute_input_manifest(tmp_path)["metadata.json"]
        != before["metadata.json"]
    )

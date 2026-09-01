# Copyright 2026 GDSFactory
"""Tests for deriving documentation notebooks from Zensical navigation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.list_docs_notebooks import (
    DEFAULT_CONFIG_PATH,
    DocsNotebookConfigError,
    list_docs_notebooks,
)

EXPECTED_NOTEBOOKS = [
    "nbs/fdtd_mmi_gpdk.ipynb",
    "nbs/fdtd_directional_coupler.ipynb",
    "nbs/fdtd_waveguide_crossing.ipynb",
    "nbs/fdtd_broadband_directional_coupler.ipynb",
    "nbs/meep_dc.ipynb",
    "nbs/meep_2d.ipynb",
    "nbs/meep_2d_xz_gc.ipynb",
    "nbs/meep_2d_xz_gc_wavelength_sweep.ipynb",
    "nbs/meep_ybranch.ipynb",
    "nbs/palace_cpw_lumped.ipynb",
    "nbs/palace_microstrip.ipynb",
    "nbs/palace_branch_line_coupler.ipynb",
    "nbs/palace_width_sweep.ipynb",
    "nbs/palace_qpdk_resonator.ipynb",
    "nbs/palace_inductor.ipynb",
    "nbs/palace_eic_ihp_hfss.ipynb",
    "nbs/palace_eic_nist_cpw.ipynb",
    "nbs/palace_cpw_via.ipynb",
    "nbs/palace_cpw_fields.ipynb",
]


def _write_config(root: Path, nav: str) -> Path:
    config_path = root / "docs" / "zensical.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(f"[project]\nnav = {nav}\n", encoding="utf-8")
    return config_path


def _write_notebook(root: Path, name: str = "example") -> None:
    notebook_path = root / "nbs" / f"{name}.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text("{}", encoding="utf-8")


def test_lists_current_docs_notebooks_in_navigation_order():
    """The checked-in config resolves to every published source notebook."""
    assert list_docs_notebooks(DEFAULT_CONFIG_PATH) == EXPECTED_NOTEBOOKS


def test_recurses_nav_and_ignores_non_notebook_pages(tmp_path):
    """Only Markdown leaves under docs/nbs map to source notebooks."""
    _write_notebook(tmp_path)
    config_path = _write_config(
        tmp_path,
        '[{ Home = "index.md" }, { Guides = [{ Demo = "nbs/example.md" }] }, '
        '{ External = "https://example.com/reference.md" }]',
    )

    assert list_docs_notebooks(config_path) == ["nbs/example.ipynb"]


@pytest.mark.parametrize(
    "nav_path",
    ["../nbs/example.md", "nbs/../example.md", "/nbs/example.md", "nbs\\example.md"],
)
def test_rejects_unsafe_navigation_paths(tmp_path, nav_path):
    """Absolute, traversing, and platform-dependent paths are rejected."""
    config_path = _write_config(tmp_path, f"[{{ Demo = {json.dumps(nav_path)} }}]")

    with pytest.raises(DocsNotebookConfigError, match="Navigation path"):
        list_docs_notebooks(config_path)


def test_rejects_duplicate_notebook_references(tmp_path):
    """A notebook may appear only once in the docs navigation."""
    _write_notebook(tmp_path)
    config_path = _write_config(
        tmp_path,
        '[{ First = "nbs/example.md" }, { Second = "nbs/./example.md" }]',
    )

    with pytest.raises(DocsNotebookConfigError, match=r"Duplicate.*nbs/example.ipynb"):
        list_docs_notebooks(config_path)


def test_rejects_missing_notebook(tmp_path):
    """Every published notebook page must have a source notebook."""
    config_path = _write_config(tmp_path, '[{ Demo = "nbs/missing.md" }]')

    with pytest.raises(DocsNotebookConfigError, match=r"does not exist.*nbs/missing"):
        list_docs_notebooks(config_path)

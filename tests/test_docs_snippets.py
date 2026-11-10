"""Docs snippets stay runnable up to the point where they would use the cloud.

The home page's Palace quick start is pulled out of ``docs/index.md`` and run
as written, up to the ``sim.run(...)`` line: it builds the geometry, validates,
meshes and writes ``config.json``. Nothing here submits a cloud job.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"

QUICKSTART_HEADING = "### Palace quick start"


def _python_block_after(markdown: str, heading: str) -> str:
    """Return the first fenced ``python`` block that follows *heading*."""
    _, found, rest = markdown.partition(heading)
    assert found, f"heading {heading!r} not found"
    match = re.search(r"```python\n(.*?)```", rest, flags=re.DOTALL)
    assert match, f"no python block after {heading!r}"
    return match.group(1)


def _before_cloud_run(source: str) -> str:
    """Return *source* up to, not including, the first line that calls ``.run(``."""
    lines = source.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if ".run(" in line:
            return "".join(lines[:index])
    pytest.fail("the snippet has no .run( line")


def test_home_page_palace_quickstart_builds_a_config(tmp_path, monkeypatch):
    source = _python_block_after(
        (DOCS / "index.md").read_text(encoding="utf-8"), QUICKSTART_HEADING
    )
    assert "sim.run(check_cache=True)" in source, "the quick start reuses cached jobs"

    monkeypatch.chdir(tmp_path)
    namespace: dict = {"__name__": "docs_snippet"}
    exec(compile(_before_cloud_run(source), "docs/index.md", "exec"), namespace)  # noqa: S102

    sim = namespace["sim"]
    assert sim.validate_config().valid
    config_path = Path(sim.write_config())
    assert config_path.is_file()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert len(config["Boundaries"]["LumpedPort"]) == 2
    assert config["Problem"]["Type"] == "Driven"


def _nav_files(entries: list) -> list[str]:
    files: list[str] = []
    for entry in entries:
        for value in entry.values():
            if isinstance(value, list):
                files.extend(_nav_files(value))
            else:
                files.append(value)
    return files


def test_nav_pages_exist():
    """Every nav page that is not a generated notebook page is a file in docs/."""
    config = tomllib.loads((DOCS / "zensical.toml").read_text(encoding="utf-8"))
    pages = [
        p for p in _nav_files(config["project"]["nav"]) if not p.startswith("nbs/")
    ]
    assert pages
    missing = [p for p in pages if not (DOCS / p).is_file()]
    assert not missing

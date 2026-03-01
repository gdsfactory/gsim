"""MkDocs hooks for gsim documentation.

Auto-generates docs/api.md from each module's ``__all__`` so the API
reference stays in sync with the source code.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).parent
API_MD = DOCS_DIR / "api.md"

# Modules to document, in display order.
# Each entry is (heading, dotted module path).
MODULES: list[tuple[str, str]] = [
    ("gsim", "gsim"),
    ("gsim.common", "gsim.common"),
    ("gsim.common.viz", "gsim.common.viz"),
    ("gsim.palace", "gsim.palace"),
    ("gsim.meep", "gsim.meep"),
    ("gsim.gcloud", "gsim.gcloud"),
]

# Names to skip (internal aliases, re-exports that duplicate another section).
SKIP_NAMES: set[str] = {
    "__version__",
    "Stack",  # backward-compat alias for LayerStack
    "MeshConfigModel",  # internal alias
}


def _is_public_api(name: str, obj: object) -> bool:
    """Return True if *name* should appear in the API docs."""
    if name.startswith("_"):
        return False
    if name in SKIP_NAMES:
        return False
    # Skip plain values (str, int, float, bool) — keep classes, functions, enums, etc.
    return not isinstance(obj, (str, int, float, bool))


def _classify(obj: object) -> str:
    """Return a section label for *obj*."""
    if isinstance(obj, type) and issubclass(obj, Exception):
        return "Exceptions"
    if isinstance(obj, type):
        return "Classes"
    if callable(obj):
        return "Functions"
    return "Data"


def _module_section(heading: str, module_path: str) -> list[str]:
    """Generate mkdocstrings directives for one module."""
    try:
        mod: ModuleType = importlib.import_module(module_path)
    except Exception:
        logger.warning("Could not import %s — skipping", module_path)
        return []

    names = getattr(mod, "__all__", None)
    if names is None:
        names = [n for n in dir(mod) if not n.startswith("_")]

    # Group names by kind
    groups: dict[str, list[str]] = {}
    for name in sorted(names):
        obj = getattr(mod, name, None)
        if obj is None or not _is_public_api(name, obj):
            continue
        kind = _classify(obj)
        groups.setdefault(kind, []).append(name)

    if not groups:
        return []

    lines: list[str] = [f"## {heading}\n"]
    # Preferred section order
    for section in ("Classes", "Functions", "Data", "Exceptions"):
        items = groups.get(section)
        if not items:
            continue
        lines.append(f"### {section}\n")
        for name in items:
            lines.append(f"::: {module_path}.{name}\n")
    return lines


def generate_api_md() -> str | None:
    """Generate the full API reference markdown.

    Returns ``None`` if any module fails to import (e.g. missing
    dependencies in CI) so callers can skip writing a partial file.
    """
    lines = ["# API Reference\n"]
    for heading, module_path in MODULES:
        try:
            importlib.import_module(module_path)
        except Exception:
            logger.warning("Could not import %s — skipping generation", module_path)
            return None
        section = _module_section(heading, module_path)
        if section:
            lines.extend(section)
            lines.append("---\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MkDocs hook entry points
# ---------------------------------------------------------------------------


def on_pre_build(config: Any, **kwargs: Any) -> None:
    """Generate api.md before the build starts."""
    api_content = generate_api_md()
    if api_content is None:
        logger.warning("Skipping api.md generation (import failure)")
        return
    API_MD.write_text(api_content)
    logger.info("Generated %s (%d bytes)", API_MD, len(api_content))

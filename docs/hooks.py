"""MkDocs hooks for gsim documentation.

Converts ugly doctest blockquote examples into syntax-highlighted
Python code blocks on API pages.
"""

from __future__ import annotations

import re
from typing import Any

from pygments import (
    highlight,  # type: ignore[import-untyped]  # ty: ignore[unresolved-import]
)
from pygments.formatters import (
    HtmlFormatter,  # type: ignore[import-untyped]  # ty: ignore[unresolved-import]
)
from pygments.lexers import (
    PythonLexer,  # type: ignore[import-untyped]  # ty: ignore[unresolved-import]
)

_EXAMPLE_RE = re.compile(
    r'<details class="example" open>\s*<summary>Example</summary>'
    r"(.*?)"
    r"</details>",
    re.DOTALL,
)

_TAG_RE = re.compile(r"<[^>]+>")

_lexer = PythonLexer()
_formatter = HtmlFormatter(nowrap=False, cssclass="codehilite")


def _example_to_code(match: re.Match) -> str:
    """Replace a doctest example block with highlighted Python."""
    inner = match.group(1)
    # Strip all HTML tags to get plain text
    text = _TAG_RE.sub("", inner)
    # Clean up >>> and ... prefixes
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("&gt;&gt;&gt;"):
            stripped = stripped[12:].strip()
        elif stripped.startswith((">>>", "...")):
            stripped = stripped[3:].strip()
        elif not stripped:
            continue
        else:
            lines.append(stripped)
            continue
        lines.append(stripped)

    code = "\n".join(lines).strip()
    if not code:
        return ""

    highlighted = highlight(code, _lexer, _formatter)
    return (
        '<details class="example" open>'
        "<summary>Example</summary>"
        f"{highlighted}"
        "</details>"
    )


def on_page_content(html: str, page: Any, **kwargs: Any) -> str:
    """Convert doctest examples to highlighted code on API pages."""
    if page.file.src_path.startswith("api/"):
        return _EXAMPLE_RE.sub(_example_to_code, html)
    return html

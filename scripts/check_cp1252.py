#!/usr/bin/env python3
"""Fail if any Python source file contains characters that cannot be encoded as cp1252.

Windows's default ANSI codepage (cp1252) is used by Python's ``open()`` and by
setuptools/pip when reading source files during install. A character outside
cp1252 (e.g. ``->``, ``sigma``) triggers ``UnicodeDecodeError`` at install time.
See issue #122 / PR #123.
"""

from __future__ import annotations

import sys
from pathlib import Path


def find_violations(path: Path) -> list[tuple[int, str, str]]:
    """Return ``(lineno, char, line)`` tuples for cp1252-incompatible chars."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    out: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        try:
            line.encode("cp1252")
        except UnicodeEncodeError as exc:
            out.append((lineno, line[exc.start : exc.end], line.rstrip()))
    return out


def main(argv: list[str]) -> int:
    """Scan ``argv`` paths, print any violations, exit 1 if found else 0."""
    any_bad = False
    for arg in argv:
        p = Path(arg)
        for lineno, ch, line in find_violations(p):
            any_bad = True
            print(f"{p}:{lineno}: cp1252-incompatible {ch!r}: {line}")
    if any_bad:
        print(
            "\nReplace the flagged characters with ASCII/cp1252 equivalents.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

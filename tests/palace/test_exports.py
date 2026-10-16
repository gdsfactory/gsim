"""Every name gsim.palace exports is importable."""

from __future__ import annotations

import gsim.palace as palace


def test_every_exported_name_exists():
    missing = [name for name in palace.__all__ if not hasattr(palace, name)]
    assert missing == []

"""One shape for every optional dependency's import guard."""

from __future__ import annotations

import sys

import pytest

from gsim.common.optional import require_module


class TestRequireModule:
    def test_it_returns_the_imported_module(self):
        assert require_module("json", extra="tcad").dumps([1]) == "[1]"

    def test_a_missing_module_names_the_packaging_extra(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "nowhere_at_all", None)
        with pytest.raises(ImportError, match=r"gsim\[femwell\]") as raised:
            require_module("nowhere_at_all", extra="femwell")
        assert "nowhere_at_all" in str(raised.value)

    def test_a_caller_supplied_hint_replaces_the_message(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "nowhere_at_all", None)
        with pytest.raises(ImportError, match="say this instead"):
            require_module("nowhere_at_all", extra="femwell", hint="say this instead")

    def test_the_original_import_error_stays_the_cause(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "nowhere_at_all", None)
        with pytest.raises(ImportError) as raised:
            require_module("nowhere_at_all", extra="tcad")
        assert isinstance(raised.value.__cause__, ImportError)

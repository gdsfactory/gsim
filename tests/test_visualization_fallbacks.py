"""Verify cleanup and headless rendering fallback at the display boundary."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import pyvista as pv

from gsim import viz


@pytest.mark.parametrize("xvfb_failure", [False, True])
def test_forced_offscreen_environment(monkeypatch, xvfb_failure):
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setenv("GSIM_FORCE_OFFSCREEN", "1")
    monkeypatch.delenv("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", raising=False)
    monkeypatch.setattr(pv, "OFF_SCREEN", False)
    startup = Mock(side_effect=RuntimeError("no display")) if xvfb_failure else Mock()
    monkeypatch.setattr(viz, "_start_xvfb", startup)
    assert viz._is_headless()
    assert viz._ensure_pyvista() is pv
    assert pv.OFF_SCREEN
    assert "DISPLAY" not in __import__("os").environ
    startup.assert_called_once()


def test_virtual_display_startup(monkeypatch):
    display = Mock()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(
        sys.modules,
        "pyvirtualdisplay",
        SimpleNamespace(Display=Mock(return_value=display)),
    )
    viz._start_xvfb()
    display.start.assert_called_once()


@pytest.mark.parametrize("individual", [False, True])
def test_failed_live_view_close_still_clears_registry(monkeypatch, caplog, individual):
    plotter = Mock()
    plotter.close.side_effect = RuntimeError("already destroyed")
    monkeypatch.setattr(viz, "_LIVE_PLOTTERS", {"view": plotter})
    if individual:
        viz.close_interactive_view("absent")
        viz.close_interactive_view("view")
    else:
        viz.close_interactive_views()
    assert viz.interactive_views() == ()
    assert "Failed to close" in caplog.text


def test_static_render_failure_closes_plotter(monkeypatch):
    plotter = Mock()
    plotter.screenshot.side_effect = RuntimeError("render unavailable")
    monkeypatch.setattr(viz, "_in_notebook", lambda: True)
    viz._show_static(plotter)
    plotter.close.assert_called_once()


def test_notebook_detection_and_view_identity(monkeypatch):
    monkeypatch.setattr("scooby.in_ipykernel", lambda: True)
    assert viz._in_notebook()
    monkeypatch.setattr(
        "scooby.in_ipykernel", Mock(side_effect=RuntimeError("no shell"))
    )
    assert not viz._in_notebook()
    assert viz._view_id(SimpleNamespace()).startswith("plotter-")

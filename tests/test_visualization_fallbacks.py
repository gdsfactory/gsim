"""Verify cleanup and headless rendering fallback at the display boundary."""

import os
import subprocess
import sys
import textwrap
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
    registered = []
    monkeypatch.setattr(viz, "_XVFB", None)
    monkeypatch.setattr(viz.atexit, "register", registered.append)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(
        sys.modules,
        "pyvirtualdisplay",
        SimpleNamespace(Display=Mock(return_value=display)),
    )
    viz._start_xvfb()
    viz._start_xvfb()
    display.start.assert_called_once()
    assert registered == [viz._stop_xvfb]
    viz._stop_xvfb()
    display.stop.assert_called_once()


def test_forked_child_leaves_parent_display_running(monkeypatch):
    display = Mock()
    monkeypatch.setattr(viz, "_XVFB", (display, os.getpid() + 1))
    viz._stop_xvfb()
    display.stop.assert_not_called()


_FAKE_PYVIRTUALDISPLAY = """
import os


class Display:
    def __init__(self, **kwargs):
        pass

    def _log(self, event):
        with open(os.environ["FAKE_XVFB_LOG"], "a") as f:
            f.write(event + "\\n")

    def start(self):
        self._log("start")
        return self

    def stop(self):
        self._log("stop")
        return self
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Xvfb is Linux-only")
def test_virtual_display_is_lazy_and_stopped_at_exit(tmp_path):
    """Import alone starts no display; rendering starts one, reused, then stopped.

    A recording stand-in replaces ``pyvirtualdisplay`` so no real Xvfb runs.
    """
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "pyvirtualdisplay.py").write_text(_FAKE_PYVIRTUALDISPLAY)
    log = tmp_path / "xvfb.log"
    script = textwrap.dedent(
        """
        import os
        from pathlib import Path

        import gsim
        from gsim import viz

        log = Path(os.environ["FAKE_XVFB_LOG"])
        assert not log.exists(), "import started a virtual display"
        for _ in range(2):
            viz._make_plotter(False).close()
        """
    )
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    env.pop("GSIM_FORCE_OFFSCREEN", None)
    env["FAKE_XVFB_LOG"] = str(log)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(fake), env.get("PYTHONPATH")])
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert log.read_text().split() == ["start", "stop"]


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

"""gsim.modulator imports with neither optional extra installed."""

from __future__ import annotations

import importlib
import sys


def test_imports_without_devsim_or_femwell(monkeypatch):
    import gsim

    # Block the optional runtimes even when they happen to be installed.
    for name in ("devsim", "femwell", "skfem"):
        monkeypatch.setitem(sys.modules, name, None)
    # Re-importing rebinds gsim.modulator on its parent package; record it so
    # the fresh module objects do not outlive this test and shadow the ones
    # other tests patch.
    monkeypatch.setattr(
        gsim, "modulator", importlib.import_module("gsim.modulator"), raising=False
    )
    for name in [m for m in sys.modules if m.startswith("gsim.modulator")]:
        monkeypatch.delitem(sys.modules, name)

    modulator = importlib.import_module("gsim.modulator")

    assert modulator.Study is not None
    assert modulator.Device is not None
    assert modulator.ChargeStage is not None
    assert modulator.CarriersStage is not None
    assert modulator.OpticalStage is not None
    assert modulator.RFStage is not None
    assert modulator.LineStage is not None
    # The Route seam is importable without either backend: picking a
    # Route is a configuration choice, not a dependency. It is off
    # ``modulator.__all__``, so it is read from its own module.
    assert modulator.PalaceMode is not None
    assert importlib.import_module("gsim.modulator.route").DEFAULT_PALACE_STRIPS >= 1
    assert modulator.OpticalStage().route == "femwell"
    assert modulator.RFStage().route == "femwell"

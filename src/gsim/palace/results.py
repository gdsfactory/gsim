"""Load and map Palace S-parameter results to port names.

Standalone utility — works with a Palace output directory, or the results
dict returned by ``sim.run()``.

Usage::

    from gsim.palace.results import load_sparams

    # From a results dict (notebook workflow)
    results = sim.run()
    df = load_sparams(results)

    # From a directory path
    df = load_sparams("./sim/output")
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_sparams(
    source: str | Path | dict,
    *,
    port_info_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load Palace S-parameter CSV with port-name-based columns.

    Reads ``port-S.csv`` and ``port_information.json``, then renames the
    numeric Palace column headers (``|S[2][1]| (dB)``) to human-readable
    names (``S_o2_o1_dB``, ``S_o2_o1_deg``).

    If ``port_information.json`` is missing or lacks ``name`` fields,
    falls back to numeric names (``S_p2_p1_dB``).

    Args:
        source: One of:

            - **results dict** returned by ``sim.run()`` — keys are
              filenames, values are ``Path`` objects.
            - **directory path** — the top-level sim dir or the
              ``output/palace/`` subdirectory.
        port_info_path: Explicit path to ``port_information.json``.
            When *None* the file is auto-discovered next to the CSV or
            in common locations relative to the output directory.

    Returns:
        DataFrame with columns: ``freq_ghz`` plus one ``S_<to>_<from>_dB``
        and one ``S_<to>_<from>_deg`` column per S-parameter entry.

    Raises:
        FileNotFoundError: If ``port-S.csv`` cannot be found.
    """
    csv_path, base_dir = _resolve_source(source)
    if csv_path is None:  # pragma: no cover — _resolve_source raises first
        msg = "port-S.csv not found"
        raise FileNotFoundError(msg)

    # --- locate port_information.json ----------------------------------------
    port_map = _load_port_map(base_dir, csv_path, port_info_path)

    # --- read CSV and rename columns -----------------------------------------
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    rename: dict[str, str] = {}
    for col in df.columns:
        new_name = _rename_column(col, port_map)
        if new_name is not None:
            rename[col] = new_name

    df = df.rename(columns=rename)

    # Rename the frequency column
    freq_col = [c for c in df.columns if c.startswith("f")]
    if freq_col:
        # Palace writes frequency in GHz — keep as-is but normalise name
        df = df.rename(columns={freq_col[0]: "freq_ghz"})

    return df


def get_port_map(source: str | Path | dict) -> dict[int, str]:
    """Return the ``{port_number: port_name}`` mapping.

    Accepts a directory path or a results dict from ``sim.run()``.
    """
    csv_path, base_dir = _resolve_source(source, require_csv=False)
    return _load_port_map(base_dir, csv_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_source(
    source: str | Path | dict,
    *,
    require_csv: bool = True,
) -> tuple[Path | None, Path]:
    """Turn *source* into ``(csv_path, base_dir)``.

    *source* can be a directory path or a results dict from ``sim.run()``.
    """
    if isinstance(source, dict):
        # Results dict: {"port-S.csv": Path(...), ...}
        csv_val = source.get("port-S.csv")
        if csv_val is not None:
            csv_path = Path(csv_val)
            return csv_path, csv_path.parent
        # Try port_information.json to derive base dir
        for val in source.values():
            p = Path(val)
            if p.exists():
                return None, p.parent
        if require_csv:
            raise FileNotFoundError("Results dict has no 'port-S.csv' entry")
        return None, Path()

    output_dir = Path(source)
    csv_path = _find_file(output_dir, "port-S.csv")
    if csv_path is None and require_csv:
        raise FileNotFoundError(
            f"port-S.csv not found in {output_dir} or its subdirectories"
        )
    return csv_path, output_dir


_S_PATTERN = re.compile(r"(\|?)S\[(\d+)\]\[(\d+)\]\|?\s*\((\w+\.?)\)")


def _rename_column(col: str, port_map: dict[int, str]) -> str | None:
    """Rename a single Palace S-parameter column header.

    ``|S[2][1]| (dB)``  -> ``S_o2_o1_dB``
    ``arg(S[2][1]) (deg.)`` -> ``S_o2_o1_deg``
    """
    # Match magnitude: |S[i][j]| (dB)
    m = re.match(r"\|S\[(\d+)\]\[(\d+)\]\|\s*\((\w+\.?)\)", col)
    if m:
        i, j, unit = int(m.group(1)), int(m.group(2)), m.group(3)
        ni = port_map.get(i, f"p{i}")
        nj = port_map.get(j, f"p{j}")
        unit_tag = unit.rstrip(".").lower()
        return f"S_{ni}_{nj}_{unit_tag}"

    # Match phase: arg(S[i][j]) (deg.)
    m = re.match(r"arg\(S\[(\d+)\]\[(\d+)\]\)\s*\((\w+\.?)\)", col)
    if m:
        i, j, unit = int(m.group(1)), int(m.group(2)), m.group(3)
        ni = port_map.get(i, f"p{i}")
        nj = port_map.get(j, f"p{j}")
        unit_tag = unit.rstrip(".").lower()
        return f"S_{ni}_{nj}_{unit_tag}"

    return None


def _load_port_map(
    output_dir: Path,
    csv_path: Path | None,
    port_info_path: str | Path | None = None,
) -> dict[int, str]:
    """Build ``{port_number: port_name}`` from ``port_information.json``."""
    if port_info_path is not None:
        info_path = Path(port_info_path)
    else:
        info_path = _find_port_info(output_dir, csv_path)

    if info_path is None or not info_path.exists():
        logger.warning(
            "port_information.json not found — using numeric port names (p1, p2, …)"
        )
        return {}

    with open(info_path) as f:
        data = json.load(f)

    port_map: dict[int, str] = {}
    for entry in data.get("ports", []):
        num = entry.get("portnumber")
        name = entry.get("name")
        if num is not None and name is not None:
            port_map[num] = name
        elif num is not None:
            # Older sims without name field
            port_map[num] = f"p{num}"

    if port_map and all(
        v.startswith("p") and v[1:].isdigit() for v in port_map.values()
    ):
        logger.info(
            "port_information.json has no 'name' fields — "
            "columns will use numeric names (p1, p2, …). "
            "Re-mesh to get named columns."
        )

    return port_map


def _find_port_info(output_dir: Path, csv_path: Path | None) -> Path | None:
    """Search common locations for ``port_information.json``."""
    candidates = [
        output_dir / "port_information.json",
        output_dir / "output" / "port_information.json",
    ]
    if csv_path is not None:
        candidates.insert(0, csv_path.parent / "port_information.json")
        # port_information.json is often written at the sim root, not in output/palace/
        candidates.append(csv_path.parent.parent / "port_information.json")
        candidates.append(csv_path.parent.parent.parent / "port_information.json")

    for p in candidates:
        if p.exists():
            return p
    return None


def _find_file(base: Path, name: str) -> Path | None:
    """Find *name* in *base* or common Palace subdirectories."""
    candidates = [
        base / name,
        base / "output" / "palace" / name,
        base / "palace" / name,
    ]
    for p in candidates:
        if p.exists():
            return p

    # Last resort: recursive glob (shallow)
    matches = list(base.rglob(name))
    return matches[0] if matches else None

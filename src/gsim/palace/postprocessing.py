"""Configure Palace energy output by mesh physical-group name."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from gsim.palace.field_viz import resolve_physical_groups


def add_domain_energy_postprocessing(
    sim_dir: str | Path, group_names: Sequence[str]
) -> dict[str, int]:
    """Request per-domain energies in ``domain-E.csv`` after ``write_config()``.

    Returns each requested group's Palace postprocessing index. Repeated calls
    preserve existing requests and do not add duplicate domain attributes.
    """
    sim_dir = Path(sim_dir)
    attributes = resolve_physical_groups(sim_dir, group_names, dimension=3)
    config_path = sim_dir / "config.json"
    config = json.loads(config_path.read_text())
    postprocessing = config["Domains"].setdefault("Postprocessing", {})
    entries = postprocessing.setdefault("Energy", [])
    indices = {
        int(attribute): int(entry["Index"])
        for entry in entries
        for attribute in entry["Attributes"]
        if len(entry["Attributes"]) == 1
    }
    next_index = max((int(entry["Index"]) for entry in entries), default=0) + 1
    requested: dict[str, int] = {}
    for name, attribute in zip(group_names, attributes, strict=True):
        if attribute not in indices:
            indices[attribute] = next_index
            entries.append({"Index": next_index, "Attributes": [attribute]})
            next_index += 1
        requested[name] = indices[attribute]
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return requested


__all__ = ["add_domain_energy_postprocessing"]

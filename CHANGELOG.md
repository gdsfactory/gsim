# Changelog

## v0.0.17 (unreleased)

### Added
- **Optional palace-toolkit dependency**: Added `[palace-toolkit]` extra in `pyproject.toml`.
  When installed, GSIM automatically discovers the bundled Palace binary.
  Install with `pip install gsim[palace-toolkit]`.
- **Palace binary resolver** (`gsim.palace.runtime`): `resolve_palace_binary()` and
  `resolve_palace_library_dir()` with tiered resolution:
  `PALACE_BIN` env → `PALACE_EXECUTABLE`/PATH → `palacetoolkit.palace_runtime`
  (when installed) → `None`.
- **NaN-free field visualization** (`gsim.palace.fields`): A new module providing
  direct-mesh rendering for Palace boundary and volume fields:
  - `plot_boundary_field()` — renders the actual boundary mesh cells directly
    via PyVista, **eliminating NaN values** that occurred with the old
    probe-grid resampling approach.
  - `load_boundary_field_data()` — load and select boundary faces by entity name,
    boundary type, or explicit attribute IDs.
  - `plot_volume_slice()` / `plot_volume_contours()` — slice + direct-mesh rendering.
  - `build_selector_context()` / `resolve_entity_attributes()` — resolve entity names
    to attribute tags from the Palace config + mesh physical-group map.
  - `extract_boundary_cells()` / `activate_vector_component()` — cell extraction and
    vector-component scalar activation.
- **Tests**: `tests/palace/test_fields.py` and `tests/palace/test_runtime.py`
  covering the new resolver and field-visualization functions.

### Changed
- **`PalaceSim.run_local()`** (`gsim.palace.base`): The `use_apptainer=False` branch
  now falls back to the runtime resolver, then to `"palace"` in PATH. When using a
  bundled binary from `palace-toolkit`, `LD_LIBRARY_PATH` is automatically set.
- **Notebook `nbs/palace_cpw_fields.ipynb`**: Added cells demonstrating NaN-free
  `plot_boundary_field` with explicit `assert np.isnan(...).sum() == 0` check and
  `plot_volume_slice` for volume slices.

### Removed
- **`src/gsim/viz.py`**: Removed dead-code duplicates
  (`_sample_topview_field_duplicate`, `_plot_topview_duplicate`,
  `_plot_cross_section_duplicate`) that were unused copies of the public API.
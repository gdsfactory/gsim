# Palace Module TODO

## API Consistency

### Update EigenmodeSim and ElectrostaticSim
The `DrivenSim` class was updated with a cleaner API. Apply the same changes to:

- [ ] `EigenmodeSim` (eigenmode.py - 776 lines)
- [ ] `ElectrostaticSim` (electrostatic.py - 621 lines)

Changes needed:
- Add `set_output_dir()` method
- Remove `output_dir` parameter from `mesh()`
- Remove `output_dir` parameter from `simulate()`
- Add `write_config()` method
- Mesh should only generate mesh, not config

### Placeholder Methods
- [ ] `simulate_local()` - Currently raises NotImplementedError. Keep as placeholder until local Palace execution is implemented.

## Code Organization

### Extract Common Base Class
Create a common base module with shared functionality:

- [ ] Create `src/gsim/palace/base.py` with `PalaceSimBase` class containing:
  - `set_output_dir()` / `output_dir` property
  - `set_geometry()`
  - `set_stack()`
  - `validate()` base implementation
  - `_resolve_stack()`
  - `write_config()` (if signature is identical)
  - Common private attributes (`_output_dir`, `_last_mesh_result`, etc.)

### Break Down Large Files
Files over 500 lines need refactoring:

#### mesh/generator.py (1137 lines)
- [ ] Extract geometry functions to `mesh/geometry.py`
- [ ] Extract physical group assignment to `mesh/groups.py`
- [ ] Extract config generation to `mesh/config.py` or `config/generator.py`
- [ ] Keep `generate_mesh()` as the main entry point

#### driven.py (1003 lines)
- [ ] Move base class methods to `base.py`
- [ ] Consider extracting port configuration to separate module
- [ ] Consider extracting mesh config building logic

#### eigenmode.py (776 lines)
- [ ] After extracting base class, should be under 500 lines

#### electrostatic.py (621 lines)
- [ ] After extracting base class, should be under 500 lines

## File Structure After Refactoring

```
src/gsim/palace/
├── __init__.py
├── base.py              # NEW: PalaceSimBase class
├── driven.py            # DrivenSim (inherits from base)
├── eigenmode.py         # EigenmodeSim (inherits from base)
├── electrostatic.py     # ElectrostaticSim (inherits from base)
├── config/
│   ├── __init__.py
│   └── generator.py     # NEW: Palace config.json generation
├── mesh/
│   ├── __init__.py
│   ├── generator.py     # Main generate_mesh() entry point
│   ├── geometry.py      # NEW: Geometry extraction
│   ├── groups.py        # NEW: Physical group assignment
│   ├── gmsh_utils.py
│   └── pipeline.py
├── models/
│   └── ...
└── ports/
    └── ...
```

## Testing
- [ ] Add tests for the new API (`set_output_dir`, `mesh`, `write_config`, `simulate` flow)
- [ ] Ensure backward compatibility is tested (or document breaking changes)

dist:
  uv run python -m build --wheel

dev:
  uv venv --python 3.12 --clear
  uv sync --dev
  uv pip install -e .
  uvx pre-commit install

# Install the release versioning tool.
tbump:
    uv tool install tbump

# bump version
bump version:
    tbump "{{ version }}"

uv:
  curl -LsSf https://astral.sh/uv/install.sh | sh

inits:
  cd src/gsim && uvx mkinit --relative --recursive --write && uvx ruff format __init__.py

ipykernel:
  uv run python -m ipykernel install --user --name gsim --display-name gsim

test:
  uv run pytest -s -n logical --cov-report=term-missing --cov-report=html --cov-report=xml --cov=src/gsim

cov:
  uv run pytest --cov=gsim --cov-report=term-missing:skip-covered --cov-report=xml

# Copy the root CHANGELOG into docs/ so zensical can build it (docs_dir is docs/)
sync-changelog:
  cp CHANGELOG.md docs/CHANGELOG.md

docs: sync-changelog
  uv run python scripts/plot_material_indices.py
  uv run zensical build -f docs/zensical.toml

# Run a notebook normally (interactive plots): just nbrun nbs/foo.ipynb
nbrun +notebooks: ipykernel
  for nb in {{notebooks}}; do \
    uv run papermill "$nb" "$nb" -k gsim; \
  done

nbclean-all:
  find . -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" -not -path "./.venv/*" | xargs just nbclean

nbclean +filenames:
  for filename in {{filenames}}; do \
    uvx nbstripout "$filename"; \
    uvx nb-clean clean --remove-empty-cells "$filename"; \
    jq --indent 1 'del(.metadata.papermill)' "$filename" > "$filename.tmp" && mv "$filename.tmp" "$filename"; \
  done

taper_sc_nc:
  uv run python samples/meep_taper_sc_nc.py

tree:
  @tree -a -I .git --gitignore

clean: nbclean-all
  rm -rf site docs/site
  rm -rf .venv
  rm -f uv.lock
  find src -name "*.c" | xargs rm -rf
  find src -name "*.pyc" | xargs rm -rf
  find src -name "*.so" | xargs rm -rf
  find src -name "*.pyd" | xargs rm -rf
  find . -name "*.egg-info" | xargs rm -rf
  find . -name ".ipynb_checkpoints" | xargs rm -rf
  find . -name ".mypy_cache" | xargs rm -rf
  find . -name ".pytest_cache" | xargs rm -rf
  find . -name ".ruff_cache" | xargs rm -rf
  find . -name "__pycache__" | xargs rm -rf
  find . -name "build" | xargs rm -rf
  find . -name "builds" | xargs rm -rf
  find . -name "dist" -not -path "*node_modules*" | xargs rm -rf

dist:
  uv run python -m build --wheel

dev:
  uv venv --python 3.12 --clear
  uv sync --all-extras
  uv pip install -e .
  uvx pre-commit install

# Version bumping
[linux,macos]
bver:
    curl -LsSf https://github.com/flaport/bver/releases/latest/download/install.sh | sh

# Version bumping
[windows]
bver:
    powershell -ExecutionPolicy ByPass -c "irm https://github.com/flaport/bver/releases/latest/download/install.ps1 | iex"

# bump version
bump version="patch":
    bver bump "{{ version }}"

uv:
  curl -LsSf https://astral.sh/uv/install.sh | sh

inits:
  cd src/gsim && uvx mkinit --relative --recursive --write && uvx ruff format __init__.py

ipykernel:
  uv run python -m ipykernel install --user --name gsim --display-name gsim

test:
  uv run pytest -s -n logical

docs:
  uv run mkdocs build

serve:
  uv run mkdocs serve -a localhost:8080

nbrun: ipykernel
  find nbs -maxdepth 1 -mindepth 1 -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" -not -path "./.venv/*" | xargs parallel -j `nproc --all` uv run papermill {} {} -k gsim :::

nbdocs:
  find nbs -maxdepth 1 -mindepth 1 -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" -not -path "./.venv/*" | xargs parallel -j `nproc --all` uv run jupyter nbconvert --to markdown --embed-images {} --output-dir docs/nbs ':::'

nbclean-all:
  find . -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" -not -path "./.venv/*" | xargs just nbclean

nbclean +filenames:
  for filename in {{filenames}}; do \
    uvx nbstripout "$filename"; \
    uvx nb-clean clean --remove-empty-cells "$filename"; \
    jq --indent 1 'del(.metadata.papermill)' "$filename" > "$filename.tmp" && mv "$filename.tmp" "$filename"; \
  done

tree:
  @tree -a -I .git --gitignore

clean: nbclean-all
  rm -rf site
  rm -rf .venv
  rm -f uv.lock
  rm -rf docs/nbs/*
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

#!/usr/bin/env bash
# Record the missing 2-D cases, execute the validation notebook against the
# recorded runs, and publish the result as the `pdk-validation` gh-pages
# variant.  Designed to run detached (systemd) for an hour or more.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}"
MEEP_PY="${GSIM_MEEP_PYTHON:-$MAMBA_ROOT_PREFIX/envs/meep/bin/python}"
if [[ -n "${GSIM_NOTEBOOK_PYTHON:-}" ]]; then
  NOTEBOOK_PY="$GSIM_NOTEBOOK_PYTHON"
elif [[ -x "$REPO/.venv/bin/python" ]]; then
  NOTEBOOK_PY="$REPO/.venv/bin/python"
else
  NOTEBOOK_PY="$MEEP_PY"
fi
BUILD="${PDK_BUILD_DIR:-/tmp/opencode/pdk-notebook-build}"
RUNS="$REPO/nbs/data/pdk_validation/runs"
VARIANT="${PDK_VARIANT_DIR:-/tmp/opencode/pdk-validation-variant}"

mkdir -p "$BUILD"
exec >>"$BUILD/job.log" 2>&1

echo "[$(date -Is)] start"
"$NOTEBOOK_PY" -m ipykernel install --user --name gsim-pdk --display-name "gsim-pdk" >/dev/null

echo "[$(date -Is)] recording missing 2-D runs (MZI, ring)"
GSIM_PDK_REUSE= "$MEEP_PY" "$REPO/scripts/run_missing_2d.py"

echo "[$(date -Is)] executing notebook against recorded runs"
export GSIM_PDK_REUSE="$RUNS"
export GSIM_PDK_REUSE_FALLBACK_2D=1
export GSIM_MEEP_PYTHON="$MEEP_PY"
cd "$REPO"
"$NOTEBOOK_PY" -m nbconvert --to notebook --execute \
  "$REPO/nbs/photonic_pdk_validation.ipynb" \
  --output "$BUILD/photonic_pdk_validation.executed.ipynb" \
  --ExecutePreprocessor.timeout=10800 \
  --ExecutePreprocessor.kernel_name=gsim-pdk

echo "[$(date -Is)] converting notebook to HTML"
"$NOTEBOOK_PY" -m nbconvert --to html \
  "$BUILD/photonic_pdk_validation.executed.ipynb" \
  --output "$BUILD/notebook.html"

echo "[$(date -Is)] building variant"
"$NOTEBOOK_PY" "$REPO/scripts/build_pdk_validation_variant.py" \
  --out "$VARIANT" \
  --notebook-html "$BUILD/notebook.html" \
  --notebook-ipynb "$BUILD/photonic_pdk_validation.executed.ipynb"

echo "[$(date -Is)] publishing"
"/tmp/opencode/gsim-pages/_tools/publish_variant.sh" pdk-validation "$VARIANT" \
  "PDK validation — notebook & layouts" \
  "Executed notebook with every available validation case plus GDS layouts"

echo "[$(date -Is)] done"

#!/bin/bash
set -euo pipefail

# Local MEEP simulation test loop:
#   1. Regenerate config files from gsim
#   2. Copy to simulation-engines Docker context
#   3. Rebuild Docker image
#   4. Run simulation in container
#   5. Display S-parameter results

GSIM_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SIM_ENGINE_DIR="$(cd "$GSIM_DIR/../simulation-engines/meep" && pwd)"
CONFIG_OUTPUT="$GSIM_DIR/nbs/meep-sim-test"
RESULTS_DIR="$(mktemp -d)"
IMAGE_NAME="meep-local"
MEEP_NP="${MEEP_NP:-4}"

cleanup() { rm -rf "$RESULTS_DIR"; }
trap cleanup EXIT

echo "=== 1. Regenerate config ==="
(cd "$GSIM_DIR" && uv run python nbs/generate_meep_config.py)

echo ""
echo "=== 2. Copy to Docker context ==="
cp "$CONFIG_OUTPUT"/{layout.gds,run_meep.py,sim_config.json} "$SIM_ENGINE_DIR/src/"
echo "Copied layout.gds, run_meep.py, sim_config.json -> $SIM_ENGINE_DIR/src/"

echo ""
echo "=== 3. Build Docker image ==="
docker build -t "$IMAGE_NAME" -f "$SIM_ENGINE_DIR/Dockerfile.local" "$SIM_ENGINE_DIR"

echo ""
echo "=== 4. Run simulation (MEEP_NP=$MEEP_NP) ==="
docker run --rm \
    -e MEEP_NP="$MEEP_NP" \
    -v "$RESULTS_DIR:/app/data" \
    "$IMAGE_NAME"

echo ""
echo "=== 5. Results ==="
CSV="$RESULTS_DIR/s_parameters.csv"
if [ -f "$CSV" ]; then
    echo "--- s_parameters.csv ---"
    column -t -s, "$CSV"

    echo ""
    echo "--- S-param summary (dB) ---"
    uv run python -c "
import csv, math
with open('$CSV') as f:
    rows = list(csv.DictReader(f))
mid = rows[len(rows)//2]
wl = float(mid['wavelength'])
params = sorted(k.replace('_mag','') for k in mid if k.endswith('_mag'))
print(f'At wavelength = {wl:.4f} um:')
total_power = 0
for p in params:
    mag = float(mid[f'{p}_mag'])
    db = 20*math.log10(mag) if mag > 0 else -999
    total_power += mag**2
    print(f'  {p} = {mag:.6f} ({db:+.1f} dB)')
print(f'  Power conservation: {total_power:.4f} (should be ~1.0)')
"

    # Copy CSV back to gsim for inspection
    cp "$CSV" "$CONFIG_OUTPUT/s_parameters.csv"
    echo ""
    echo "CSV copied to $CONFIG_OUTPUT/s_parameters.csv"
else
    echo "ERROR: No s_parameters.csv found in $RESULTS_DIR"
    echo "Files in results dir:"
    ls -la "$RESULTS_DIR"
    exit 1
fi

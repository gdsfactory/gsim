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
MEEP_NP="${MEEP_NP:-8}"
LOG_FILE="$CONFIG_OUTPUT/docker_meep.log"

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
    "$IMAGE_NAME" 2>&1 | tee "$LOG_FILE"
echo ""
echo "Docker log saved to $LOG_FILE"

# Collect diagnostic PNGs from results
for PNG in "$RESULTS_DIR"/meep_*.png; do
    [ -f "$PNG" ] || continue
    cp "$PNG" "$CONFIG_OUTPUT/"
    echo "Copied $(basename "$PNG") -> $CONFIG_OUTPUT/"
done

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

    # Copy and display debug JSON if present
    DEBUG_JSON="$RESULTS_DIR/meep_debug.json"
    if [ -f "$DEBUG_JSON" ]; then
        cp "$DEBUG_JSON" "$CONFIG_OUTPUT/meep_debug.json"
        echo ""
        echo "--- meep_debug.json summary ---"
        uv run python -c "
import json
with open('$DEBUG_JSON') as f:
    d = json.load(f)
meta = d.get('metadata', {})
print(f'Resolution: {meta.get(\"resolution\")} pixels/um')
print(f'Cell size: {meta.get(\"cell_size\")}')
print(f'Wall time: {meta.get(\"wall_seconds\", 0):.1f}s')
print(f'MEEP time: {meta.get(\"meep_time\", 0):.1f}')
print(f'Timesteps: {meta.get(\"timesteps\", 0)}')
print(f'Stopping mode: {meta.get(\"stopping_mode\")}')
print()
for port, info in d.get('eigenmode_info', {}).items():
    n_effs = info.get('n_eff', [])
    if n_effs:
        mid = n_effs[len(n_effs)//2]
        print(f'Port {port} n_eff (center freq): {mid:.4f}')
pcons = d.get('power_conservation', [])
if pcons:
    mid = pcons[len(pcons)//2]
    print(f'Power conservation (center freq): {mid:.4f}')
"
        echo "Debug JSON copied to $CONFIG_OUTPUT/meep_debug.json"
    else
        echo ""
        echo "WARNING: No meep_debug.json found (eigenmode diagnostics unavailable)"
        echo "Files in results dir:"
        ls -la "$RESULTS_DIR"
    fi
else
    echo "ERROR: No s_parameters.csv found in $RESULTS_DIR"
    echo "Files in results dir:"
    ls -la "$RESULTS_DIR"
    echo ""
    echo "Check docker log: $LOG_FILE"
    exit 1
fi

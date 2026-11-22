#!/bin/bash
# Wipes the active run's generated state (raw data, archived history, logs, nets, backups, datasets) and writes fresh
# random-init checkpoints, so the training scripts have something to load. Asks first. Not touched: data/act_checkpoints
# (the act-boundary saves), runs/run2 and runs/base (earlier nets), data/legacy_sts2cli.
set -euo pipefail

vakuu="${vakuu:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$vakuu"
export PYTHONPATH="$vakuu${PYTHONPATH:+:$PYTHONPATH}"
RUN="${SPIRE_RUN:-$vakuu/runs/spire}"
DATA="${SPIRE_DATA:-$vakuu/data/spire}"

read -r -p "This deletes everything under $DATA and the nets, backups and datasets under $RUN. Continue? [y/N] " confirm
if [[ "$confirm" != [yY] ]]; then
    echo "Aborted."
    exit 1
fi

echo "[1/3] Removing raw data, archived history and logs..."
rm -rf "$DATA"
mkdir -p "$DATA"

echo "[2/3] Removing model checkpoints, backups and materialized datasets..."
rm -rf "$RUN"
mkdir -p "$RUN/claws" "$RUN/compass"

echo "[3/3] Writing fresh random-init checkpoints..."
python -m vakuu.claws.train --epochs 0 --init "" --out "$RUN/claws/claws.pt"
python -m vakuu.golden_compass.train --epochs 0 --init "" --out "$RUN/compass/golden_compass.pt"

echo "Reset complete."

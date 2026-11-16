#!/bin/bash
# Wipes all generated training state (raw trial data, archived history, stats history, model
# checkpoints/backups/datasets) and reinitializes claws.pt and golden_compass.pt
# with random weights so the whispering_earring policy + training scripts have
# something to load. Run with `conda activate sts2` active, same as training_*.sh.
set -euo pipefail

vakuu="${vakuu:-/Users/kylelau519/Programming/Vakuu}"
we_data="$vakuu/agents/whispering_earring/data"

read -r -p "This deletes all raw trial data, archived history, stats history and .pt files under $vakuu. Continue? [y/N] " confirm
if [[ "$confirm" != [yY] ]]; then
    echo "Aborted."
    exit 1
fi

echo "[1/3] Removing raw trial data and archived history..."
rm -rf "$we_data/raw" "$we_data/raw_runs" "$we_data/history"
rm -f "$we_data/stats_history.jsonl" "$we_data/stats_history_choices.jsonl"

echo "[2/3] Removing model checkpoints, backups and materialized datasets..."
find "$vakuu/agents/claws/model" "$vakuu/agents/golden_compass" -name '*.pt' -delete
rm -rf "$vakuu/agents/claws/model/backups" "$vakuu/agents/golden_compass/backups"

echo "[3/3] Writing fresh random-init checkpoints..."
python "$vakuu/agents/claws/model/train.py" \
    --epochs 0 --init "" --out "$vakuu/agents/claws/model/claws.pt"
python "$vakuu/agents/golden_compass/train.py" \
    --epochs 0 --init "" --out "$vakuu/agents/golden_compass/golden_compass.pt"

echo "Reset complete."

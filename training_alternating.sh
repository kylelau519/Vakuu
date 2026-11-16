#!/bin/bash
# One round = Claws (mainline + turn forks -> train claws.pt), then Golden Compass (real-HP runs,
# played by the freshly trained Claws -> train golden_compass.pt). Alternating keeps each model's
# data on-policy for the other: the hazard net's labels depend on how well Claws fights, and Claws's
# mainline map/card picks come from the hazard net.
#
# Run with `conda activate sts2` active (or via watchdog_overnight.sh).
shopt -s nullglob
ulimit -n 65536  # materialize's worker pool hands tensors back through fd-backed shm segments

export vakuu="${vakuu:-/Users/kylelau519/Programming/Vakuu}"
export OMP_NUM_THREADS=1
export DOTNET_gcServer=0

CLAWS_DIR="$vakuu/agents/claws/model"
GC_DIR="$vakuu/agents/golden_compass"
WE_DATA="$vakuu/agents/whispering_earring/data"
HISTORY_WINDOW="${HISTORY_WINDOW:-3}"

source "$vakuu/training_lib.sh"

ROUND=1
while true; do
    echo "=========================================== Round $ROUND: claws"
    export_checkpoints
    mkdir -p "$WE_DATA/raw"
    find "$WE_DATA/raw" -maxdepth 1 -name '*.json' -delete

    python "$vakuu/agents/whispering_earring/facility_claws.py" \
        --trials 5 --branch-workers 15 --forks-per-turn 40 --main-workers 3 \
        --ascension 1 --raw-dir "$WE_DATA/raw"
    python "$CLAWS_DIR/materialize.py" "$WE_DATA/raw/" --out "$CLAWS_DIR/dataset.pt" \
        --target "${CLAWS_TARGET:-return}" --jobs 8
    python "$CLAWS_DIR/train.py" "$CLAWS_DIR/dataset.pt" --epochs 40 \
        --out "$CLAWS_DIR/claws.pt" --init "${CLAWS_CHECKPOINT:-}"
    backup_checkpoint "$CLAWS_DIR/claws.pt"
    cleanup_procs

    echo "=========================================== Round $ROUND: golden compass"
    export_checkpoints   # picks up the claws.pt just trained
    mkdir -p "$WE_DATA/raw_runs"
    archive_round_data "$WE_DATA/raw_runs" "$WE_DATA/history/runs" "$HISTORY_WINDOW"

    python "$vakuu/agents/whispering_earring/facility_compass.py" \
        --trials "${COMPASS_TRIALS:-200}" --branch-workers 15 --ascension 1 --raw-dir "$WE_DATA/raw_runs"
    python "$GC_DIR/materialize.py" "$WE_DATA/raw_runs" "$WE_DATA/history/runs"/*/ --out "$GC_DIR/dataset.pt"
    python "$GC_DIR/train.py" "$GC_DIR/dataset.pt" --epochs 20 \
        --out "$GC_DIR/golden_compass.pt" --init "${GOLDEN_COMPASS_CHECKPOINT:-}"
    backup_checkpoint "$GC_DIR/golden_compass.pt"
    cleanup_procs

    ((ROUND++))
done

#!/bin/bash
# Golden Compass loop: real-HP runs (facility_compass.py) -> materialize -> train the hazard net.
# Combat is played by whatever CLAWS_CHECKPOINT points at (random until one exists), so these runs
# are only valid labels for that agent -- retrain whenever Claws changes. Map and card-reward
# choices come from the hazard net itself once golden_compass.pt exists (with exploration:
# COMPASS_EPS / COMPASS_TAU, see agents/golden_compass/compass.py), tier list / random before.
# Progress per round: agents/whispering_earring/data/progress_compass.jsonl (progress.py).
#
# Run with `conda activate sts2` active (or via watchdog_overnight.sh).
shopt -s nullglob
ulimit -n 65536

export vakuu="${vakuu:-/Users/kylelau519/Programming/Vakuu}"
export OMP_NUM_THREADS=1      # one torch thread per worker process -- see training_claws.sh
export DOTNET_gcServer=0      # workstation GC in every worker's in-process CoreCLR -- see training_claws.sh

GC_DIR="$vakuu/agents/golden_compass"
RAW="$vakuu/agents/whispering_earring/data/raw_runs"
HISTORY="$vakuu/agents/whispering_earring/data/history/runs"
HISTORY_WINDOW="${HISTORY_WINDOW:-3}"  # past rounds kept and trained on alongside the fresh one
TRIALS="${COMPASS_TRIALS:-200}"
WORKERS="${COMPASS_WORKERS:-15}"

source "$vakuu/training_lib.sh"

# SEEDS: the pool of seeds every run draws from, so progression shows on a few fixed maps. Default is
# five; SEEDS="a b c" picks your own, SEEDS=random goes back to a fresh seed per run. One character
# (TRAIN_CHARACTER, default Ironclad) because a seed plus a character is one map.
SEEDS="${SEEDS:-train-1 train-2 train-3 train-4 train-5}"
SEED_ARGS=()
[ "$SEEDS" != "random" ] && SEED_ARGS=(--seeds $SEEDS --characters "${TRAIN_CHARACTER:-Ironclad}")

ROUND=1
while true; do
    echo "=========================================== Compass round $ROUND"
    export_checkpoints
    mkdir -p "$RAW"

    echo "[1/4] Archiving last round's runs (keeping $HISTORY_WINDOW)..."
    archive_round_data "$RAW" "$HISTORY" "$HISTORY_WINDOW"

    echo "[2/4] Generating $TRIALS real-HP runs..."
    python "$vakuu/agents/whispering_earring/facility_compass.py" \
        --trials "$TRIALS" --branch-workers "$WORKERS" --ascension 1 --raw-dir "$RAW" "${SEED_ARGS[@]}"
    python "$GC_DIR/progress.py" "$RAW" --label "$ROUND" \
        --log "$vakuu/agents/whispering_earring/data/progress_compass.jsonl"

    echo "[3/4] Materializing..."
    python "$GC_DIR/materialize.py" "$RAW" "$HISTORY"/*/ --out "$GC_DIR/dataset.pt"

    echo "[4/4] Training..."
    python "$GC_DIR/train.py" "$GC_DIR/dataset.pt" --epochs 20 \
        --out "$GC_DIR/golden_compass.pt" --init "${GOLDEN_COMPASS_CHECKPOINT:-}"
    backup_checkpoint "$GC_DIR/golden_compass.pt"

    cleanup_procs
    ((ROUND++))
done

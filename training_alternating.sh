#!/bin/bash
# One round = Claws (mainline + turn forks -> train claws.pt), then Golden Compass (real-HP runs,
# played by the freshly trained Claws -> train golden_compass.pt). Alternating keeps each model's
# data on-policy for the other: the hazard net's labels depend on how well Claws fights, and Claws's
# mainline map/card picks come from the hazard net.
#
# Sizing (defaults suit a big machine): WORKERS (processes, default 15), CLAWS_TRIALS (mainline runs per
# round, 5), CLAWS_FORKS (forks per combat turn, 40), CLAWS_MAIN_WORKERS (3), COMPASS_TRIALS (200).
#
# Run with `conda activate sts2` active (or via watchdog_overnight.sh).
shopt -s nullglob
ulimit -n 65536  # materialize's worker pool hands tensors back through fd-backed shm segments

export vakuu="${vakuu:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export OMP_NUM_THREADS=1
export DOTNET_gcServer=0

CLAWS_DIR="$vakuu/agents/claws/model"
GC_DIR="$vakuu/agents/golden_compass"
WE_DATA="$vakuu/agents/whispering_earring/data"
HISTORY_WINDOW="${HISTORY_WINDOW:-3}"

source "$vakuu/training_lib.sh"
install_traps

SEEDS="${SEEDS:-random}"
SEED_ARGS=()
[ "$SEEDS" != "random" ] && SEED_ARGS=(--seeds $SEEDS --characters "${TRAIN_CHARACTER:-Ironclad}")

ROUND=1
while true; do
    echo "=========================================== Round $ROUND: claws"
    export_checkpoints
    mkdir -p "$WE_DATA/raw"
    find "$WE_DATA/raw" -maxdepth 1 -name '*.json' -delete

    python "$vakuu/agents/whispering_earring/facility_claws.py" \
        --trials "${CLAWS_TRIALS:-5}" --branch-workers "${WORKERS:-17}" \
        --forks-per-turn "${CLAWS_FORKS:-40}" --main-workers "${CLAWS_MAIN_WORKERS:-5}" \
        --ascension 1 --raw-dir "$WE_DATA/raw" --dedup-forks "${SEED_ARGS[@]}"
    python "$CLAWS_DIR/materialize.py" "$WE_DATA/raw/" --out "$CLAWS_DIR/dataset.pt" \
        --target "${CLAWS_TARGET:-return}" --jobs "${JOBS:-18}"
    python "$CLAWS_DIR/train.py" "$CLAWS_DIR/dataset.pt" --epochs 40 --device cuda --batch-size "${CLAWS_BATCH:-512}" --lr "${CLAWS_LR:-1.5e-5}" \
        --out "$CLAWS_DIR/claws.pt" --init "${CLAWS_CHECKPOINT:-}"
    backup_checkpoint "$CLAWS_DIR/claws.pt"

    # Is Claws actually better? Held-out eval seeds, the compass on, no exploration noise (each row records the compass checkpoint)
    # (agents/whispering_earring/eval_claws.py). EVAL_RUNS=0 skips it; rows accumulate in data/claws_eval.jsonl.
    if [ "${EVAL_RUNS:-40}" -gt 0 ]; then
        [ "$ROUND" = 1 ] && python "$vakuu/agents/whispering_earring/eval_claws.py" --ckpt random \
            --label "base" --runs "${EVAL_RUNS:-40}" --workers "${WORKERS:-8}" | tail -2
        python "$vakuu/agents/whispering_earring/eval_claws.py" --ckpt "$CLAWS_DIR/claws.pt" \
            --label "$(date +%d.%H%M)#$ROUND" --runs "${EVAL_RUNS:-40}" --workers "${WORKERS:-8}" | sed -n '/--- log/,$p'
    fi
    cleanup_procs

    echo "=========================================== Round $ROUND: golden compass"
    export_checkpoints   # picks up the claws.pt just trained
    mkdir -p "$WE_DATA/raw_runs"
    archive_round_data "$WE_DATA/raw_runs" "$WE_DATA/history/runs" "$HISTORY_WINDOW"

    python "$vakuu/agents/whispering_earring/facility_compass.py" \
        --trials "${COMPASS_TRIALS:-200}" --branch-workers "${WORKERS:-17}" --ascension 1 --raw-dir "$WE_DATA/raw_runs" "${SEED_ARGS[@]}"
    python "$GC_DIR/progress.py" "$WE_DATA/raw_runs" --label "$ROUND" --log "$WE_DATA/progress_compass.jsonl"
    python "$GC_DIR/materialize.py" "$WE_DATA/raw_runs" "$WE_DATA/history/runs"/*/ --out "$GC_DIR/dataset.pt"
    python "$GC_DIR/train.py" "$GC_DIR/dataset.pt" --epochs 20 \
        --out "$GC_DIR/golden_compass.pt" --init "${GOLDEN_COMPASS_CHECKPOINT:-}"
    backup_checkpoint "$GC_DIR/golden_compass.pt"
    cleanup_procs

    ((ROUND++))
done

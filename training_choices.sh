#!/bin/bash

export vakuu="/Users/kylelau519/Programming/Vakuu"
ROUND=1

while true; do
    echo "==========================================="
    echo " Starting Choices Paradox Self-Play Loop: Round $ROUND"
    echo "==========================================="

    export CLAWS_CHECKPOINT="$vakuu/agents/claws/model/claws.pt"
    [ -f "$CLAWS_CHECKPOINT" ] || unset CLAWS_CHECKPOINT  

    CHOICES_CHECKPOINT="$vakuu/agents/choices_paradox/model/choices_paradox.pt"
    [ -f "$CHOICES_CHECKPOINT" ] || CHOICES_CHECKPOINT=""  

    export CHOICES_PARADOX_CHECKPOINT="$CHOICES_CHECKPOINT" 
    raw_dir="$vakuu/agents/whispering_earring/data/raw_choices"

    echo "[1/6] Cleaning up old raw data..."
    find "$raw_dir/" -maxdepth 1 -name '*.json' -delete
    find "$raw_dir/checkpoints/" -maxdepth 1 -name '*.json' -delete

    echo "[2/6] Running facility_choices_paradox.py (Data Generation)..."
    python "$vakuu/agents/whispering_earring/facility_choices_paradox.py" \
        --trials 1 --branch-workers 15 --main-workers 1 \
        --branches-per-turn 4 --meta-branches-per-trial 3 \
        --raw-dir "$raw_dir"

    echo "[3/6] Meta-decision stats for this round..."
    python "$vakuu/agents/whispering_earring/stats_choices.py" \
        --raw-dir "$raw_dir" \
        --round "$ROUND" --no-card-stats

    echo "[4/6] Materializing dataset..."
    python "$vakuu/agents/choices_paradox/model/materialize.py" \
        "$raw_dir/" \
        --out "$vakuu/agents/choices_paradox/model/dataset.pt" \
        --jobs 8

    echo "[5/6] Training model..."
    python "$vakuu/agents/choices_paradox/model/train.py" \
        "$vakuu/agents/choices_paradox/model/dataset.pt" --epochs 50 \
        --out "$vakuu/agents/choices_paradox/model/choices_paradox.pt" \
        --init "$CHOICES_CHECKPOINT" \
        --warm-start-from "${CLAWS_CHECKPOINT:-}"

    echo "[6/6] Backing up checkpoint..."
    # Timestamped, never overwritten -- see training_claws.sh's own comment; same reasoning here.
    backup_dir="$vakuu/agents/choices_paradox/model/backups"
    mkdir -p "$backup_dir"
    cp "$vakuu/agents/choices_paradox/model/choices_paradox.pt" \
        "$backup_dir/choices_paradox_$(date +%Y%m%d_%H%M%S).pt"

    # Deliberately kill all the python multiprocessing to avoid zombie processes
    pkill -9 -f "multiprocessing.spawn"

    echo "==========================================="
    echo " Round $ROUND complete! Restarting in 3s..."
    echo "==========================================="
    sleep 3
    ((ROUND++))
done

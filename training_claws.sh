#!/bin/bash

export vakuu="/Users/kylelau519/Programming/Vakuu"
ROUND=1

while true; do
    echo "==========================================="
    echo " Starting Self-Play Loop: Round $ROUND"
    echo "==========================================="

    export CLAWS_CHECKPOINT="$vakuu/agents/claws/model/claws.pt"
    [ -f "$CLAWS_CHECKPOINT" ] || unset CLAWS_CHECKPOINT

    # policy.py's card_reward decisions upgrade to this trained net (if one exists yet) instead
    # of the static tier-list heuristic -- see agents/whispering_earring/policy.py's
    # _choose_card_reward_decision.
    CHOICES_CHECKPOINT="$vakuu/agents/choices_paradox/model/choices_paradox.pt"
    [ -f "$CHOICES_CHECKPOINT" ] || CHOICES_CHECKPOINT=""
    export CHOICES_PARADOX_CHECKPOINT="$CHOICES_CHECKPOINT"

    echo "[1/6] Cleaning up old raw data..."
    find "$vakuu/agents/whispering_earring/data/raw/" -maxdepth 1 -name '*.json' -delete
    find "$vakuu/agents/whispering_earring/data/raw/checkpoints/" -maxdepth 1 -name '*.json' -delete

    echo "[2/6] Running facility_claws.py (Data Generation)..."
    python "$vakuu/agents/whispering_earring/facility_claws.py" \
        --trials 5 --branch-workers 15 --branches-per-turn 40 --main-workers 3 \
        --raw-dir "$vakuu/agents/whispering_earring/data/raw"

    echo "[3/6] Floor/combat stats for this round..."
    python "$vakuu/agents/whispering_earring/stats.py" \
        --raw-dir "$vakuu/agents/whispering_earring/data/raw" \
        --round "$ROUND" --no-card-stats

    echo "[4/6] Materializing dataset..."
    python "$vakuu/agents/claws/model/materialize.py" \
        "$vakuu/agents/whispering_earring/data/raw/" \
        --out "$vakuu/agents/claws/model/dataset.pt" \
        --jobs 8

    echo "[5/6] Training model..."
    python "$vakuu/agents/claws/model/train.py" \
        "$vakuu/agents/claws/model/dataset.pt" --epochs 40 \
        --out "$vakuu/agents/claws/model/claws.pt" \
        --init "${CLAWS_CHECKPOINT:-}"

    echo "[6/6] Backing up checkpoint..."
    backup_dir="$vakuu/agents/claws/model/backups"
    mkdir -p "$backup_dir"
    cp "$vakuu/agents/claws/model/claws.pt" \
        "$backup_dir/claws_$(date +%Y%m%d_%H%M%S).pt"

    # Deliberately kill all the python multiprocessing to avoid zombie processes
    pkill -9 -f "multiprocessing.spawn"
    # Belt-and-suspenders for the same problem facility_common.py's Facility.stop() now handles
    # at the source (each worker is its own process group, killed via os.killpg so its `dotnet
    # run` Sts2Session children die with it) -- see training_choices.sh's matching line for the
    # full explanation (confirmed by hand: an orphaned one of these can sit around for DAYS,
    # invisible to the pkill above since that only ever matches the Python side).
    pkill -9 -f "sts2-cli/src/Sts2Headless"

    echo "==========================================="
    echo " Round $ROUND complete! Restarting in 3s..."
    echo "==========================================="
    sleep 3
    ((ROUND++))
done
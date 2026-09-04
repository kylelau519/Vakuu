#!/bin/bash

export vakuu="/Users/kylelau519/Programming/Vakuu"
ROUND=1

cleanup_procs() {
    pkill -9 -f "multiprocessing.spawn"
    pkill -9 -f "sts2-cli/src/Sts2Headless"
    sleep 5
}

while true; do
    echo "==========================================="
    echo " Starting Claws Self-Play Loop: Round $ROUND"
    echo "==========================================="

    export CLAWS_CHECKPOINT="$vakuu/agents/claws/model/claws.pt"
    [ -f "$CLAWS_CHECKPOINT" ] || unset CLAWS_CHECKPOINT

    # policy.py's card_reward decisions upgrade to this trained net (if one exists yet) instead of
    # the static tier-list heuristic -- see agents/whispering_earring/policy.py's
    # _choose_card_reward_decision.
    CHOICES_CHECKPOINT="$vakuu/agents/choices_paradox/model/choices_paradox.pt"
    [ -f "$CHOICES_CHECKPOINT" ] || CHOICES_CHECKPOINT=""
    export CHOICES_PARADOX_CHECKPOINT="$CHOICES_CHECKPOINT"

    # map_select decisions upgrade to this trained net (if one exists yet) instead of
    # golden_compass/policy.py's DP heuristic -- see agents/whispering_earring/policy.py's
    # _choose_map_action. Trained at the end of this same round below (Training Golden Compass
    # section), off this round's own data/raw + data/raw_choices output.
    GOLDEN_COMPASS_CHECKPOINT_PATH="$vakuu/agents/golden_compass/model/golden_compass.pt"
    [ -f "$GOLDEN_COMPASS_CHECKPOINT_PATH" ] || GOLDEN_COMPASS_CHECKPOINT_PATH=""
    export GOLDEN_COMPASS_CHECKPOINT="$GOLDEN_COMPASS_CHECKPOINT_PATH"

    echo "[claws 1/6] Cleaning up old raw data..."
    find "$vakuu/agents/whispering_earring/data/raw/" -maxdepth 1 -name '*.json' -delete
    find "$vakuu/agents/whispering_earring/data/raw/checkpoints/" -maxdepth 1 -name '*.json' -delete

    echo "[claws 2/6] Running facility_claws.py (Data Generation)..."
    python "$vakuu/agents/whispering_earring/facility_claws.py" \
        --trials 5 --branch-workers 15 --branches-per-turn 30 --main-workers 3 \
        --raw-dir "$vakuu/agents/whispering_earring/data/raw"

    echo "[claws 3/6] Floor/combat stats for this round..."
    python "$vakuu/agents/whispering_earring/stats.py" \
        --raw-dir "$vakuu/agents/whispering_earring/data/raw" \
        --round "$ROUND" --no-card-stats

    echo "[claws 4/6] Materializing dataset..."
    python "$vakuu/agents/claws/model/materialize.py" \
        "$vakuu/agents/whispering_earring/data/raw/" \
        --out "$vakuu/agents/claws/model/dataset.pt" \
        --jobs 8

    echo "[claws 5/6] Training model..."
    python "$vakuu/agents/claws/model/train.py" \
        "$vakuu/agents/claws/model/dataset.pt" --epochs 40 \
        --out "$vakuu/agents/claws/model/claws.pt" \
        --init "${CLAWS_CHECKPOINT:-}"

    echo "[claws 6/6] Backing up checkpoint..."
    backup_dir="$vakuu/agents/claws/model/backups"
    mkdir -p "$backup_dir"
    cp "$vakuu/agents/claws/model/claws.pt" \
        "$backup_dir/claws_$(date +%Y%m%d_%H%M%S).pt"

    cleanup_procs

    echo "==========================================="
    echo " Starting Choices Paradox Self-Play Loop: Round $ROUND"
    echo "==========================================="

    # Refresh: the claws round above may have just written a new checkpoint.
    export CLAWS_CHECKPOINT="$vakuu/agents/claws/model/claws.pt"
    [ -f "$CLAWS_CHECKPOINT" ] || unset CLAWS_CHECKPOINT

    CHOICES_CHECKPOINT="$vakuu/agents/choices_paradox/model/choices_paradox.pt"
    [ -f "$CHOICES_CHECKPOINT" ] || CHOICES_CHECKPOINT=""

    export CHOICES_PARADOX_CHECKPOINT="$CHOICES_CHECKPOINT"

    # Refresh: unchanged since the claws phase above (golden compass is only trained at the end
    # of this same round, below), kept here for the same reasoning as CLAWS_CHECKPOINT's refresh.
    GOLDEN_COMPASS_CHECKPOINT_PATH="$vakuu/agents/golden_compass/model/golden_compass.pt"
    [ -f "$GOLDEN_COMPASS_CHECKPOINT_PATH" ] || GOLDEN_COMPASS_CHECKPOINT_PATH=""
    export GOLDEN_COMPASS_CHECKPOINT="$GOLDEN_COMPASS_CHECKPOINT_PATH"

    raw_dir="$vakuu/agents/whispering_earring/data/raw_choices"

    echo "[choices 1/6] Cleaning up old raw data..."
    find "$raw_dir/" -maxdepth 1 -name '*.json' -delete
    find "$raw_dir/checkpoints/" -maxdepth 1 -name '*.json' -delete

    echo "[choices 2/6] Running facility_choices_paradox.py (Data Generation)..."
    python "$vakuu/agents/whispering_earring/facility_choices_paradox.py" \
        --trials 6 --branch-workers 10 --main-workers 3 \
        --branches-per-turn 4 --meta-branches-per-trial 6\
        --raw-dir "$raw_dir"

    echo "[choices 3/6] Meta-decision stats for this round..."
    python "$vakuu/agents/whispering_earring/stats_choices.py" \
        --raw-dir "$raw_dir" \
        --round "$ROUND"

    echo "[choices 4/6] Materializing dataset..."
    python "$vakuu/agents/choices_paradox/model/materialize.py" \
        "$raw_dir/" \
        --out "$vakuu/agents/choices_paradox/model/dataset.pt" \
        --jobs 8

    echo "[choices 5/6] Training model..."
    python "$vakuu/agents/choices_paradox/model/train.py" \
        "$vakuu/agents/choices_paradox/model/dataset.pt" --epochs 50 \
        --out "$vakuu/agents/choices_paradox/model/choices_paradox.pt" \
        --init "$CHOICES_CHECKPOINT" \
        --warm-start-from "${CLAWS_CHECKPOINT:-}"

    echo "[choices 6/6] Backing up checkpoint..."
    backup_dir="$vakuu/agents/choices_paradox/model/backups"
    mkdir -p "$backup_dir"
    cp "$vakuu/agents/choices_paradox/model/choices_paradox.pt" \
        "$backup_dir/choices_paradox_$(date +%Y%m%d_%H%M%S).pt"

    cleanup_procs

    echo "==========================================="
    echo " Training Golden Compass: Round $ROUND"
    echo "==========================================="

    # No dedicated facility run -- every select_map_node decision on this round's own claws +
    # choices main lines already carries its own trial's +1/-1 terminal outcome via
    # facility_common.py's pending_map_steps wiring, so this just materializes off the raw data
    # both phases above already produced this round -- see
    # agents/golden_compass/model/README.md's Design section.
    echo "[golden 1/2] Materializing dataset from claws + choices raw data..."
    python "$vakuu/agents/golden_compass/model/materialize.py" \
        "$vakuu/agents/whispering_earring/data/raw/" \
        "$raw_dir/" \
        --out "$vakuu/agents/golden_compass/model/dataset.pt" \
        --jobs 8

    echo "[golden 2/2] Training model..."
    python "$vakuu/agents/golden_compass/model/train.py" \
        "$vakuu/agents/golden_compass/model/dataset.pt" --epochs 20 \
        --out "$vakuu/agents/golden_compass/model/golden_compass.pt" \
        --init "$GOLDEN_COMPASS_CHECKPOINT_PATH"

    echo "Backing up Golden Compass checkpoint..."
    backup_dir="$vakuu/agents/golden_compass/model/backups"
    mkdir -p "$backup_dir"
    cp "$vakuu/agents/golden_compass/model/golden_compass.pt" \
        "$backup_dir/golden_compass_$(date +%Y%m%d_%H%M%S).pt"

    echo "==========================================="
    echo " Round $ROUND complete (claws + choices + golden compass)! Restarting in 3s..."
    echo "==========================================="
    sleep 3
    ((ROUND++))
done

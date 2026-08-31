#!/bin/bash

export vakuu="/Users/kylelau519/Programming/Vakuu"
ROUND=1

while true; do
    echo "==========================================="
    echo " Starting Self-Play Loop: Round $ROUND"
    echo "==========================================="

    export CLAWS_CHECKPOINT="$vakuu/agents/claws/model/claws.pt"
    [ -f "$CLAWS_CHECKPOINT" ] || unset CLAWS_CHECKPOINT  # no checkpoint yet on the first run --
    # facility.py's TD-bootstrap value term (_play_turn_only) falls back to 0.0 without one, a
    # known limitation for round 1 only. Once this script's own train.py step below writes
    # claws.pt, the next run of this same script picks it up automatically -- this is what
    # turns repeated runs into a self-play improvement loop.

    echo "[1/4] Cleaning up old raw data..."
    rm -f "$vakuu/agents/whispering_earring/data/raw/"*.json

    echo "[2/4] Running facility.py (Data Generation)..."
    python "$vakuu/agents/whispering_earring/facility.py" \
        --trials 3 --branch-workers 30 --branches-per-turn 40 --main-workers 3 \
        --raw-dir "$vakuu/agents/whispering_earring/data/raw"
    # branches_per_turn spends roughly branches_per_turn x turns-per-combat branch-worker-
    # seconds per combat now (branching happens once per turn, not once per whole combat) --
    # retune deliberately rather than assuming this number means what it used to.

    echo "[3/4] Floor/combat stats for this round..."
    python "$vakuu/agents/whispering_earring/stats.py" \
        --raw-dir "$vakuu/agents/whispering_earring/data/raw"

    echo "[4/4] Materializing dataset..."
    python "$vakuu/agents/claws/model/materialize.py" \
        "$vakuu/agents/whispering_earring/data/raw/"*.json \
        --out "$vakuu/agents/claws/model/dataset.pt"

    echo "[4/4] Training model..."
    # --init resumes from the previous round's weights -- without it, every run trains a fresh
    # randomly initialized network from scratch and discards it, since CLAWS_CHECKPOINT above only
    # feeds facility.py's generation step, not training; empty/unset is fine too (train.py treats ""
    # as "no checkpoint yet"). No --value-only: MCTS was dropped from the live generation path
    # entirely (too expensive against this engine's replay-only checkpointing -- see
    # agents/whispering_earring/policy.py's module docstring), but materialize.py now aggregates a
    # real target policy per turn from branch-and-adopt's own discarded branches (see its module
    # docstring) and tags it policy_real -- train.py only ever backprops the policy term on those,
    # so the rest of the (still-placeholder) recorded `policy` values are automatically excluded.
    python "$vakuu/agents/claws/model/train.py" \
        "$vakuu/agents/claws/model/dataset.pt" --epochs 50 \
        --out "$vakuu/agents/claws/model/claws.pt" \
        --init "${CLAWS_CHECKPOINT:-}"

    echo "==========================================="
    echo " Round $ROUND complete! Restarting in 3s..."
    echo "==========================================="
    sleep 3
    ((ROUND++))
done
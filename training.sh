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

    echo "[1/5] Cleaning up old raw data..."
    # A shell-expanded "*.json" here can blow past the OS's exec argument-length limit once
    # data/raw/ holds tens of thousands of per-branch files (E2BIG happens in the shell before
    # rm even runs, so -f can't save it, and the script has no set -e to catch the failure) --
    # find -delete never builds one giant argument list, so it can't hit that ceiling.
    find "$vakuu/agents/whispering_earring/data/raw/" -maxdepth 1 -name '*.json' -delete

    echo "[2/5] Running facility.py (Data Generation)..."
    python "$vakuu/agents/whispering_earring/facility.py" \
        --trials 5 --branch-workers 15 --branches-per-turn 40 --main-workers 3 \
        --raw-dir "$vakuu/agents/whispering_earring/data/raw"
    # branches_per_turn spends roughly branches_per_turn x turns-per-combat branch-worker-
    # seconds per combat now (branching happens once per turn, not once per whole combat) --
    # retune deliberately rather than assuming this number means what it used to.

    echo "[3/5] Floor/combat stats for this round..."
    python "$vakuu/agents/whispering_earring/stats.py" \
        --raw-dir "$vakuu/agents/whispering_earring/data/raw" \
        --round "$ROUND"
    # Cumulative stats persist to data/stats_history.jsonl (a sibling of data/raw, so it
    # survives the "Cleaning up old raw data" wipe above) -- that's what to check mid-run to
    # see whether the model is actually improving over the day, not just this round's numbers.

    echo "[4/5] Materializing dataset..."
    # Pass the raw dir itself, not a shell-expanded "*.json" glob -- materialize.py globs
    # internally now so this one argument can't hit the same E2BIG ceiling as step [1/5].
    python "$vakuu/agents/claws/model/materialize.py" \
        "$vakuu/agents/whispering_earring/data/raw/" \
        --out "$vakuu/agents/claws/model/dataset.pt" \
        --jobs 8

    echo "[5/5] Training model..."
    python "$vakuu/agents/claws/model/train.py" \
        "$vakuu/agents/claws/model/dataset.pt" --epochs 40 \
        --out "$vakuu/agents/claws/model/claws.pt" \
        --init "${CLAWS_CHECKPOINT:-}" 

    echo "==========================================="
    echo " Round $ROUND complete! Restarting in 3s..."
    echo "==========================================="
    sleep 3
    ((ROUND++))
done
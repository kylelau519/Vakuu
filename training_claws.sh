#!/bin/bash

export vakuu="/Users/kylelau519/Programming/Vakuu"
ROUND=1

while true; do
    echo "==========================================="
    echo " Starting Self-Play Loop: Round $ROUND"
    echo "==========================================="

    export CLAWS_CHECKPOINT="$vakuu/agents/claws/model/claws.pt"
    [ -f "$CLAWS_CHECKPOINT" ] || unset CLAWS_CHECKPOINT  # no checkpoint yet on the first run --
    # facility_common.py's TD-bootstrap value term (_play_turn_only) falls back to 0.0 without
    # one, a known limitation for round 1 only. Once this script's own train.py step below writes
    # claws.pt, the next run of this same script picks it up automatically -- this is what
    # turns repeated runs into a self-play improvement loop.

    echo "[1/6] Cleaning up old raw data..."
    # A shell-expanded "*.json" here can blow past the OS's exec argument-length limit once
    # data/raw/ holds tens of thousands of per-branch files (E2BIG happens in the shell before
    # rm even runs, so -f can't save it, and the script has no set -e to catch the failure) --
    # find -delete never builds one giant argument list, so it can't hit that ceiling.
    find "$vakuu/agents/whispering_earring/data/raw/" -maxdepth 1 -name '*.json' -delete
    # Native combat checkpoints, plain .json (see facility_common.py's checkpoint_path:
    # "{main_id}_c{combat_index:03d}.json") -- wiped alongside the round's trial files since a
    # checkpoint from a wiped round can never be resumed from again.
    find "$vakuu/agents/whispering_earring/data/raw/checkpoints/" -maxdepth 1 -name '*.json' -delete

    echo "[2/6] Running facility_claws.py (Data Generation)..."
    python "$vakuu/agents/whispering_earring/facility_claws.py" \
        --trials 5 --branch-workers 15 --branches-per-turn 40 --main-workers 3 \
        --raw-dir "$vakuu/agents/whispering_earring/data/raw"
    # branches_per_turn spends roughly branches_per_turn x turns-per-combat branch-worker-
    # seconds per combat now (branching happens once per turn, not once per whole combat) --
    # retune deliberately rather than assuming this number means what it used to.

    echo "[3/6] Floor/combat stats for this round..."
    # --no-card-stats: the card offer/pick table needs a replay pass per turn-branch group
    # (torch + CLR bridge), expensive enough to slow down every round for a table nobody's
    # watching live -- run stats.py by hand without this flag when you actually want it.
    python "$vakuu/agents/whispering_earring/stats.py" \
        --raw-dir "$vakuu/agents/whispering_earring/data/raw" \
        --round "$ROUND" --no-card-stats
    # Cumulative stats persist to data/stats_history.jsonl (a sibling of data/raw, so it
    # survives the "Cleaning up old raw data" wipe above) -- that's what to check mid-run to
    # see whether the model is actually improving over the day, not just this round's numbers.

    echo "[4/6] Materializing dataset..."
    # Pass the raw dir itself, not a shell-expanded "*.json" glob -- materialize.py globs
    # internally now so this one argument can't hit the same E2BIG ceiling as step [1/6].
    python "$vakuu/agents/claws/model/materialize.py" \
        "$vakuu/agents/whispering_earring/data/raw/" \
        --out "$vakuu/agents/claws/model/dataset.pt" \
        --jobs 8

    echo "[5/6] Training model..."
    python "$vakuu/agents/claws/model/train.py" \
        "$vakuu/agents/claws/model/dataset.pt" --epochs 30 \
        --out "$vakuu/agents/claws/model/claws.pt" \
        --init "${CLAWS_CHECKPOINT:-}"

    echo "[6/6] Backing up checkpoint..."
    # Timestamped, never overwritten -- so a bad round (or a manual mistake in between runs of
    # this script, see train.py's --init guards) can be recovered from by hand instead of losing
    # every checkpoint before it. Not pruned here; keep an eye on backups/ disk usage over a long
    # run and thin it out by hand if it grows too large.
    backup_dir="$vakuu/agents/claws/model/backups"
    mkdir -p "$backup_dir"
    cp "$vakuu/agents/claws/model/claws.pt" \
        "$backup_dir/claws_$(date +%Y%m%d_%H%M%S).pt"

    echo "==========================================="
    echo " Round $ROUND complete! Restarting in 3s..."
    echo "==========================================="
    sleep 3
    ((ROUND++))
done
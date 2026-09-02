#!/bin/bash

export vakuu="/Users/kylelau519/Programming/Vakuu"
ROUND=1

while true; do
    echo "==========================================="
    echo " Starting Choices Paradox Self-Play Loop: Round $ROUND"
    echo "==========================================="

    export CLAWS_CHECKPOINT="$vakuu/agents/claws/model/claws.pt"
    [ -f "$CLAWS_CHECKPOINT" ] || unset CLAWS_CHECKPOINT  # combat quality + facility_common.py's
    # TD-bootstrap value term fall back to uniform-random/0.0 without one -- fine for round 1,
    # picked up automatically once training_claws.sh (a separate loop) has written claws.pt.

    CHOICES_CHECKPOINT="$vakuu/agents/choices_paradox/model/choices_paradox.pt"
    [ -f "$CHOICES_CHECKPOINT" ] || CHOICES_CHECKPOINT=""  # empty means train.py inits from
    # scratch (round 1) -- see step [4/5]'s --init below

    raw_dir="$vakuu/agents/whispering_earring/data/raw_choices"
    # Deliberately NOT the same directory training_claws.sh uses -- claws/model/materialize.py
    # has no branch_kind filter and would try to treat this loop's MetaBranchJob branch files as
    # combat turn-branches (wrong prefix/action shape) if the two ever shared a raw-dir, and the
    # two loops' own "delete last round's json" steps would race each other if run concurrently.

    echo "[1/5] Cleaning up old raw data..."
    # find -delete (not a shell-expanded *.json glob) -- see training_claws.sh's own comment on
    # this, same E2BIG concern once raw_choices/ holds many thousands of per-branch files.
    find "$raw_dir/" -maxdepth 1 -name '*.json' -delete

    echo "[2/5] Running facility_choices_paradox.py (Data Generation)..."
    python "$vakuu/agents/whispering_earring/facility_choices_paradox.py" \
        --trials 1 --branch-workers 15 --main-workers 3 \
        --branches-per-turn 3 \
        --meta-options-replicates 3 --meta-horizon-combats 1 --meta-branches-per-trial 5 \
        --raw-dir "$raw_dir"
    # branches-per-turn is lower here than training_claws.sh's 40: this loop's shared
    # branch-worker pool also has to serve meta-decision jobs, and each of those is much more
    # expensive than a turn-branch job (no native-checkpoint fast path -- every replicate pays a
    # full replay from StartRun, growing with run depth -- times num_options x
    # meta-options-replicates per card_reward decision; see choices_paradox/model/README.md's
    # Data section). meta-branches-per-trial caps the worst case (unlimited would branch every
    # card_reward in every trial) rather than leaving it open-ended. All four of these are
    # starting points, not measured-optimal -- retune once real per-round throughput is known.

    echo "[3/5] Materializing dataset..."
    python "$vakuu/agents/choices_paradox/model/materialize.py" \
        "$raw_dir/" \
        --out "$vakuu/agents/choices_paradox/model/dataset.pt" \
        --jobs 8

    echo "[4/5] Training model..."
    # --warm-start-from only actually applies when --init is empty (round 1, see train.py) --
    # harmless to always pass once CLAWS_CHECKPOINT exists, since a resumed net (--init
    # non-empty) ignores it. Seeds card-identity embeddings from Claws's own trained ones rather
    # than starting this net's id_emb from noise -- see model.py's warm_started_from.
    python "$vakuu/agents/choices_paradox/model/train.py" \
        "$vakuu/agents/choices_paradox/model/dataset.pt" --epochs 30 \
        --out "$vakuu/agents/choices_paradox/model/choices_paradox.pt" \
        --init "$CHOICES_CHECKPOINT" \
        --warm-start-from "${CLAWS_CHECKPOINT:-}"

    echo "[5/5] Backing up checkpoint..."
    # Timestamped, never overwritten -- see training_claws.sh's own comment; same reasoning here.
    backup_dir="$vakuu/agents/choices_paradox/model/backups"
    mkdir -p "$backup_dir"
    cp "$vakuu/agents/choices_paradox/model/choices_paradox.pt" \
        "$backup_dir/choices_paradox_$(date +%Y%m%d_%H%M%S).pt"

    echo "==========================================="
    echo " Round $ROUND complete! Restarting in 3s..."
    echo "==========================================="
    sleep 3
    ((ROUND++))
done

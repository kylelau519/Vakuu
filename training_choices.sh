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
    # scratch (round 1) -- see step [5/6]'s --init below

    export CHOICES_PARADOX_CHECKPOINT="$CHOICES_CHECKPOINT"  # read by
    # facility_choices_paradox.py's _load_choices_paradox_net -- the net's own softmax+Dirichlet
    # prediction is what the real trial actually commits to at each divergence (see
    # _choose_meta_option_claws); empty/missing falls back to the tier-list heuristic, same
    # "harmless round 1" fallback CLAWS_CHECKPOINT already follows above.

    raw_dir="$vakuu/agents/whispering_earring/data/raw_choices"
    # Deliberately NOT the same directory training_claws.sh uses -- claws/model/materialize.py
    # has no branch_kind filter and would try to treat this loop's own meta-branch files as
    # combat turn-branches (wrong prefix/action shape) if the two ever shared a raw-dir, and the
    # two loops' own "delete last round's json" steps would race each other if run concurrently.

    echo "[1/6] Cleaning up old raw data..."
    # find -delete (not a shell-expanded *.json glob) -- see training_claws.sh's own comment on
    # this, same E2BIG concern once raw_choices/ holds many thousands of per-branch files.
    find "$raw_dir/" -maxdepth 1 -name '*.json' -delete
    find "$raw_dir/checkpoints/" -maxdepth 1 -name '*.json.gz' -delete

    echo "[2/6] Running facility_choices_paradox.py (Data Generation)..."
    python "$vakuu/agents/whispering_earring/facility_choices_paradox.py" \
        --trials 1 --branch-workers 15 --main-workers 3 \
        --branches-per-turn 3 --meta-branches-per-trial 3 \
        --raw-dir "$raw_dir"
    # Change of strategy (see facility_choices_paradox.py's own module docstring): every option at
    # a card_reward divergence now plays to natural death using the SAME turn-branch-and-adopt
    # combat quality as the main line (branches_per_turn), not a single short-horizon rollout --
    # no more meta-options-replicates/meta-horizon-combats knobs, both retired. That makes each
    # divergence far more expensive than Phase 1's (num_options chains, each potentially several
    # combats deep, each combat its own branches_per_turn-way turn-branching) -- BUT every option's
    # chain now runs concurrently (one thread per option, each with its own results_queue -- see
    # that module's docstring), so a divergence's wall-clock cost is roughly the SLOWEST option's
    # chain, not the sum of all of them. Peak worker-pool pressure is correspondingly higher than
    # Phase 1's (every option's turn-branches contend for the shared branch-workers pool at once,
    # on top of every other main worker's own combat) -- raise --branch-workers if turn-branches
    # start timing out under real load. meta-branches-per-trial dropped from 5 to 3 accordingly.
    # Both knobs are starting points, not measured-optimal -- retune once real per-round throughput
    # is known under the new cost shape.

    echo "[3/6] Meta-decision stats for this round..."
    # --no-card-stats: same tradeoff as training_claws.sh's stats.py step -- the card
    # offer/pick table needs a replay pass per turn-branch group (torch + CLR bridge), too
    # expensive to pay every round for a table nobody's watching live. Cumulative stats
    # persist to data/stats_history_choices.jsonl (a sibling of raw_dir, so it survives the
    # "Cleaning up old raw data" wipe above) -- check that mid-run to see whether the
    # meta-option net is actually improving over the day.
    python "$vakuu/agents/whispering_earring/stats_choices.py" \
        --raw-dir "$raw_dir" \
        --round "$ROUND" --no-card-stats

    echo "[4/6] Materializing dataset..."
    python "$vakuu/agents/choices_paradox/model/materialize.py" \
        "$raw_dir/" \
        --out "$vakuu/agents/choices_paradox/model/dataset.pt" \
        --jobs 8

    echo "[5/6] Training model..."
    # --warm-start-from only actually applies when --init is empty (round 1, see train.py) --
    # harmless to always pass once CLAWS_CHECKPOINT exists, since a resumed net (--init
    # non-empty) ignores it. Seeds card-identity embeddings from Claws's own trained ones rather
    # than starting this net's id_emb from noise -- see model.py's warm_started_from.
    python "$vakuu/agents/choices_paradox/model/train.py" \
        "$vakuu/agents/choices_paradox/model/dataset.pt" --epochs 30 \
        --out "$vakuu/agents/choices_paradox/model/choices_paradox.pt" \
        --init "$CHOICES_CHECKPOINT" \
        --warm-start-from "${CLAWS_CHECKPOINT:-}"

    echo "[6/6] Backing up checkpoint..."
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

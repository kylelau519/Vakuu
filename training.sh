export vakuu="/Users/kylelau519/Programming/Vakuu"

export CLAWS_CHECKPOINT="$vakuu/agents/claws/model/claws.pt"
[ -f "$CLAWS_CHECKPOINT" ] || unset CLAWS_CHECKPOINT  # no checkpoint yet on the first run --
    # facility.py's TD-bootstrap value term (_play_turn_only) falls back to 0.0 without one, a
    # known limitation for round 1 only. Once this script's own train.py step below writes
    # claws.pt, the next run of this same script picks it up automatically -- this is what
    # turns repeated runs into a self-play improvement loop.

rm -f "$vakuu/agents/whispering_earring/data/raw/"*.json

python "$vakuu/agents/whispering_earring/facility.py" \
    --trials 1 --branch-workers 10 --branches-per-turn 30 --main-workers 1 \
    --raw-dir "$vakuu/agents/whispering_earring/data/raw"
    # branches_per_turn spends roughly branches_per_turn x turns-per-combat branch-worker-
    # seconds per combat now (branching happens once per turn, not once per whole combat) --
    # retune deliberately rather than assuming this number means what it used to.

python "$vakuu/agents/claws/model/materialize.py" \
    "$vakuu/agents/whispering_earring/data/raw/"*.json \
    --out "$vakuu/agents/claws/model/dataset.pt"

# --init resumes from the previous round's weights -- without it, every run trains a fresh
# randomly initialized network from scratch and discards it, since CLAWS_CHECKPOINT above only
# feeds facility.py's generation step, not training; empty/unset is fine too (train.py treats ""
# as "no checkpoint yet"). --value-only is permanent, not a temporary flag: MCTS was dropped
# from the live generation path entirely (too expensive against this engine's replay-only
# checkpointing -- see agents/whispering_earring/policy.py's module docstring), so recorded
# `policy` is always the softmax-sample/uniform-random placeholder, never a real search
# distribution -- there's no future round where policy training would become meaningful.
python "$vakuu/agents/claws/model/train.py" \
    "$vakuu/agents/claws/model/dataset.pt" --epochs 50 \
    --out "$vakuu/agents/claws/model/claws.pt" \
    --init "$CLAWS_CHECKPOINT" \
    --value-only

export vakuu="/Users/kylelau519/Programming/Vakuu"
# export CLAWS_MCTS=1
unset CLAWS_MCTS  # policy.py checks truthiness of the raw env var string -- CLAWS_MCTS=0 would

export CLAWS_CHECKPOINT="$vakuu/agents/claws/model/claws.pt"
[ -f "$CLAWS_CHECKPOINT" ] || unset CLAWS_CHECKPOINT  # no checkpoint yet on the first run --
    # CLAWS_MCTS still works without one, falling back to a freshly initialized (untrained)
    # network as the search's evaluator (policy.py's _load_mcts). Once this script's own
    # train.py step below writes claws.pt, the next run of this same script picks it up
    # automatically -- this is what turns repeated runs into a self-play improvement loop.

rm -f "$vakuu/agents/whispering_earring/data/raw/"*.json

python "$vakuu/agents/whispering_earring/facility.py" \
    --trials 1 --branch-workers 10 --branches-per-combat 30 --main-workers 1 \
    --raw-dir "$vakuu/agents/whispering_earring/data/raw"

python "$vakuu/agents/claws/model/materialize.py" \
    "$vakuu/agents/whispering_earring/data/raw/"*.json \
    --out "$vakuu/agents/claws/model/dataset.pt"

# --init resumes from the previous round's weights -- without it, every run trains a fresh
# randomly initialized network from scratch and discards it, since CLAWS_CHECKPOINT above only
# feeds facility.py's generation step, not training; empty/unset is fine too (train.py treats ""
# as "no checkpoint yet"). --value-only: CLAWS_MCTS is off above, so recorded `policy` here is
# uniform-random/raw-softmax noise, not real search signal -- drop --value-only once a later
# round actually generates data with CLAWS_MCTS=1.
python "$vakuu/agents/claws/model/train.py" \
    "$vakuu/agents/claws/model/dataset.pt" --epochs 50 \
    --out "$vakuu/agents/claws/model/claws.pt" \
    --init "$vakuu/agents/claws/model/claws.pt" \
    --value-only

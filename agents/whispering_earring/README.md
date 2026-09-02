# whispering_earring

Self-play facility: plays full runs end-to-end and records every decision, to generate training
data for [`claws`](../claws/), the in-combat ML suggestion model. This package doesn't decide
combat well itself -- it drives the game forward so that trajectories exist to train on.

See `facility.py`'s module docstring for the branching scheme (N concurrent main-line workers,
each forking off a checkpoint at every *turn* to explore alternate lines via a shared pool of
branch workers, discarded after that one turn resolves -- a main line adopts whichever of its
branches scores best, per `_turn_score_key`, once all have reported or timed out) and `policy.py`
for the current action-selection policy: combat is uniform-random exploration by default
(non-combat decisions delegate to the Choices Paradox and Golden Compass placeholder policies,
since a random meta/path policy would produce decks and runs too weak to make combat outcomes
meaningful) -- or, with `CLAWS_CHECKPOINT` set, samples from a loaded `claws/model/ClawsNet`
checkpoint's own softmax policy instead (still no search, just the raw network). This
softmax-sample/uniform-random diversity across a turn's parallel branches is what gives
branch-and-adopt something to select between.

Each adopted turn is scored with a TD-bootstrapped value, not a flat per-combat constant (see
`facility.py`'s `_play_turn_only` and `_combat_value`): a turn that doesn't end the combat gets
`-(hp_lost_this_turn)/max_hp + V_net(state_after_turn)` (the current `ClawsNet`'s own value-head
estimate, one forward pass, no further simulation); a turn that does end the combat gets the
existing terminal HP-loss formula (`_combat_value`) instead of a bootstrap. This is what lets a
turn that took damage to set up a good follow-up score differently from a turn that didn't,
which a single scalar backfilled onto the whole combat couldn't do.

Branch workers drive their own turn via sts2-cli's buffer protocol (`buffer_env.Sts2BufferSession`,
in-process, no subprocess/JSON/pydantic decode) instead of the JSON `Sts2Session` -- see sts2-cli's
`src/Sts2Headless/README.md` for the wire format and `policy.py`'s combat/card-select choosers for how
they branch on the decision type (`isinstance(decision, BufferDecision)`) to read fields via
`buffer_layout.py` instead of a pydantic attribute tree. The main self-play line still uses
`Sts2Session`/pydantic `Decision` throughout, since it never plays `combat_play` directly (see
`_resolve_turn`) -- only branch workers do.

## Layout

| File | Role |
|---|---|
| `facility.py` | Orchestrates main + branch workers; `Facility`/`FacilityConfig`; CLI entry point |
| `policy.py` | `choose_action(decision, session, purpose=None)` -- combat is random, or ClawsNet-sampled if `CLAWS_CHECKPOINT` is set; meta decisions delegate out. `purpose`/the returned next-purpose thread `card_select`'s remove/upgrade disambiguation across calls -- see the function's docstring |
| `recorder.py` | `CompactRecorder` -- per-trial JSON capture (action log, policy, per-turn value) |
| `checkpoint.py` | `BranchJob` -- the replay-prefix format passed to branch workers via an in-memory queue |
| `sts2_path.py` | Adds sts2-cli's `python/` to `sys.path`; import before anything from sts2-cli |
| `data/raw/` | Recorded trial JSON per trial (gitignored) |

## Running

```
conda activate sts2
python facility.py --trials 10 --branch-workers 4 --branches-per-turn 3

# multiple concurrent main lines sharing that same branch-worker pool, so branch workers stay
# fed even while any given main line is off doing map/shop/reward decisions between combats
python facility.py --trials 10 --branch-workers 12 --branches-per-turn 3 --main-workers 4
```

To drive combat from a trained (or just initialized -- see `claws/model/train.py --epochs 0`)
checkpoint instead of uniform-random -- this also feeds the TD-bootstrap value term's `V_net`
estimate, not just action selection:

```sh
CLAWS_CHECKPOINT=../claws/model/claws.pt python facility.py --trials 10 ...
```

Requires `sts2-cli` checked out next to this repo (or `STS2_CLI_ROOT` set) and set up per its own
`./setup.sh`.

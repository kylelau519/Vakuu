# whispering_earring

Self-play facility: plays full runs end-to-end and records every decision, to generate training
data for [`claws`](../claws/), the in-combat ML suggestion model. This package doesn't decide
combat well itself -- it drives the game forward so that trajectories exist to train on.

See `facility.py`'s module docstring for the branching scheme (N concurrent main-line workers,
each forking off a checkpoint at every combat to explore alternate lines via a shared pool of
branch workers, discarded after that one combat resolves -- a main line adopts whichever of its
branches reports a win *first*, without waiting on the rest) and `policy.py` for the current
action-selection policy: combat is
uniform-random exploration by default (non-combat decisions delegate to the Choices Paradox and
Golden Compass placeholder policies, since a random meta/path policy would produce decks and runs
too weak to make combat outcomes meaningful) -- or, with `CLAWS_CHECKPOINT` set, samples from a
loaded `claws/model/ClawsNet` checkpoint's own softmax policy instead (still no search, just the
raw network) -- or, with `CLAWS_MCTS` set, is driven by `claws/model/mcts.py`'s real PUCT search
over that same network (see `policy.py`'s module docstring; this is the expensive path, up to
~200 sts2-cli subprocess spawns per combat decision).

## Layout

| File | Role |
|---|---|
| `facility.py` | Orchestrates main + branch workers; `Facility`/`FacilityConfig`; CLI entry point |
| `policy.py` | `choose_action(decision, session)` -- combat is random (ClawsNet-sampled if `CLAWS_CHECKPOINT` is set, or MCTS-searched if `CLAWS_MCTS` is set), meta decisions delegate out |
| `recorder.py` | `CompactRecorder` -- per-trial JSON capture (action log, policy, per-combat HP-loss value) |
| `checkpoint.py` | `BranchJob` -- the replay-prefix format passed to branch workers via an in-memory queue |
| `sts2_path.py` | Adds sts2-cli's `python/` to `sys.path`; import before anything from sts2-cli |
| `data/raw/` | Recorded trial JSON per trial (gitignored) |

## Running

```
conda activate sts2
python facility.py --trials 10 --branch-workers 4 --branches-per-combat 3

# multiple concurrent main lines sharing that same branch-worker pool, so branch workers stay
# fed even while any given main line is off doing map/shop/reward decisions between combats
python facility.py --trials 10 --branch-workers 12 --branches-per-combat 3 --main-workers 4
```

To drive combat from a trained (or just initialized -- see `claws/model/train.py --epochs 0`)
checkpoint instead of uniform-random:

```sh
CLAWS_CHECKPOINT=../claws/model/claws.pt python facility.py --trials 10 ...
```

To drive combat with a real MCTS search instead (much slower -- up to ~200 sts2-cli subprocess
spawns per combat decision, see `policy.py`'s module docstring; `CLAWS_CHECKPOINT` is optional
here too, an untrained net is used if omitted):

```sh
CLAWS_MCTS=1 CLAWS_CHECKPOINT=../claws/model/claws.pt python facility.py --trials 10 ...
```

Requires `sts2-cli` checked out next to this repo (or `STS2_CLI_ROOT` set) and set up per its own
`./setup.sh`.

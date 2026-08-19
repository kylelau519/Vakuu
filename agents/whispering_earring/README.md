# whispering_earring

Self-play facility: plays full runs end-to-end and records every decision, to generate training
data for [`claws`](../claws/), the in-combat ML suggestion model. This package doesn't decide
combat well itself -- it drives the game forward so that trajectories exist to train on.

See `facility.py`'s module docstring for the branching scheme (one main-line worker per run, N
sub-workers that fork off a checkpoint at each combat to explore an alternate line, discarded
after that one combat resolves) and `policy.py` for the current action-selection policy: combat is
uniform-random exploration (no trained network exists yet -- that's `claws`'s job once there's
data to train it on); non-combat decisions delegate to the Choices Paradox and Golden Compass
placeholder policies, since a random meta/path policy would produce decks and runs too weak to
make combat outcomes meaningful.

## Layout

| File | Role |
|---|---|
| `facility.py` | Orchestrates main + branch workers; `Facility`/`FacilityConfig`; CLI entry point |
| `policy.py` | `choose_action(decision, session)` -- combat is random, meta decisions delegate out |
| `recorder.py` | `TrialRecorder` -- per-trial JSONL capture (states, actions, errors) |
| `checkpoint.py` | `BranchJob` -- the replay-prefix format branch workers pick up off the queue |
| `sts2_path.py` | Adds sts2-cli's `python/` to `sys.path`; import before anything from sts2-cli |
| `data/raw/` | Recorded trial JSONL/JSON per trial (gitignored) |
| `data/queue/` | Pending `BranchJob` files, consumed by branch workers (gitignored) |

## Running

```
conda activate sts2
python facility.py --trials 10 --branch-workers 4 --branches-per-combat 3
```

Requires `sts2-cli` checked out next to this repo (or `STS2_CLI_ROOT` set) and set up per its own
`./setup.sh`.

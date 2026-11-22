# Vakuu

An AI that plays Slay the Spire 2, trained by self-play on [spire-zero](https://github.com/shinnku-nikaidou/spire-zero)'s
engine.

Two models, deliberately kept separate:

| Subsystem | Decides | Training |
|---|---|---|
| **Claws** (`vakuu/claws`) | Playing combat itself: a policy + value network over one combat decision | Self-play turn-branching: every combat turn of a mainline run is forked in the engine, each fork's cost is compared against its siblings; value is TD-bootstrapped |
| **Golden Compass** (`vakuu/golden_compass`) | The run-level choices: map path, card rewards, shops, rests, upgrade / remove prompts | A per-floor death-hazard net over run-state afterstates, trained on whole real-HP runs (Claws plays the combats), with paired branch evaluations as an extra signal |

`vakuu/macro` carries a run through the screens neither net decides (rewards, treasure, plain proceed) and is the fallback while
no compass checkpoint exists. Whether the two nets stay separate or converge on a shared policy-value network is an open question;
the split is how the problem is carved up today, not a committed architecture.

The long-term goal is a live in-game suggestion mod (spire-zero ships its own). Deployment would be a single forward pass per
decision, with no live search, so training doesn't need deep per-decision search.

## How it works

The game is [spire-zero](https://github.com/shinnku-nikaidou/spire-zero)'s: the real game (v0.113.1) hosted headless by its
`SimWorker` (deterministic, single-threaded pump), plus a Rust port of the combat rules. `vakuu/engine` talks to both:

- **`SpireHost`** (`engine/host.py`): one game process, driven over its wire protocol. It asks (one *decision* per thing the player
  must answer: map, rewards, card reward, rest, shop, event, card prompt, combat), Python answers with one of the decision's own legal
  action strings. A run can start from floor 0 or from an act-boundary save.
- **`World`** (`engine/engine.py`): the Rust combat engine over ctypes. A live combat the host exports is imported, cloned and played
  out any number of times, with no game and no reload. This is what makes turn-branching cheap.

Every decision the nets see is encoded into one fixed-width float32 **buffer record** (`decision.arr`; layout in
`buffer/layout.py`, writers `buffer/encode.py` for combat and `buffer/run_encode.py` for the run-level screens). Token numbers come
from spire-zero's own id tables (`engine/data/`), so a net is tied to the game version it trained on; `common/extend_vocab.py` moves
a net onto a newer version's vocabulary.

Runs are deterministic (seed + action log), and so is every fork of a saved combat. The branches feed a REINFORCE-style baseline in
`claws/materialize.py`.

```
claws/facility.py        ->  data/spire/raw/*.pkl       claws/materialize.py   ->  dataset.pt  ->  claws/train.py        ->  claws.pt
golden_compass/facility  ->  data/spire/raw_runs/*.json golden_compass/materialize.py -> dataset.pt -> golden_compass/train.py -> golden_compass.pt
branch/facility          ->  data/spire/raw_branch/*.json          (extra signal for golden_compass/train.py --branch-dir)
         ^                                                                                |
         '-- the facilities load the nets (CLAWS_CHECKPOINT, GOLDEN_COMPASS_CHECKPOINT) <-'
```

## Layout

| Path | Contents |
|---|---|
| `vakuu/engine/` | spire-zero binding: `host.py` (game process), `engine.py` (Rust combat engine), `pools.py` (encounter pools), `actions.py`, `tables.py` + `data/` (id tables of the game version) |
| `vakuu/buffer/` | The decision record: `layout.py` (offsets and accessors), `decision.py` (`BufferDecision`, token decode), `encode.py`, `run_encode.py`, `state_key.py` (position equality for fork dedup) |
| `vakuu/common/` | Entity vocabulary (`vocab_ids.json`, `tokenizer.py`), embeddings, `extend_vocab.py` |
| `vakuu/claws/` | `model.py` (`ClawsNet`), `train.py`, `job.py` + `facility.py` (data), `materialize.py`, `evaluate.py` + `eval_panel.py`, `policy.py` (checkpoint and sampling settings), `fast_tokens.py`. See [`README.md`](vakuu/claws/README.md) (the feature table) and [`MODEL.md`](vakuu/claws/MODEL.md) |
| `vakuu/golden_compass/` | `hazard_net.py`, `run_state.py`, `room_search.py`, `agent.py` (decision time), `policy.py`, `job.py` + `facility.py` (data), `materialize.py`, `train.py`, `compare.py` (the gate), `progress.py`. See its [README](vakuu/golden_compass/README.md) |
| `vakuu/branch/` | Paired branch evaluation of card-reward and route choices (`job.py`, `facility.py`) |
| `vakuu/macro/` | `heuristics.py`, `player.py` (compass or heuristics), `tier_policy.py` + `tier_data/` (shop / remove / upgrade tier lists) |
| `vakuu/paths.py` | Default locations (`SPIRE_DATA`, `SPIRE_RUN` override) |
| `scripts/` | `train_claws.sh`, `train_compass.sh`, `train_alternating.sh` (generate -> materialize -> train loops), `watchdog.sh`, `reset.sh`, `setup_env.sh`, `lib.sh` |
| `tests/`, `tools/` | Engine-free tests; `tools/verify_encode.py` round-trips the encoder over spire-zero's combat fixtures |
| `data/` | Generated game data (gitignored; see `data/README.md`). `act_checkpoints/` holds the act-boundary saves runs resume from, `spire/` the active run's raw data, history and logs |
| `runs/` | Trained nets (gitignored): `spire/` is the active run (`claws/claws.pt`, `compass/golden_compass.pt`, with backups); `run2/` and `base/` are earlier nets used as warm starts |

## Setup

Prerequisites: Slay the Spire 2 v0.113.1 on Steam, the .NET 9+ SDK and a Rust toolchain (for spire-zero), Python 3.11+.

```sh
git clone https://github.com/shinnku-nikaidou/spire-zero.git ~/Programming/spire-zero   # then build it: see its README
./scripts/setup_env.sh      # .venv, requirements, and a check that spire-zero is built
```

Vakuu looks for spire-zero at `$SPIRE_ZERO` (default `~/Programming/spire-zero`) and needs its
`engine/target/release/libsts2ai_engine.so` and `src/Sts2Ai.SimWorker`. Its binaries and the game's assemblies are never vendored here.

Everything runs from the repo root: `python -m vakuu.<package>.<module>`.

```sh
python -m pytest tests            # no engine needed
```

## Running

Smoke test (a couple of runs from act-2 saves, tiny settings):

```sh
CLAWS_CHECKPOINT=runs/spire/claws/claws.pt python -m vakuu.claws.facility --trials 2 --branch-workers 2 \
    --forks-per-turn 2 --raw-dir /tmp/smoke_raw --ckpt-dir data/act_checkpoints --resume-act 2
```

Training loops, each repeating generate -> materialize -> train per round (`MAX_ROUNDS=1` is a dry run; every knob is an environment
variable, listed at the top of each script):

| Script | Trains |
|---|---|
| `scripts/train_claws.sh` | Claws only (macro decisions from the heuristics or the compass if one exists) |
| `scripts/train_compass.sh` | The compass (real-HP runs played by the current Claws), with the gate |
| `scripts/train_alternating.sh` | Both in one loop: Claws trains in the background while the compass stage runs; the gate decides whether a candidate compass replaces the accepted one |

`scripts/watchdog.sh train_alternating.sh 3600` runs a loop under a hard wall-clock limit and restarts it (there is a known,
not-yet-root-caused hang inside a round). `scripts/reset.sh` wipes the active run and writes random-init nets.

`CLAWS_CHECKPOINT` / `GOLDEN_COMPASS_CHECKPOINT` make the facilities play with trained nets; `CLAWS_GREEDY` and `CLAWS_NOISE_EPS`
set how Claws samples (see `vakuu/claws/policy.py`).

## Status

Active, pre-deployment. The turn-branching Claws loop, the compass with its gate and the branch evaluation all run end to end on
spire-zero. Open problems: a known hang inside training rounds, the act 1 boss wall (which may come from deck quality rather than
combat skill), and the open question of a unified network.

Decompiled game source is used as reference during development but is never checked into this repository.

# Vakuu

An AI that plays Slay the Spire 2.

Four subsystems, currently kept separate:

| Subsystem | Decides | Training |
|---|---|---|
| **Choices Paradox** | Meta options: card reward take/skip, removal, upgrade, rest site, event choices | Self-play branch-and-score |
| **Golden Compass** | Which map path to take | Whole-run win/loss reward, with a local Elite/Boss shaping bonus |
| **Claws** | Playing combat itself (policy + value network) | Self-play turn-branching, TD-bootstrapped value |
| **Whispering Earring** | Nothing itself. The self-play facility that plays full runs and records every decision as training data for the other three | n/a |

Whether these stay separate models or converge on a shared policy-value network is an open question.
The split is how the problem is carved up today, not a committed architecture.

The long-term goal is a live in-game suggestion mod. Deployment would be a single forward pass per
decision, with no live search, so training doesn't need deep per-decision search.

## How it works

The game is driven by [`sts2-cli`](https://github.com/kylelau519/sts2-cli), a fork of
[wuhao21/sts2-cli](https://github.com/wuhao21/sts2-cli). It runs `sts2.dll` on bare .NET with no
Godot process. This repo talks to it in two ways:

- **JSON session** (`Sts2Session`): pydantic-typed decisions. Used by the main self-play line for
  map, shop and reward decisions.
- **Buffer session** (`Sts2BufferSession`): an in-process CoreCLR host with no subprocess, JSON or
  pydantic decode. Used by branch workers for combat. It supports native checkpoints
  (`save`/`load_save`) and `HealAndEnterCombat`.

Runs are deterministic: seed + action log reproduces a run. Branching works by forking from a
checkpoint at every combat turn (Claws) or every meta decision (Choices Paradox), exploring
alternatives in a shared pool of worker processes, and recording every branch. The branches feed a
REINFORCE-style baseline in each subsystem's `materialize.py`.

```
whispering_earring/facility_*.py  ->  data/raw*/ (trial JSON)
        ^                                  |
        |                       <subsystem>/model/materialize.py  ->  dataset.pt
   policy.py loads                         |
   the checkpoints  <------  <subsystem>/model/train.py  ->  *.pt
```

## Layout

| Path | Contents |
|---|---|
| `agents/claws/` | Combat model (`ClawsNet`): feature table in the README, transformer trunk, policy/value heads, `materialize.py`, `train.py` |
| `agents/choices_paradox/` | Deckbuilding/meta model (`ChoicesParadoxNet`) plus the tier-list heuristic used until a checkpoint exists |
| `agents/golden_compass/` | Map-path model plus the DP heuristic in `policy.py` used until a checkpoint exists |
| `agents/whispering_earring/` | Self-play facility: `facility_claws.py`, `facility_choices_paradox.py`, `facility_compass.py`, shared core in `facility_common.py`, `policy.py`, `recorder.py`, `checkpoint.py`, and `debug/` repro scripts |
| `agents/common/` | Shared tokenizer, embeddings and vocab |
| `training_*.sh` | Generate -> materialize -> train loops (see below) |
| `watchdog_overnight.sh` | Runs a training loop under a hard wall-clock limit and restarts it (there is a known, not-yet-root-caused hang inside a round) |
| `setup_env.sh`, `reset.sh` | Fresh-machine setup; wipe all generated state and reinitialize random checkpoints |
| `data/` | Generated trial data. Gitignored; the recipe is versioned, not the data (see `data/README.md`) |
| `mod/` | Placeholder for a C# Godot mod. Probably unnecessary now that `sts2-cli` drives the game; see `mod/README.md` |

Each subsystem directory has its own README with the design and open questions. Start with
[`agents/whispering_earring/README.md`](agents/whispering_earring/README.md).

## Setup

Prerequisites: .NET 9+ SDK, Slay the Spire 2 installed via Steam, Python 3.11+.

```sh
# From a directory that should hold both repos as siblings
./Vakuu/setup_env.sh   # clones Vakuu + sts2-cli, creates .venv, installs deps, runs sts2-cli/setup.sh
```

`sts2-cli` must sit next to this repo (or set `STS2_CLI_ROOT`). Its `setup.sh` copies and IL-patches
game DLLs from your Steam install; those binaries are never vendored here.

## Running

Smoke test:

```sh
python agents/whispering_earring/facility_choices_paradox.py --trials 1 --branch-workers 1 \
  --main-workers 1 --branches-per-turn 1 --meta-branches-per-trial 1 --ascension 1 \
  --raw-dir /tmp/smoke_test
```

Training loops, each repeating generate -> materialize -> train per round:

| Script | Trains |
|---|---|
| `training_claws.sh` | Claws |
| `training_choices.sh` | Choices Paradox |
| `training_golden_compass.sh` | Golden Compass (reuses the trials the other two generate) |
| `training_alternating.sh` | Rotates through the above |
| `training_debug.sh` | Small configuration for debugging the loop |

Edit the `vakuu` path at the top of each script for your machine. `reset.sh` wipes all generated
data and checkpoints and writes fresh random-init `.pt` files.

Set `CLAWS_CHECKPOINT` / `CHOICES_PARADOX_CHECKPOINT` to make `policy.py` sample from a trained
network instead of the placeholder heuristics.

## Status

Active, pre-deployment. The self-play facility, buffer-protocol branch workers, native checkpoint
branching and all three training pipelines are wired end to end. Open problems include a known hang
inside training rounds, an Act 1 boss wall that may come from deck quality rather than combat skill,
and the open question of a unified network.

A prior prototype of Choices Paradox (per-character XGBoost hurdle models over scraped run
histories) lives in a separate `ChoicesParadox` repo. It is being redesigned, not carried over.

Decompiled game source is used as reference during development but is never checked into this
repository.

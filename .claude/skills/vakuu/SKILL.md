---
name: vakuu
description: Orients work on Vakuu, the Slay the Spire 2 autoplay project (Claws in-combat policy/value net, Golden Compass run-level hazard net, the macro heuristics, and the self-play facilities that train them on spire-zero). Use when working on the spire-zero host/engine binding, the buffer decision record and its encoders, self-play data generation (mainline + engine forks), the training loops and gates, RNG determinism, or when reading the decompiled STS2 source to answer "how does the game actually do X".
---

# Vakuu — Slay the Spire 2 autoplay

## The project

**Vakuu** is an AI that plays STS2, trained by self-play on [spire-zero](https://github.com/shinnku-nikaidou/spire-zero)'s engine.
Two nets, deliberately not yet merged — whether they stay separate is an open question, not a settled architecture:

| Subsystem | Package | Decides |
|---|---|---|
| **Claws** | `vakuu/claws` | Combat: a policy + value net over one combat decision. Trained by turn-branching: every combat turn of a mainline run is forked in the Rust engine, forks are compared against siblings |
| **Golden Compass** | `vakuu/golden_compass` | Map path, card rewards, shops, rests, upgrade/remove prompts: a per-floor death-hazard net over run-state afterstates, trained on whole real-HP runs; paired branch evaluations (`vakuu/branch`) are an extra signal |

`vakuu/macro` carries a run through the screens neither net decides, and is the fallback when no compass checkpoint exists.
**Whispering Earring** (`vakuu/whispering_earring`) is the self-play facility: it decides nothing itself, it plays full runs and
records every decision as training data for the two nets. (The older "Choices Paradox" name is gone: the compass absorbed it. The
retired sts2-cli engine and its facility code are in git history before the `spire_only_refactor` branch.)

Python env: `source .venv/bin/activate` (see `scripts/setup_env.sh`). Everything runs from the repo root as `python -m vakuu...`.

## Repo layout

```
vakuu/engine/          spire-zero binding: host.py (game process, wire protocol), engine.py (Rust combat engine, ctypes),
                       pools.py, actions.py, tables.py + data/ (id tables of the game version)
vakuu/buffer/          the fixed-width float32 decision record (decision.arr): layout.py, decision.py, encode.py (combat),
                       run_encode.py (run-level screens), state_key.py (fork dedup)
vakuu/common/          entity vocabulary, embeddings, extend_vocab.py
vakuu/claws/           model, train, job (data), materialize, evaluate, policy (CLAWS_* env), fast_tokens
vakuu/golden_compass/  hazard_net, run_state, room_search, agent, policy, job (data), materialize, train, compare (gate)
vakuu/whispering_earring/  facility_claws, facility_compass, facility_branch (the self-play pools), workers
vakuu/branch/          card-reward / route branch evaluation
vakuu/macro/           heuristics, player, tier_policy + tier_data
scripts/               train_claws.sh, train_compass.sh, train_alternating.sh, watchdog.sh, reset.sh, setup_env.sh, lib.sh
tests/  tools/         engine-free tests; verify_encode.py (encoder round trip over spire-zero's fixtures)
data/  runs/           generated, gitignored: act saves + raw data + logs; trained nets (runs/spire is the active run)
```

**Paths outside this repo** (`~/Programming/`):

- `spire-zero` — the **engine**, third-party (`shinnku-nikaidou/spire-zero`). Two parts Vakuu uses: the real game v0.113.1 hosted
  headless by `src/Sts2Ai.SimWorker` (the `SpireHost` wire protocol: `hello`, one `decision` per question, answered by `ready` /
  `recommendation`), and `engine/` — a Rust port of the combat rules (`libsts2ai_engine.so`) that imports a live combat, clones it
  and plays it out. It builds against the game's own assemblies (never vendor them here). Read `engine/DESIGN.md` and its README
  before digging into the decompiled source: it is a translation of the game's own code, already organised for this job.
  `SPIRE_ZERO` overrides the location; `vakuu/engine/host.py` and `engine.py` read it.
- `sts2-decompiled` — read-only decompiled game source. **Never copy it into this repo and never commit any of it**; it's derived
  from a copyrighted game and Vakuu has a public remote. Reference it in place; `references/engine-map.md` maps it.
- `ChoicesParadox` — a **prototype**: per-character XGBoost hurdle models over scraped run histories. Prior art, not architecture;
  don't carry over its structure by default.

## How data flows

1. A facility (`whispering_earring/facility_claws.py`, `facility_compass.py`, `facility_branch.py`) starts a pool of spawned worker processes;
   each owns one game host and plays runs from an act-boundary save (`data/act_checkpoints`, `--resume-act N`) or from floor 0.
2. Claws data: the mainline's every combat turn is exported into the Rust engine and cloned `--forks-per-turn` times
   (`claws/job.py`); forks reaching a seen end-of-turn state are dropped (`buffer/state_key.py`). Output: `raw/*.pkl`, already tokenized.
3. Compass data: whole real-HP runs with the compass deciding, logging a `RunState` before every macro decision
   (`golden_compass/job.py`). Output: `raw_runs/*.json`.
4. `materialize.py` -> `dataset.pt` -> `train.py` -> a `.pt`; the next round's facilities load it through
   `CLAWS_CHECKPOINT` / `GOLDEN_COMPASS_CHECKPOINT`. The compass has a gate (`compare.py`): a candidate replaces the accepted compass
   only if it beats it on fresh runs.
5. Determinism: seed + action log reproduces a run, and a fork of a saved combat is exact. A net's token numbers come from the
   game version's id tables, so nets are tied to the version they trained on (`common/extend_vocab.py` converts).

## Checking a change

- `python -m pytest tests` — no engine needed.
- A real round in miniature: `MAX_ROUNDS=1 CLAWS_TRIALS=2 WORKERS=2 CLAWS_FORKS=2 CLAWS_EPOCHS=1 EVAL_RUNS=2 HELDOUT_RUNS=2 RESUME_ACT=2
  SPIRE_RUN=/tmp/x/run SPIRE_DATA=/tmp/x/data scripts/train_claws.sh` (and `train_compass.sh`, `train_alternating.sh`) — point
  `SPIRE_RUN` / `SPIRE_DATA` at scratch dirs so the real nets and data are untouched.
- A refactor must keep materialized datasets bit-identical on fixed raw files and the trainers' first held-out score identical on fixed weights.

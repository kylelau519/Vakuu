# golden_compass

Decides the non-combat run: which map node to enter and which card to take. It replaces the old
Golden Compass (path-combination DP / net) and Choices Paradox (tier-list / meta net). The idea is
borrowed from spire-zero's run value net; the combat is **not** brute-forced: Claws plays it, so
the labels say "survives if Claws is the one fighting".

## The model

`HazardNet` reads a `RunState` and predicts, for every floor `g`, the chance the run **dies on `g`
given it is alive there**. Survival from a state is the product of `(1 - hazard)` over the floors
still to come. Every floor survived is a label, so it learns even while nothing wins.

- **Deck**: per card `embed(id) + embed(upgrade) + embed(enchant)` through a shared MLP, pooled as
  `[mean, sum/16]` (DeepSets). No hand-built deck features.
- **Relics, potions, boss, remaining encounters, character, act**: summed embeddings.
- **Map**: a bag of `"<Type>+<rows ahead>"` tokens, one per node reachable from the current one.
  **No edges.** "More options ahead" and "a rest site before the elite" are just more or different
  tokens, so the net learns them from outcomes instead of memorising route shapes.
- **Numbers**: floor, HP fraction, max HP, HP, gold, ascension, potion slots, deck size, relic count.
- **Heads**: 64-floor hazard (masked BCE, one vote per run) and a 6-way auxiliary head, the share of
  max HP each of the next six combats costs, which gives the body a dense low-variance signal.

## How a choice is scored

Every choice becomes the `RunState` it leads to (an afterstate) and that state is scored:

| Decision | Options | Afterstate |
|---|---|---|
| `map_select` | each reachable child | that node entered: it becomes `"<Type>+0"`, everything reachable from it is measured from it |
| `card_reward` / `event_card_reward` | each card, plus skip | deck with / without the card |
| `rest_site` | Heal, Smith | Heal: +30% max HP (+15 with Regal Pillow). Smith: worth its best card to upgrade |
| `shop` | leave, or the first step of a buy / removal sequence | a beam search over sequences of buys and one removal (below) |
| `card_select` after a Smith / shop removal | each distinct card | that card upgraded / removed |

Score = expected further floors survived (`hazard_net.expected_floors`), not win chance, which is ~0
for every option while the combat agent is still weak. Sampling is a softmax over scores
(`COMPASS_TAU`, default 0.5) mixed with `COMPASS_EPS` (default 0.1) uniform. With no checkpoint,
`GOLDEN_COMPASS_CHECKPOINT` unset or missing, maps are uniform random and card rewards use the tier
list (`whispering_earring/tier_policy.py`).

### In-room search (`room_search.py`)

Same idea as spire-zero's room search -- apply each option, keep going through the room's further
screens, keep the best positions -- but on token states, since the engine can't snapshot a room
(native saves are map-level). The shop search is a beam of 16 states with at most 1,024 states scored;
the value of a root option (leave / buy a card / buy a relic / buy a potion / remove) is the best state
reachable through it, and the next shop screen re-searches with what is left. A Smith is worth its
best upgrade, found the same way. Everything is scored in batches by one net call per level.

What this can't see: anything an option does that a `RunState` doesn't hold (a relic's effect, apart
from the relic being present).

Rest-site options are identified by position: index 0 is Heal and 1 is Smith, because relics only
append options (Shovel's Dig, Meat Cleaver's Cook, ...). The Midas run modifier removes Heal and would
break that. Appended options are never chosen while Heal or Smith is available.

Not covered, still random: events, and card selections that are not a single upgrade / removal (e.g.
"remove 2").

## Data and training

```sh
python agents/whispering_earring/facility_compass.py --trials 200 --branch-workers 15 --ascension 1 \
    --raw-dir agents/whispering_earring/data/raw_runs        # RunJob: real-HP runs, nothing healed
python agents/golden_compass/materialize.py <run dirs...> --out agents/golden_compass/dataset.pt
python agents/golden_compass/train.py agents/golden_compass/dataset.pt --epochs 20 \
    --out agents/golden_compass/golden_compass.pt --init ""      # or --init the previous checkpoint
```

`training_compass.sh` loops this; `training_alternating.sh` alternates it with Claws.

A run file (`role: "run"`) holds a `RunState` per macro decision plus a `map_entered` afterstate for
every map move, each combat's HP in and out, the outcome (`lost`, `won`, `censored` for a stalled
combat) and the floor the run ended on. Labels are derived from those; no replay is needed.

**Labels are on-policy for Claws.** Retrain the hazard net whenever Claws changes. Round 0 hazard is
mostly "how badly does a random combat agent die", so expect path and deck effects to appear only
after a few alternating rounds.

## Remaining enemies (planned)

`RunState.remaining` is the encounter ids the act can still roll: beat elite A and you will only meet
B or C, which moves the hazard a lot. The net already embeds it (the same encounter table the boss
uses); what's missing is the data. The buffer doesn't export the act's encounter pool, so `remaining`
is `[]` today. Filling it needs either an sts2-cli export of the pool's unseen entries or a
Python-side tracker of the encounters fought so far plus the act's pool table. Neither changes the
model or the dataset format.

## Layout

| File | Role |
|---|---|
| `run_state.py` | `RunState`, buffer encoder, map tokens, afterstate builders. Pure Python; only `encode_from_buffer` touches sts2-cli |
| `hazard_net.py` | `Encoder` (RunState -> tensors), `HazardNet`, survival maths, losses |
| `compass.py` | Decision time: scoring, exploration, per-process checkpoint load |
| `room_search.py` | Rest / shop / card_select options and the shop beam search |
| `materialize.py`, `train.py` | run files -> `dataset.pt` -> `golden_compass.pt` |

Flat, uniquely named modules on purpose: `whispering_earring/policy.py` also puts `claws/model/` on
`sys.path` and imports a bare `model`, so nothing here is called `model.py`.

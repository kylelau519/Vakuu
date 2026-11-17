# Claws model

Claws suggests combat actions. A single forward pass over a `combat_play` decision gives a policy
(one logit per legal action) and `V(state)`: **the HP the player is expected to lose from this state
until the combat ends** (0 once it is won). There is no search at deployment.

- `model.py` -- tokenizer -> transformer -> policy/value heads (`ClawsNet`), reading the buffer
  protocol (`BufferDecision`) directly.
- `materialize.py` -- turns `whispering_earring`'s mainline + fork files into training examples.
- `train.py` -- trains `ClawsNet` on them.

## Value and Q

- `V(s)`: expected HP still to be lost before the combat ends. In HP, never negative
  (`ClawsNet.value_in_hp`). The target is what actually happened: the HP lost from that decision
  to the end of the combat, in the game that was played (Monte Carlo, no bootstrapping).
- A turn's cost: `Q = HP lost this turn + V(state at the next turn's start)`, with `V = 0` once
  the combat is won. `Q` is the one-step version of the same quantity `V` regresses on.
- Combats are measured **from full HP**: a mainline records the HP the run has, heals to full,
  plays the combat, then deducts what it cost from the recorded HP. `V` therefore never has to
  absorb "and then you die" -- a run dies only if the deduction takes it to 0 (or the player dies
  at full HP, which costs max HP). See `whispering_earring/jobs.py`'s `MainlineJob`.

## Data

`whispering_earring/facility_claws.py` plays a run straight from the policy (the *mainline*), then
forks every turn of every combat it played `--forks-per-turn` times: each fork reloads the same
checkpoint (the combat's room-entry native save plus the actions up to that turn) and plays the
combat out. The mainline never adopts a fork; each job writes its own file.

For one combat turn the *group* is {the mainline's own turn} + {its forks}: the same state, different
continuations. Each group member's cost is either the HP it actually lost from that turn's start to
the combat's end (`--target return`), or `Q` from the `V` it recorded at generation time
(`--target q`, needs a `CLAWS_CHECKPOINT` when the data was generated). Its **advantage** is the
group's mean cost minus its own, so positive means it lost less HP than its siblings. Use `return`
while `V` is untrained; `q` has lower variance but inherits `V`'s errors.

`materialize.py` emits one example per `combat_play` decision:

| field | meaning |
|---|---|
| tokens, options | the state and its legal actions |
| `value` | HP actually lost from this decision to the combat's end |
| `action_index`, `advantage` | set only inside a group's compared turn: the action taken and its advantage |

A fork's later turns have no group of their own, so only their turn-start decisions are kept, for `V`.
States are rebuilt by replaying each mainline once (taking a native save before every
`select_map_node`) and reloading each fork from those saves, so a fork costs one combat of replay.

```
python materialize.py ../../whispering_earring/data/raw --out dataset.pt --target return
python train.py dataset.pt --epochs 40 --out claws.pt --init ""      # round 1: random init
python train.py dataset.pt --epochs 40 --out claws.pt --init claws.pt
```

`V` is on-policy: it estimates the loss under the policy that produced the data, so regenerate data
after every training round and don't mix rounds from different checkpoints. A checkpoint trained
before this design has a different value head (bounded `tanh`, not HP) and should be retrained from
`--init ""`.

## Representation

Self-attention over typed tokens: `tok = id_emb[global_id] + type_emb[type] + feat_proj[type](features)`
(numeric features flat `x / 10`, no log; hp keeps its hp/max_hp ratio), pre-norm transformer blocks, then:

- policy: one query per legal action (`play_card` = hand-card token, plus the target enemy's token for
  an aimed card; `use_potion` likewise; `end_turn` = a learned query plus the player token) -> MLP -> logit.
- value: the player token -> MLP -> `V_SCALE * softplus`.

Token types today: player (hp ratio, hp/10, block/10, energy/10, max energy/10), hand cards, enemies (hp, block, intent), relics, potions, player powers, enemy powers (one token per power, amount/10; the owner enemy's id embedding is added so the power knows whose it is),
and one pooled token each for the draw, discard and exhaust piles. A pile token is
`rho(masked_mean(phi(card)) ++ summaries)` (`PileEncoder`): `phi` sees the shared `id_emb`, cost,
upgraded, copies/10 and (draw pile) copies/draw_count; the summaries are count/50, counts by card type,
0-cost count and mean cost. Mean-pooling with no position embedding hides the draw pile's true order.
Datasets materialized before piles existed lack the `pile_*` tensors and must be re-materialized;
older checkpoints still load (`load_compatible`), with the pile weights freshly initialized.

## Known gaps

- **No multi-hit intent, orbs or history counters in the tokens.** What the enemies are
  buffed with drives the HP still to be lost, so `V` is noisier than it needs to be.
- **`V` is a mean.** A hazard model downstream needs how likely a loss at least as large as the player's
  HP is; a categorical head over loss buckets (with `V` as its mean) would give that.
- **HP is the only cost.** Potions are free, so the policy has no reason to keep them; record them per combat
  (the mainline already does, as `potions_used`) and price them before relying on that.
- **Combats are capped** at `MAX_COMBAT_TURNS` (`jobs.py`); a stalled combat is dropped from the groups.
- **Training always starts combats at full HP**, while a live game won't.

# Claws model

Claws suggests combat actions. A single forward pass over a `combat_play` decision gives a policy
(one logit per legal action) and `V(state)`: **the HP the player is expected to lose from this state
until the combat ends** (0 once it is won). There is no search at deployment.

- `model.py` -- tokenizer -> transformer -> policy/value heads (`ClawsNet`), reading the buffer
  record (`BufferDecision`) directly.
- `policy.py` -- the checkpoint and the sampling settings (`CLAWS_CHECKPOINT`, `CLAWS_GREEDY`, `CLAWS_NOISE_EPS`).
- `job.py` / `facility.py` -- data generation: one mainline run and its forks per worker process.
- `materialize.py` -- turns the mainline + fork files into training examples.
- `train.py` -- trains `ClawsNet` on them.
- `evaluate.py` / `eval_panel.py` -- greedy real-HP evaluation and its text report.

## Value and Q

- `V(s)`: expected HP still to be lost before the combat ends. In HP, never negative
  (`ClawsNet.value_in_hp`). The target is what actually happened: the HP lost from that decision
  to the end of the combat, in the game that was played (Monte Carlo, no bootstrapping).
- A turn's cost: `Q = HP lost this turn + V(state at the next turn's start)`, with `V = 0` once
  the combat is won. `Q` is the one-step version of the same quantity `V` regresses on.
- Combats are measured **from full HP**: a mainline records the HP the run has, heals to full,
  plays the combat, then deducts what it cost from the recorded HP. `V` therefore never has to
  absorb "and then you die" -- a run dies only if the deduction takes it to 0 (or the player dies
  at full HP, which costs max HP). See `job.py`'s `ClawsRun`.

## Data

`facility.py` starts `--branch-workers` processes; each plays mainline runs (`ClawsRun`) from random act-boundary saves in
`--ckpt-dir` (`--resume-act N`) or from floor 0. The mainline plays straight from the policy; at every turn of every combat it
exports the live combat into spire-zero's Rust engine and clones it `--forks-per-turn` times, each clone playing the turn out with the
policy sampled differently (later turns greedy). Forks that reach an already-seen end-of-turn state are dropped (`--dedup-forks`,
`buffer/state_key.py`). No reload, no replay: a fork costs one combat of engine time. The mainline never adopts a fork. One
`raw_dir/{main_id}.pkl` per run holds the per-turn costs and the already tokenized examples.

A fork's later turns have no group of their own, so only their turn-start decisions are kept, for `V`.

```
python -m vakuu.claws.materialize data/spire/raw --out dataset.pt --advantage turn
python -m vakuu.claws.train dataset.pt --epochs 40 --out claws.pt --init ""      # round 1: random init
python -m vakuu.claws.train dataset.pt --epochs 40 --out claws.pt --init claws.pt
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

- **No orbs in the tokens.** The player and enemy tokens carry the history counters
  (18 pairs: this turn, this combat), a card token the game's damage preview per target and an enemy token its next three
  turns by the game's own move machine (expected damage, chances of attack/defend/other, uncertainty; the record's tail,
  `Look` in the observation). `FEATURE_DIM` is 74 (a card token also carries the game's block preview, an enemy token its attack's hits, damage per hit and intent types) and an action's own numbers (damage against its
  target, hits, product) enter its query through `act_proj`; a checkpoint of the 6-wide features loads with the new columns
  at zero, but data tokenized with them must be made again.
- **`V` is a mean.** A hazard model downstream needs how likely a loss at least as large as the player's
  HP is; a categorical head over loss buckets (with `V` as its mean) would give that.
- **HP is the only cost.** Potions are free, so the policy has no reason to keep them; record them per combat
  (the mainline already does, as `potions_used`) and price them before relying on that.
- **Combats are capped** at `MAX_COMBAT_TURNS` (`jobs.py`); a stalled combat is dropped from the groups.
- **Training always starts combats at full HP**, while a live game won't.

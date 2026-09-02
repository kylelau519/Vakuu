# Model



Four pieces exist so far, all v0/untrained -- wired together end to end, but none of it has been
trained on real data yet, so none of it has actually improved combat play. See each module's
docstring for specifics and open questions.

- `model.py` -- tokenizer -> transformer -> policy/value heads (`ClawsNet`), proving the
  Architecture section below's shapes compose against a real captured `combat_play` decision.
  Deliberately out of scope for v0: piles, `history` counters, enemy powers, orbs/stars/osty.
- `materialize.py` -- replays a `whispering_earring` trial file through a fresh sts2-cli session to
  reconstruct the state at each policy-bearing decision, and packs it with that decision's recorded
  `policy`/`value` into a training example.
- `train.py` -- trains `ClawsNet` on `materialize.py`'s output: policy cross-entropy + value MSE,
  batch size 1 (no padding -- see its docstring for why). Verified end to end (loss goes down,
  `--init` correctly resumes from a saved checkpoint) on a single small trial's worth of data --
  not a real training run, there isn't enough materialized data for one yet.
  `(p, v)` evaluator. A chance node is created for `play_card` on a card that draws mid-turn (e.g.
  Pommel Strike), not `end_turn` -- the once-per-turn draw is already covered by simulation
  diversity across the tree, but a same-turn draw effect revisited from the same node would
  otherwise always resolve to the same drawn card every time; see `draws_cards`'s docstring. Two
  uses `sqrt(1 + sum(N))`, not `sqrt(sum(N))`, so a freshly expanded node's first-ever selection
  still weighs the network's priors instead of every edge scoring 0 and ties defaulting to
  option 0 regardless of `P(s,a)`. Wired into `whispering_earring/policy.py` behind `CLAWS_MCTS`
  -- opt-in, not the default, since it means spawning sts2-cli subprocesses for every genuinely
  new tree node from inside an already-running self-play worker, a much bigger cost than the
  raw-network wiring below. `N_SIMULATIONS` (200) is a per-*combat* budget, not per-decision --
  `whispering_earring/policy.py`'s `CombatSimBudget` spends it down in chunks across a combat's
  decisions and falls back to the raw network softmax once it's exhausted, rather than re-running
  a full 200-simulation search at every single decision in the combat.

`whispering_earring/policy.py` samples combat actions from a `ClawsNet` checkpoint's raw softmax
policy (no search) when `CLAWS_CHECKPOINT` is set -- see its module
docstring. `claws.pt` (an untrained init checkpoint, `train.py --epochs 0`) exists to prove this
path end to end before there's anything trained; using it doesn't produce better play than
uniform-random, only differently-random play with a real (if meaningless) probability distribution
recorded as `policy` instead of a one-hot placeholder. 

## Prediction
Predict an action capture the meaning of play_card("bash", "cultist_1"), re-analyze and give potion("block_potion, self) (because self is the only legal target), at some point "end_turn"

## Representation

- AZ is trained on chess, Go, and Shogi, which heavily rely on neighboorhood information, and therefore CNN is used. However, in sts2 where neighboorhood are non-existence, other representations will be a better candidate. 
- The interrelationships between relics, deck, hand, enemies and the rest can be represented by attention based transformers. 

## Architecture

- A dedicated tokenizer is made in `/common`. After tokenizing the game data, they will be embedded strategically. Typed dictionary should provide all the information.
- Two embedding will be used to distinguish the type. IDs embedding, which is the standard vocabulary identifier, and a type embedding, which is much smaller, to capture the type like "relic", "enemy" and make the embedded item category absolutely clear. This is performed because unlike text embedding everything is a text.
- To add extra information about the object, like "enemy health", feature projection is performed through projection layer. Numerical values like health will be normalized with `log1p`. Finally, layer normalization is applied.
- Once the process is done, a form of token like `tok = self.emb[global_id] + self.type_emb[type_id] + self.feat_proj[type_id](features)` will be used.
- Followed by multiple self-attention, cross-attention layers, to get a transformer encoded information.
- Something like this `x = x + attention(layernorm(x))`, `x = x + ffn(layernorm(x))` should be implemented.
- Policy head: To evaluate action "play(Bash, target=Cultist)", `q = W_card(h[7]) + W_tgt(h[3]) + type_emb[CARD] # it is implied the type CARD means play, other action like card_select, the token will be source_token + x + attention(x)`, followed by `logit = mlp(q)` with the masking we can collect from the card typed dictionary. For example, when we play true-grit, we enter card_select for exhaust. We shall inform the network through adding the `true_grit_token` to the vector.
- Value head: HP loss is used as the metric. Every combat allows 5-7 hp loss as chip damage to avoid strongly discouraging values.


## Data
- Every attempt's full trajectory — win or loss — is kept as real training data; a loss is valid signal and is not discarded.
- Generated by [`whispering_earring`](../../whispering_earring/); see `recorder.py`'s `CompactRecorder` for the actual writer this section describes.

Each entry in a trial's `actions` list is the action taken, plus (for `combat_play` decisions only)
the policy that produced it:
```
{
  "cmd": "action",
  "action": "play_card",
  "args": {"card_index": 1, "target_index": 0},
  "policy": [
    {"action": "end_turn", "args": {}, "prob": 0.05},
    {"action": "play_card", "args": {"card_index": 1, "target_index": 0}, "prob": 0.71},
    {"action": "play_card", "args": {"card_index": 1, "target_index": 1}, "prob": 0.19},
    {"action": "play_card", "args": {"card_index": 3}, "prob": 0.05}
  ],
  "value": -1.0
}
```
`policy` is a **list** of `{action, args, prob}` records over the legal options at that decision, not
a dict keyed by action/index — a dict can't tell apart two options that share an action name but
differ in `args`, which happens whenever an `AnyEnemy` card can be aimed at more than one alive
enemy (`play_card` with `card_index: 1` against two different `target_index` values, as above). The
default placeholder/softmax policies (`whispering_earring/policy.py`, `CLAWS_CHECKPOINT` unset) only ever sample one candidate target per card, so trial files from those
paths have at most one entry per `card_index`. The list shape is what lets a real search populate
more than one 

`value` is scoped per turn, not per trial or per combat (see `recorder.py`'s
`begin_turn()`/`end_turn()`, backfilled onto just that turn's actions once it resolves) — so a
turn that took damage to set up a good follow-up scores differently from one that didn't, instead
of the whole combat being flattened to one scalar. The turn that actually ends a combat is scored
by `whispering_earring/facility_common.py`'s `_combat_value`, graded on the *resulting* HP fraction of
max HP, not loss relative to that combat's own starting HP: at or above `_HEALTHY_HP_FRACTION`
(0.5) the grade is just that fraction (`1.0` at full health, gently lower as it drops); below it
the same formula continues at a steeper slope (`_LOW_HP_SLOPE`), so death (HP fraction `0`) falls
out of that same curve at exactly `-1.0` rather than being a separately injected penalty on top of
whatever the fall to low HP already cost. A combat a trial never reaches (e.g. it errors out
mid-run) gets no value on any of its actions: an aborted combat says nothing about whether the
actions in it were good.

Non-`combat_play` actions in the same file (`map_select`, `card_reward`, ...) never carry a `policy`
or `value` — only combat is currently policy-driven; meta decisions delegate to Choices Paradox /
Golden Compass placeholders instead (see `whispering_earring/policy.py`).

Full state at a given `combat_play` decision is never stored in the trial file itself — it's
reconstructed by replaying the trial's action log (`character`, `seed`, `actions[:i]`) through a
fresh `sts2-cli` session (see `whispering_earring/checkpoint.py`, `Sts2Session.open_from_checkpoint`).
Materializing tokenized training examples from a trial file is therefore a separate, offline step,
not part of this format.

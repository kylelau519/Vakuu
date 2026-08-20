# Model

AlphaZero(AZ) Monte Carlo Tree Search (MCTS) approach will be used as the backbone of the combat play. However, it is known that AZ is trained on minimax style, an opponent makes a move will diminish your value. Sts2 is expectimax style, we make decision base on the expectation of the uncertainty.

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
- `mcts.py` -- the PUCT search from the MCTS strategy section below, wrapping `ClawsNet` as the
  `(p, v)` evaluator. A chance node is created for `play_card` on a card that draws mid-turn (e.g.
  Pommel Strike), not `end_turn` -- the once-per-turn draw is already covered by simulation
  diversity across the tree, but a same-turn draw effect revisited from the same node would
  otherwise always resolve to the same drawn card every time; see `draws_cards`'s docstring. Two
  things from the doc are still genuinely underspecified (the dynamical formula's constants, what
  "argmax[Q(s,a)]" means for final action selection) -- `mcts.py`'s module docstring documents the
  judgment call made for each rather than silently guessing. Wired into
  `whispering_earring/policy.py` behind `CLAWS_MCTS` -- opt-in, not the default, since it means
  spawning ~200 more sts2-cli subprocesses per combat decision from inside an already-running
  self-play worker, a much bigger cost than the raw-network wiring below.

`whispering_earring/policy.py` samples combat actions from a `ClawsNet` checkpoint's raw softmax
policy (no search) when `CLAWS_CHECKPOINT` is set (and `CLAWS_MCTS` isn't) -- see its module
docstring. `claws.pt` (an untrained init checkpoint, `train.py --epochs 0`) exists to prove this
path end to end before there's anything trained; using it doesn't produce better play than
uniform-random, only differently-random play with a real (if meaningless) probability distribution
recorded as `policy` instead of a one-hot placeholder. `CLAWS_MCTS` uses the same checkpoint (or
the same untrained fallback) as the search's evaluator instead, recording the search's actual
visit-count distribution as `policy` rather than a raw one-shot softmax.

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

## MCTS strategy
- Each decision is a node. Expectimax strategy will be used to handle uncertainty. At a chance node, the only uncertainty is draw order. Child nodes can shuffle the draw order through the `Sts2session` after restoring the state. 
- Not every decision must have child nodes, but when we decide to create an explicit chance node 80 children at most can be made. Dynamical formula `k(N) = ⌈C·N^α⌉` will be used for highly visited node to have even more children.
- Each child node end at the end of combat, win, lose, or terminated. To preserve resource, there's main branch and sub-branches. 
- Among the K attempts, if any won, use that attempt's resulting state (rewards, path choice, deck) as the one the run-level pipeline continues from into the next combat as a "main branch".
- If all K attempts lose, accept the loss and end that lineage there, same as a real run would.
- Node statistics: Visit count N(s,a), total backed-up value W(s,a), Q(s,a)=W/N, and the policy prior P(s,a) from the network. 
- Standard PUCT strategy will be used to do exploration.
- Each combat, run about 200 simulations (not per combat).
- Backup strategy: at every `edge (s, a)` along that path, update:
`N(s,a) ← N(s,a) + 1`
`W(s,a) ← W(s,a) + v`
`Q(s,a) ← W(s,a) / N(s,a)`
- When a simulation's selection phase reaches a node with no children yet, expand it and call the network once — `(p, v) = f_θ(state)` — then immediately back up that v, without continuing the simulation further.
- To protect the resources, if a combat is not ended in 100 decisions, terminate it by evaluating policy and value, and back propagate the value.
- The Live game play will use `armgax[v]` as the decision.

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
default placeholder/softmax policies (`whispering_earring/policy.py`, `CLAWS_CHECKPOINT` unset or
set without `CLAWS_MCTS`) only ever sample one candidate target per card, so trial files from those
paths have at most one entry per `card_index`. The list shape is what lets a real search populate
more than one — trial files recorded with `CLAWS_MCTS` set do exactly that, no format change
needed.

`value` is omitted from an action's entry until the trial's outcome is known, then backfilled onto
every entry in that trial in one pass at save time (flat Monte-Carlo return over the whole
trajectory, no discount — a stand-in for a learned value target until one exists, per
`CompactRecorder`'s `_OUTCOME_VALUE`). An `"error"` trial gets no value on any entry: an aborted
trial says nothing about whether the actions in it were good.

Non-`combat_play` actions in the same file (`map_select`, `card_reward`, ...) never carry a `policy`
or `value` — only combat is currently policy-driven; meta decisions delegate to Choices Paradox /
Golden Compass placeholders instead (see `whispering_earring/policy.py`).

Full state at a given `combat_play` decision is never stored in the trial file itself — it's
reconstructed by replaying the trial's action log (`character`, `seed`, `actions[:i]`) through a
fresh `sts2-cli` session (see `whispering_earring/checkpoint.py`, `Sts2Session.open_from_checkpoint`).
Materializing tokenized training examples from a trial file is therefore a separate, offline step,
not part of this format.

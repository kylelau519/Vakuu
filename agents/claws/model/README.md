# Model

AlphaZero(AZ) Monte Carlo Tree Search (MCTS) approach will be used as the backbone of the combat play. However, it is known that AZ is trained on minimax style, an opponent makes a move will diminish your value. Sts2 is expectimax style, we make decision base on the expectation of the uncertainty.

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

Data are in the format of :
```
{
  "cmd": "action",
  "action": "play_card",
  "args": {"card_index": 1},
  "policy": {"0": 0.05, "1": 0.71, "3": 0.19, "end_turn": 0.05},
  "value": null
}
```
Value stays null until the game ends, then you backfill it across the trajectory (or store it once per run and let the loader broadcast/discount it). 
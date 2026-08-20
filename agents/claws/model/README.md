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
- Not every decision must have child nodes, but when we decide to create an explicit chance node 80 children can be made.
- Each child node end at the end of combat, either lose or win.
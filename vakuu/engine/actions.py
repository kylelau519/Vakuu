"""Translation between the agents' action vocabulary (play_card / use_potion / end_turn ..., indexes into the buffer's slots)
and the host/engine's action strings (play:i[:enemy_uid], potion:i[:enemy_uid], end_turn, pick:i,j)."""
from __future__ import annotations

import random


def alive(obs: dict) -> list[dict]:
    """The alive enemies in the order encode_combat puts them in the buffer's enemy slots."""
    return [e for e in obs["enemies"] if e["hp"] > 0][:5]


def to_action(name: str, args: dict, obs: dict, legal: list[str]) -> str:
    """`legal` is the host's/engine's list for this decision: a card the engine lists only with a target (a single-target card
    with one enemy left) is aimed at that enemy, though the agents' option has no target_index then."""
    if name == "end_turn":
        return "end_turn"
    target = args.get("target_index")
    uid = alive(obs)[target]["uid"] if target is not None else None
    if name == "play_card":
        base = f"play:{obs['hand'][args['card_index']]['uid']}"  # the host's play:<card uid>
    elif name == "use_potion":
        base = f"potion:{args['potion_index']}"
    else:
        raise ValueError(f"no host action for {name} {args}")
    if uid is not None:
        return f"{base}:{uid}"
    if base in legal:
        return base
    aimed = [a for a in legal if a.startswith(base + ":")]
    if aimed:
        return aimed[0]
    return base


def random_pick(prompt_min: int, prompt_max: int, n_options: int, rng: random.Random | None = None) -> str:
    """A card prompt answered at random: between min and max of the options, as 'pick:i,j' (canonical indexes)."""
    r = rng or random
    hi = min(prompt_max, n_options)
    k = r.randint(prompt_min, hi) if hi >= prompt_min else 0
    return "pick:" + ",".join(str(i) for i in sorted(r.sample(range(n_options), k)))

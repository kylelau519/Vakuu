"""Combat is still uniform-random exploration -- the simplest possible action selection, used
for the first self-play pass before any trained network exists. Non-combat (meta) decisions
delegate to the Choices Paradox placeholder policy (agents/choices_paradox/policy.py), which
scores cards off Baalorlord's STS2 tier lists instead of picking blindly -- a random meta
policy produces decks too weak to make combat outcomes meaningful, which defeats the point of
training on them. map_select likewise delegates to Golden Compass (agents/golden_compass/
policy.py), which value-to-go DPs the full map instead of picking a reachable node blindly.

Returns (action_name, kwargs, policy) rather than a built command, so callers pass the first
two straight into session.step(name, **kwargs) -- the session builds/validates the actual
command via actions.action(), no need to duplicate that here. `session` is threaded through
(rather than defaulted away) because map_select needs session.get_map() for full-map
lookahead -- decision.choices alone is only the one-hop list.

policy is None for every decision type except combat_play: it's a placeholder for the
per-option distribution an MCTS search would produce, standing in as a one-hot weight on
whichever option uniform-random happened to pick, so recorder.py's `policy` field has a
consistent shape from the start.

If the CLAWS_CHECKPOINT env var is set, combat instead samples from ClawsNet's own softmax
policy (temperature 1, no search) -- see _choose_combat_action_claws. That's a cheap first
integration step, not the real target: mcts.py's PUCT search is what's supposed to drive combat
eventually, but wiring that in here means spawning ~200 more sts2-cli subprocesses per combat
decision from inside an already-running self-play worker, a much bigger change than swapping in
a raw (even untrained) network. Unset CLAWS_CHECKPOINT (the default) to keep the uniform-random
path -- nothing else about the record shape changes either way.
"""
from __future__ import annotations

import importlib.util
import os
import random
import sys
from pathlib import Path

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the import below)
from env import Sts2Session
from models import CombatPlay, Decision

_CLAWS_CHECKPOINT = os.environ.get("CLAWS_CHECKPOINT")
_claws_net = None  # lazy-loaded once per worker process, on first combat decision -- see below

# Loaded by explicit file path, not sys.path + import, because this file is *also* named
# policy.py -- an sys.path-based `import policy` would risk resolving to itself instead of
# agents/choices_paradox/policy.py or agents/golden_compass/policy.py depending on path order.
_choices_paradox_path = Path(__file__).resolve().parents[1] / "choices_paradox" / "policy.py"
_spec = importlib.util.spec_from_file_location("choices_paradox_policy", _choices_paradox_path)
assert _spec is not None and _spec.loader is not None
choices_paradox = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(choices_paradox)

_golden_compass_path = Path(__file__).resolve().parents[1] / "golden_compass" / "policy.py"
_spec = importlib.util.spec_from_file_location("golden_compass_policy", _golden_compass_path)
assert _spec is not None and _spec.loader is not None
golden_compass = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden_compass)

_REST_HEAL_HP_THRESHOLD = 35

# Set by _choose_shop_action/_choose_rest_site_action right before returning an action that
# triggers a follow-up card_select, and consumed by the card_select branch below. The protocol
# is strictly request/response -- the game emits that follow-up as its immediate next decision,
# with no way to distinguish it from any other card_select by shape alone (see CardSelect's
# docstring in sts2-cli's models.py) -- so tracking "what did we just do" here is the simplest
# reliable way to tell a remove-card select apart from an upgrade select.
_pending_card_select_purpose: str | None = None

# session is passed around incase we need extra info from the session and we can get through env.py
def choose_action(decision: Decision, session: Sts2Session) -> tuple[str, dict, dict | None]:
    if isinstance(decision, CombatPlay):
        return _choose_combat_action(decision)
    name, kwargs = _choose_generic_action(decision, session)
    return name, kwargs, None


def _load_claws_net():
    """None if CLAWS_CHECKPOINT isn't set -- torch and claws/model are only imported (and only
    need to be installed/present) when a checkpoint is actually configured, so the plain
    uniform-random path stays lightweight."""
    global _claws_net
    if _claws_net is not None or not _CLAWS_CHECKPOINT:
        return _claws_net

    claws_model_dir = Path(__file__).resolve().parents[1] / "claws" / "model"
    if str(claws_model_dir) not in sys.path:
        sys.path.insert(0, str(claws_model_dir))
    from model import ClawsNet  # noqa: E402  (path just inserted above)

    _claws_net = ClawsNet.from_checkpoint(_CLAWS_CHECKPOINT)
    return _claws_net


def _choose_combat_action_claws(decision: CombatPlay, net) -> tuple[str, dict, dict]:
    """Samples (not argmax) from the network's own softmax over model.py's legal_options() --
    every (card, target) pair, not just one random target per card like the fallback below, so
    the recorded `policy` list is already the richer shape materialize.py/train.py expect.
    Sampling, not greedy, because this path still feeds self-play data generation: the point is
    exploration, same as the uniform-random fallback, just weighted by whatever the network has
    learned instead of picked blindly."""
    import torch

    options, logits, _value = net.forward(decision)
    probs = torch.softmax(logits, dim=-1)
    idx = int(torch.multinomial(probs, 1).item())
    policy = [
        {"action": a, "args": args, "prob": p} for (a, args), p in zip(options, probs.tolist())
    ]
    name, args = options[idx]
    return name, args, policy


def _choose_combat_action(decision: CombatPlay) -> tuple[str, dict, dict]:
    net = _load_claws_net()
    if net is not None:
        return _choose_combat_action_claws(decision, net)

    options: list[tuple[str, dict]] = [("end_turn", {})]

    for card in decision.hand:
        if not card.can_play or card.cost > decision.energy:
            continue
        args = {"card_index": card.index}
        if card.target_type == "AnyEnemy" and len(decision.enemies) >= 2:
            args["target_index"] = random.choice(decision.enemies).index
        options.append(("play_card", args))

    for potion in decision.player.potions:
        args = {"potion_index": potion.index}
        if potion.target_type == "AnyEnemy" and len(decision.enemies) >= 2:
            args["target_index"] = random.choice(decision.enemies).index
        options.append(("use_potion", args))

    chosen = random.choice(options)
    # Placeholder: a one-hot weight on the option uniform-random actually picked, standing
    # in for an MCTS visit distribution over the same option list. A list of records rather
    # than a dict keyed by action, since kwargs (e.g. target_index) can differ between two
    # options that share an action name.
    policy = [
        {"action": name, "args": args, "prob": 1.0 if (name, args) == chosen else 0.0}
        for name, args in options
    ]
    name, args = chosen
    return name, args, policy


def _choose_generic_action(decision: Decision, session: Sts2Session) -> tuple[str, dict]:
    """Non-combat decisions: pick a uniformly random legal option, or the simplest safe
    default (leave/skip) where "random" isn't a meaningful choice yet. Good enough to
    keep a run moving forward -- combat_play is what's actually being trained right now."""
    dec = decision.decision

    if dec == "map_select":
        return _choose_map_action(decision, session)

    if dec == "card_reward":
        index = choices_paradox.choose_card_reward(
            decision.cards, decision.can_skip, decision.player.deck_size or 0
        )
        if index is None:
            return "skip_card_reward", {}
        return "select_card_reward", {"card_index": index}

    if dec == "bundle_select":
        return "select_bundle", {"bundle_index": random.choice(decision.bundles).index}

    if dec == "card_select":
        return _choose_card_select_action(decision)

    if dec == "shop":
        return _choose_shop_action(decision)

    if dec == "rest_site":
        return _choose_rest_site_action(decision)

    if dec == "event_choice":
        unlocked = [o for o in decision.options if not o.is_locked]
        if not unlocked:
            return "leave_room", {}
        return "choose_option", {"option_index": random.choice(unlocked).index}

    raise ValueError(f"no policy for decision type {dec!r}")


def _choose_map_action(decision: Decision, session: Sts2Session) -> tuple[str, dict]:
    """Delegates to Golden Compass's value-to-go DP over the full act map -- decision.choices
    alone is only the one-hop list, not enough for the DP, hence session.get_map() (cached on
    the session; see env.py) rather than reading straight off `decision`. Falls back to a
    random legal choice if there's no full-map decision to make yet (pre-act, single forced
    choice) or the DP's pick isn't among what's actually offered (map desynced from the cache)."""
    hp_frac = decision.player.hp / decision.player.max_hp if decision.player.max_hp else 1.0
    target = golden_compass.choose_next_node(session.get_map(), hp_frac=hp_frac)
    if target is not None:
        for choice in decision.choices:
            if (choice.col, choice.row) == target:
                return "select_map_node", {"col": choice.col, "row": choice.row}

    pick = random.choice(decision.choices)
    return "select_map_node", {"col": pick.col, "row": pick.row}


def _choose_card_select_action(decision: Decision) -> tuple[str, dict]:
    """Dispatches on _pending_card_select_purpose, set by whichever shop/rest_site action
    just triggered this follow-up. Anything else (True Grit's exhaust-select, Wish's
    deck-tutor-select, ...) falls back to a uniformly random legal pick -- CardSelect's shape
    alone can't tell those apart from a remove/upgrade select, and no purpose was staged for
    them."""
    global _pending_card_select_purpose
    purpose, _pending_card_select_purpose = _pending_card_select_purpose, None

    if purpose == "remove" and decision.cards:
        index = choices_paradox.choose_removal_target(decision.cards)
        return "select_cards", {"indices": [index]}

    if purpose == "upgrade" and decision.cards:
        index = choices_paradox.choose_upgrade_target(decision.cards)
        return "select_cards", {"indices": [index]}

    lo, hi = decision.min_select, min(decision.max_select, len(decision.cards))
    n_pick = random.randint(lo, hi) if hi >= lo else 0
    if n_pick == 0:
        return "skip_select", {}
    picks = random.sample(decision.cards, n_pick)
    return "select_cards", {"indices": [c.index for c in picks]}


def _choose_shop_action(decision: Decision) -> tuple[str, dict]:
    """Priority: remove a card if the removal service is affordable (targeting the deck's
    worst-tier card via the card_select that follows), then top up potions, then buy the
    highest-tier card still within budget. Leaves once nothing is affordable."""
    global _pending_card_select_purpose
    gold = decision.player.gold

    if decision.card_removal_cost is not None and decision.card_removal_cost <= gold:
        _pending_card_select_purpose = "remove"
        return "remove_card", {}

    if len(decision.player.potions) < decision.player.max_potions:
        affordable_potions = [p for p in decision.potions if p.is_stocked and p.cost <= gold]
        if affordable_potions:
            return "buy_potion", {"potion_index": random.choice(affordable_potions).index}

    affordable_cards = [c for c in decision.cards if c.is_stocked and c.cost <= gold]
    if affordable_cards:
        index = choices_paradox.choose_shop_card(affordable_cards, decision.player.deck_size or 0)
        if index is not None:
            return "buy_card", {"card_index": index}

    return "leave_room", {}


def _choose_rest_site_action(decision: Decision) -> tuple[str, dict]:
    """Heal below the HP threshold, otherwise upgrade a card -- targeting whichever offered
    card the tier list flags good_upgrade via the card_select that follows."""
    global _pending_card_select_purpose
    enabled = {o.option_id: o for o in decision.options if o.is_enabled}
    if not enabled:
        return "leave_room", {}

    preferred = "HEAL" if decision.player.hp < _REST_HEAL_HP_THRESHOLD else "SMITH"
    option = enabled[preferred] if preferred in enabled else random.choice(list(enabled.values()))
    if option.option_id == "SMITH":
        _pending_card_select_purpose = "upgrade"
    return "choose_option", {"option_index": option.index}

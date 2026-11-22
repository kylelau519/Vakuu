"""The run-level decisions an agent makes on the buffer record (decision.arr): the hazard net when GOLDEN_COMPASS_CHECKPOINT names one
(map moves, card rewards, shops, rests, upgrade / remove prompts; see agent.py), the tier list and uniform-random picks otherwise.
Returns the agents' action vocabulary (select_map_node, select_card_reward, buy_card, ...), which buffer/run_encode.py turns into a
host action string.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from vakuu.buffer import layout as bl
from vakuu.buffer.decision import BufferDecision, decode_token
from vakuu.golden_compass import agent
from vakuu.macro import tier_policy


@dataclass
class _BufferCard:
    index: int
    id: str | None


def choose_action(decision: BufferDecision, session, purpose: str | None = None) -> tuple[str, dict, dict | None, str | None]:
    """-> (action name, kwargs, the net's policy over the options or None, the purpose of the follow-up card prompt or None).
    `session` supplies `.character` and `.ascension` (the buffer does not carry the character); `purpose` is what the previous
    shop / rest action wants its card prompt to do ("remove", "upgrade")."""
    name, kwargs, next_purpose, policy = _choose(decision, purpose, session)
    return name, kwargs, policy, next_purpose


def _tier_card_reward(cards: list, can_skip: bool, deck_size: int) -> tuple[str, dict]:
    """The tier-list card pick -- the fallback while no hazard net is loaded."""
    index = tier_policy.choose_card_reward(cards, can_skip, deck_size)
    if index is None:
        return "skip_card_reward", {}
    return "select_card_reward", {"card_index": index}


def _choose(decision: BufferDecision, purpose: str | None, session) -> tuple[str, dict, str | None, dict | None]:
    dec = decision.decision
    arr = decision.arr
    character, ascension = getattr(session, "character", None), getattr(session, "ascension", 0)

    if dec == "map_select":
        name, kwargs, policy = agent.choose_map(decision, character, ascension)
        return name, kwargs, None, policy
    if dec in ("card_reward", "event_card_reward"):
        name, kwargs, policy = _choose_card_reward(decision, character, ascension)
        return name, kwargs, None, policy
    if dec == "bundle_select":
        return "select_bundle", {"bundle_index": random.choice([b for b, _ in bl.bundles(arr)])}, None, None
    if dec == "card_select":
        picked = agent.choose_card_select(decision, purpose, character, ascension)
        if picked is not None:
            name, kwargs, policy = picked
            return name, kwargs, None, policy
        name, kwargs = _choose_card_select(decision)
        return name, kwargs, None, None
    if dec == "shop":
        picked = agent.choose_shop(decision, character, ascension)
        if picked is not None:
            return picked
        name, kwargs, next_purpose = _choose_shop(arr)
        return name, kwargs, next_purpose, None
    if dec == "rest_site":
        picked = agent.choose_rest(decision, character, ascension)
        if picked is not None:
            return picked
        name, kwargs, next_purpose = _choose_rest_site(arr)
        return name, kwargs, next_purpose, None
    if dec == "event_choice":
        unlocked = [i for i, is_locked in bl.event_options(arr) if not is_locked]
        if not unlocked:
            return "leave_room", {}, None, None
        return "choose_option", {"option_index": random.choice(unlocked)}, None, None

    raise ValueError(f"no run-level policy for decision type {dec!r}")


def _choose_card_reward(decision: BufferDecision, character, ascension) -> tuple[str, dict, dict | None]:
    """The hazard net's pick (afterstate = deck with / without each card) when a checkpoint is loaded, else the tier list."""
    arr = decision.arr
    if bl.decision_name(arr) == "event_card_reward":  # the blessing-triggered reward, always skippable
        slots, can_skip = list(bl.event_card_reward_cards(arr)), True
    else:  # the ordinary post-combat reward
        slots, can_skip = list(bl.reward_cards(arr)), bl.can_skip_reward(arr)
    offered = [(i, decode_token(bl.card_token(slot)), int(slot[bl.CARD_UPGRADED])) for i, slot in slots]

    picked = agent.choose_card_reward(decision, offered, can_skip, character, ascension)
    if picked is not None:
        return picked
    cards = [_BufferCard(i, cid) for i, cid, _ in offered]
    name, kwargs = _tier_card_reward(cards, can_skip, bl.player_deck_size(arr))
    return name, kwargs, None


def _choose_shop(arr) -> tuple[str, dict, str | None]:
    gold = bl.player_gold(arr)

    removal_cost = bl.shop_removal_cost(arr)
    if removal_cost != -1 and removal_cost <= gold:
        return "remove_card", {}, "remove"

    if len(list(bl.player_potions(arr))) < bl.player_max_potions(arr):
        affordable_potions = [i for i, slot in bl.shop_potions(arr) if bool(slot[2]) and slot[1] <= gold]
        if affordable_potions:
            return "buy_potion", {"potion_index": random.choice(affordable_potions)}, None

    cards = [
        _BufferCard(i, decode_token(bl.card_token(slot)))
        for i, slot in bl.shop_cards(arr)
        if bool(slot[bl.CARD_SIZE + 1]) and slot[bl.CARD_SIZE] <= gold  # is_stocked, cost
    ]
    if cards:
        index = tier_policy.choose_shop_card(cards, bl.player_deck_size(arr))
        if index is not None:
            return "buy_card", {"card_index": index}, None

    return "leave_room", {}, None


def _choose_rest_site(arr) -> tuple[str, dict, str | None]:
    enabled = [i for i, slot in bl.rest_options(arr) if int(slot[1]) == 1]
    if not enabled:
        return "leave_room", {}, None
    return "choose_option", {"option_index": random.choice(enabled)}, None


def _choose_card_select(decision: BufferDecision) -> tuple[str, dict]:
    """A card prompt the hazard net did not answer: a uniformly random legal pick of the allowed size."""
    arr = decision.arr
    indices = [i for i, _ in bl.select_cards(arr)]
    lo, hi = bl.min_select(arr), min(bl.max_select(arr), len(indices))
    n_pick = random.randint(lo, hi) if hi >= lo else 0
    if n_pick == 0:
        return "skip_select", {}
    return "select_cards", {"indices": random.sample(indices, n_pick)}

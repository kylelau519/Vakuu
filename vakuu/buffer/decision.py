"""One decoded-on-demand decision: `BufferDecision` wraps the float32 record (buffer/layout.py) a decision was encoded into, and
`decode_token` turns a token number back into the game's entry id.

A token is a position in engine/tables.py's TOKEN_MAP (built from the game's own id tables), so a net is tied to the tables it was
trained on. decode_buffer / BufferDecision.to_dict are the human-readable decode, for debugging and logging only: never call them from
a hot path (that object-graph cost is what the buffer protocol exists to avoid).
"""
from __future__ import annotations

import numpy as np

from vakuu.buffer import layout as bl
from vakuu.engine.tables import TOKEN_MAP


def decode_token(token: int) -> str | None:
    """token -> the raw game entry id (e.g. 5 -> "STRIKE"), or None for UnknownToken (0) / an empty-slot sentinel (-1) / an
    unrecognized token. TOKEN_MAP holds "CATEGORY:ID" strings."""
    if token <= 0:
        return None
    key = TOKEN_MAP.get(token)
    if key is None or key == "UNKNOWN":
        return None
    _, _, entry = key.partition(":")
    return entry


def _decode_card(slot: np.ndarray) -> dict | None:
    """slot is a raw CARD_SIZE block, or the first CARD_SIZE floats of a longer row (a hand/pile/
    shop/etc. slot) """
    if bl.card_token(slot) == 0:
        return None
    enchant_tok = int(slot[bl.CARD_ENCHANT_TOKEN])
    afflict_tok = int(slot[bl.CARD_AFFLICT_TOKEN])
    mask = int(slot[bl.CARD_KEYWORD_MASK])
    return {
        "id": decode_token(bl.card_token(slot)),
        "cost": float(slot[bl.CARD_COST]),
        "type": int(slot[bl.CARD_TYPE]),
        "rarity": int(slot[bl.CARD_RARITY]),
        "upgraded": bool(slot[bl.CARD_UPGRADED]),
        "upgradable": bool(slot[bl.CARD_UPGRADABLE]),
        "enchantment": decode_token(enchant_tok),
        "enchantment_amount": float(slot[bl.CARD_ENCHANT_AMOUNT]) if enchant_tok != -1 else None,
        "affliction": decode_token(afflict_tok),
        "affliction_amount": float(slot[bl.CARD_AFFLICT_AMOUNT]) if afflict_tok != -1 else None,
        "keywords": [name for name, bit in bl.CARD_KEYWORD_BITS.items() if mask & (1 << bit)],
    }


def _decode_hand_slot(slot: np.ndarray) -> dict | None:
    card = _decode_card(slot[:bl.CARD_SIZE])
    if card is None:
        return None
    card["can_play"] = bl.hand_can_play(slot)
    card["star_cost"] = bl.hand_star_cost(slot)
    card["target_type"] = bl.hand_target_type(slot)
    card["stats"] = {k: bl.hand_stat(slot, k) for k in bl.HAND_STAT_KEYS if bl.hand_stat(slot, k) != -1}
    card["damage_by_target"] = {}
    for i in range(bl.MAX_ENEMIES):
        dmg, repeat = bl.damage_by_target(slot, i)
        if dmg != -1:
            card["damage_by_target"][i] = {"damage": dmg, "repeat": repeat}
    return card


def _decode_enemy_slot(slot: np.ndarray) -> dict:
    intents = [
        {"type": itype, "damage": dmg}
        for i in range(bl.MAX_INTENTS_PER_ENEMY)
        for itype, dmg in [bl.enemy_intent(slot, i)] if itype != -1
    ]
    powers = [
        {"id": decode_token(int(pw[0])), "amount": float(pw[1])} for _, pw in bl.enemy_powers(slot)
    ]
    return {
        "monster": decode_token(bl.enemy_token(slot)),
        "hp": bl.enemy_hp(slot),
        "max_hp": bl.enemy_max_hp(slot),
        "block": bl.enemy_block(slot),
        "intends_attack": bl.enemy_intends_attack(slot),
        "intents": intents,
        "powers": powers,
    }


def _decode_player_summary(arr: np.ndarray) -> dict:
    return {
        "hp": bl.player_hp(arr),
        "max_hp": bl.player_max_hp(arr),
        "block": bl.player_block(arr),
        "gold": bl.player_gold(arr),
        "max_potions": bl.player_max_potions(arr),
        "deck_size": bl.player_deck_size(arr),
        "relics": [decode_token(int(slot[bl.RELIC_TOKEN])) for _, slot in bl.player_relics(arr)],
        "potions": [decode_token(int(slot[bl.POTION_TOKEN])) for _, slot in bl.player_potions(arr)],
        "deck": [_decode_card(slot) for _, slot in bl.deck_cards(arr)],
    }


def _decode_run_context(arr: np.ndarray) -> dict:
    raw = bl.run_context(arr)
    return {
        "act_index": raw["act_index"],
        "act_id": decode_token(raw["act_id_token"]),
        "floor": raw["floor"],
        "boss1": decode_token(raw["boss1_token"]),
        "boss2": decode_token(raw["boss2_token"]),
    }


def _decode_combat_play(arr: np.ndarray) -> dict:
    stars = bl.regent_stars(arr)
    osty = bl.osty(arr)
    alive_enemy_indices = [i for i, _ in bl.enemies(arr)]
    return {
        "round": bl.round_number(arr),
        "energy": bl.energy(arr),
        "max_energy": bl.max_energy(arr),
        "hand": [_decode_hand_slot(slot) for _, slot in bl.hand_cards(arr)],
        "enemies": {i: _decode_enemy_slot(slot) for i, slot in bl.enemies(arr)},
        "player_powers": [
            {"id": decode_token(int(slot[0])), "amount": float(slot[1])}
            for _, slot in bl.player_powers(arr)
        ],
        "draw_pile": {
            "count": bl.draw_pile_count(arr),
            "cards": [_decode_card(slot) for _, slot in bl.draw_pile_cards(arr)],
        },
        "discard_pile": {
            "count": bl.discard_pile_count(arr),
            "cards": [_decode_card(slot) for _, slot in bl.discard_pile_cards(arr)],
        },
        "exhaust_pile": {
            "count": bl.exhaust_pile_count(arr),
            "cards": [_decode_card(slot) for _, slot in bl.exhaust_pile_cards(arr)],
        },
        "orbs": [
            {"id": decode_token(int(slot[0])), "passive": float(slot[1]), "evoke": float(slot[2])}
            for _, slot in bl.orbs(arr)
        ],
        "orb_capacity": bl.orb_capacity(arr),
        "regent_stars": stars if stars != -1 else None,
        "osty": osty if osty["hp"] != -1 else None,
        "player_history": bl.player_history(arr),
        "card_generated_by_creator": bl.card_generated_by_creator(arr),
        "enemy_history": {i: bl.enemy_history(arr, i) for i in alive_enemy_indices},
    }


def _decode_map(arr: np.ndarray) -> dict:
    rows: dict[int, list[dict]] = {}
    boss = None
    for row, col, slot in bl.map_points(arr):
        point = {
            "type": bl.map_point_type(slot),
            "visited": bool(slot[1]),
            "current": bool(slot[2]),
            "children": list(bl.map_point_children(slot)),
        }
        if row == -1:
            boss = point
        else:
            point["col"] = col
            rows.setdefault(row, []).append(point)
    return {"rows": rows, "boss": boss}


def _decode_card_select(arr: np.ndarray) -> dict:
    return {
        "min_select": bl.min_select(arr),
        "max_select": bl.max_select(arr),
        "options": [_decode_card(slot) for _, slot in bl.select_cards(arr)],
    }


def _decode_shop(arr: np.ndarray) -> dict:
    cards = []
    for _, slot in bl.shop_cards(arr):
        card = _decode_card(slot[:bl.CARD_SIZE])
        assert card is not None  # shop_cards() already filters to occupied slots
        card["price"] = float(slot[bl.CARD_SIZE])
        card["is_stocked"] = bool(slot[bl.CARD_SIZE + 1])
        card["is_on_sale"] = bool(slot[bl.CARD_SIZE + 2])
        cards.append(card)
    return {
        "cards": cards,
        "relics": [
            {"id": decode_token(int(slot[0])), "price": float(slot[1]), "is_stocked": bool(slot[2])}
            for _, slot in bl.shop_relics(arr)
        ],
        "potions": [
            {"id": decode_token(int(slot[0])), "price": float(slot[1]), "is_stocked": bool(slot[2])}
            for _, slot in bl.shop_potions(arr)
        ],
        "removal_cost": bl.shop_removal_cost(arr),
    }


def _decode_rest_site(arr: np.ndarray) -> dict:
    return {
        "options": [
            {"index": i, "is_enabled": bool(slot[1])} for i, slot in bl.rest_options(arr)
        ]
    }


def _decode_card_reward(arr: np.ndarray) -> dict:
    return {
        "can_skip": bl.can_skip_reward(arr),
        "gold_delta": bl.reward_gold_delta(arr),
        "cards": [_decode_card(slot) for _, slot in bl.reward_cards(arr)],
    }


def _decode_event_choice(arr: np.ndarray) -> dict:
    return {
        "event": decode_token(bl.event_token(arr)),
        "options": [{"index": i, "is_locked": locked} for i, locked in bl.event_options(arr)],
    }


def _decode_bundle_select(arr: np.ndarray) -> dict:
    return {b: [_decode_card(slot) for _, slot in cards] for b, cards in bl.bundles(arr)}


_SECTION_DECODERS = {
    "combat_play": ("combat_play", _decode_combat_play),
    "map_select": ("map", _decode_map),
    "card_select": ("card_select", _decode_card_select),
    "shop": ("shop", _decode_shop),
    "rest_site": ("rest_site", _decode_rest_site),
    "card_reward": ("card_reward", _decode_card_reward),
    "event_choice": ("event_choice", _decode_event_choice),
    "event_card_reward": ("event_card_reward", lambda arr: {
        "cards": [_decode_card(slot) for _, slot in bl.event_card_reward_cards(arr)]
    }),
    "bundle_select": ("bundle_select", _decode_bundle_select),
    "game_over": ("game_over", lambda arr: {"victory": bl.victory(arr)}),
}


def decode_buffer(arr: np.ndarray) -> dict:
    """Full human-readable decode of one raw buffer record: every tokenized id resolved to its
    game entry string (via decode_token). Debugging/logging only """
    decision = bl.decision_name(arr)
    out: dict = {
        "decision": decision,
        "player": _decode_player_summary(arr),
        "run_context": _decode_run_context(arr),
    }
    entry = _SECTION_DECODERS.get(decision)
    if entry is not None:
        key, decoder = entry
        out[key] = decoder(arr)
    return out


class BufferDecision:
    """arr: a numpy float32 array this instance owns exclusively (the encoders write a fresh record per decision and never
    alias a reused buffer)."""

    __slots__ = ("arr", "decision")

    def __init__(self, arr: np.ndarray):
        self.arr = arr
        self.decision = bl.decision_name(arr)

    @property
    def player_hp(self) -> int:
        return bl.player_hp(self.arr)

    @property
    def player_max_hp(self) -> int:
        return bl.player_max_hp(self.arr)

    @property
    def victory(self) -> bool:
        return bl.victory(self.arr) if self.decision == "game_over" else False
    
    @property
    def floor(self) -> int:
        return bl.run_context(self.arr)["floor"]
    
    @property
    def act(self) -> int:
        return bl.run_context(self.arr)["act_index"]

    def is_last_of_act(self) -> bool:
        """Check if current location is at the last map node (highest row before boss at -1)."""
        current_row = None
        max_row = -1

        for row, _, slot in bl.map_points(self.arr):
            if row == -1:  # boss point
                continue
            point_type = int(slot[0])
            if point_type != -1:  # valid map point
                if int(slot[2]):  # current flag
                    current_row = row
                max_row = max(max_row, row)

        return current_row is not None and current_row == max_row

    def get_enemies_alive_mask(self) -> np.ndarray:
        """5-element bool array: True if enemy at index i is alive, False if dead."""
        return np.array([bl.enemy_token(bl.enemy_slot(self.arr, i)) != -1 for i in range(bl.MAX_ENEMIES)])

    def get_enemies_hp(self) -> np.ndarray:
        """Current HP for all 5 enemies (use alive_mask to filter)."""
        return np.array([bl.enemy_hp(bl.enemy_slot(self.arr, i)) for i in range(bl.MAX_ENEMIES)])

    def get_enemies_max_hp(self) -> np.ndarray:
        """Max HP for all 5 enemies (use alive_mask to filter)."""
        return np.array([bl.enemy_max_hp(bl.enemy_slot(self.arr, i)) for i in range(bl.MAX_ENEMIES)])

    def get_enemies_block(self) -> np.ndarray:
        """Current block for all 5 enemies (use alive_mask to filter)."""
        return np.array([bl.enemy_block(bl.enemy_slot(self.arr, i)) for i in range(bl.MAX_ENEMIES)])

    def get_enemies_token(self) -> np.ndarray:
        """Token (ID) for all 5 enemies (use alive_mask to filter)."""
        return np.array([bl.enemy_token(bl.enemy_slot(self.arr, i)) for i in range(bl.MAX_ENEMIES)])

    def get_enemies_intends_attack(self) -> np.ndarray:
        """Attack intent for all 5 enemies (use alive_mask to filter)."""
        return np.array([bl.enemy_intends_attack(bl.enemy_slot(self.arr, i)) for i in range(bl.MAX_ENEMIES)])
    
    def get_enemies_ids(self) -> np.ndarray[str]:
        enemies_token = np.array([bl.enemy_token(bl.enemy_slot(self.arr, i)) for i in range(bl.MAX_ENEMIES)])
        return np.array([decode_token(tok) for tok in enemies_token])

    def to_dict(self) -> dict:
        """Full human-readable decode -- see decode_buffer()'s docstring. For debugging/logging
        (printing a step, dumping to a log file, diffing two decisions) -- never call this from a
        training loop's hot path; that's exactly the per-step object-graph cost the buffer
        protocol exists to avoid."""
        return decode_buffer(self.arr)

    def __repr__(self) -> str:
        return f"BufferDecision({self.decision!r})"

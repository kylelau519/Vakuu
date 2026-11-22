"""The fixed-width float32 record every decision is encoded into (`decision.arr`): offsets, sizes and accessors.

Originally written as the Python mirror of an earlier headless driver's C# buffer writer; the engine now is spire-zero, whose
observations buffer/encode.py and buffer/run_encode.py write into this layout, so the models read one record format for every
decision. Every accessor is a pure function of (arr, ...) -> a numpy scalar or slice: there is deliberately no per-decision object
graph (no Card/Enemy/... class) between the raw array and whatever reads it (claws/model.py's tokenize_from_buffer, the compass's
run_state) -- building one was measured as the actual bottleneck of the JSON + pydantic path this replaced.

Changing a constant here changes what every trained net reads: treat it as a breaking change that goes with a retrain.
"""
from __future__ import annotations

import numpy as np

# --- decision type tag (buffer[0]) --------------------------------------------------------------

DECISION_NAMES = {
    0: "error", 1: "game_over", 2: "map_select", 3: "combat_play", 4: "card_reward",
    5: "event_choice", 6: "rest_site", 7: "shop", 8: "bundle_select", 9: "event_card_reward",
    10: "card_select", 11: "unknown",
}

# --- Card(11) block -- RunSimulator.cs's private static Card(CardModel?, float[], ref int) -------

CARD_SIZE = 11
CARD_TOKEN, CARD_COST, CARD_TYPE, CARD_RARITY, CARD_UPGRADED, CARD_UPGRADABLE, \
    CARD_ENCHANT_TOKEN, CARD_ENCHANT_AMOUNT, CARD_AFFLICT_TOKEN, CARD_AFFLICT_AMOUNT, \
    CARD_KEYWORD_MASK = range(CARD_SIZE)

CARD_KEYWORD_BITS = {
    "exhaust": 0, "ethereal": 1, "innate": 2, "unplayable": 3, "retain": 4, "sly": 5, "eternal": 6,
}

RELIC_SIZE = 2
RELIC_TOKEN, RELIC_COUNTER = range(RELIC_SIZE)

POTION_SIZE = 2
POTION_TOKEN, POTION_TARGET_TYPE = range(POTION_SIZE)
POTION_TARGET_TYPES = {0: "Self", 1: "AnyAlly", 2: "AnyEnemy", -1: None}

MAX_RELICS = 40
MAX_POTIONS = 10
MAX_DECK_CARDS = 50  # deck cap (PlayerSummary) and the draw/discard/exhaust pile cap and
                      # card_select's option cap -- see each section below
MAX_HAND_CARDS = 10  # the real hand-size cap (Sts2 never deals more than 10 cards into a hand) --
                      # used only for CombatPlay's hand slots, distinct from MAX_DECK_CARDS above

# --- PlayerSummary -- always valid, refreshed on every single decision ---------------------------

PLAYER_CHAR_TOKEN = 0
PLAYER_HP = 1
PLAYER_MAX_HP = 2
PLAYER_BLOCK = 3
PLAYER_GOLD = 4
PLAYER_MAX_POTIONS = 5
PLAYER_DECK_SIZE = 6
_PLAYER_HEADER_SIZE = 7
_PLAYER_RELICS_OFFSET = _PLAYER_HEADER_SIZE
_PLAYER_POTIONS_OFFSET = _PLAYER_RELICS_OFFSET + MAX_RELICS * RELIC_SIZE
_PLAYER_DECK_OFFSET = _PLAYER_POTIONS_OFFSET + MAX_POTIONS * POTION_SIZE
PLAYER_SUMMARY_SIZE = _PLAYER_DECK_OFFSET + MAX_DECK_CARDS * CARD_SIZE
assert PLAYER_SUMMARY_SIZE == 657

MAX_MAP_ROWS = 20
MAX_MAP_COLS = 7
MAX_MAP_CHILDREN = 3
MAP_POINT_SIZE = 3 + MAX_MAP_CHILDREN * 2  # type, visited, current, then (col,row) per child
MAP_SIZE = MAX_MAP_ROWS * MAX_MAP_COLS * MAP_POINT_SIZE + MAP_POINT_SIZE  # + 1 trailing boss point
assert MAP_SIZE == 1269

MAP_POINT_TYPES = {
    -1: None, 0: "Unassigned", 1: "Unknown", 2: "Shop", 3: "Treasure", 4: "RestSite",
    5: "Monster", 6: "Elite", 7: "Boss", 8: "Ancient",
}

RUN_CONTEXT_SIZE = 5
# Fixed 2026-08-27: this used to read RUN_CONTEXT_ACT, RUN_CONTEXT_FLOOR, RUN_CONTEXT_ROOM_TYPE,
# RUN_CONTEXT_BOSS1, RUN_CONTEXT_BOSS2 = range(5) -- mislabeled against RunSimulator.cs's actual
# RunContext() write order (act_index, act_id, floor, boss1, boss2; there is no "room_type" field
# in this section at all). Never read via these names before the fix (grep confirmed), so this was
# a landmine, not a live bug.
RUN_CONTEXT_ACT_INDEX, RUN_CONTEXT_ACT_ID, RUN_CONTEXT_FLOOR, RUN_CONTEXT_BOSS1, RUN_CONTEXT_BOSS2 = range(5)

# --- CombatPlay ------------------------------------------------------------------------------------

MAX_ENEMIES = 5
MAX_ENEMY_POWERS = 8
MAX_PLAYER_POWERS = 16
MAX_INTENTS_PER_ENEMY = 2
MAX_ORBS = 10

HAND_STAT_KEYS = (
    "damage", "block", "magicnumber", "secondarydamage", "secondarymagicnumber",
    "calculateddamage", "repeat", "heal", "cards",
)
# Per hand slot: Card(11) + can_play + star_cost + target_type + len(HAND_STAT_KEYS) resolved
# stats + MAX_ENEMIES*(damage, repeat) per-target damage rows.
_HAND_CAN_PLAY = CARD_SIZE
_HAND_STAR_COST = CARD_SIZE + 1
_HAND_TARGET_TYPE = CARD_SIZE + 2
_HAND_STATS_OFFSET = CARD_SIZE + 3
_HAND_DAMAGE_BY_TARGET_OFFSET = _HAND_STATS_OFFSET + len(HAND_STAT_KEYS)
HAND_SLOT_SIZE = _HAND_DAMAGE_BY_TARGET_OFFSET + MAX_ENEMIES * 2
assert HAND_SLOT_SIZE == 33

# TargetType enum ordinals (MegaCrit.Sts2.Core.Entities.Cards.TargetType) -- the buffer writes the
# raw C# enum int (see RunSimulator.States.Buffer.cs's hand-card loop), not a token, since it's a
# fixed compiled enum, not a game data table Tokenizer knows about.
TARGET_TYPE_NAMES = {
    0: "None", 1: "Self", 2: "AnyEnemy", 3: "AllEnemies", 4: "RandomEnemy", 5: "AnyPlayer",
    6: "AnyAlly", 7: "AllAllies", 8: "TargetedNoCreature", 9: "Osty", -1: None,
}

# Per enemy slot: token, hp, max_hp, block, intends_attack, MAX_INTENTS_PER_ENEMY*(type, damage),
# MAX_ENEMY_POWERS*(token, amount).
_ENEMY_HEADER_SIZE = 5
_ENEMY_TOKEN, _ENEMY_HP, _ENEMY_MAX_HP, _ENEMY_BLOCK, _ENEMY_INTENDS_ATTACK = range(_ENEMY_HEADER_SIZE)
_ENEMY_INTENTS_OFFSET = _ENEMY_HEADER_SIZE
_ENEMY_POWERS_OFFSET = _ENEMY_INTENTS_OFFSET + MAX_INTENTS_PER_ENEMY * 2
ENEMY_SLOT_SIZE = _ENEMY_POWERS_OFFSET + MAX_ENEMY_POWERS * 2
assert ENEMY_SLOT_SIZE == 25

HISTORY_COUNTER_NAMES = (
    "block_gained", "card_afflicted", "card_discarded", "card_drawn", "card_exhausted",
    "card_generated", "card_played", "creature_attacked", "damage_received", "damage_dealt",
    "energy_spent", "monster_move_performed", "orb_channeled", "potion_used", "power_received",
    "power_applied", "stars_modified", "summoned",
)
HISTORY_CREATURE_SIZE = len(HISTORY_COUNTER_NAMES) * 2  # (this_turn, this_combat) per counter
assert HISTORY_CREATURE_SIZE == 36

_CP_HEADER_SIZE = 3  # round, energy, max_energy
_CP_ROUND, _CP_ENERGY, _CP_MAX_ENERGY = range(3)
_CP_HAND_OFFSET = _CP_HEADER_SIZE
_CP_ENEMIES_OFFSET = _CP_HAND_OFFSET + MAX_HAND_CARDS * HAND_SLOT_SIZE
_CP_PLAYER_POWERS_OFFSET = _CP_ENEMIES_OFFSET + MAX_ENEMIES * ENEMY_SLOT_SIZE
_CP_PILE_COUNTS_OFFSET = _CP_PLAYER_POWERS_OFFSET + MAX_PLAYER_POWERS * 2
_CP_DRAW_PILE_OFFSET = _CP_PILE_COUNTS_OFFSET + 3
_CP_DISCARD_PILE_OFFSET = _CP_DRAW_PILE_OFFSET + MAX_DECK_CARDS * CARD_SIZE
_CP_EXHAUST_PILE_OFFSET = _CP_DISCARD_PILE_OFFSET + MAX_DECK_CARDS * CARD_SIZE
_CP_ORBS_OFFSET = _CP_EXHAUST_PILE_OFFSET + MAX_DECK_CARDS * CARD_SIZE
_CP_ORB_CAPACITY_OFFSET = _CP_ORBS_OFFSET + MAX_ORBS * 3
_CP_STARS_OFFSET = _CP_ORB_CAPACITY_OFFSET + 1
_CP_OSTY_OFFSET = _CP_STARS_OFFSET + 1
_CP_PLAYER_HISTORY_OFFSET = _CP_OSTY_OFFSET + 4
_CP_CARD_GENERATED_BY_CREATOR_OFFSET = _CP_PLAYER_HISTORY_OFFSET + HISTORY_CREATURE_SIZE
_CP_ENEMY_HISTORY_OFFSET = _CP_CARD_GENERATED_BY_CREATOR_OFFSET + 2
COMBAT_PLAY_SIZE = _CP_ENEMY_HISTORY_OFFSET + MAX_ENEMIES * HISTORY_CREATURE_SIZE
assert COMBAT_PLAY_SIZE == 2397

# --- CardSelect --------------------------------------------------------------------------------

_CS_MIN_SELECT, _CS_MAX_SELECT = 0, 1
_CS_CARDS_OFFSET = 2
CARD_SELECT_SIZE = _CS_CARDS_OFFSET + MAX_DECK_CARDS * CARD_SIZE
assert CARD_SELECT_SIZE == 552

# --- Shop ----------------------------------------------------------------------------------------

MAX_SHOP_CARDS = 15
MAX_SHOP_RELICS = 6
MAX_SHOP_POTIONS = 5
_SHOP_CARD_SLOT_SIZE = CARD_SIZE + 3  # Card(11) + cost + is_stocked + on_sale
_SHOP_RELIC_SLOT_SIZE = 3  # token + cost + is_stocked
_SHOP_POTION_SLOT_SIZE = 3  # token + cost + is_stocked
_SHOP_CARDS_OFFSET = 0
_SHOP_RELICS_OFFSET = _SHOP_CARDS_OFFSET + MAX_SHOP_CARDS * _SHOP_CARD_SLOT_SIZE
_SHOP_POTIONS_OFFSET = _SHOP_RELICS_OFFSET + MAX_SHOP_RELICS * _SHOP_RELIC_SLOT_SIZE
_SHOP_REMOVAL_COST_OFFSET = _SHOP_POTIONS_OFFSET + MAX_SHOP_POTIONS * _SHOP_POTION_SLOT_SIZE
SHOP_SIZE = _SHOP_REMOVAL_COST_OFFSET + 1
assert SHOP_SIZE == 244

# --- RestSite --------------------------------------------------------------------------------------

MAX_REST_OPTIONS = 6
REST_SITE_SIZE = MAX_REST_OPTIONS * 2  # (present, is_enabled) per option
assert REST_SITE_SIZE == 12

# --- CardReward ------------------------------------------------------------------------------------

MAX_REWARD_CARDS = 6
_CR_CAN_SKIP, _CR_GOLD_DELTA = 0, 1
_CR_CARDS_OFFSET = 2
CARD_REWARD_SIZE = _CR_CARDS_OFFSET + MAX_REWARD_CARDS * CARD_SIZE
assert CARD_REWARD_SIZE == 68

# --- EventChoice -----------------------------------------------------------------------------------

MAX_EVENT_OPTIONS = 5
_EC_EVENT_TOKEN = 0
_EC_OPTIONS_OFFSET = 1
EVENT_CHOICE_SIZE = _EC_OPTIONS_OFFSET + MAX_EVENT_OPTIONS
assert EVENT_CHOICE_SIZE == 6

# --- EventCardReward -------------------------------------------------------------------------------

_ECR_CARDS_OFFSET = 0
EVENT_CARD_REWARD_SIZE = MAX_REWARD_CARDS * CARD_SIZE
assert EVENT_CARD_REWARD_SIZE == 66

# --- BundleSelect ----------------------------------------------------------------------------------

MAX_BUNDLES = 4
MAX_BUNDLE_CARDS = 8
BUNDLE_SELECT_SIZE = MAX_BUNDLES * MAX_BUNDLE_CARDS * CARD_SIZE
assert BUNDLE_SELECT_SIZE == 352

# --- GameOver --------------------------------------------------------------------------------------

_GO_VICTORY = 0
GAME_OVER_SIZE = 1

# --- absolute section offsets, in BufferLayout's own declared order ------------------------------

DECISION_TYPE = 0
PLAYER_SUMMARY = DECISION_TYPE + 1
MAP = PLAYER_SUMMARY + PLAYER_SUMMARY_SIZE
RUN_CONTEXT = MAP + MAP_SIZE
COMBAT_PLAY = RUN_CONTEXT + RUN_CONTEXT_SIZE
CARD_SELECT = COMBAT_PLAY + COMBAT_PLAY_SIZE
SHOP = CARD_SELECT + CARD_SELECT_SIZE
REST_SITE = SHOP + SHOP_SIZE
CARD_REWARD = REST_SITE + REST_SITE_SIZE
EVENT_CHOICE = CARD_REWARD + CARD_REWARD_SIZE
EVENT_CARD_REWARD = EVENT_CHOICE + EVENT_CHOICE_SIZE
BUNDLE_SELECT = EVENT_CARD_REWARD + EVENT_CARD_REWARD_SIZE
GAME_OVER = BUNDLE_SELECT + BUNDLE_SELECT_SIZE
TOTAL_SIZE = GAME_OVER + GAME_OVER_SIZE


# --- accessors ---------------------------------------------------------------------------------
# Pure functions of (arr, ...): no object graph, just offset math + numpy indexing. `arr` is the
# fixed-width float32 record -- every function below indexes into it directly, matching whatever the encoders
# (buffer/encode.py, buffer/run_encode.py) last wrote for that section.


def decision_name(arr: np.ndarray) -> str:
    return DECISION_NAMES.get(int(arr[DECISION_TYPE]), "unknown")


def card_token(slot: np.ndarray, offset: int = 0) -> int:
    """slot is a Card(11) block (or an array containing one starting at `offset`, e.g. a hand
    slot). Card() writes token 0 for an empty/padding slot -- 0 is also Tokenizer.UnknownToken,
    but a real card can never encode to it (LoadCategory starts numbering categories at 1), so
    `!= 0` is a safe "is this slot occupied" check wherever Card()'s own convention applies
    (unlike enemies()/player_powers()/orbs(), which use -1 instead -- see their own docstrings)."""
    return int(slot[offset + CARD_TOKEN])


# --- PlayerSummary (section base = PLAYER_SUMMARY, but combat_play/etc. re-embed the same shape
# at their own player offset -- callers pass the section's own base, not the constant, hence
# these all take `base` explicitly rather than hardcoding PLAYER_SUMMARY) ------------------------

def player_hp(arr: np.ndarray) -> int:
    return int(arr[PLAYER_SUMMARY + PLAYER_HP])


def player_max_hp(arr: np.ndarray) -> int:
    return int(arr[PLAYER_SUMMARY + PLAYER_MAX_HP])


def player_block(arr: np.ndarray) -> int:
    return int(arr[PLAYER_SUMMARY + PLAYER_BLOCK])


def player_gold(arr: np.ndarray) -> int:
    return int(arr[PLAYER_SUMMARY + PLAYER_GOLD])


def player_max_potions(arr: np.ndarray) -> int:
    return int(arr[PLAYER_SUMMARY + PLAYER_MAX_POTIONS])


def player_relic_slot(arr: np.ndarray, i: int) -> np.ndarray:
    base = PLAYER_SUMMARY + _PLAYER_RELICS_OFFSET + i * RELIC_SIZE
    return arr[base:base + RELIC_SIZE]


def player_relics(arr: np.ndarray):
    for i in range(MAX_RELICS):
        slot = player_relic_slot(arr, i)
        if int(slot[RELIC_TOKEN]) != 0:
            yield i, slot


def player_potion_slot(arr: np.ndarray, i: int) -> np.ndarray:
    base = PLAYER_SUMMARY + _PLAYER_POTIONS_OFFSET + i * POTION_SIZE
    return arr[base:base + POTION_SIZE]


def player_potions(arr: np.ndarray):
    """Yields (index, slot) for every occupied potion slot -- index is the position among occupied
    slots only (0, 1, 2, ...), NOT the raw fixed buffer slot. This matches what use_potion's
    potion_index arg actually needs: RunSimulator.Actions.cs's DoUsePotion indexes into
    `player.Potions` (Player.cs: `_potionSlots.Where(p => p != null)`, gap-filtered), not
    `player.PotionSlots` (the raw fixed array this buffer section itself is written from -- see
    RunSimulator.cs's PlayerSummary()). Despite models.py's Potion.index docstring describing the
    JSON path's `.index` as "the original slot, not list position" -- empirically it's assigned
    via `player.Potions.Select((p, i) => ... index = i ...)` in PlayerSummaryJson, i.e. the SAME
    gap-filtered position DoUsePotion expects, not a stable raw slot -- so this intentionally
    diverges from the raw slot number the buffer itself encodes, to stay a valid action argument.
    (Confirmed the hard way: passing a raw slot number here caused a live combat_play decision to
    loop for 1000+ steps with mismatched potion state before this was traced back and fixed.)"""
    n = 0
    for i in range(MAX_POTIONS):
        slot = player_potion_slot(arr, i)
        if int(slot[POTION_TOKEN]) != 0:
            yield n, slot
            n += 1


def player_deck_size(arr: np.ndarray) -> int:
    return int(arr[PLAYER_SUMMARY + PLAYER_DECK_SIZE])


def deck_slot(arr: np.ndarray, i: int) -> np.ndarray:
    base = PLAYER_SUMMARY + _PLAYER_DECK_OFFSET + i * CARD_SIZE
    return arr[base:base + CARD_SIZE]


def deck_cards(arr: np.ndarray):
    """Yields (index, slot) for every real (non-padding) deck card -- the full deck, not the hand
    (see hand_cards() below for that)."""
    for i in range(MAX_DECK_CARDS):
        slot = deck_slot(arr, i)
        if card_token(slot) != 0:
            yield i, slot


# --- Map -------------------------------------------------------------------------------------------

def map_point_slot(arr: np.ndarray, row: int, col: int) -> np.ndarray:
    base = MAP + (row * MAX_MAP_COLS + col) * MAP_POINT_SIZE
    return arr[base:base + MAP_POINT_SIZE]


def map_boss_point(arr: np.ndarray) -> np.ndarray:
    base = MAP + MAX_MAP_ROWS * MAX_MAP_COLS * MAP_POINT_SIZE
    return arr[base:base + MAP_POINT_SIZE]


def map_point_type(slot: np.ndarray) -> str | None:
    return MAP_POINT_TYPES.get(int(slot[0]))


def map_point_children(slot: np.ndarray):
    """Yields (col, row) for each present child (a grid cell's PointType is -1 for an empty cell,
    but a *real* point with fewer than MAX_MAP_CHILDREN children still pads the rest -1,-1)."""
    for i in range(MAX_MAP_CHILDREN):
        col, row = int(slot[3 + i * 2]), int(slot[3 + i * 2 + 1])
        if col != -1:
            yield col, row


def map_points(arr: np.ndarray):
    """Yields (row, col, slot) for every non-empty grid cell (PointType != -1) plus a final
    (-1, -1, slot) entry for the trailing boss point -- mirrors GetFullMap()'s JSON `boss` field
    being separate from its `rows` grid."""
    for row in range(MAX_MAP_ROWS):
        for col in range(MAX_MAP_COLS):
            slot = map_point_slot(arr, row, col)
            if int(slot[0]) != -1:
                yield row, col, slot
    boss = map_boss_point(arr)
    if int(boss[0]) != -1:
        yield -1, -1, boss


# --- RunContext ------------------------------------------------------------------------------------

def run_context(arr: np.ndarray) -> dict:
    """Raw ints (act index, tokenized act/boss ids, floor) -- same convention as everywhere else in
    this module: token -> name resolution is decision.py's decode_token (engine/tables.py's TOKEN_MAP),
    not here."""
    return {
        "act_index": int(arr[RUN_CONTEXT + RUN_CONTEXT_ACT_INDEX]),
        "act_id_token": int(arr[RUN_CONTEXT + RUN_CONTEXT_ACT_ID]),
        "floor": int(arr[RUN_CONTEXT + RUN_CONTEXT_FLOOR]),
        "boss1_token": int(arr[RUN_CONTEXT + RUN_CONTEXT_BOSS1]),
        "boss2_token": int(arr[RUN_CONTEXT + RUN_CONTEXT_BOSS2]),
    }


# --- CombatPlay ------------------------------------------------------------------------------------

def round_number(arr: np.ndarray) -> int:
    return int(arr[COMBAT_PLAY + _CP_ROUND])


def energy(arr: np.ndarray) -> int:
    return int(arr[COMBAT_PLAY + _CP_ENERGY])


def max_energy(arr: np.ndarray) -> int:
    return int(arr[COMBAT_PLAY + _CP_MAX_ENERGY])


def hand_slot(arr: np.ndarray, i: int) -> np.ndarray:
    base = COMBAT_PLAY + _CP_HAND_OFFSET + i * HAND_SLOT_SIZE
    return arr[base:base + HAND_SLOT_SIZE]


def hand_cards(arr: np.ndarray):
    """Yields (index, slot) for every real (non-padding) hand card. `slot` is the full
    HAND_SLOT_SIZE row -- slot[:CARD_SIZE] is the Card(11) block, slot[_HAND_CAN_PLAY] is the
    can_play flag, etc.; damage_by_target(slot, enemy_i) below reads the per-target rows."""
    for i in range(MAX_HAND_CARDS):
        slot = hand_slot(arr, i)
        if card_token(slot) != 0:
            yield i, slot


def hand_can_play(slot: np.ndarray) -> bool:
    return bool(slot[_HAND_CAN_PLAY])


def hand_star_cost(slot: np.ndarray) -> int:
    return int(slot[_HAND_STAR_COST])


def hand_target_type(slot: np.ndarray) -> str | None:
    return TARGET_TYPE_NAMES.get(int(slot[_HAND_TARGET_TYPE]))


def hand_stat(slot: np.ndarray, key: str) -> float:
    """-1 if this stat wasn't resolved for this card -- same convention as the JSON path's
    absent-key-means-not-applicable (see HandCardStatKeys in RunSimulator.States.Buffer.cs)."""
    return float(slot[_HAND_STATS_OFFSET + HAND_STAT_KEYS.index(key)])


def damage_by_target(slot: np.ndarray, enemy_index: int) -> tuple[float, float]:
    """(damage, repeat) for this card against the enemy at `enemy_index` (an alive-enemy slot
    index, matching enemy_slot's own `i`), or (-1, -1) if not applicable."""
    off = _HAND_DAMAGE_BY_TARGET_OFFSET + enemy_index * 2
    return float(slot[off]), float(slot[off + 1])


def enemy_slot(arr: np.ndarray, i: int) -> np.ndarray:
    base = COMBAT_PLAY + _CP_ENEMIES_OFFSET + i * ENEMY_SLOT_SIZE
    return arr[base:base + ENEMY_SLOT_SIZE]


def enemies(arr: np.ndarray):
    """Yields (index, slot) for every alive enemy -- index matches damage_by_target's
    enemy_index and is NOT stable across turns once something dies (same caveat as the JSON
    path's Enemy.index). Unlike Card/Relic/Potion, an empty enemy slot's token is -1, not 0
    (WriteCombatPlay's enemy loop defaults to -1) -- don't reuse is_real_card's convention here."""
    for i in range(MAX_ENEMIES):
        slot = enemy_slot(arr, i)
        if int(slot[_ENEMY_TOKEN]) != -1:
            yield i, slot


def enemy_token(slot: np.ndarray) -> int:
    return int(slot[_ENEMY_TOKEN])


def enemy_hp(slot: np.ndarray) -> int:
    return int(slot[_ENEMY_HP])


def enemy_max_hp(slot: np.ndarray) -> int:
    return int(slot[_ENEMY_MAX_HP])


def enemy_block(slot: np.ndarray) -> int:
    return int(slot[_ENEMY_BLOCK])


def enemy_intends_attack(slot: np.ndarray) -> bool:
    return bool(slot[_ENEMY_INTENDS_ATTACK])


def enemy_intent(slot: np.ndarray, i: int) -> tuple[int, int]:
    """(intent_type, damage) for the i-th intent, or (-1, -1) if absent. intent_type is
    Intent.IntentType's raw enum int -- decode via a category lookup only if a caller actually
    needs the name; nothing on the hot path does today."""
    off = _ENEMY_INTENTS_OFFSET + i * 2
    return int(slot[off]), int(slot[off + 1])


def enemy_power_slot(slot: np.ndarray, i: int) -> np.ndarray:
    base = _ENEMY_POWERS_OFFSET + i * 2
    return slot[base:base + 2]


def enemy_powers(slot: np.ndarray):
    """slot is one enemy's own ENEMY_SLOT_SIZE row (from enemy_slot/enemies), not the top-level
    combat_play array. Empty power slots are token -1, same convention as enemies()."""
    for i in range(MAX_ENEMY_POWERS):
        pw = enemy_power_slot(slot, i)
        if int(pw[0]) != -1:
            yield i, pw


def player_power_slot(arr: np.ndarray, i: int) -> np.ndarray:
    base = COMBAT_PLAY + _CP_PLAYER_POWERS_OFFSET + i * 2
    return arr[base:base + 2]


def player_powers(arr: np.ndarray):
    """Empty power slots are token -1 (WriteCombatPlay's player-powers loop), not 0 -- same
    convention as enemies()/enemy_powers(), unlike Card/Relic/Potion's 0-for-empty."""
    for i in range(MAX_PLAYER_POWERS):
        slot = player_power_slot(arr, i)
        if int(slot[0]) != -1:
            yield i, slot


def orb_slot(arr: np.ndarray, i: int) -> np.ndarray:
    base = COMBAT_PLAY + _CP_ORBS_OFFSET + i * 3
    return arr[base:base + 3]


def orbs(arr: np.ndarray):
    """(token, passive_val, evoke_val) per slot; empty slots are token -1 )."""
    for i in range(MAX_ORBS):
        slot = orb_slot(arr, i)
        if int(slot[0]) != -1:
            yield i, slot


def orb_capacity(arr: np.ndarray) -> int:
    return int(arr[COMBAT_PLAY + _CP_ORB_CAPACITY_OFFSET])


def draw_pile_count(arr: np.ndarray) -> int:
    return int(arr[COMBAT_PLAY + _CP_PILE_COUNTS_OFFSET])


def discard_pile_count(arr: np.ndarray) -> int:
    return int(arr[COMBAT_PLAY + _CP_PILE_COUNTS_OFFSET + 1])


def exhaust_pile_count(arr: np.ndarray) -> int:
    return int(arr[COMBAT_PLAY + _CP_PILE_COUNTS_OFFSET + 2])


def _pile_slot(arr: np.ndarray, pile_offset: int, i: int) -> np.ndarray:
    base = COMBAT_PLAY + pile_offset + i * CARD_SIZE
    return arr[base:base + CARD_SIZE]


def _pile_cards(arr: np.ndarray, pile_offset: int):
    for i in range(MAX_DECK_CARDS):
        slot = _pile_slot(arr, pile_offset, i)
        if card_token(slot) != 0:
            yield i, slot


def draw_pile_cards(arr: np.ndarray):
    """Note: WritePile always pads to MAX_DECK_CARDS regardless of the actual pile size -- use
    draw_pile_count(arr) for the true count, this just yields whichever cards are actually
    present (there's no ordering guarantee beyond whatever the engine's pile order was)."""
    return _pile_cards(arr, _CP_DRAW_PILE_OFFSET)


def discard_pile_cards(arr: np.ndarray):
    return _pile_cards(arr, _CP_DISCARD_PILE_OFFSET)


def exhaust_pile_cards(arr: np.ndarray):
    return _pile_cards(arr, _CP_EXHAUST_PILE_OFFSET)


def regent_stars(arr: np.ndarray) -> int:
    """-1 if the player isn't Regent."""
    return int(arr[COMBAT_PLAY + _CP_STARS_OFFSET])


def osty(arr: np.ndarray) -> dict:
    """hp/max_hp/block are -1 if there's no Osty and the player isn't Necrobinder. `alive` is 1/0
    if there's a live/dead Osty, 0 if Necrobinder with no current Osty, or -1 if not Necrobinder --
    see WriteCombatPlay's osty lines for the exact tri-state logic."""
    base = COMBAT_PLAY + _CP_OSTY_OFFSET
    return {
        "hp": int(arr[base]), "max_hp": int(arr[base + 1]),
        "block": int(arr[base + 2]), "alive": int(arr[base + 3]),
    }


def _history_dict(arr: np.ndarray, base: int) -> dict:
    return {
        name: (int(arr[base + i * 2]), int(arr[base + i * 2 + 1]))
        for i, name in enumerate(HISTORY_COUNTER_NAMES)
    }


def player_history(arr: np.ndarray) -> dict:
    """{counter_name: (this_turn, this_combat)} for the player -- see HISTORY_COUNTER_NAMES."""
    return _history_dict(arr, COMBAT_PLAY + _CP_PLAYER_HISTORY_OFFSET)


def card_generated_by_creator(arr: np.ndarray) -> tuple[int, int]:
    """(this_turn, this_combat) count of cards generated BY this player as the creator (e.g. via
    a relic/power effect) -- keyed differently from player_history's own card_generated (which is
    keyed by Actor, not Creator). See WriteHistoryCount's doc comment in RunSimulator.cs."""
    base = COMBAT_PLAY + _CP_CARD_GENERATED_BY_CREATOR_OFFSET
    return int(arr[base]), int(arr[base + 1])


def enemy_history(arr: np.ndarray, i: int) -> dict:
    """{counter_name: (this_turn, this_combat)} for the enemy at slot `i` (same index as
    enemies()/enemy_slot()) -- an absent/dead enemy slot reads -1,-1 for every counter."""
    base = COMBAT_PLAY + _CP_ENEMY_HISTORY_OFFSET + i * HISTORY_CREATURE_SIZE
    return _history_dict(arr, base)


# --- CardSelect ------------------------------------------------------------------------------------

def min_select(arr: np.ndarray) -> int:
    return int(arr[CARD_SELECT + _CS_MIN_SELECT])


def max_select(arr: np.ndarray) -> int:
    return int(arr[CARD_SELECT + _CS_MAX_SELECT])


def select_card_slot(arr: np.ndarray, i: int) -> np.ndarray:
    base = CARD_SELECT + _CS_CARDS_OFFSET + i * CARD_SIZE
    return arr[base:base + CARD_SIZE]


def select_cards(arr: np.ndarray):
    """Yields (index, slot) for every real (non-padding) offered card -- index is the position to
    pass back in select_cards' `indices` arg."""
    for i in range(MAX_DECK_CARDS):
        slot = select_card_slot(arr, i)
        if card_token(slot) != 0:
            yield i, slot


# --- Shop ------------------------------------------------------------------------------------------

def shop_card_slot(arr: np.ndarray, i: int) -> np.ndarray:
    base = SHOP + _SHOP_CARDS_OFFSET + i * _SHOP_CARD_SLOT_SIZE
    return arr[base:base + _SHOP_CARD_SLOT_SIZE]


def shop_cards(arr: np.ndarray):
    """Yields (index, slot) for every stocked-or-not shop card entry -- slot[:CARD_SIZE] is the
    Card(11) block, slot[CARD_SIZE] is cost, slot[CARD_SIZE+1] is_stocked, slot[CARD_SIZE+2]
    is_on_sale. Character + colorless entries are concatenated, same order as the JSON path."""
    for i in range(MAX_SHOP_CARDS):
        slot = shop_card_slot(arr, i)
        if card_token(slot) != 0:
            yield i, slot


def shop_relic_slot(arr: np.ndarray, i: int) -> np.ndarray:
    base = SHOP + _SHOP_RELICS_OFFSET + i * _SHOP_RELIC_SLOT_SIZE
    return arr[base:base + _SHOP_RELIC_SLOT_SIZE]


def shop_relics(arr: np.ndarray):
    """(token, cost, is_stocked) per slot; empty slots are token -1."""
    for i in range(MAX_SHOP_RELICS):
        slot = shop_relic_slot(arr, i)
        if int(slot[0]) != -1:
            yield i, slot


def shop_potion_slot(arr: np.ndarray, i: int) -> np.ndarray:
    base = SHOP + _SHOP_POTIONS_OFFSET + i * _SHOP_POTION_SLOT_SIZE
    return arr[base:base + _SHOP_POTION_SLOT_SIZE]


def shop_potions(arr: np.ndarray):
    """(token, cost, is_stocked) per slot; empty slots are token -1."""
    for i in range(MAX_SHOP_POTIONS):
        slot = shop_potion_slot(arr, i)
        if int(slot[0]) != -1:
            yield i, slot


def shop_removal_cost(arr: np.ndarray) -> int:
    """-1 if card removal isn't available."""
    return int(arr[SHOP + _SHOP_REMOVAL_COST_OFFSET])


# --- RestSite --------------------------------------------------------------------------------------

def rest_option_slot(arr: np.ndarray, i: int) -> np.ndarray:
    base = REST_SITE + i * 2
    return arr[base:base + 2]


def rest_options(arr: np.ndarray):
    """(present, is_enabled) per slot -- present/is_enabled are 1/0, or -1 if no such option."""
    for i in range(MAX_REST_OPTIONS):
        slot = rest_option_slot(arr, i)
        if int(slot[0]) != -1:
            yield i, slot


# --- CardReward ------------------------------------------------------------------------------------

def can_skip_reward(arr: np.ndarray) -> bool:
    return bool(arr[CARD_REWARD + _CR_CAN_SKIP])


def reward_gold_delta(arr: np.ndarray) -> int:
    """Gold gained this combat (player.Gold - _goldBeforeCombat)."""
    return int(arr[CARD_REWARD + _CR_GOLD_DELTA])


def reward_card_slot(arr: np.ndarray, i: int) -> np.ndarray:
    base = CARD_REWARD + _CR_CARDS_OFFSET + i * CARD_SIZE
    return arr[base:base + CARD_SIZE]


def reward_cards(arr: np.ndarray):
    for i in range(MAX_REWARD_CARDS):
        slot = reward_card_slot(arr, i)
        if card_token(slot) != 0:
            yield i, slot


# --- EventChoice -----------------------------------------------------------------------------------

def event_token(arr: np.ndarray) -> int:
    """-1 if no event is active."""
    return int(arr[EVENT_CHOICE + _EC_EVENT_TOKEN])


def event_option_locked(arr: np.ndarray, i: int) -> int:
    """1/0 if the option exists (locked/unlocked), or -1 if no such option."""
    return int(arr[EVENT_CHOICE + _EC_OPTIONS_OFFSET + i])


def event_options(arr: np.ndarray):
    """Yields (index, is_locked) for every present event option."""
    for i in range(MAX_EVENT_OPTIONS):
        v = event_option_locked(arr, i)
        if v != -1:
            yield i, bool(v)


# --- EventCardReward -------------------------------------------------------------------------------

def event_card_reward_slot(arr: np.ndarray, i: int) -> np.ndarray:
    base = EVENT_CARD_REWARD + _ECR_CARDS_OFFSET + i * CARD_SIZE
    return arr[base:base + CARD_SIZE]


def event_card_reward_cards(arr: np.ndarray):
    for i in range(MAX_REWARD_CARDS):
        slot = event_card_reward_slot(arr, i)
        if card_token(slot) != 0:
            yield i, slot


# --- BundleSelect ------------------------------------------------------------------------------------

def bundle_card_slot(arr: np.ndarray, bundle_index: int, i: int) -> np.ndarray:
    base = BUNDLE_SELECT + (bundle_index * MAX_BUNDLE_CARDS + i) * CARD_SIZE
    return arr[base:base + CARD_SIZE]


def bundles(arr: np.ndarray):
    """Yields (bundle_index, [(card_index, slot), ...]) for every non-empty bundle (Scroll Box
    'pick 1 of N packs') -- an empty bundle slot (no pending selection, or fewer than
    MAX_BUNDLES offered) is skipped entirely rather than yielded with an empty card list."""
    for b in range(MAX_BUNDLES):
        cards = [
            (i, bundle_card_slot(arr, b, i))
            for i in range(MAX_BUNDLE_CARDS)
            if card_token(bundle_card_slot(arr, b, i)) != 0
        ]
        if cards:
            yield b, cards


# --- GameOver --------------------------------------------------------------------------------------

def victory(arr: np.ndarray) -> bool:
    return bool(arr[GAME_OVER + _GO_VICTORY])

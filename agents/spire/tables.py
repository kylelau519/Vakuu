"""Token map and static card data for the buffer encoder, built from spire-zero's own tables (data/game_ids.json and
data/card_table.json, written by `SimWorker tables` and `SimWorker cardtable`: see refresh_tables.sh).

A token is the 1-based position of "CATEGORY:ID" in the same order sts2-cli's Tokenizer used: the categories in
CATEGORY_ORDER, each category's ids sorted ordinally, then the characters. Token 0 is unknown. A new game version
shifts them, so nets are tied to the tables they trained on."""
from __future__ import annotations

import json
import os

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

CATEGORY_ORDER = ("afflictions", "acts", "card_keywords", "cards", "enchantments", "encounters", "events", "intents",
                  "monsters", "orbs", "potions", "powers", "relics")
# game_ids.json's name for each category (keywords there, card_keywords here)
_IDS_KEY = {"card_keywords": "keywords"}

with open(os.path.join(DATA, "game_ids.json")) as _f:
    _IDS = json.load(_f)
with open(os.path.join(DATA, "card_table.json")) as _f:
    CARD_TABLE = json.load(_f)
CARDS: dict[str, dict] = CARD_TABLE["cards"]
INTENT_TYPES: list[str] = CARD_TABLE["intent_types"]
INTENT_INDEX = {n: i for i, n in enumerate(INTENT_TYPES)}

_token_of: dict[tuple[str, str], int] = {}
TOKEN_MAP: dict[int, str] = {0: "UNKNOWN"}  # int -> "CATEGORY:ID", what buffer_env.decode_token reads


def _build() -> None:
    n = 1
    def add(cat: str, ids) -> None:
        nonlocal n
        for i in sorted({x.upper() for x in ids}):
            _token_of[(cat, i)] = n
            TOKEN_MAP[n] = f"{cat.upper()}:{i}"
            n += 1
    for cat in CATEGORY_ORDER:
        add(cat, _IDS.get(_IDS_KEY.get(cat, cat), []))
    add("characters", _IDS.get("characters", []))


_build()


def token(category: str, entry_id: str | None) -> int:
    """The token of an id ("+" upgrade marks stripped), 0 when it is unknown."""
    if not entry_id:
        return 0
    return _token_of.get((category, entry_id.rstrip("+").upper()), 0)


def install_decode_map() -> None:
    """Point buffer_env.decode_token at this table instead of sts2-cli's CLR one."""
    import buffer_env
    buffer_env._token_map = dict(TOKEN_MAP)

"""Shared entity vocabulary: every card, relic, potion, power, keyword, enemy, orb, enchantment,
affliction, encounter, event, and event-option id in the game, mapped to a stable integer index.
Both Vakuu nets see these same entities inside `PlayerSummary`/`RunContext` (deck,
relics, potions -- Golden Compass needs them same as Claws does
for combat), so one vocabulary shared across both means a card's embedding index is the same
wherever it's consumed, and later lets their embedding tables be pretrained or tied together if
they converge on one network.

Indices come from vocab_ids.json, committed to this repo (ids only, no display text). Loading is index-stable across runs *as long
as vocab_ids.json doesn't change*. A new game version is handled by common/extend_vocab.py, which keeps every old id at its old index
and APPENDS the new ids (from engine/data/game_ids.json), then moves each net's embedding tables onto the new layout. Never
re-sort it: that shifts every index above an insertion point and silently invalidates any embedding table trained against the old
file. Treat a vocab_ids.json update as a breaking change to go with a retrain (or an extend_vocab conversion), not a routine regen.

Every category is a plain id -> index lookup. `Bundle.cards` (from `bundle_select`) carries no ids: encode_card returns UNK for those.

Usage:
    from vakuu.common.tokenizer import EntityVocab
    vocab = EntityVocab.load()
    vocab.encode_card(card_id)                   # "CARD.STRIKE_IRONCLAD" -> int
    vocab.encode_id("power", power_id)
    vocab.encode_id("relic", relic_id)
    vocab.encode_id("event", event_id)
    vocab.encode_id("encounter", encounter_id)
    vocab.encode_id("event_option", option_text_key)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_IDS_PATH = Path(os.environ.get("VOCAB_IDS") or Path(__file__).parent / "vocab_ids.json")  # VOCAB_IDS: another game version's vocabulary (common/extend_vocab.py)


class EntityVocab:
    PAD_IDX = 0
    UNKNOWN_IDX = 1

    def __init__(self, ids_by_category: dict[str, list[str]]):
        self._index: dict[str, dict[str, int]] = {
            category: {entry_id: i + 2 for i, entry_id in enumerate(ids)}
            for category, ids in ids_by_category.items()
        }

    @classmethod
    def load(cls, path: str | Path | None = None) -> "EntityVocab":
        return cls(json.loads(Path(path or _IDS_PATH).read_text()))

    @property
    def categories(self) -> list[str]:
        return list(self._index)

    def size(self, category: str) -> int:
        """Embedding table size for this category, including the PAD/UNK rows."""
        return len(self._index[category]) + 2

    def encode_id(self, category: str, entry_id: str | None) -> int:
        """Unrecognized or missing ids map to UNK rather than raising, since an unseen id (game
        patch, locale drift, a context that genuinely has no id -- Bundle.cards) is a
        training-time input, not a bug to crash on."""
        if entry_id is None:
            return self.UNKNOWN_IDX
        return self._index[category].get(entry_id, self.UNKNOWN_IDX)

    def encode_card(self, raw_id: str | None) -> int:
        """a card id arrives as "CARD:ENTRY" or "CARD.ENTRY" (both have been used
        separators across versions) -- strip the "CARD" prefix and look up the remainder."""
        if not raw_id:
            return self.UNKNOWN_IDX
        entry_id = raw_id[5:] if raw_id[:4] == "CARD" and len(raw_id) > 4 and raw_id[4] in ":." else raw_id
        return self.encode_id("card", entry_id)

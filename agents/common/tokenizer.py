"""Shared entity vocabulary: every card, relic, potion, power, keyword, enemy, orb, enchantment,
affliction, encounter, event, and event-option id in the game, mapped to a stable integer index.
All three Vakuu subsystems see these same entities inside `PlayerSummary`/`RunContext` (deck,
relics, potions -- Choices Paradox and Golden Compass need them same as Whispering Earring does
for combat), so one vocabulary shared across all three means a card's embedding index is the same
wherever it's consumed, and later lets their embedding tables be pretrained or tied together if
the three converge on one network (see agents/README.md).

Indices come from vocab_ids.json, committed to this repo (see common/README.md for why that file
holds ids only, no display text). Loading is index-stable across runs *as long as
vocab_ids.json doesn't change* -- rebuilding it via build_vocab.py re-sorts and can shift every
index above an insertion point, silently invalidating any embedding table trained against the
old file. Treat a vocab_ids.json update as a breaking change to go with a retrain, not a routine
regen.

Every category is a plain id -> index lookup. That wasn't always true: relic/potion/enemy/orb/
event used to only expose a localized display name on the wire, which needed a live reverse
lookup against sts2-cli's localization_eng at runtime -- see git history (encode_name/_name_map)
for that version, including the bug it had (multiple entry ids sharing one display name, e.g.
every character's "Strike", silently collapsed to a single arbitrary id). A local sts2-cli patch
(see its python/models.py -- `Relic.id`/`Potion.id`/`Enemy.id`/`Orb.id`/`EventChoice.event_id`
added, mirroring `Power.id`/`Boss.id`/`Card.id`, which already existed) closed that gap at the
source instead: every category now carries a stable id, so this module no longer needs to read
sts2-cli at runtime at all, only the committed vocab_ids.json. Verified end-to-end against a real
session (event_id/relic.id/enemy.id all confirmed present); potion.id/orb.id use the identical
code pattern, not separately exercised.

One category still can't reach an id even with the patch: `Bundle.cards` (from `bundle_select`)
is documented as `Card` minus `id` -- encode_card correctly returns UNK for those rather than
guessing from the display name.

Usage:
    from vocab import EntityVocab
    vocab = EntityVocab.load()
    vocab.encode_card(hand_card.id)              # "CARD.STRIKE_IRONCLAD" -> int
    vocab.encode_id("power", power.id)
    vocab.encode_id("relic", relic.id)
    vocab.encode_id("event", decision.event_id)
    vocab.encode_id("encounter", encounter_id)   # e.g. sts2-cli's `encounter` request param
    vocab.encode_id("event_option", option.text_key)
"""
from __future__ import annotations

import json
from pathlib import Path

_IDS_PATH = Path(__file__).parent / "vocab_ids.json"


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
        """schema Card.id arrives as "CARD:ENTRY" or "CARD.ENTRY" (sts2-cli has used both
        separators across versions) -- strip the "CARD" prefix and look up the remainder."""
        if not raw_id:
            return self.UNKNOWN_IDX
        entry_id = raw_id[5:] if raw_id[:4] == "CARD" and len(raw_id) > 4 and raw_id[4] in ":." else raw_id
        return self.encode_id("card", entry_id)

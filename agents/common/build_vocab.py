"""Generates vocab_ids.json from sts2-cli's localization_eng/*.json. Run manually:

    cd agents/common && python build_vocab.py   # sts2-cli must be checked out next to Vakuu
Parses sts2-cli's localization_eng/*.json -- convert to a single stable vocab_ids.json 
"""
from __future__ import annotations
import os

import json
from pathlib import Path


CATEGORY_FILES: dict[str, tuple[str, ...]] = {
    "card": ("cards.json",),
    "relic": ("relics.json",),
    "potion": ("potions.json",),
    "power": ("powers.json",),
    "keyword": ("card_keywords.json",),
    "enemy": ("monsters.json",),
    "orb": ("orbs.json",),
    "enchantment": ("enchantments.json",),
    "affliction": ("afflictions.json",),
    "encounter": ("encounters.json",),
    "event": ("ancients.json", "events.json"),
}

_DEFAULT_STS2_CLI_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "sts2-cli")
)


def localization_root(override: str | None = None) -> Path:
    """Directory containing cards.json/relics.json/etc. Override with STS2_CLI_ROOT if sts2-cli
    isn't checked out next to Vakuu."""
    root = override or os.environ.get(
        "STS2_CLI_ROOT") or _DEFAULT_STS2_CLI_ROOT
    loc = Path(root) / "localization_eng"
    if not loc.is_dir():
        raise FileNotFoundError(
            f"no localization_eng/ under {root!r} -- check out sts2-cli next to Vakuu or set "
            "STS2_CLI_ROOT to point at it"
        )
    return loc


def load_entries(category: str, root: Path) -> list[str]:
    """Returns sorted entry ids for one category, merged across all of CATEGORY_FILES[category]
    -- one file for most categories, two for "event"."""
    ids: set[str] = set()
    for filename in CATEGORY_FILES[category]:
        data = json.loads((root / filename).read_text())
        for key in data:
            if "." not in key:
                continue  # e.g. card_keywords.json's "PERIOD" -- a formatting string, not an entry
            ids.add(key.split(".", 1)[0])
    return sorted(ids)


def load_event_option_ids(root: Path) -> list[str]:
    """EventOption.text_key (e.g. "ABYSSAL_BATHS.pages.ALL.options.EXIT_BATHS") is already a
    stable id -- no display-name resolution needed, unlike event_name. Collected as every
    ancients.json/events.json key path containing ".options." with its trailing ".title"/
    ".description" stripped.

    Not exhaustive: some events (confirmed for Neow's opening choice) generate their options
    dynamically from other game content (a relic id, a card pack) rather than pre-written
    per-option text, so their real text_key never appears in localization_eng at all. Those fall
    back to UNK like any other id this vocab hasn't seen -- expected, not a bug."""
    ids: set[str] = set()
    for filename in CATEGORY_FILES["event"]:
        data = json.loads((root / filename).read_text())
        for key in data:
            if ".options." not in key:
                continue
            for field in ("title", "description"):
                suffix = f".{field}"
                if key.endswith(suffix):
                    ids.add(key[: -len(suffix)])
                    break
    return sorted(ids)



_OUT_PATH = Path(__file__).parent / "vocab_ids.json"


def build() -> dict[str, list[str]]:
    root = localization_root()
    ids_by_category = {category: load_entries(category, root) for category in CATEGORY_FILES}
    ids_by_category["event_option"] = load_event_option_ids(root)
    return ids_by_category


def main() -> None:
    ids_by_category = build()
    _OUT_PATH.write_text(json.dumps(ids_by_category, indent=2, sort_keys=True) + "\n")
    for category, ids in sorted(ids_by_category.items()):
        print(f"{category}: {len(ids)} entries")
    print(f"wrote {_OUT_PATH}")


if __name__ == "__main__":
    main()

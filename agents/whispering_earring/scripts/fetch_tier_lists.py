#!/usr/bin/env python3
"""Fetches Baalorlord's per-character STS2 card tier lists from untapped.gg and merges them
into one flat card_id -> placement lookup.

untapped.gg is a Next.js app: each tier-list page server-renders its full data into a
`self.__next_f.push([1, "..."])` script chunk before any client JS runs, so the page HTML
already contains everything needed -- no headless browser required. This script extracts that
chunk, JS-unescapes it, then pulls out the `"tierList": {...}` object by brace-balancing (it's
embedded inside a larger flight-protocol payload, not a standalone JSON document).

Only the factual ranking (tier, order, good_upgrade, max_copies) is kept -- the human-written
`commentary` text is proprietary editorial content from untapped.gg, not code, so it isn't
vendored into this repo.

Run: python agents/choices_paradox/scripts/fetch_tier_lists.py
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

# Baalorlord's tier lists, v0.111.0 -- see agents/choices_paradox/data/README.md for how to
# find the current URLs if these characters get re-published under new uuids.
TIER_LIST_URLS = {
    "ironclad": "https://sts2.untapped.gg/en/tier-list/004de170-026a-4dd4-a280-3b904be0b5d6/beta",
    "silent": "https://sts2.untapped.gg/en/tier-list/6d61ea21-0552-4c49-8bb5-a5c15530fc00/beta",
    "defect": "https://sts2.untapped.gg/en/tier-list/5a512e04-4583-4a16-9271-d46864c6cb4c/beta",
    "regent": "https://sts2.untapped.gg/en/tier-list/0e6c1e23-bec6-4887-a9e0-dbf49ede974d/beta",
    "necrobinder": "https://sts2.untapped.gg/en/tier-list/43d0b41f-7d6d-4ce9-928e-c1310a413983/beta",
    "colorless": "https://sts2.untapped.gg/en/tier-list/0a626c22-dc49-433e-ac6e-76cc9abf5684/beta",
}

# Merge order: colorless first as a baseline, then each class list layered on top so a
# class-specific placement for a colorless card (evaluated in that class's context) wins over
# the generic one. Card ids are unique within a class's own pool, so class-vs-class collisions
# don't happen -- only colorless-vs-class ones do.
MERGE_ORDER = ["colorless", "ironclad", "silent", "defect", "regent", "necrobinder"]

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "card_tiers.json"


def _fetch_html(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _extract_next_chunks(html: str) -> list[str]:
    """Pulls every `self.__next_f.push([1, "<chunk>"])` payload out of the page and
    JS-unescapes each one back to plain text."""
    chunks = []
    for m in re.finditer(r'self\.__next_f\.push\(\[1,"', html):
        start = m.end()
        i = start
        in_escape = False
        while i < len(html):
            if html[i] == "\\":
                i += 2
                continue
            if html[i] == '"':
                break
            i += 1
        raw = html[start:i]
        try:
            chunks.append(json.loads('"' + raw + '"'))
        except json.JSONDecodeError:
            continue
    return chunks


def _extract_balanced_object(s: str, start: int) -> str:
    """s[start] must be '{' -- returns the substring through its matching '}', respecting
    string literals (so braces inside card descriptions don't throw off the depth count)."""
    depth = 0
    in_str = False
    i = start
    while i < len(s):
        ch = s[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
        i += 1
    raise ValueError("unbalanced braces")


def fetch_tier_list(character: str, url: str) -> dict:
    html = _fetch_html(url)
    for chunk in _extract_next_chunks(html):
        idx = chunk.find('"tierList":')
        if idx == -1:
            continue
        start = idx + len('"tierList":')
        if start >= len(chunk) or chunk[start] != "{":
            continue
        data = json.loads(_extract_balanced_object(chunk, start))
        if data.get("canonical") == character:
            return data
    raise RuntimeError(f"no tierList payload found for {character} at {url}")


def merge(tier_lists: dict[str, dict]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for character in MERGE_ORDER:
        data = tier_lists[character]
        for tier in data["tiers"]:
            for card in tier["cards"]:
                card_id = card["card_id"]
                merged[card_id] = {
                    "tier": tier["name"],
                    "order": card["order"],
                    "good_upgrade": card.get("good_upgrade"),
                    "max_copies": card.get("max_copies"),
                    "source_character": character,
                }
    return merged


def main() -> None:
    tier_lists = {}
    for character, url in TIER_LIST_URLS.items():
        print(f"fetching {character}...")
        tier_lists[character] = fetch_tier_list(character, url)

    merged = merge(tier_lists)

    payload = {
        "source": "untapped.gg (Baalorlord tier lists)",
        "version": tier_lists["ironclad"].get("version"),
        "characters": {c: d.get("version") for c, d in tier_lists.items()},
        "updated": {c: d.get("updated") for c, d in tier_lists.items()},
        "cards": merged,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(merged)} cards to {OUT_PATH}")


if __name__ == "__main__":
    main()

"""The entity vocabulary for the game spire-zero hosts, and the nets' embedding tables moved onto it.

The vocabulary common/vocab_ids.json used to be built from an older game's localization (that file is now in git history only): a card, relic, power... the game has since added is UNKNOWN to
every net trained on it. This keeps every old id at its old index and APPENDS the ids of data/game_ids.json that are missing, per
category, into common/vocab_ids.json, the repo's one vocabulary. A checkpoint is then converted: each embedding table gets
its rows moved to the new layout, and a new id's row starts as a copy of the UNK row (plus a little noise to break symmetry), so
the converted net plays exactly as the old one until it has trained on the new ids.

    python -m vakuu.common.extend_vocab vocab                      # write the vocabulary
    python -m vakuu.common.extend_vocab claws SRC.pt DST.pt        # Claws (one global id table)
    python -m vakuu.common.extend_vocab compass SRC.pt DST.pt      # Golden Compass (one table per category)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
NEW = HERE / "vocab_ids.json"          # this game version's vocabulary (the repo's one vocabulary)
# The vocabulary the pre-v0.113.1 nets were trained on is no longer a file: it is this blob of the repo's history (git cat-file -p).
LEGACY_BLOB = "510fc3c9d09a6abbe2b957b1ab61985330134a42"
GAME = HERE.parent / "engine" / "data" / "game_ids.json"
# vocabulary category -> game_ids.json key (the others keep their old ids)
FROM_GAME = {"card": "cards", "relic": "relics", "potion": "potions", "power": "powers", "enemy": "monsters", "orb": "orbs",
             "enchantment": "enchantments", "affliction": "afflictions", "encounter": "encounters"}


def legacy_vocab() -> dict[str, list[str]]:
    """The old vocabulary, read from git history (needed only to move an old net onto the new vocabulary)."""
    import subprocess
    out = subprocess.run(["git", "cat-file", "-p", LEGACY_BLOB], cwd=HERE, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def build_vocab() -> dict[str, list[str]]:
    old = legacy_vocab()
    game = json.loads(GAME.read_text())
    new = {}
    for cat, ids in old.items():
        extra = []
        if cat in FROM_GAME:
            have = set(ids)
            extra = sorted(i for i in game[FROM_GAME[cat]] if i not in have)
        new[cat] = list(ids) + extra
    return new


def _sizes(vocab: dict[str, list[str]]) -> dict[str, int]:
    return {c: len(v) + 2 for c, v in vocab.items()}  # PAD and UNK


def _grow(old_rows: torch.Tensor, new_n: int, gen: torch.Generator) -> torch.Tensor:
    """old rows kept; each added row = the UNK row (index 1) + 1% noise."""
    out = torch.empty(new_n, old_rows.shape[1], dtype=old_rows.dtype)
    out[: old_rows.shape[0]] = old_rows
    extra = new_n - old_rows.shape[0]
    if extra > 0:
        unk = old_rows[1].unsqueeze(0).expand(extra, -1)
        out[old_rows.shape[0]:] = unk + 0.01 * torch.randn(extra, old_rows.shape[1], generator=gen, dtype=old_rows.dtype)
    return out


def convert_claws(src: str, dst: str) -> None:
    old, new = legacy_vocab(), json.loads(NEW.read_text())
    sd = torch.load(src, map_location="cpu", weights_only=False)
    emb = sd["id_emb.weight"]
    so, sn = _sizes(old), _sizes(new)
    assert emb.shape[0] == sum(so.values()), f"{src}: id_emb has {emb.shape[0]} rows, the old vocabulary {sum(so.values())}"
    gen = torch.Generator().manual_seed(0)
    parts, o = [], 0
    for cat in old:
        parts.append(_grow(emb[o: o + so[cat]], sn[cat], gen))
        o += so[cat]
    sd["id_emb.weight"] = torch.cat(parts)
    torch.save(sd, dst)
    print(f"claws: id_emb {emb.shape[0]} -> {sd['id_emb.weight'].shape[0]} rows -> {dst}")


def convert_compass(src: str, dst: str) -> None:
    old, new = legacy_vocab(), json.loads(NEW.read_text())
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd, cfg = ck["state_dict"], ck["config"]
    gen = torch.Generator().manual_seed(0)
    table = {"card": "card", "enchantment": "enchant", "relic": "relic", "potion": "potion", "encounter": "encounter"}
    for cat, name in table.items():
        w = sd[f"{name}.weight"]
        assert w.shape[0] == len(old[cat]) + 2, f"{src}: {name} has {w.shape[0]} rows, the old vocabulary {len(old[cat]) + 2}"
        sd[f"{name}.weight"] = _grow(w, len(new[cat]) + 2, gen)
        cfg["sizes"][cat] = len(new[cat]) + 2
    torch.save({"state_dict": sd, "config": cfg}, dst)
    print(f"compass: {', '.join(f'{c} {len(old[c]) + 2}->{len(new[c]) + 2}' for c in table)} -> {dst}")


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "vocab":
        new = build_vocab()
        NEW.write_text(json.dumps(new, indent=1) + "\n")
        old = legacy_vocab()
        print(f"wrote {NEW}: " + ", ".join(f"{c} {len(old[c])}->{len(v)}" for c, v in new.items() if len(v) != len(old[c])))
    elif cmd == "claws":
        convert_claws(sys.argv[2], sys.argv[3])
    elif cmd == "compass":
        convert_compass(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()

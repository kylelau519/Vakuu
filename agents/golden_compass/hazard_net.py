"""HazardNet: a run-state -> per-floor death-hazard model (the "run value net" of spire-zero, rebuilt
for this repo). sigmoid(logit[g]) is the chance the run dies ON floor g given it is alive at g; the
chance to survive from a state is the product of (1 - hazard) over the floors still to come.

Why hazards and not a win probability: every floor a run survives is a label, so even while the
combat agent is weak (round 0: nothing wins) the model still learns what makes a run last longer.

Why this scores map choices and card picks alike: a choice is turned into the RunState it leads to
(run_state.py's afterstates) and that state is scored. Nothing in here knows what a "choice" is.

Auxiliary head: the share of max HP the next AUX combats cost. It is not used at decision time; it
exists so the body gets a dense, low-variance signal besides the sparse death label.

Checkpoint = torch.save({"state_dict", "config"}); load with HazardNet.from_checkpoint (weights_only safe).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # same two-OpenMP-runtimes workaround as claws/model/model.py

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for `common`
from common.tokenizer import EntityVocab  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_state import MAP_TYPES, MAX_DELTA, RunState  # noqa: E402

FLOORS = 64
AUX = 6  # combats whose HP cost the auxiliary head predicts
ACT1_BOSS_FLOOR = 17  # floors restart in every act; act 1's boss is floor 17 (act 2 restarts at 2)
FINAL_FLOOR = ACT1_BOSS_FLOOR + 1  # expected_floors() stops counting here: the compass looks only to the end of act 1
NUM_FEATURES = 9

CHARACTERS = ["IRONCLAD", "SILENT", "DEFECT", "REGENT", "NECROBINDER"]
_CHAR_IDX = {c: i + 2 for i, c in enumerate(CHARACTERS)}  # 0 = pad, 1 = unknown
MAP_TOKENS = [f"{t}+{d}" for t in MAP_TYPES for d in range(MAX_DELTA + 1)]
_MAP_IDX = {tok: i + 1 for i, tok in enumerate(MAP_TOKENS)}  # 0 = pad


class Encoder:
    """RunState(s) -> padded id/number tensors. Ids come from common/vocab_ids.json (PAD=0, UNK=1)."""

    def __init__(self, vocab: EntityVocab | None = None):
        self.vocab = vocab or EntityVocab.load()

    def sizes(self) -> dict[str, int]:
        v = self.vocab
        return {
            "card": v.size("card"), "relic": v.size("relic"), "potion": v.size("potion"),
            "enchantment": v.size("enchantment"), "encounter": v.size("encounter"),
            "character": len(CHARACTERS) + 2, "map": len(MAP_TOKENS) + 1,
        }

    def _pad(self, rows: list[list[int]]) -> torch.Tensor:
        width = max(1, max((len(r) for r in rows), default=1))
        return torch.tensor([r + [0] * (width - len(r)) for r in rows], dtype=torch.long)

    def collate(self, states: list[RunState]) -> dict[str, torch.Tensor]:
        v = self.vocab
        deck_id, deck_up, deck_en, relic, potion, boss, remaining, mp = [], [], [], [], [], [], [], []
        char, act, start, nums = [], [], [], []
        for s in states:
            deck_id.append([v.encode_card(c[0]) for c in s.deck])
            deck_up.append([min(int(c[1]), 2) for c in s.deck])
            deck_en.append([v.encode_id("enchantment", c[2]) if c[2] else 0 for c in s.deck])
            relic.append([v.encode_id("relic", r[0]) for r in s.relics])
            potion.append([v.encode_id("potion", p) for p in s.potions])
            boss.append([v.encode_id("encounter", b) for b in s.boss])
            remaining.append([v.encode_id("encounter", e) for e in s.remaining])
            mp.append([_MAP_IDX[t] for t in s.map if t in _MAP_IDX])
            char.append(_CHAR_IDX.get(s.character, 1))
            act.append(min(max(s.act, 0), 3) + 1)
            start.append(min(max(s.floor, 0), FLOORS - 1))
            hp_frac = s.hp / s.max_hp if s.max_hp else 0.0
            nums.append([
                s.floor / 50.0, hp_frac, s.max_hp / 100.0, s.hp / 100.0, s.gold / 300.0,
                s.ascension / 10.0, s.max_potions / 5.0, len(s.deck) / 30.0, len(s.relics) / 20.0,
            ])
        return {
            "deck_id": self._pad(deck_id), "deck_up": self._pad(deck_up), "deck_en": self._pad(deck_en),
            "relic": self._pad(relic), "potion": self._pad(potion), "boss": self._pad(boss),
            "remaining": self._pad(remaining), "map": self._pad(mp),
            "char": torch.tensor(char), "act": torch.tensor(act), "start": torch.tensor(start),
            "nums": torch.tensor(nums, dtype=torch.float),
        }


def _bag(emb: nn.Embedding, ids: torch.Tensor) -> torch.Tensor:
    return emb(ids).sum(1)  # padding_idx rows are zero, so a plain sum is the masked sum


class HazardNet(nn.Module):
    def __init__(self, sizes: dict[str, int], dim: int = 8, hidden: int = 128, body: int = 128):
        super().__init__()
        self.config = {"sizes": sizes, "dim": dim, "hidden": hidden, "body": body}
        pad = {"padding_idx": 0}
        self.card = nn.Embedding(sizes["card"], dim, **pad)
        self.upgrade = nn.Embedding(3, dim)
        self.enchant = nn.Embedding(sizes["enchantment"], dim, **pad)
        self.phi = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim), nn.ReLU())
        self.relic = nn.Embedding(sizes["relic"], dim, **pad)
        self.potion = nn.Embedding(sizes["potion"], dim, **pad)
        self.encounter = nn.Embedding(sizes["encounter"], dim, **pad)  # boss and remaining share it
        self.map = nn.Embedding(sizes["map"], dim, **pad)
        self.character = nn.Embedding(sizes["character"], dim)
        self.act = nn.Embedding(5, dim)
        width = dim * 9 + NUM_FEATURES  # deck mean+sum, relic, potion, boss, remaining, map, char, act
        self.norm = nn.LayerNorm(width)
        self.body = nn.Sequential(nn.Linear(width, body), nn.ReLU(), nn.Linear(body, body), nn.ReLU())
        self.hazard = nn.Linear(body, FLOORS)
        self.aux = nn.Linear(body, AUX)

    def forward(self, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (hazard logits [B, FLOORS], aux HP-share predictions [B, AUX])."""
        mask = (b["deck_id"] != 0).float().unsqueeze(-1)
        cards = self.phi(self.card(b["deck_id"]) + self.upgrade(b["deck_up"]) + self.enchant(b["deck_en"])) * mask
        n = mask.sum(1).clamp(min=1)
        deck = torch.cat([cards.sum(1) / n, cards.sum(1) / 16.0], dim=-1)
        x = torch.cat([
            deck, _bag(self.relic, b["relic"]), _bag(self.potion, b["potion"]),
            _bag(self.encounter, b["boss"]), _bag(self.encounter, b["remaining"]),
            _bag(self.map, b["map"]) / 4.0, self.character(b["char"]), self.act(b["act"]), b["nums"],
        ], dim=-1)
        h = self.body(self.norm(x))
        return self.hazard(h), self.aux(h)

    @classmethod
    def build(cls, encoder: Encoder | None = None, **kw) -> "HazardNet":
        return cls((encoder or Encoder()).sizes(), **kw)

    def save(self, path: str) -> None:
        torch.save({"state_dict": {k: t.cpu() for k, t in self.state_dict().items()}, "config": self.config}, path)

    @classmethod
    def from_checkpoint(cls, path: str) -> "HazardNet":
        blob = torch.load(path, weights_only=True, map_location="cpu")
        net = cls(**blob["config"])
        net.load_state_dict(blob["state_dict"])
        return net.eval()


# --- survival maths -------------------------------------------------------------------------------

def _live(logits: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
    floors = torch.arange(logits.shape[1], device=logits.device)[None, :]
    return (floors >= start[:, None]).float()


def win_chance(logits: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
    """P(survive every floor from `start` on) -- the product of (1 - hazard)."""
    return torch.exp((F.logsigmoid(-logits) * _live(logits, start)).sum(1))


def expected_floors(logits: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
    """Expected number of further floors survived (capped at FINAL_FLOOR). Unlike win_chance this
    isn't ~0 for every option while the combat agent is still weak, so it still ranks choices."""
    live = _live(logits, start)
    live[:, FINAL_FLOOR:] = 0.0
    surv = torch.exp(torch.cumsum(F.logsigmoid(-logits) * live, dim=1))
    return (surv * live).sum(1)


def hazard_targets(start, end, outcome, floors: int = FLOORS) -> tuple[torch.Tensor, torch.Tensor]:
    """(target, mask) [B, floors]. outcome: 0 died on `end`, 1 won, 2 censored (stopped alive at `end`).
    Died: 0 on start..end-1, 1 on end. Won: 0 everywhere from start. Censored: 0 on start..end-1 only."""
    g = torch.arange(floors)[None, :]
    start, end, outcome = start[:, None], end[:, None], outcome[:, None]
    died, won = outcome == 0, outcome == 1
    mask = (g >= start) & (won | (g < end) | (died & (g == end)))
    target = (died & (g == end)).float()
    return target, mask.float()


def hazard_loss(logits, start, end, outcome, weight) -> torch.Tensor:
    target, mask = hazard_targets(start, end, outcome, logits.shape[1])
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none") * mask
    return (bce.sum(1) * weight).sum() / weight.sum().clamp(min=1e-6)


def aux_loss(pred, target, mask, weight) -> torch.Tensor:
    err = ((pred - target) ** 2) * mask
    return (err.sum(1) * weight).sum() / (mask.sum(1) * weight).sum().clamp(min=1e-6)

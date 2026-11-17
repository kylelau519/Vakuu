"""ClawsNet: tokenizer -> typed-token transformer -> policy/value heads for combat_play decisions
(see README.md). Reads the buffer protocol (BufferDecision) directly.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# The `sts2` conda env has two OpenMP runtimes loaded in the same process -- conda's
# llvm-openmp (a numpy/scipy dependency) and the copy PyPI's torch wheel bundles itself
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # for `common`
from common.tokenizer import EntityVocab  # noqa: E402

import sts2_path  # noqa: E402,F401  (sets sys.path to sts2-cli/python before the import below)
import buffer_layout as bl  # noqa: E402  (sts2-cli's python/buffer_layout.py)
from buffer_env import BufferDecision, decode_token  # noqa: E402

# --- token types ------------------------------------------------------------------------------

BASE_TOKEN_TYPES = ["PLAYER", "CARD", "ENEMY", "RELIC", "POTION", "PLAYER_POWER", "ENEMY_POWER"]
# One pooled token per pile (see PileEncoder). Appended after the base types so the base types'
# indices -- and a pre-pile checkpoint's type_emb rows -- are unchanged.
PILE_TOKEN_TYPES = ["DRAW_PILE", "DISCARD_PILE", "EXHAUST_PILE"]
TOKEN_TYPES = BASE_TOKEN_TYPES + PILE_TOKEN_TYPES
_TYPE_TO_IDX = {t: i for i, t in enumerate(TOKEN_TYPES)}
N_PILES = len(PILE_TOKEN_TYPES)
MAX_ENEMY_GROUPS = 16  # Enemy.index is 0..k over alive enemies; 1 + index is the group id
MAX_PILE = bl.MAX_DECK_CARDS  # the buffer pads every pile to this
N_CARD_TYPES = 7  # CardType enum: None, Attack, Skill, Power, Status, Curse, Quest
PILE_KEYS = ("pile_ids", "pile_cost", "pile_upg", "pile_type", "pile_mask", "pile_counts")

FEATURE_DIM = 6
V_SCALE = 20.0  # HP: the value head's output is V_SCALE * softplus(raw), so raw ~ 1 means ~ 20 HP


def _s10(x: float | int | None) -> float:
    """Flat x / 10 -- no log. Keeps additive arithmetic additive (lethal check = hp - (intent -
    block)) and keeps 1-vs-2-vs-3 steps. Signed: powers like Strength can be negative."""
    return float(x or 0) / 10.0


# --- global id space ---------------------------------------------------------------------------


class GlobalVocab:
    """EntityVocab (common/tokenizer.py) indexes each category from 2 independently -- card id 5
    and potion id 5 are unrelated entities that collide if fed straight into one shared embedding
    table. """

    def __init__(self, vocab: EntityVocab):
        self.vocab = vocab
        self._offset: dict[str, int] = {}
        total = 0
        for category in vocab.categories:
            self._offset[category] = total
            total += vocab.size(category)
        self.total_size = total

    def id_for(self, category: str, local_id: int) -> int:
        return self._offset[category] + local_id

    def card(self, raw_id: str | None) -> int:
        return self.id_for("card", self.vocab.encode_card(raw_id))

    def entity(self, category: str, raw_id: str | None) -> int:
        return self.id_for(category, self.vocab.encode_id(category, raw_id))


# --- tokenizer -----------------------------------------------------------------------------------


@dataclass
class TokenBatch:
    """Tokens for a single decision (batch size 1 -- see ClawsNet.forward's docstring for why
    batching is deferred)."""

    type_ids: torch.Tensor  # [N] long
    global_ids: torch.Tensor  # [N] long
    features: torch.Tensor  # [N, FEATURE_DIM] float
    player_pos: int
    # [N] long: 1 + Enemy.index on an ENEMY token and on each of its ENEMY_POWER tokens, 0 on every
    # other token. ClawsNet adds one shared embedding per value, so a power is bound to exactly
    # its own enemy even when two enemies are the same type.
    group_ids: torch.Tensor = None  # type: ignore[assignment]
    hand_pos: dict[int, int] = field(default_factory=dict)  # HandCard.index -> token row
    enemy_pos: dict[int, int] = field(default_factory=dict)  # Enemy.index -> token row
    potion_pos: dict[int, int] = field(default_factory=dict)  # Potion.index -> token row
    # PILE_KEYS -> tensors, each with a leading [3] pile dim
    # (draw, discard, exhaust) and padded to MAX_PILE cards. Unpooled on purpose -- ClawsNet's
    # PileEncoder does the pooling. The three pile tokens are in type_ids/global_ids/features too.
    piles: dict[str, torch.Tensor] = field(default_factory=dict)


def _pile_tensors(arr, gvocab: GlobalVocab) -> dict[str, torch.Tensor]:
    """The draw/discard/exhaust piles as padded [3, MAX_PILE] tensors (PILE_KEYS). pile_mask marks
    real cards (the buffer pads each pile to MAX_PILE); pile_counts are the buffer's true sizes.
    Slot order is the buffer's (the draw pile's real order) -- PileEncoder is order-invariant, so
    that order never reaches the net."""
    ids = torch.zeros(N_PILES, MAX_PILE, dtype=torch.long)
    cost = torch.zeros(N_PILES, MAX_PILE)
    upg = torch.zeros(N_PILES, MAX_PILE)
    ctype = torch.zeros(N_PILES, MAX_PILE, dtype=torch.long)
    mask = torch.zeros(N_PILES, MAX_PILE, dtype=torch.bool)
    sources = (
        (bl.draw_pile_cards(arr), bl.draw_pile_count(arr)),
        (bl.discard_pile_cards(arr), bl.discard_pile_count(arr)),
        (bl.exhaust_pile_cards(arr), bl.exhaust_pile_count(arr)),
    )
    counts = []
    for p, (cards, count) in enumerate(sources):
        counts.append(float(count))
        for i, slot in cards:
            ids[p, i] = gvocab.entity("card", decode_token(bl.card_token(slot, 0)))
            cost[p, i] = float(slot[bl.CARD_COST])
            upg[p, i] = float(bool(slot[bl.CARD_UPGRADED]))
            ctype[p, i] = min(max(int(slot[bl.CARD_TYPE]), 0), N_CARD_TYPES - 1)
            mask[p, i] = True
    return {"pile_ids": ids, "pile_cost": cost, "pile_upg": upg, "pile_type": ctype,
            "pile_mask": mask, "pile_counts": torch.tensor(counts)}


def tokenize_from_buffer(buf: BufferDecision, gvocab: GlobalVocab) -> TokenBatch:
    """Tokens for one combat_play decision, read directly off buf.arr via buffer_layout's offset
    accessors. Ends with the three pile tokens (placeholder rows, filled in by ClawsNet's
    PileEncoder from the padded pile tensors in TokenBatch.piles)."""
    arr = buf.arr
    types: list[int] = []
    ids: list[int] = []
    feats: list[list[float]] = []

    owners: list[int] = []

    def add(type_name: str, global_id: int, feature: list[float], group: int = 0) -> int:
        pos = len(types)
        types.append(_TYPE_TO_IDX[type_name])
        ids.append(global_id)
        owners.append(group)
        assert len(feature) == FEATURE_DIM
        feats.append(feature)
        return pos

    hp, max_hp = bl.player_hp(arr), bl.player_max_hp(arr)
    player_pos = add(
        "PLAYER",
        EntityVocab.PAD_IDX,
        [
            hp / max_hp if max_hp else 0.0,
            _s10(hp),
            _s10(bl.player_block(arr)),
            _s10(bl.energy(arr)),
            _s10(bl.max_energy(arr)),
            0.0,
        ],
    )

    hand_pos: dict[int, int] = {}
    for idx, slot in bl.hand_cards(arr):
        entry = decode_token(bl.card_token(slot, 0))
        pos = add(
            "CARD",
            gvocab.entity("card", entry),
            [0.0, 0.0, 0.0, float(slot[bl.CARD_COST]), float(bl.hand_can_play(slot)),
             float(bool(slot[bl.CARD_UPGRADED]))],
        )
        hand_pos[idx] = pos

    enemy_pos: dict[int, int] = {}
    for idx, slot in bl.enemies(arr):
        _, intent_dmg = bl.enemy_intent(slot, 0)
        intent_damage = intent_dmg if intent_dmg >= 0 else 0
        eh, emh = bl.enemy_hp(slot), bl.enemy_max_hp(slot)
        entry = decode_token(int(slot[0]))
        pos = add(
            "ENEMY",
            gvocab.entity("enemy", entry),
            [
                eh / emh if emh else 0.0,
                _s10(eh),
                _s10(bl.enemy_block(slot)),
                _s10(intent_damage),
                float(bl.enemy_intends_attack(slot)),
                0.0,
            ],
            group=idx + 1,
        )
        enemy_pos[idx] = pos
        for _, pw in bl.enemy_powers(slot):
            add("ENEMY_POWER", gvocab.entity("power", decode_token(int(pw[0]))),
                [0.0, 0.0, 0.0, _s10(pw[1]), 0.0, 0.0], group=idx + 1)

    for _, slot in bl.player_relics(arr):
        entry = decode_token(int(slot[bl.RELIC_TOKEN]))
        counter = slot[bl.RELIC_COUNTER]
        add(
            "RELIC", gvocab.entity("relic", entry),
            [0.0, 0.0, 0.0, _s10(counter if counter != -1 else None), 0.0, 0.0],
        )

    potion_pos: dict[int, int] = {}
    for idx, slot in bl.player_potions(arr):
        entry = decode_token(int(slot[bl.POTION_TOKEN]))
        pos = add("POTION", gvocab.entity("potion", entry), [0.0] * FEATURE_DIM)
        potion_pos[idx] = pos

    for _, slot in bl.player_powers(arr):
        entry = decode_token(int(slot[0]))
        add("PLAYER_POWER", gvocab.entity("power", entry), [0.0, 0.0, 0.0, _s10(slot[1]), 0.0, 0.0])

    for name in PILE_TOKEN_TYPES:
        add(name, EntityVocab.PAD_IDX, [0.0] * FEATURE_DIM)
    pile_tensors = _pile_tensors(arr, gvocab)

    return TokenBatch(
        type_ids=torch.tensor(types, dtype=torch.long),
        global_ids=torch.tensor(ids, dtype=torch.long),
        group_ids=torch.tensor(owners, dtype=torch.long),
        features=torch.tensor(feats, dtype=torch.float),
        player_pos=player_pos,
        hand_pos=hand_pos,
        enemy_pos=enemy_pos,
        potion_pos=potion_pos,
        piles=pile_tensors,
    )


# --- legal options (the policy head's action space) -----------------------------------------------


def legal_options_from_buffer(buf: BufferDecision) -> list[tuple[str, dict]]:
    """The policy head's action space at a combat_play decision."""
    arr = buf.arr
    options: list[tuple[str, dict]] = [("end_turn", {})]
    en = bl.energy(arr)
    enemy_indices = [i for i, _ in bl.enemies(arr)]

    for idx, slot in bl.hand_cards(arr):
        if not bl.hand_can_play(slot) or int(slot[bl.CARD_COST]) > en:
            continue
        if bl.hand_target_type(slot) == "AnyEnemy" and len(enemy_indices) >= 2:
            for enemy_idx in enemy_indices:
                options.append(("play_card", {"card_index": idx, "target_index": enemy_idx}))
        else:
            options.append(("play_card", {"card_index": idx}))

    for idx, slot in bl.player_potions(arr):
        target_type = bl.POTION_TARGET_TYPES.get(int(slot[bl.POTION_TARGET_TYPE]))
        if target_type == "AnyEnemy" and len(enemy_indices) >= 2:
            for enemy_idx in enemy_indices:
                options.append(("use_potion", {"potion_index": idx, "target_index": enemy_idx}))
        else:
            options.append(("use_potion", {"potion_index": idx}))

    return options


# --- network -------------------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """x = x + attn(LN(x)); x = x + ffn(LN(x)) -- pre-norm, per README.md's Architecture section."""

    def __init__(self, dim: int, n_heads: int = 4, ffn_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Linear(dim * ffn_mult, dim)
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class PileEncoder(nn.Module):
    """One vector per pile: rho(masked_mean_over_cards(phi(card)) ++ summaries).

    Order-invariant by construction -- a masked mean over cards with no position embedding -- so
    the draw pile's true order (which the live game never shows) cannot leak in. Padding slots are
    excluded from the mean and from every summary, so a pile padded to 50 gives bit-for-bit what a
    smaller pile with no padding would.

    Per card, phi sees: its id embedding (ClawsNet.id_emb, passed in as card_emb), cost/3,
    upgraded, copies/10 (how many copies of that exact card, same upgrade state, are in this pile)
    and, for the draw pile only, copies/draw_pile_count -- the chance that card is drawn next,
    a ratio a linear layer can't build itself. Raw counts, not log1p: 1-vs-2-vs-3 copies matter.
    Pile-level summaries: count/50, counts by card type /10, 0-cost count/10, mean cost/3.
    """

    N_SUMMARY = 1 + (N_CARD_TYPES - 2) + 2  # count, Attack..Curse, zero-cost, mean cost

    def __init__(self, dim: int):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(dim + 4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.rho = nn.Sequential(nn.Linear(dim + self.N_SUMMARY, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(
        self, card_emb: torch.Tensor, ids: torch.Tensor, cost: torch.Tensor, upg: torch.Tensor,
        ctype: torch.Tensor, mask: torch.Tensor, counts: torch.Tensor,
    ) -> torch.Tensor:
        """card_emb [B,3,P,dim]; ids/ctype [B,3,P] long; cost/upg [B,3,P] float; mask [B,3,P]
        bool (True = real card); counts [B,3] float (true pile sizes). Returns [B,3,dim]."""
        m = mask.float()
        n = m.sum(-1, keepdim=True)  # [B,3,1] -- real cards, derived from the mask itself
        cost = cost.clamp(min=0.0) * m
        upg = upg * m

        key = ids * 2 + upg.long()
        same = (key.unsqueeze(-1) == key.unsqueeze(-2)) & mask.unsqueeze(-2)  # [B,3,P,P]
        copies = same.sum(-1).float() * m  # [B,3,P]
        p_next = torch.zeros_like(copies)
        p_next[:, 0] = copies[:, 0] / counts[:, 0:1].clamp(min=1.0)  # draw pile only

        x = torch.cat(
            [card_emb, (cost / 3.0).unsqueeze(-1), upg.unsqueeze(-1), (copies / 10.0).unsqueeze(-1),
             p_next.unsqueeze(-1)], dim=-1,
        )
        pooled = (self.phi(x) * m.unsqueeze(-1)).sum(-2) / n.clamp(min=1.0)  # [B,3,dim]

        type_counts = torch.stack(
            [((ctype == t) & mask).float().sum(-1) for t in range(1, N_CARD_TYPES - 1)], dim=-1
        )  # Attack, Skill, Power, Status, Curse
        zero_cost = ((cost == 0) & mask).float().sum(-1, keepdim=True)
        mean_cost = cost.sum(-1, keepdim=True) / n.clamp(min=1.0)
        summary = torch.cat(
            [counts.unsqueeze(-1) / 50.0, type_counts / 10.0, zero_cost / 10.0, mean_cost / 3.0], dim=-1
        )
        return self.rho(torch.cat([pooled, summary], dim=-1))


class ClawsNet(nn.Module):
    @classmethod
    def from_checkpoint(cls, path: str, **kwargs) -> "ClawsNet":
        """Loads a train.py-saved state_dict against a freshly loaded vocab."""
        net = cls(GlobalVocab(EntityVocab.load()), **kwargs)
        net.load_compatible(torch.load(path, weights_only=True))
        net.eval()
        return net

    def __init__(self, gvocab: GlobalVocab, dim: int = 64, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.gvocab = gvocab
        self.dim = dim
        self.id_emb = nn.Embedding(gvocab.total_size, dim, padding_idx=EntityVocab.PAD_IDX)
        self.type_emb = nn.Embedding(len(TOKEN_TYPES), dim)
        # Pile tokens get no feature projection: their content comes from PileEncoder.
        self.feat_proj = nn.ModuleDict(
            {t: nn.Linear(FEATURE_DIM, dim) for t in BASE_TOKEN_TYPES}
        )
        self.group_emb = nn.Embedding(MAX_ENEMY_GROUPS + 1, dim, padding_idx=0)
        self.pile_encoder = PileEncoder(dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, n_heads) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(dim)

        self.end_turn_query = nn.Parameter(torch.randn(dim) * 0.02)
        self.policy_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.value_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def load_compatible(self, state: dict[str, torch.Tensor]) -> None:
        """load_state_dict that accepts a checkpoint saved before pile tokens existed: its
        type_emb has fewer rows (the new rows keep their fresh init) and it has no pile_encoder
        weights. Any other mismatch still raises."""
        state = dict(state)
        old, cur = state["type_emb.weight"], self.type_emb.weight
        if old.shape[0] < cur.shape[0]:
            state["type_emb.weight"] = torch.cat([old, cur.detach().cpu()[old.shape[0]:]])
        res = self.load_state_dict(state, strict=False)
        bad = [k for k in res.missing_keys if not k.startswith(("pile_encoder.", "group_emb."))]
        if bad or res.unexpected_keys:
            raise RuntimeError(f"checkpoint mismatch: missing {bad}, unexpected {res.unexpected_keys}")

    @staticmethod
    def value_in_hp(raw: torch.Tensor) -> torch.Tensor:
        """V(state) = the HP the player is expected to lose from this state until the combat ends
        (0 once it is won). Never negative; V_SCALE only sets the output's natural size."""
        return V_SCALE * F.softplus(raw)

    def encode(self, batch: TokenBatch) -> torch.Tensor:
        """Returns per-token hidden states [N, dim] for one decision."""
        p = {k: v.unsqueeze(0) for k, v in batch.piles.items()}
        h = self.encode_batched(
            batch.type_ids.unsqueeze(0), batch.global_ids.unsqueeze(0), batch.features.unsqueeze(0),
            torch.zeros(1, batch.type_ids.shape[0], dtype=torch.bool, device=batch.type_ids.device), p,
            batch.group_ids.unsqueeze(0),
        )
        return h.squeeze(0)

    def encode_batched(
        self, type_ids: torch.Tensor, global_ids: torch.Tensor, features: torch.Tensor,
        pad_mask: torch.Tensor, piles: dict[str, torch.Tensor], group_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Batched, padded encoder -- used by train.py (_collate/_batch_step) and, with B=1, by
        encode() for live decisions. type_ids/global_ids: [B, N] long, features: [B, N,
        FEATURE_DIM] float, pad_mask: [B, N] bool, True marking a padded position -- passed
        straight through as MultiheadAttention's key_padding_mask so padded rows are never
        attended to. group_ids: [B, N] long (TokenBatch.group_ids). piles: PILE_KEYS -> [B, 3, ...] tensors (see _pile_tensors); every example
        has exactly one DRAW/DISCARD/EXHAUST token row, whose vector PileEncoder supplies.
        Returns [B, N, dim]; padded rows' own outputs are never read (every real position index
        recorded by tokenize_from_buffer is < that example's own token count)."""
        tok = self.id_emb(global_ids) + self.type_emb(type_ids) + self.group_emb(group_ids)
        for t in BASE_TOKEN_TYPES:
            mask = (type_ids == _TYPE_TO_IDX[t]) & ~pad_mask
            if mask.any():
                tok = tok.clone()
                tok[mask] = tok[mask] + self.feat_proj[t](features[mask])

        pile_vec = self.pile_encoder(
            self.id_emb(piles["pile_ids"]), piles["pile_ids"], piles["pile_cost"], piles["pile_upg"],
            piles["pile_type"], piles["pile_mask"], piles["pile_counts"],
        )  # [B, 3, dim]
        for k, name in enumerate(PILE_TOKEN_TYPES):
            mask = (type_ids == _TYPE_TO_IDX[name]) & ~pad_mask
            if int(mask.sum()) != type_ids.shape[0]:
                raise ValueError(f"expected exactly one {name} token per example")
            tok = tok.clone()
            tok[mask] = tok[mask] + pile_vec[:, k]

        x = tok
        for block in self.blocks:
            x = block(x, key_padding_mask=pad_mask)
        return self.final_ln(x)

    def score(
        self, batch: TokenBatch, options: list[tuple[str, dict]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared by forward() (live decision) and a training loop reading materialize.py's
        saved batches """
        h = self.encode(batch)
        queries = []
        for action, args in options:
            if action == "end_turn":
                q = self.end_turn_query + h[batch.player_pos]
            elif action == "play_card":
                q = h[batch.hand_pos[args["card_index"]]]
                if "target_index" in args:
                    q = q + h[batch.enemy_pos[args["target_index"]]]
            elif action == "use_potion":
                q = h[batch.potion_pos[args["potion_index"]]]
                if "target_index" in args:
                    q = q + h[batch.enemy_pos[args["target_index"]]]
            else:
                raise ValueError(f"unhandled combat action {action!r}")
            queries.append(q)

        logits = self.policy_mlp(torch.stack(queries)).squeeze(-1)
        return logits, self.value_in_hp(self.value_mlp(h[batch.player_pos]).squeeze(-1))

    def forward(
        self, decision: BufferDecision
    ) -> tuple[list[tuple[str, dict]], torch.Tensor, torch.Tensor]:
        """(legal options, one logit per option, V in HP) for a combat_play decision."""
        batch = tokenize_from_buffer(decision, self.gvocab)
        options = legal_options_from_buffer(decision)
        logits, value = self.score(batch, options)
        return options, logits, value

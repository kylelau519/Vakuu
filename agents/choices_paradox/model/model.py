"""ChoicesParadoxNet: typed-token transformer -> policy/value heads, for card_reward and (once
Phase 2 lands) card_select's remove/upgrade purposes. Mirrors claws/model/model.py's
tokenizer -> transformer -> per-option-query -> shared-head shape (see this package's
README.md's Architecture section for why the pattern is copied but the weights aren't shared).

Phase 1 scope: only `decision_kind == "card_reward"` is exercised end to end
(whispering_earring/facility_choices_paradox.py only ever branches card_reward decisions so far).
`card_select_remove`/`card_select_upgrade` are
already wired through tokenize/legal_options/score below -- mode_emb is sized for all three from
the start precisely so Phase 2 doesn't need an embedding-table resize -- but nothing yet drives a
CardSelect decision through here.

Deliberately out of scope for v0, same spirit as claws/model/model.py's own scope note:
- Act/floor progress. CardReward/CardSelect carry no RunContext (only CombatPlay does -- see
  sts2-cli's models.py), so "how much of the act is left" isn't available here without
  facility_choices_paradox.py explicitly threading it through MetaBranchJob, the way BranchJob threads
  is_last_of_act/boss_ids for combat's own boss-fight shaping. Left out until shown to matter.

Run directly (`python model.py`, `sts2` conda env) for a smoke test against a real captured
card_reward decision.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# The `sts2` conda env has two OpenMP runtimes loaded in the same process -- conda's
# llvm-openmp (a numpy/scipy dependency) and the copy PyPI's torch wheel bundles itself
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # for `common`
from common.tokenizer import EntityVocab  # noqa: E402

import sts2_path  # noqa: E402,F401  (sets sys.path to sts2-cli/python before the imports below)
from models import CardReward, CardSelect  # noqa: E402
import buffer_layout as bl  # noqa: E402
from buffer_env import BufferDecision, decode_token  # noqa: E402

# --- token types --------------------------------------------------------------------------------

TOKEN_TYPES = ["PLAYER", "MODE", "DECK_CARD", "OFFER_CARD", "RELIC"]
_TYPE_TO_IDX = {t: i for i, t in enumerate(TOKEN_TYPES)}

FEATURE_DIM = 4

# Sized for all three from the start (not just card_reward, the only one wired through
# facility_choices_paradox.py so far) -- see this module's docstring / the package README's Phasing section for
# why: adding card_select's two purposes later must not resize this table and break checkpoint
# compatibility with whatever's already been trained.
DECISION_KINDS = ["card_reward", "card_select_remove", "card_select_upgrade"]
_KIND_TO_IDX = {k: i for i, k in enumerate(DECISION_KINDS)}


def _log1p(x: float | int | None) -> float:
    """Signed log1p -- some of what feeds this can go negative (e.g. a relic counter reset)."""
    x = float(x or 0)
    return math.copysign(math.log1p(abs(x)), x)


# --- global id space -----------------------------------------------------------------------------


class GlobalVocab:
    """Same shape as claws/model/model.py's GlobalVocab, duplicated rather than cross-imported
    -- see this package's README's "Separate network" section for why the two model packages
    stay decoupled. Both load the same committed EntityVocab (common/vocab_ids.json), so a given
    card's global id is identical in both nets -- what makes warm-starting id_emb (see
    ChoicesParadoxNet.warm_started_from) a plain row copy rather than a remapping."""

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


# --- tokenizer -------------------------------------------------------------------------------------


@dataclass
class TokenBatch:
    """Tokens for one card_reward/card_select decision (batch size 1, same reasoning as claws'
    own TokenBatch -- see train.py's docstring)."""

    type_ids: torch.Tensor  # [N] long
    global_ids: torch.Tensor  # [N] long
    features: torch.Tensor  # [N, FEATURE_DIM] float
    mode_idx: int  # DECISION_KINDS index -- looked up once against mode_emb, not per-token
    player_pos: int
    mode_pos: int
    offer_pos: dict[int, int] = field(default_factory=dict)  # offered card's own .index -> row

    @classmethod
    def from_saved(cls, d: dict) -> "TokenBatch":
        """Inverse of materialize.py's per-example dict -- see claws/model/model.py's
        TokenBatch.from_saved for why this can't just be torch.load-ed back directly."""
        return cls(
            type_ids=d["type_ids"], global_ids=d["global_ids"], features=d["features"],
            mode_idx=d["mode_idx"], player_pos=d["player_pos"], mode_pos=d["mode_pos"],
            offer_pos=d["offer_pos"],
        )


def tokenize(decision: CardReward | CardSelect, decision_kind: str, gvocab: GlobalVocab) -> TokenBatch:
    types: list[int] = []
    ids: list[int] = []
    feats: list[list[float]] = []

    def add(type_name: str, global_id: int, feature: list[float]) -> int:
        pos = len(types)
        types.append(_TYPE_TO_IDX[type_name])
        ids.append(global_id)
        assert len(feature) == FEATURE_DIM
        feats.append(feature)
        return pos

    player = decision.player
    player_pos = add(
        "PLAYER",
        EntityVocab.PAD_IDX,  # no id space of its own -- global row 0 is always a zero vector
        [
            player.hp / player.max_hp if player.max_hp else 0.0,
            _log1p(player.hp),
            _log1p(player.gold),
            _log1p(player.deck_size),
        ],
    )
    mode_pos = add("MODE", EntityVocab.PAD_IDX, [0.0] * FEATURE_DIM)

    for card in player.deck or []:
        add(
            "DECK_CARD", gvocab.card(card.id),
            [_log1p(card.cost), float(bool(card.upgraded)), 0.0, 0.0],
        )

    offer_pos: dict[int, int] = {}
    for card in decision.cards:
        pos = add(
            "OFFER_CARD", gvocab.card(card.id),
            [_log1p(card.cost), float(bool(card.upgraded)), 0.0, 0.0],
        )
        offer_pos[card.index] = pos

    for relic in player.relics:
        add("RELIC", gvocab.entity("relic", relic.id), [0.0, 0.0, _log1p(relic.counter), 0.0])

    return TokenBatch(
        type_ids=torch.tensor(types, dtype=torch.long),
        global_ids=torch.tensor(ids, dtype=torch.long),
        features=torch.tensor(feats, dtype=torch.float),
        mode_idx=_KIND_TO_IDX[decision_kind],
        player_pos=player_pos,
        mode_pos=mode_pos,
        offer_pos=offer_pos,
    )


def tokenize_from_buffer(buf: BufferDecision, decision_kind: str, gvocab: GlobalVocab) -> TokenBatch:
    """Buffer counterpart to tokenize() -- same token layout/feature math, read via
    buffer_layout's offset accessors. card_reward's offered cards come from bl.reward_cards();
    card_select's (either purpose) come from bl.select_cards() -- both yield (index, slot) pairs
    of the same Card(11) shape as bl.deck_cards()/bl.hand_cards()."""
    arr = buf.arr
    types: list[int] = []
    ids: list[int] = []
    feats: list[list[float]] = []

    def add(type_name: str, global_id: int, feature: list[float]) -> int:
        pos = len(types)
        types.append(_TYPE_TO_IDX[type_name])
        ids.append(global_id)
        assert len(feature) == FEATURE_DIM
        feats.append(feature)
        return pos

    hp, max_hp = bl.player_hp(arr), bl.player_max_hp(arr)
    player_pos = add(
        "PLAYER", EntityVocab.PAD_IDX,
        [
            hp / max_hp if max_hp else 0.0,
            _log1p(hp),
            _log1p(bl.player_gold(arr)),
            _log1p(bl.player_deck_size(arr)),
        ],
    )
    mode_pos = add("MODE", EntityVocab.PAD_IDX, [0.0] * FEATURE_DIM)

    for _, slot in bl.deck_cards(arr):
        entry = decode_token(bl.card_token(slot))
        add(
            "DECK_CARD", gvocab.entity("card", entry),
            [_log1p(slot[bl.CARD_COST]), float(bool(slot[bl.CARD_UPGRADED])), 0.0, 0.0],
        )

    offer_iter = bl.reward_cards(arr) if decision_kind == "card_reward" else bl.select_cards(arr)
    offer_pos: dict[int, int] = {}
    for idx, slot in offer_iter:
        entry = decode_token(bl.card_token(slot))
        pos = add(
            "OFFER_CARD", gvocab.entity("card", entry),
            [_log1p(slot[bl.CARD_COST]), float(bool(slot[bl.CARD_UPGRADED])), 0.0, 0.0],
        )
        offer_pos[idx] = pos

    for _, slot in bl.player_relics(arr):
        entry = decode_token(int(slot[bl.RELIC_TOKEN]))
        counter = slot[bl.RELIC_COUNTER]
        add(
            "RELIC", gvocab.entity("relic", entry),
            [0.0, 0.0, _log1p(counter if counter != -1 else None), 0.0],
        )

    return TokenBatch(
        type_ids=torch.tensor(types, dtype=torch.long),
        global_ids=torch.tensor(ids, dtype=torch.long),
        features=torch.tensor(feats, dtype=torch.float),
        mode_idx=_KIND_TO_IDX[decision_kind],
        player_pos=player_pos,
        mode_pos=mode_pos,
        offer_pos=offer_pos,
    )


# --- legal options (the policy head's action space) -----------------------------------------------


def legal_options(decision: CardReward | CardSelect, decision_kind: str) -> list[tuple[str, dict]]:
    """Every (action, args) the policy head scores. card_reward: one per offered card, plus skip
    if allowed. card_select (either purpose): both purposes are min_select=max_select=1 per
    choices_paradox/policy.py's existing choose_removal_target/choose_upgrade_target convention,
    so one option per offered card, plus skip if min_select allows 0 -- multi-select (>1) isn't
    handled here, consistent with this package's README noting combinatorial subset enumeration
    as an unresolved design question, not attempted for v1/v2."""
    if decision_kind == "card_reward":
        options: list[tuple[str, dict]] = [
            ("select_card_reward", {"card_index": c.index}) for c in decision.cards
        ]
        if decision.can_skip:
            options.append(("skip_card_reward", {}))
        return options

    options = [("select_cards", {"indices": [c.index]}) for c in decision.cards]
    if decision.min_select == 0:
        options.append(("skip_select", {}))
    return options


def legal_options_from_buffer(buf: BufferDecision, decision_kind: str) -> list[tuple[str, dict]]:
    """Buffer counterpart to legal_options() -- same enumeration, read off buf.arr."""
    arr = buf.arr
    if decision_kind == "card_reward":
        options: list[tuple[str, dict]] = [
            ("select_card_reward", {"card_index": i}) for i, _ in bl.reward_cards(arr)
        ]
        if bl.can_skip_reward(arr):
            options.append(("skip_card_reward", {}))
        return options

    options = [("select_cards", {"indices": [i]}) for i, _ in bl.select_cards(arr)]
    if bl.min_select(arr) == 0:
        options.append(("skip_select", {}))
    return options


# --- network -------------------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """x = x + attn(LN(x)); x = x + ffn(LN(x)) -- pre-norm, same shape as claws/model/model.py's
    own TransformerBlock, duplicated rather than cross-imported (see this module's docstring)."""

    def __init__(self, dim: int, n_heads: int = 4, ffn_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Linear(dim * ffn_mult, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class ChoicesParadoxNet(nn.Module):
    @classmethod
    def from_checkpoint(cls, path: str, **kwargs) -> "ChoicesParadoxNet":
        """Loads a train.py-saved state_dict against a freshly loaded vocab."""
        net = cls(GlobalVocab(EntityVocab.load()), **kwargs)
        net.load_state_dict(torch.load(path, weights_only=True))
        net.eval()
        return net

    @classmethod
    def warm_started_from(cls, claws_checkpoint_path: str | None, **kwargs) -> "ChoicesParadoxNet":
        """Copies ClawsNet's own trained card-identity embedding rows into a freshly
        initialized net -- a one-time copy at construction, not a live shared tensor (see this
        package's README's "Separate network" section for why: training this net needs Claws
        pinned to a frozen checkpoint for the round, which a shared live tensor would break).
        Falls back to a plain random init if claws_checkpoint_path is falsy or doesn't exist yet
        (e.g. round 1, before Claws has anything trained -- warm-starting from noise buys
        nothing). Silently skips the copy (same fallback) if the checkpoint's id_emb doesn't
        cover this net's card id range, rather than raising -- a vocab_ids.json size mismatch
        here is a training-data problem to notice via a smaller-than-expected warm start, not a
        crash."""
        net = cls(GlobalVocab(EntityVocab.load()), **kwargs)
        if not claws_checkpoint_path or not os.path.exists(claws_checkpoint_path):
            return net
        claws_state = torch.load(claws_checkpoint_path, weights_only=True)
        claws_id_emb = claws_state.get("id_emb.weight")
        if claws_id_emb is None or claws_id_emb.shape[1] != net.dim:
            # Dim mismatch (e.g. ClawsNet's default dim=64 vs. this net's smaller default
            # dim=32 -- see this class's docstring on why the trunk is smaller) -- a row copy
            # can't reconcile different embedding widths, and there's no principled way to
            # project one onto the other worth building for a v1 warm start. Skip rather than
            # raise, same "noisy but harmless fallback" reasoning as the missing-checkpoint case
            # above -- pass matching `dim` kwargs to actually get the warm start.
            return net
        card_lo = net.gvocab._offset["card"]
        card_hi = card_lo + net.gvocab.vocab.size("card")
        n = min(card_hi, claws_id_emb.shape[0]) - card_lo
        if n > 0:
            with torch.no_grad():
                net.id_emb.weight[card_lo:card_lo + n] = claws_id_emb[card_lo:card_lo + n]
        return net

    def __init__(self, gvocab: GlobalVocab, dim: int = 64, n_layers: int = 2, n_heads: int = 4):
        # dim matches ClawsNet's own default (agents/claws/model/model.py) so
        # warm_started_from's id_emb row-copy works without passing matching kwargs by hand --
        # see that classmethod's docstring for what happens on a deliberate mismatch.
        super().__init__()
        self.gvocab = gvocab
        self.dim = dim
        self.id_emb = nn.Embedding(gvocab.total_size, dim, padding_idx=EntityVocab.PAD_IDX)
        self.type_emb = nn.Embedding(len(TOKEN_TYPES), dim)
        self.mode_emb = nn.Embedding(len(DECISION_KINDS), dim)
        self.feat_proj = nn.ModuleDict(
            {t: nn.Linear(FEATURE_DIM, dim) for t in TOKEN_TYPES}
        )
        self.blocks = nn.ModuleList([TransformerBlock(dim, n_heads) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(dim)

        # "skip" has no card of its own to read a query off, same reasoning as ClawsNet's
        # end_turn_query -- a dedicated learned parameter instead of overloading a real token.
        self.skip_query = nn.Parameter(torch.randn(dim) * 0.02)
        self.policy_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.value_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def encode(self, batch: TokenBatch) -> torch.Tensor:
        """Returns per-token hidden states [N, dim] for one decision."""
        tok = self.id_emb(batch.global_ids) + self.type_emb(batch.type_ids)
        for t in TOKEN_TYPES:
            mask = batch.type_ids == _TYPE_TO_IDX[t]
            if mask.any():
                tok = tok.clone()
                tok[mask] = tok[mask] + self.feat_proj[t](batch.features[mask])
        tok = tok.clone()
        tok[batch.mode_pos] = tok[batch.mode_pos] + self.mode_emb(
            torch.tensor(batch.mode_idx, dtype=torch.long)
        )
        x = tok.unsqueeze(0)  # MultiheadAttention wants a batch dim
        for block in self.blocks:
            x = block(x)
        return self.final_ln(x).squeeze(0)

    def score(
        self, batch: TokenBatch, options: list[tuple[str, dict]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared by forward() (live decision) and train.py's loop reading materialize.py's
        saved batches. Returns (logits [len(options)], value scalar); logits unnormalized."""
        h = self.encode(batch)
        queries = []
        for action, args in options:
            if action in ("skip_card_reward", "skip_select"):
                q = self.skip_query + h[batch.player_pos]
            elif action == "select_card_reward":
                q = h[batch.offer_pos[args["card_index"]]]
            elif action == "select_cards":
                q = h[batch.offer_pos[args["indices"][0]]]
            else:
                raise ValueError(f"unhandled meta action {action!r}")
            queries.append(q)

        logits = self.policy_mlp(torch.stack(queries)).squeeze(-1)
        value = torch.tanh(self.value_mlp(h[batch.player_pos]).squeeze(-1))
        return logits, value

    def forward(
        self, decision: CardReward | CardSelect | BufferDecision, decision_kind: str
    ) -> tuple[list[tuple[str, dict]], torch.Tensor, torch.Tensor]:
        """Returns (options, logits [len(options)], value scalar) for one live decision. Accepts
        either shape decision can arrive in, same convention as ClawsNet.forward."""
        if isinstance(decision, BufferDecision):
            batch = tokenize_from_buffer(decision, decision_kind, self.gvocab)
            options = legal_options_from_buffer(decision, decision_kind)
        else:
            batch = tokenize(decision, decision_kind, self.gvocab)
            options = legal_options(decision, decision_kind)
        logits, value = self.score(batch, options)
        return options, logits, value


# --- smoke test ------------------------------------------------------------------------------------

if __name__ == "__main__":
    import itertools
    import json

    fixture = (
        Path(__file__).resolve().parents[3]
        / "schema" / "tests" / "fixtures" / "ironclad_run_01.jsonl"
    )
    lines = fixture.read_text().splitlines()
    raw = json.loads(lines[23])  # a card_reward decision, 3 offered cards, can_skip=True
    decision = CardReward.model_validate(raw)

    vocab = EntityVocab.load()
    gvocab = GlobalVocab(vocab)
    net = ChoicesParadoxNet(gvocab)

    options, logits, value = net(decision, "card_reward")
    probs = torch.softmax(logits, dim=-1)

    print(f"{len(options)} legal options, value={value.item():+.3f}")
    for (action, args), p in itertools.islice(zip(options, probs.tolist()), 8):
        print(f"  {p:5.3f}  {action} {args}")

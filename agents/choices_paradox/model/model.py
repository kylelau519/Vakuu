"""ChoicesParadoxNet: typed-token transformer -> policy/value heads, for card_reward and (once
Phase 2 lands) card_select's remove/upgrade purposes. Mirrors claws/model/model.py's
tokenizer -> transformer -> per-option-query -> shared-head shape, encode_batched() included --
see that method's docstring and train.py's _collate/_batch_step for the padded-batch training
path; forward() (live decisions) still goes through the single-example encode().
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
from models import CardReward, CardSelect, EventChoice, RestSite  # noqa: E402
import buffer_layout as bl  # noqa: E402
from buffer_env import BufferDecision, decode_token  # noqa: E402

# --- token types --------------------------------------------------------------------------------

TOKEN_TYPES = [
    "PLAYER", "MODE", "DECK_CARD", "OFFER_CARD", "RELIC", "REST_OPTION", "EVENT_OPTION", "BOSS",
]
_TYPE_TO_IDX = {t: i for i, t in enumerate(TOKEN_TYPES)}

FEATURE_DIM = 4

DECISION_KINDS = [
    "card_reward", "card_select_remove", "card_select_upgrade", "rest_site", "event_choice",
]
_KIND_TO_IDX = {k: i for i, k in enumerate(DECISION_KINDS)}


def _log1p(x: float | int | None) -> float:
    """Signed log1p -- some of what feeds this can go negative (e.g. a relic counter reset)."""
    x = float(x or 0)
    return math.copysign(math.log1p(abs(x)), x)


# --- global id space -----------------------------------------------------------------------------


class GlobalVocab:
    """Same shape as claws/model/model.py's GlobalVocab, duplicated rather than cross-imported """
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
    """Tokens for a single decision -- forward() only ever scores one live decision at a time,
    so no batch dimension here. Training's own batching goes through encode_batched() instead,
    via train.py's _collate/_batch_step."""

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


def tokenize(
    decision: CardReward | CardSelect | RestSite | EventChoice, decision_kind: str,
    gvocab: GlobalVocab,
) -> TokenBatch:
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
    if decision_kind == "rest_site":
        for option in decision.options:
            if not option.is_enabled:
                continue
            pos = add("REST_OPTION", EntityVocab.PAD_IDX, [0.0] * FEATURE_DIM)
            offer_pos[option.index] = pos
    elif decision_kind == "event_choice":
        for option in decision.options:
            if option.is_locked:
                continue
            pos = add("EVENT_OPTION", EntityVocab.PAD_IDX, [0.0] * FEATURE_DIM)
            offer_pos[option.index] = pos
    else:
        for card in decision.cards:
            pos = add(
                "OFFER_CARD", gvocab.card(card.id),
                [_log1p(card.cost), float(bool(card.upgraded)), 0.0, 0.0],
            )
            offer_pos[card.index] = pos

    for relic in player.relics:
        add("RELIC", gvocab.entity("relic", relic.id), [0.0, 0.0, _log1p(relic.counter), 0.0])

    # No BOSS token here -- unlike tokenize_from_buffer(), none of CardReward/CardSelect/RestSite/
    # EventChoice carry a RunContext (act boss info lives only in the buffer protocol's always-
    # present RUN_CONTEXT section, read via bl.run_context()). Not wired through the pydantic/JSON
    # path since materialize.py/training only ever uses tokenize_from_buffer() -- see there.

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
    of the same Card(11) shape as bl.deck_cards()/bl.hand_cards(). """
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

    offer_pos: dict[int, int] = {}
    if decision_kind == "rest_site":
        # No option-id-equivalent token in the buffer protocol yet (see tokenize()'s matching
        # comment and model/README.md's Open questions) -- identity-less REST_OPTION tokens,
        # positional offer_pos only.
        for i, slot in bl.rest_options(arr):
            if int(slot[1]) != 1:
                continue
            pos = add("REST_OPTION", EntityVocab.PAD_IDX, [0.0] * FEATURE_DIM)
            offer_pos[i] = pos
    elif decision_kind == "event_choice":
        # Same identity-less reasoning -- no per-option id/title/vars in the buffer either.
        for i, is_locked in bl.event_options(arr):
            if is_locked:
                continue
            pos = add("EVENT_OPTION", EntityVocab.PAD_IDX, [0.0] * FEATURE_DIM)
            offer_pos[i] = pos
    else:
        if decision_kind == "card_reward":
            offer_iter = (
                bl.event_card_reward_cards(arr) if buf.decision == "event_card_reward"
                else bl.reward_cards(arr)
            )
        else:
            offer_iter = bl.select_cards(arr)
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

    # RUN_CONTEXT is written regardless of decision type (same as PlayerSummary), so this act's
    # boss encounter(s) are available at every decision, not just rest_site's -- deck-building and
    # rest/upgrade choices can both reasonably depend on what's coming. boss2 is most acts' "NONE"
    # (a real sentinel string, not an absent slot -- RunSimulator.cs always writes it), which
    # decode_token/gvocab.entity resolve to the same UNKNOWN_IDX as any other unrecognized id, no
    # special-casing needed. The feature flags which of the two this token is (irrelevant for
    # single-boss acts, since boss2 is UNKNOWN anyway).
    ctx = bl.run_context(arr)
    for key, is_primary in (("boss1_token", 1.0), ("boss2_token", 0.0)):
        entry = decode_token(ctx[key])
        add("BOSS", gvocab.entity("encounter", entry), [is_primary, 0.0, 0.0, 0.0])

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


def legal_options(
    decision: CardReward | CardSelect | RestSite | EventChoice, decision_kind: str
) -> list[tuple[str, dict]]:
    if decision_kind == "card_reward":
        options: list[tuple[str, dict]] = [
            ("select_card_reward", {"card_index": c.index}) for c in decision.cards
        ]
        if decision.can_skip:
            options.append(("skip_card_reward", {}))
        return options

    if decision_kind == "rest_site":
        options = [
            ("choose_option", {"option_index": o.index}) for o in decision.options if o.is_enabled
        ]
        options.append(("leave_room", {}))
        return options

    if decision_kind == "event_choice":
        options = [
            ("choose_option", {"option_index": o.index}) for o in decision.options
            if not o.is_locked
        ]
        options.append(("leave_room", {}))
        return options

    options = [("select_cards", {"indices": [c.index]}) for c in decision.cards]
    if decision.min_select == 0:
        options.append(("skip_select", {}))
    return options


def legal_options_from_buffer(buf: BufferDecision, decision_kind: str) -> list[tuple[str, dict]]:
    """Buffer counterpart to legal_options() -- same enumeration, read off buf.arr. """
    arr = buf.arr
    if decision_kind == "rest_site":
        options: list[tuple[str, dict]] = [
            ("choose_option", {"option_index": i}) for i, slot in bl.rest_options(arr)
            if int(slot[1]) == 1
        ]
        options.append(("leave_room", {}))
        return options
    if decision_kind == "event_choice":
        options = [
            ("choose_option", {"option_index": i}) for i, is_locked in bl.event_options(arr)
            if not is_locked
        ]
        options.append(("leave_room", {}))
        return options
    if decision_kind == "card_reward":
        if buf.decision == "event_card_reward":
            options: list[tuple[str, dict]] = [
                ("select_card_reward", {"card_index": i}) for i, _ in bl.event_card_reward_cards(arr)
            ]
            # Always skippable -- WriteEventCardReward hardcodes can_skip=true on the JSON side
            # unconditionally (an engine invariant, not per-instance state), and the buffer's own
            # EVENT_CARD_REWARD section has no can_skip flag to read at all.
            options.append(("skip_card_reward", {}))
            return options
        options = [
            ("select_card_reward", {"card_index": i}) for i, _ in bl.reward_cards(arr)
        ]
        if bl.can_skip_reward(arr):
            options.append(("skip_card_reward", {}))
        return options

    options = [("select_cards", {"indices": [i]}) for i, _ in bl.select_cards(arr)]
    if bl.min_select(arr) == 0:
        options.append(("skip_select", {}))
    return options


def load_state_dict_allowing_growth(net: nn.Module, state_dict: dict) -> None:
    """Loads a checkpoint saved before TOKEN_TYPES/DECISION_KINDS grew (e.g. rest_site added to
    an already-training-in-production card_reward-only model, adding REST_OPTION to TOKEN_TYPES
    and rest_site to DECISION_KINDS). A brand-new feat_proj.<TYPE> entry is simply absent from an
    old checkpoint -- strict=False already tolerates that as a missing key, leaving it at its
    fresh random init. type_emb.weight/mode_emb.weight are different: they're still PRESENT keys,
    just with fewer rows than this process's TOKEN_TYPES/DECISION_KINDS now expects, which a plain
    load_state_dict (or a bare strict=False) hard-fails on as a shape mismatch rather than
    tolerating. This preserves every old row in place and leaves new ones at the net's own random
    init, so growing either list never discards a live training run's accumulated weights."""
    own = net.state_dict()
    to_load = dict(state_dict)
    for key in ("type_emb.weight", "mode_emb.weight"):
        saved = to_load.get(key)
        if saved is None or saved.shape == own[key].shape:
            continue
        n = min(saved.shape[0], own[key].shape[0])
        with torch.no_grad():
            own[key][:n] = saved[:n]
        del to_load[key]
    net.load_state_dict(to_load, strict=False)


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

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class ChoicesParadoxNet(nn.Module):
    @classmethod
    def from_checkpoint(cls, path: str, **kwargs) -> "ChoicesParadoxNet":
        """Loads a train.py-saved state_dict against a freshly loaded vocab."""
        net = cls(GlobalVocab(EntityVocab.load()), **kwargs)
        load_state_dict_allowing_growth(net, torch.load(path, weights_only=True))
        net.eval()
        return net

    @classmethod
    def warm_started_from(cls, claws_checkpoint_path: str | None, **kwargs) -> "ChoicesParadoxNet":
        """Copies ClawsNet's own trained card-identity embedding rows into a freshly
        initialized net."""
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

    def encode_batched(
        self, type_ids: torch.Tensor, global_ids: torch.Tensor, features: torch.Tensor,
        pad_mask: torch.Tensor, mode_idx: torch.Tensor, mode_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Batched, padded counterpart to encode() -- for training only (see train.py's
        _collate/_batch_step); live decisions still go through the single-example encode()
        above via forward(). type_ids/global_ids: [B, N] long, features: [B, N, FEATURE_DIM]
        float, pad_mask: [B, N] bool, True marking a padded position -- passed straight through
        as MultiheadAttention's key_padding_mask so padded rows are never attended to.
        mode_idx/mode_pos: [B] long -- each example's DECISION_KINDS index and the token row to
        add it to, mirroring encode()'s single-example mode_emb add. Returns [B, N, dim];
        padded rows' own outputs are never read (every real position index recorded by
        tokenize_from_buffer is < that example's own token count) so their content doesn't
        matter."""
        tok = self.id_emb(global_ids) + self.type_emb(type_ids)
        for t in TOKEN_TYPES:
            mask = (type_ids == _TYPE_TO_IDX[t]) & ~pad_mask
            if mask.any():
                tok = tok.clone()
                tok[mask] = tok[mask] + self.feat_proj[t](features[mask])
        b_idx = torch.arange(tok.shape[0])
        tok = tok.clone()
        tok[b_idx, mode_pos] = tok[b_idx, mode_pos] + self.mode_emb(mode_idx)
        x = tok
        for block in self.blocks:
            x = block(x, key_padding_mask=pad_mask)
        return self.final_ln(x)

    def score(
        self, batch: TokenBatch, options: list[tuple[str, dict]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared by forward() (live decision) and train.py's loop reading materialize.py's
        saved batches. Returns (logits [len(options)], value scalar); logits unnormalized."""
        h = self.encode(batch)
        queries = []
        for action, args in options:
            if action in ("skip_card_reward", "skip_select", "leave_room"):
                q = self.skip_query + h[batch.player_pos]
            elif action == "select_card_reward":
                q = h[batch.offer_pos[args["card_index"]]]
            elif action == "select_cards":
                q = h[batch.offer_pos[args["indices"][0]]]
            elif action == "choose_option":
                q = h[batch.offer_pos[args["option_index"]]]
            else:
                raise ValueError(f"unhandled meta action {action!r}")
            queries.append(q)

        logits = self.policy_mlp(torch.stack(queries)).squeeze(-1)
        # No tanh here: facility_choices_paradox.py's option-chain
        # value is a sum of _WIN_BONUS + _combat_value over every combat won along the whole
        # chain,(observed training targets: ~4 to ~14+).
        value = self.value_mlp(h[batch.player_pos]).squeeze(-1)
        return logits, value

    def forward(
        self,
        decision: CardReward | CardSelect | RestSite | EventChoice | BufferDecision,
        decision_kind: str,
    ) -> tuple[list[tuple[str, dict]], torch.Tensor, torch.Tensor]:
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

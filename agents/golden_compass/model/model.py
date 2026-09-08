"""GoldenCompassNet: typed-token transformer -> policy/value heads, for map_select.

Scores map_select's own physical one-hop `choices` DIRECTLY -- one token, one logit, per real
child coordinate. This replaces an earlier design that scored pooled multi-floor per-room-type
histograms instead (see git history / golden_compass/model/README.md's Design section for why
that was dropped): it discarded exactly the signal that matters most for judging a path -- an
Elite two floors ahead at 90% HP and the same Elite at 20% HP produced the identical token, so the
net had no way to learn the HP-vs-risk tradeoff golden_compass/policy.py's DP heuristic already
hard-codes via _ELITE_HP_FLOOR. Now that map_select decisions are scored via MapOptionJob branch
walks (facility_common.py's _run_map_option_job/_resolve_map_decision) instead of whole-trajectory
tagging, there's no need for this net to enumerate several floors of hypothetical futures itself --
the branch walk *is* the lookahead, played out for real against the live combat/meta policy stack,
so scoring the literal one-hop children directly is both simpler and strictly more informative.

Mirrors choices_paradox/model/model.py's shape: forward() dispatches on isinstance(decision,
BufferDecision) between the pydantic and buffer-protocol tokenizers/option-enumerators, returning
(options, logits, value) with options as (action_name, args) tuples -- same convention
legal_options()/legal_options_from_buffer() use there, so policy.py's callers don't need to know
which decision type they got.

Value head has NO tanh (same reasoning as ChoicesParadoxNet's own): the training target is a
MapOptionJob branch's own terminal state score (_combat_value's hp-fraction-shaped output, or
-1.0 on death within the branch's room horizon -- see facility_common.py's _run_map_option_job),
not a fixed +1/-1. That target rarely strays far outside roughly [-1, 1], but nothing guarantees
it never will, and tanh's asymptotic squash bought nothing here beyond saturated gradients for a
target the earlier design didn't actually keep in range either.
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
import buffer_layout as bl  # noqa: E402
from buffer_env import BufferDecision, decode_token  # noqa: E402
from models import MapSelect  # noqa: E402

# Loading golden_compass/policy.py by path, same importlib shim whispering_earring/policy.py uses
# to reach this same file -- a bare `import policy` would collide with claws'/choices_paradox's
# own sibling-named policy.py modules depending on sys.path order.
import importlib.util  # noqa: E402

_golden_compass_policy_path = Path(__file__).resolve().parents[1] / "policy.py"
_spec = importlib.util.spec_from_file_location("golden_compass_policy", _golden_compass_policy_path)
assert _spec is not None and _spec.loader is not None
golden_compass_policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden_compass_policy)

Coord = tuple[int, int]

# --- token types --------------------------------------------------------------------------------

TOKEN_TYPES = ["PLAYER", "OPTION", "RELIC"]
_TYPE_TO_IDX = {t: i for i, t in enumerate(TOKEN_TYPES)}

# MapPointType strings as returned by MapChoice.type / buffer_layout.map_point_type -- same set
# golden_compass/policy.py's own _NODE_VALUE heuristic keys off.
NODE_TYPES = ["Monster", "Elite", "RestSite", "Shop", "Unknown", "Treasure", "Ancient", "Boss"]
_NODE_TYPE_TO_IDX = {t: i for i, t in enumerate(NODE_TYPES)}
FEATURE_DIM = len(NODE_TYPES)


def _log1p(x: float | int | None) -> float:
    """Signed log1p -- some of what feeds this can go negative (e.g. a relic counter reset)."""
    x = float(x or 0)
    return math.copysign(math.log1p(abs(x)), x)


def _type_feature(node_type: str | None) -> list[float]:
    """One node's own type as a one-hot-ish FEATURE_DIM vector (log1p(1) at its own slot, 0
    elsewhere) -- same encoding the old design used for a "just this node" singleton combination,
    now the ONLY encoding, since every OPTION token is exactly one physical node."""
    feat = [0.0] * FEATURE_DIM
    if node_type in _NODE_TYPE_TO_IDX:
        feat[_NODE_TYPE_TO_IDX[node_type]] = _log1p(1)
    return feat


# --- global id space -----------------------------------------------------------------------------


class GlobalVocab:
    """Same shape as claws/model/model.py's and choices_paradox/model/model.py's GlobalVocab,
    duplicated rather than cross-imported (see those modules' docstrings for why)."""
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

    def entity(self, category: str, raw_id: str | None) -> int:
        return self.id_for(category, self.vocab.encode_id(category, raw_id))


# --- one-hop option enumeration ------------------------------------------------------------------


def legal_options(decision: MapSelect) -> list[tuple[str, dict]]:
    return [("select_map_node", {"col": c.col, "row": c.row}) for c in decision.choices]


def legal_options_from_buffer(decision: BufferDecision) -> list[tuple[str, dict]]:
    """Buffer counterpart to legal_options() -- see golden_compass/policy.py's
    _current_children_from_buffer for the same enumeration (children of the current position, or
    every row-0 point pre-act-entry, or the (-1, -1) boss sentinel at a childless junction)."""
    children = golden_compass_policy._current_children_from_buffer(decision.arr)
    return [("select_map_node", {"col": c, "row": r}) for c, r in children]


def _node_type_at(col: int, row: int, decision: MapSelect) -> str | None:
    chosen = next((c for c in decision.choices if (c.col, c.row) == (col, row)), None)
    return chosen.type if chosen is not None else None


def _node_type_at_buffer(col: int, row: int, arr) -> str | None:
    slot = bl.map_boss_point(arr) if (col, row) == (-1, -1) else bl.map_point_slot(arr, row, col)
    return bl.map_point_type(slot)


# --- tokenizer -------------------------------------------------------------------------------------


@dataclass
class TokenBatch:
    """Tokens for one map_select decision (batch size 1, same reasoning as claws'/choices_paradox's
    own TokenBatch)."""

    type_ids: torch.Tensor  # [N] long
    global_ids: torch.Tensor  # [N] long
    features: torch.Tensor  # [N, FEATURE_DIM] float
    player_pos: int
    option_pos: dict[int, int] = field(default_factory=dict)  # options-list index -> token row

    @classmethod
    def from_saved(cls, d: dict) -> "TokenBatch":
        """Inverse of materialize.py's per-example dict -- see claws/model/model.py's
        TokenBatch.from_saved for why this can't just be torch.load-ed back directly."""
        return cls(
            type_ids=d["type_ids"], global_ids=d["global_ids"], features=d["features"],
            player_pos=d["player_pos"], option_pos=d["option_pos"],
        )


def _tokenize_common(
    gvocab: GlobalVocab, hp: float, max_hp: float, gold: float, deck_size: float, floor: float,
    relics: list[tuple[str | None, float]], option_types: list[str | None],
) -> TokenBatch:
    """Shared by tokenize()/tokenize_from_buffer() -- both just extract these plain values from
    their own decision shape first."""
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

    player_pos = add(
        "PLAYER",
        EntityVocab.PAD_IDX,  # no id space of its own -- global row 0 is always a zero vector
        [
            hp / max_hp if max_hp else 0.0,
            _log1p(gold),
            _log1p(deck_size),
            floor / 50.0,  # rough run-progress scale, same order of magnitude as
            # sts2-rl-agent's own OBS_TOTAL_FLOOR_SCALE
        ] + [0.0] * (FEATURE_DIM - 4),
    )

    option_pos: dict[int, int] = {}
    for i, node_type in enumerate(option_types):
        option_pos[i] = add("OPTION", EntityVocab.PAD_IDX, _type_feature(node_type))

    for raw_id, counter in relics:
        add(
            "RELIC", gvocab.entity("relic", raw_id),
            [0.0, 0.0, _log1p(counter), 0.0] + [0.0] * (FEATURE_DIM - 4),
        )

    return TokenBatch(
        type_ids=torch.tensor(types, dtype=torch.long),
        global_ids=torch.tensor(ids, dtype=torch.long),
        features=torch.tensor(feats, dtype=torch.float),
        player_pos=player_pos,
        option_pos=option_pos,
    )


def tokenize(
    decision: MapSelect, gvocab: GlobalVocab, options: list[tuple[str, dict]] | None = None,
) -> TokenBatch:
    """`options`, when given (materialize.py replaying a saved trial), must be legal_options()'s
    own output for this same decision -- lets a caller that already computed it once reuse the
    same ordering rather than recomputing here."""
    if options is None:
        options = legal_options(decision)
    option_types = [_node_type_at(a["col"], a["row"], decision) for _, a in options]
    player = decision.player
    return _tokenize_common(
        gvocab, player.hp, player.max_hp, player.gold, player.deck_size, decision.floor,
        [(r.id, r.counter) for r in player.relics], option_types,
    )


def tokenize_from_buffer(
    decision: BufferDecision, gvocab: GlobalVocab, options: list[tuple[str, dict]] | None = None,
) -> TokenBatch:
    arr = decision.arr
    if options is None:
        options = legal_options_from_buffer(decision)
    option_types = [_node_type_at_buffer(a["col"], a["row"], arr) for _, a in options]
    relics = []
    for _, slot in bl.player_relics(arr):
        entry = decode_token(int(slot[bl.RELIC_TOKEN]))
        counter = slot[bl.RELIC_COUNTER]
        relics.append((entry, counter if counter != -1 else None))
    return _tokenize_common(
        gvocab, bl.player_hp(arr), bl.player_max_hp(arr), bl.player_gold(arr),
        bl.player_deck_size(arr), bl.run_context(arr)["floor"], relics, option_types,
    )


# --- network -------------------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """x = x + attn(LN(x)); x = x + ffn(LN(x)) -- pre-norm, same shape as claws/model/model.py's
    and choices_paradox/model/model.py's own TransformerBlock, duplicated rather than
    cross-imported."""

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


class GoldenCompassNet(nn.Module):
    @classmethod
    def from_checkpoint(cls, path: str, **kwargs) -> "GoldenCompassNet":
        """Loads a train.py-saved state_dict against a freshly loaded vocab."""
        net = cls(GlobalVocab(EntityVocab.load()), **kwargs)
        net.load_state_dict(torch.load(path, weights_only=True))
        net.eval()
        return net

    def __init__(self, gvocab: GlobalVocab, dim: int = 64, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.gvocab = gvocab
        self.dim = dim
        self.id_emb = nn.Embedding(gvocab.total_size, dim, padding_idx=EntityVocab.PAD_IDX)
        self.type_emb = nn.Embedding(len(TOKEN_TYPES), dim)
        self.feat_proj = nn.ModuleDict(
            {t: nn.Linear(FEATURE_DIM, dim) for t in TOKEN_TYPES}
        )
        self.blocks = nn.ModuleList([TransformerBlock(dim, n_heads) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(dim)

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
        x = tok.unsqueeze(0)  # MultiheadAttention wants a batch dim
        for block in self.blocks:
            x = block(x)
        return self.final_ln(x).squeeze(0)

    def encode_batched(
        self, type_ids: torch.Tensor, global_ids: torch.Tensor, features: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Batched, padded counterpart to encode() -- for training only (see train.py's
        _collate/_batch_step); live decisions still go through the single-example encode() above
        via forward(). type_ids/global_ids: [B, N] long, features: [B, N, FEATURE_DIM] float,
        pad_mask: [B, N] bool, True marking a padded position -- passed straight through as
        MultiheadAttention's key_padding_mask so padded rows are never attended to. See
        claws/model/model.py's own encode_batched -- identical reasoning, duplicated rather than
        cross-imported."""
        tok = self.id_emb(global_ids) + self.type_emb(type_ids)
        for t in TOKEN_TYPES:
            mask = (type_ids == _TYPE_TO_IDX[t]) & ~pad_mask
            if mask.any():
                tok = tok.clone()
                tok[mask] = tok[mask] + self.feat_proj[t](features[mask])
        x = tok
        for block in self.blocks:
            x = block(x, key_padding_mask=pad_mask)
        return self.final_ln(x)

    def score(self, batch: TokenBatch, n_options: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared by forward() (live decision) and train.py's loop reading materialize.py's
        saved batches. Returns (logits [n_options], value scalar); logits unnormalized."""
        h = self.encode(batch)
        queries = [h[batch.option_pos[i]] for i in range(n_options)]
        logits = self.policy_mlp(torch.stack(queries)).squeeze(-1)
        value = self.value_mlp(h[batch.player_pos]).squeeze(-1)  # no tanh -- see module docstring
        return logits, value

    def forward(
        self, decision: MapSelect | BufferDecision,
    ) -> tuple[list[tuple[str, dict]], torch.Tensor, torch.Tensor]:
        """Returns (options, logits, value) -- options/logits are index-aligned (softmax index i
        <-> options[i]), each option a plain (action_name, args) tuple ready to hand to
        session.step(), same convention ChoicesParadoxNet.forward uses."""
        if isinstance(decision, BufferDecision):
            options = legal_options_from_buffer(decision)
            batch = tokenize_from_buffer(decision, self.gvocab, options=options)
        else:
            options = legal_options(decision)
            batch = tokenize(decision, self.gvocab, options=options)
        logits, value = self.score(batch, len(options))
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
    decision = None
    for line in lines:
        raw = json.loads(line)
        if raw.get("decision") == "map_select":
            decision = MapSelect.model_validate(raw)
            break
    if decision is None:
        raise SystemExit("no map_select decision found in fixture")

    vocab = EntityVocab.load()
    gvocab = GlobalVocab(vocab)
    net = GoldenCompassNet(gvocab)

    options, logits, value = net(decision)
    probs = torch.softmax(logits, dim=-1)

    print(f"{len(options)} options, value={value.item():+.3f}")
    for (name, args), p in itertools.islice(zip(options, probs.tolist()), 8):
        print(f"  {p:5.3f}  {name} {args}")

"""GoldenCompassNet: typed-token transformer -> policy/value heads, for map_select.

This net does NOT score "which physical node to walk into next." It scores COMBINATIONS: for
every path reachable within _LOOKAHEAD_HORIZON floors of the current position -- pooled across
ALL of map_select's one-hop `choices`, not kept separate per branch -- compute that path's own
per-room-type count histogram (_path_combination), then group paths that produce the identical
histogram into one candidate (combination_options). Two paths through completely different
physical nodes that happen to pass the same rooms (2 Elites + a Shop, say) are literally the same
candidate to this net; a candidate with a unique histogram is its own singleton group. The network
scores each DISTINCT combination once (one token, one logit) -- how many physical paths happen to
share it is invisible to it by design.

Once a combination is picked (sampled from the softmax, at decision time), it has to be resolved
back to a real engine action: pick any one of the paths that produced it (they're interchangeable
to the net -- see resolve_combination_action) and walk ONLY that path's first hop. The rest of the
path is thrown away on purpose: this is re-planned from scratch at every single map_select, not
committed to upfront (facility_common.py's main_worker_loop already calls into this fresh at every
map_select on its own, so "walk one hop, replan" falls out of the existing control flow for free
-- nothing there needs to change). That also means WHICH combination was actually chosen has to be
recorded explicitly (see _choose_map_action_net's policy["chosen"] in whispering_earring/policy.py)
-- the resolved (col, row) alone doesn't disambiguate it, since more than one combination can
share a first hop.



Value head IS tanh-bounded, this net's training target is the run's own
terminal reward (+1 win / -1 death, plus a local Elite/Boss-win shaping bonus -- see
facility_common.py's pending_map_steps / recorder.set_value wiring), not a growing per-chain sum.
"""
from __future__ import annotations

import math
import os
import random
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
from env import Sts2Session  # noqa: E402
from models import MapSelect  # noqa: E402

Coord = tuple[int, int]
Combination = tuple[float, ...]

# --- token types --------------------------------------------------------------------------------

TOKEN_TYPES = ["PLAYER", "OPTION", "RELIC"]
_TYPE_TO_IDX = {t: i for i, t in enumerate(TOKEN_TYPES)}

# MapPointType strings as returned by MapChoice.type -- same set golden_compass/policy.py's own
# _NODE_VALUE heuristic keys off.
NODE_TYPES = ["Monster", "Elite", "RestSite", "Shop", "Unknown", "Treasure", "Ancient", "Boss"]
FEATURE_DIM = len(NODE_TYPES)
_LOOKAHEAD_HORIZON = 5


def _log1p(x: float | int | None) -> float:
    """Signed log1p -- some of what feeds this can go negative (e.g. a relic counter reset)."""
    x = float(x or 0)
    return math.copysign(math.log1p(abs(x)), x)


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


# --- map lookahead: path enumeration + combination pooling ---------------------------------------


def _map_node_index(map_json: dict) -> dict[Coord, dict]:
    """(col, row) -> node dict. Same `map` shape golden_compass/policy.py's own DP heuristic
    already walks (RunSimulator.GetFullMap) -- see that module's docstring. `boss` is reported as
    its own top-level key, not one of `rows`' own entries (see that module's _best_path_values) --
    folded in here too, so a path reaching it is walked/counted like any other room."""
    index = {(n["col"], n["row"]): n for row in map_json["rows"] for n in row}
    boss = map_json.get("boss")
    if boss is not None:
        index[(boss["col"], boss["row"])] = boss
    return index


def _enumerate_paths(
    coord: Coord, node_index: dict[Coord, dict], horizon: int = _LOOKAHEAD_HORIZON,
) -> list[list[Coord]]:
    """Every coordinate path starting at `coord`, walking the map's own `children` adjacency, up
    to `horizon` floors deep -- terminating early wherever a path dead-ends (Boss, or no
    children) before that."""
    node = node_index.get(coord)
    if node is None:
        return [[coord]]
    children = node.get("children") or []
    if node.get("type") == "Boss" or not children or horizon <= 0:
        return [[coord]]
    paths: list[list[Coord]] = []
    for child in children:
        c = (child["col"], child["row"])
        for rest in _enumerate_paths(c, node_index, horizon - 1):
            paths.append([coord] + rest)
    return paths


def _path_combination(path: list[Coord], node_index: dict[Coord, dict]) -> Combination:
    """log1p-compressed per-type room count along one path."""
    counts = {t: 0 for t in NODE_TYPES}
    for c in path:
        node = node_index.get(c)
        if node is not None and node.get("type") in counts:
            counts[node["type"]] += 1
    return tuple(_log1p(counts[t]) for t in NODE_TYPES)


def combination_options(
    decision: MapSelect, node_index: dict[Coord, dict] | None, horizon: int = _LOOKAHEAD_HORIZON,
) -> dict[Combination, list[list[Coord]]]:
    """combination -> [paths that produce it], pooled across EVERY one-hop `choices` candidate's
    own subtree -- this dict IS the network's action space: the
    ordered list of its keys is what tokenize()/GoldenCompassNet score, one token per distinct
    combination, and the dict itself is how a chosen combination resolves back to a real action
    (see resolve_combination_action)."""
    combos: dict[Combination, list[list[Coord]]] = {}
    for choice in decision.choices:
        coord = (choice.col, choice.row)
        if node_index is not None:
            paths = _enumerate_paths(coord, node_index, horizon)
            for path in paths:
                combo = _path_combination(path, node_index)
                combos.setdefault(combo, []).append(path)
        else:
            counts = {t: 0 for t in NODE_TYPES}
            if choice.type in counts:
                counts[choice.type] += 1
            combo = tuple(_log1p(counts[t]) for t in NODE_TYPES)
            combos.setdefault(combo, []).append([coord])
    return combos


def resolve_combination_action(
    combo: Combination, combo_paths: dict[Combination, list[list[Coord]]],
    rng: random.Random | None = None,
) -> tuple[str, dict]:
    """Any path sharing the chosen combination is equally valid to this net by construction"""
    paths = combo_paths[combo]
    path = (rng or random).choice(paths)
    coord = path[0]
    return "select_map_node", {"col": coord[0], "row": coord[1]}


# --- tokenizer -------------------------------------------------------------------------------------


@dataclass
class TokenBatch:
    """Tokens for one map_select decision (batch size 1, same reasoning as claws'/choices_paradox's
    own TokenBatch)."""

    type_ids: torch.Tensor  # [N] long
    global_ids: torch.Tensor  # [N] long
    features: torch.Tensor  # [N, FEATURE_DIM] float
    player_pos: int
    option_pos: dict[int, int] = field(default_factory=dict)  # combos-list index -> token row

    @classmethod
    def from_saved(cls, d: dict) -> "TokenBatch":
        """Inverse of materialize.py's per-example dict -- see claws/model/model.py's
        TokenBatch.from_saved for why this can't just be torch.load-ed back directly."""
        return cls(
            type_ids=d["type_ids"], global_ids=d["global_ids"], features=d["features"],
            player_pos=d["player_pos"], option_pos=d["option_pos"],
        )


def tokenize(
    decision: MapSelect, gvocab: GlobalVocab, session: Sts2Session | None = None,
    combos: list[Combination] | None = None,
) -> TokenBatch:
    """`session`, when given (and `combos` isn't already supplied), is used for one `get_map()`
    call (cheap -- see env.py's own _map_cache) to build combination_options(); without it, a
    candidate degrades to a singleton "just this node's own type" combination (see
    combination_options' own docstring) -- so callers with no live session (e.g. this module's own
    __main__ smoke test, replaying an isolated fixture) still get a valid batch.

    `combos` lets a caller that already computed combination_options() (materialize.py, replaying
    a saved trial and needing the SAME ordering to recover which index was actually chosen) pass
    its ordered keys straight in instead of recomputing them here.

    PLAYER's and RELIC's own feature vectors are shorter than FEATURE_DIM (which is sized for
    OPTION's own per-type-count histogram, see that constant's comment) -- zero-padded in the
    unused trailing slots, same "pad rather than resize per type" convention `feat_proj`
    (GoldenCompassNet.encode) already assumes by sharing one Linear width across all three token
    types."""
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
            _log1p(player.gold),
            _log1p(player.deck_size),
            decision.floor / 50.0,  # rough run-progress scale, same order of magnitude as
            # sts2-rl-agent's own OBS_TOTAL_FLOOR_SCALE
        ] + [0.0] * (FEATURE_DIM - 4),
    )

    if combos is None:
        node_index = _map_node_index(session.get_map()) if session is not None else None
        combos = list(combination_options(decision, node_index).keys())

    option_pos: dict[int, int] = {}
    for i, combo in enumerate(combos):
        option_pos[i] = add("OPTION", EntityVocab.PAD_IDX, list(combo))

    for relic in player.relics:
        add(
            "RELIC", gvocab.entity("relic", relic.id),
            [0.0, 0.0, _log1p(relic.counter), 0.0] + [0.0] * (FEATURE_DIM - 4),
        )

    return TokenBatch(
        type_ids=torch.tensor(types, dtype=torch.long),
        global_ids=torch.tensor(ids, dtype=torch.long),
        features=torch.tensor(feats, dtype=torch.float),
        player_pos=player_pos,
        option_pos=option_pos,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
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

    def score(self, batch: TokenBatch, n_options: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared by forward() (live decision) and train.py's loop reading materialize.py's
        saved batches. Returns (logits [n_options], value scalar); logits unnormalized."""
        h = self.encode(batch)
        queries = [h[batch.option_pos[i]] for i in range(n_options)]
        logits = self.policy_mlp(torch.stack(queries)).squeeze(-1)
        # tanh-bounded: the training target is the run's own terminal +1/-1 reward, not an
        # unbounded sum -- see this module's docstring.
        value = torch.tanh(self.value_mlp(h[batch.player_pos]).squeeze(-1))
        return logits, value

    def forward(
        self, decision: MapSelect, session: Sts2Session | None = None,
    ) -> tuple[list[Combination], torch.Tensor, torch.Tensor, dict[Combination, list[list[Coord]]]]:
        """Returns (combinations, logits, value, combo_paths) -- combinations/logits are index-
        aligned (softmax index i <-> combinations[i]); combo_paths resolves a chosen combination
        back to a real action (see resolve_combination_action). `session` is passed straight
        through to combination_options() -- see tokenize()'s own docstring for why it's optional
        here too."""
        node_index = _map_node_index(session.get_map()) if session is not None else None
        combo_paths = combination_options(decision, node_index)
        combos = list(combo_paths.keys())
        batch = tokenize(decision, self.gvocab, combos=combos)
        logits, value = self.score(batch, len(combos))
        return combos, logits, value, combo_paths


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

    combos, logits, value, combo_paths = net(decision)
    probs = torch.softmax(logits, dim=-1)

    print(f"{len(combos)} distinct combinations, value={value.item():+.3f}")
    for combo, p in itertools.islice(zip(combos, probs.tolist()), 8):
        readable = {t: c for t, c in zip(NODE_TYPES, combo) if c > 0}
        print(f"  {p:5.3f}  {readable}  ({len(combo_paths[combo])} path(s))")

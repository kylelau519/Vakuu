"""The network described in README.md -- tokenizer -> typed-token
transformer -> policy/value heads. Exists to prove the shapes described in the README actually
compose (a `CombatPlay` decision in, a policy over legal options + a scalar value out), not as a
trained or tuned model. Run directly (`python model.py`, `sts2` conda env) for a smoke test against
a real captured decision from schema/tests/fixtures/ironclad_run_01.jsonl.

"""
from __future__ import annotations

import itertools
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

import sts2_path  # noqa: E402,F401  (sets sys.path to sts2-cli/python before the import below)
from models import CombatPlay  # noqa: E402  (sts2-cli's python/models.py -- see ../README.md)
import buffer_layout as bl  # noqa: E402  (sts2-cli's python/buffer_layout.py)
from buffer_env import BufferDecision, decode_token  # noqa: E402

# --- token types ------------------------------------------------------------------------------

TOKEN_TYPES = ["PLAYER", "CARD", "ENEMY", "RELIC", "POTION", "PLAYER_POWER"]
_TYPE_TO_IDX = {t: i for i, t in enumerate(TOKEN_TYPES)}

FEATURE_DIM = 6


def _log1p(x: float | int | None) -> float:
    """Signed log1p,  -- some of what feeds this can go negative (Strength and
    other player_powers can drop below 0 from debuffs/curses) """
    x = float(x or 0)
    return math.copysign(math.log1p(abs(x)), x)


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
    hand_pos: dict[int, int] = field(default_factory=dict)  # HandCard.index -> token row
    enemy_pos: dict[int, int] = field(default_factory=dict)  # Enemy.index -> token row
    potion_pos: dict[int, int] = field(default_factory=dict)  # Potion.index -> token row

    @classmethod
    def from_saved(cls, d: dict) -> "TokenBatch":
        """Inverse of materialize.py's per-example dict -- materialize.py saves plain dicts
        rather than TokenBatch instances (see its module docstring: pickling a dataclass by
        import path breaks across scripts), so a training loop reads examples back through
        this instead of torch.load handing back TokenBatch objects directly."""
        return cls(
            type_ids=d["type_ids"], global_ids=d["global_ids"], features=d["features"],
            player_pos=d["player_pos"], hand_pos=d["hand_pos"], enemy_pos=d["enemy_pos"],
            potion_pos=d["potion_pos"],
        )


def tokenize(decision: CombatPlay, gvocab: GlobalVocab) -> TokenBatch:
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
                               # (id_emb is built with padding_idx=EntityVocab.PAD_IDX=0)
        [
            player.hp / player.max_hp if player.max_hp else 0.0,
            _log1p(player.hp),
            _log1p(player.block),
            0.0,
            0.0,
            decision.energy / decision.max_energy if decision.max_energy else 0.0,
        ],
    )

    hand_pos: dict[int, int] = {}
    for card in decision.hand:
        pos = add(
            "CARD",
            gvocab.card(card.id),
            [0.0, 0.0, 0.0, float(card.cost), float(card.can_play), float(bool(card.upgraded))],
        )
        hand_pos[card.index] = pos

    enemy_pos: dict[int, int] = {}
    for enemy in decision.enemies:
        intent_damage = 0
        if enemy.intents:
            intent_damage = enemy.intents[0].total_damage or enemy.intents[0].damage or 0
        pos = add(
            "ENEMY",
            gvocab.entity("enemy", enemy.id),
            [
                enemy.hp / enemy.max_hp if enemy.max_hp else 0.0,
                _log1p(enemy.hp),
                _log1p(enemy.block),
                _log1p(intent_damage),
                float(enemy.intends_attack),
                0.0,
            ],
        )
        enemy_pos[enemy.index] = pos

    for relic in player.relics:
        add(
            "RELIC",
            gvocab.entity("relic", relic.id),
            [0.0, 0.0, 0.0, _log1p(relic.counter), 0.0, 0.0],
        )

    potion_pos: dict[int, int] = {}
    for potion in player.potions:
        pos = add("POTION", gvocab.entity("potion", potion.id), [0.0] * FEATURE_DIM)
        potion_pos[potion.index] = pos

    for power in decision.player_powers or []:
        add(
            "PLAYER_POWER",
            gvocab.entity("power", power.id),
            [0.0, 0.0, 0.0, _log1p(power.amount), 0.0, 0.0],
        )

    return TokenBatch(
        type_ids=torch.tensor(types, dtype=torch.long),
        global_ids=torch.tensor(ids, dtype=torch.long),
        features=torch.tensor(feats, dtype=torch.float),
        player_pos=player_pos,
        hand_pos=hand_pos,
        enemy_pos=enemy_pos,
        potion_pos=potion_pos,
    )


def tokenize_from_buffer(buf: BufferDecision, gvocab: GlobalVocab) -> TokenBatch:
    """Buffer counterpart to tokenize() -- same token layout and feature math, read directly off
    buf.arr via buffer_layout's offset accessors instead of a pydantic CombatPlay's attribute
    tree."""
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
    en, max_en = bl.energy(arr), bl.max_energy(arr)
    player_pos = add(
        "PLAYER",
        EntityVocab.PAD_IDX,
        [
            hp / max_hp if max_hp else 0.0,
            _log1p(hp),
            _log1p(bl.player_block(arr)),
            0.0,
            0.0,
            en / max_en if max_en else 0.0,
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
                _log1p(eh),
                _log1p(bl.enemy_block(slot)),
                _log1p(intent_damage),
                float(bl.enemy_intends_attack(slot)),
                0.0,
            ],
        )
        enemy_pos[idx] = pos

    for _, slot in bl.player_relics(arr):
        entry = decode_token(int(slot[bl.RELIC_TOKEN]))
        counter = slot[bl.RELIC_COUNTER]
        add(
            "RELIC", gvocab.entity("relic", entry),
            [0.0, 0.0, 0.0, _log1p(counter if counter != -1 else None), 0.0, 0.0],
        )

    potion_pos: dict[int, int] = {}
    for idx, slot in bl.player_potions(arr):
        entry = decode_token(int(slot[bl.POTION_TOKEN]))
        pos = add("POTION", gvocab.entity("potion", entry), [0.0] * FEATURE_DIM)
        potion_pos[idx] = pos

    for _, slot in bl.player_powers(arr):
        entry = decode_token(int(slot[0]))
        add("PLAYER_POWER", gvocab.entity("power", entry), [0.0, 0.0, 0.0, _log1p(slot[1]), 0.0, 0.0])

    return TokenBatch(
        type_ids=torch.tensor(types, dtype=torch.long),
        global_ids=torch.tensor(ids, dtype=torch.long),
        features=torch.tensor(feats, dtype=torch.float),
        player_pos=player_pos,
        hand_pos=hand_pos,
        enemy_pos=enemy_pos,
        potion_pos=potion_pos,
    )


# --- legal options (the policy head's action space) -----------------------------------------------


def legal_options(decision: CombatPlay) -> list[tuple[str, dict]]:
    """Every (action, args) the policy head scores -- one entry per legal target, unlike
    whispering_earring/policy.py's single-random-target placeholder (see module docstring)."""
    options: list[tuple[str, dict]] = [("end_turn", {})]

    for card in decision.hand:
        if not card.can_play or card.cost > decision.energy:
            continue
        if card.target_type == "AnyEnemy" and len(decision.enemies) >= 2:
            for enemy in decision.enemies:
                options.append(("play_card", {"card_index": card.index, "target_index": enemy.index}))
        else:
            options.append(("play_card", {"card_index": card.index}))

    for potion in decision.player.potions:
        if potion.target_type == "AnyEnemy" and len(decision.enemies) >= 2:
            for enemy in decision.enemies:
                options.append(("use_potion", {"potion_index": potion.index, "target_index": enemy.index}))
        else:
            options.append(("use_potion", {"potion_index": potion.index}))

    return options


def legal_options_from_buffer(buf: BufferDecision) -> list[tuple[str, dict]]:
    """Buffer counterpart to legal_options() -- same enumeration, read off buf.arr. Needs
    hand_target_type() (see buffer_layout.py) to tell AnyEnemy cards (fan out per enemy target)
    apart from AllEnemies/Self/None cards (single no-target option)."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class ClawsNet(nn.Module):
    @classmethod
    def from_checkpoint(cls, path: str, **kwargs) -> "ClawsNet":
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

        self.end_turn_query = nn.Parameter(torch.randn(dim) * 0.02)
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

    def score(
        self, batch: TokenBatch, options: list[tuple[str, dict]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared by forward() (live decision) and a training loop reading materialize.py's
        saved batches -- both need the same option-query + head logic, just from a differently
        sourced TokenBatch. Returns (logits [len(options)], value scalar); logits unnormalized,
        softmax them for a distribution comparable to a trial file's `policy` list."""
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
        value = torch.tanh(self.value_mlp(h[batch.player_pos]).squeeze(-1))
        return logits, value

    def forward(
        self, decision: CombatPlay | BufferDecision
    ) -> tuple[list[tuple[str, dict]], torch.Tensor, torch.Tensor]:
        """Returns (options, logits [len(options)], value scalar) for one live decision. Accepts
        either shape decision can arrive in: a pydantic CombatPlay (the main self-play line,
        materialize.py's saved batches) or a buffer_env.BufferDecision (whispering_earring's
        branch-worker/MCTS hot path -- see tokenize_from_buffer's docstring for why that path
        avoids the pydantic one)."""
        if isinstance(decision, BufferDecision):
            batch = tokenize_from_buffer(decision, self.gvocab)
            options = legal_options_from_buffer(decision)
        else:
            batch = tokenize(decision, self.gvocab)
            options = legal_options(decision)
        logits, value = self.score(batch, options)
        return options, logits, value


# --- smoke test ------------------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    fixture = (
        Path(__file__).resolve().parents[3]
        / "schema" / "tests" / "fixtures" / "ironclad_run_01.jsonl"
    )
    lines = fixture.read_text().splitlines()
    raw = json.loads(lines[63])  # a 2-enemy, 1-potion combat_play -- exercises target branching
    decision = CombatPlay.model_validate(raw)

    vocab = EntityVocab.load()
    gvocab = GlobalVocab(vocab)
    net = ClawsNet(gvocab)

    options, logits, value = net(decision)
    probs = torch.softmax(logits, dim=-1)

    print(f"{len(options)} legal options, value={value.item():+.3f}")
    for (action, args), p in itertools.islice(zip(options, probs.tolist()), 8):
        print(f"  {p:5.3f}  {action} {args}")

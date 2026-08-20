"""Untrained skeleton of the network described in README.md -- tokenizer -> typed-token
transformer -> policy/value heads. Exists to prove the shapes described in the README actually
compose (a `CombatPlay` decision in, a policy over legal options + a scalar value out), not as a
trained or tuned model. Run directly (`python model.py`, `sts2` conda env) for a smoke test against
a real captured decision from schema/tests/fixtures/ironclad_run_01.jsonl.

Scope deliberately cut from this v0 to keep the skeleton small: draw/discard/exhaust pile cards,
`history` scaling counters, enemy `powers`, orbs/stars/osty. None of those are wired into tokens or
features yet -- add a token type + feature block for each the same way CARD/ENEMY/RELIC/POTION/
PLAYER_POWER are done below when they're needed.

Legal-option enumeration (`legal_options`) is the model's own action space, not a copy of
whispering_earring/policy.py's -- that module's `_choose_combat_action` only ever samples one
random candidate target per AnyEnemy card (see its docstring), since it's a placeholder for
generating *some* self-play data, not a definition of the action space. This module enumerates
every (card, target) and (potion, target) pair, matching what an MCTS expansion would actually
need to score as separate children (see ../README.md's MCTS strategy section). The two will only
ever produce different-shaped option lists for multi-target decisions -- expected, not a bug to
reconcile, until MCTS replaces the placeholder policy.
"""
from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # for `common`
from common.tokenizer import EntityVocab  # noqa: E402

import sts2_path  # noqa: E402,F401  (sets sys.path to sts2-cli/python before the import below)
from models import CombatPlay  # noqa: E402  (sts2-cli's python/models.py -- see ../README.md)

# --- token types ------------------------------------------------------------------------------

TOKEN_TYPES = ["PLAYER", "CARD", "ENEMY", "RELIC", "POTION", "PLAYER_POWER"]
_TYPE_TO_IDX = {t: i for i, t in enumerate(TOKEN_TYPES)}

# Every token gets the same fixed-width feature vector, zero-padded where a type doesn't use a
# given slot -- simplest way to let one nn.Linear-per-type (feat_proj) share input shape without
# a ragged per-type schema. Slots, by index: [hp_frac, log1p_hp, log1p_block, cost_or_amount,
# can_play_or_intends_attack, upgraded_or_energy_frac].
FEATURE_DIM = 6


def _log1p(x: float | int | None) -> float:
    return float(torch.log1p(torch.tensor(float(x or 0))))


# --- global id space ---------------------------------------------------------------------------


class GlobalVocab:
    """EntityVocab (common/tokenizer.py) indexes each category from 2 independently -- card id 5
    and potion id 5 are unrelated entities that collide if fed straight into one shared embedding
    table. README.md's `self.emb[global_id]` wants one table, so this offsets each category by
    the cumulative size of the categories before it, giving every (category, local_id) pair a
    unique row. Wrapping EntityVocab rather than changing it -- Choices Paradox/Golden Compass
    still want the plain per-category indices for their own (non-shared-embedding) use."""

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
    batching is deferred). Index maps point from a schema field's own `index` (HandCard.index,
    Enemy.index, Potion.index) to that entity's row in the token tensors, since those indices
    aren't dense/0-based-contiguous once options are filtered (see legal_options)."""

    type_ids: torch.Tensor  # [N] long
    global_ids: torch.Tensor  # [N] long
    features: torch.Tensor  # [N, FEATURE_DIM] float
    player_pos: int
    hand_pos: dict[int, int] = field(default_factory=dict)  # HandCard.index -> token row
    enemy_pos: dict[int, int] = field(default_factory=dict)  # Enemy.index -> token row
    potion_pos: dict[int, int] = field(default_factory=dict)  # Potion.index -> token row


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
        """Returns per-token hidden states [N, dim] for one decision (no batch dim -- MCTS calls
        this once per leaf node with a single freshly-tokenized state, so there's no natural
        batch axis to pad token counts against; revisit if training throughput needs batching)."""
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

    def forward(self, decision: CombatPlay) -> tuple[list[tuple[str, dict]], torch.Tensor, torch.Tensor]:
        """Returns (options, logits [len(options)], value scalar). logits are unnormalized --
        softmax over them for a policy distribution comparable to a trial file's `policy` list."""
        batch = tokenize(decision, self.gvocab)
        h = self.encode(batch)

        options = legal_options(decision)
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

"""One nn.Embedding per entity category, keyed by the same EntityVocab all three subsystems
share -- so a card/relic/power looked up from Choices Paradox, Golden Compass, or Whispering
Earring lands on the same row index, whether or not the tables' weights end up tied or
pretrained jointly later.

Usage:
    from vocab import EntityVocab
    from embeddings import EntityEmbeddings

    vocab = EntityVocab.load()
    embed = EntityEmbeddings(vocab, dim=32)
    card_vecs = embed("card", torch.tensor([vocab.encode_card(c.id) for c in decision.hand]))
"""
from __future__ import annotations

import torch
from torch import nn

from agents.common.tokenizer import EntityVocab


class EntityEmbeddings(nn.Module):
    def __init__(self, vocab: EntityVocab, dim: int = 32):
        super().__init__()
        self.vocab = vocab
        self.dim = dim
        self.tables = nn.ModuleDict(
            {
                category: nn.Embedding(vocab.size(category), dim, padding_idx=EntityVocab.PAD_IDX)
                for category in vocab.categories
            }
        )

    def forward(self, category: str, ids: torch.Tensor) -> torch.Tensor:
        return self.tables[category](ids)

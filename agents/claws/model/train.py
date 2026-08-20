"""Trains ClawsNet on materialize.py's output.

Batch size 1, no padding: each example has a different token count and option count
(materialize.py's output isn't rectangular), the same constraint ClawsNet.encode/score already
assume -- see their docstrings, MCTS calls them one leaf at a time too. Revisit if batch-size-1
throughput actually becomes a bottleneck; padding/masking to batch examples is a real complexity
jump not worth taking preemptively.

Loss = policy cross-entropy against the recorded (possibly sparse -- see materialize.py's
_target_probs) distribution, plus value MSE, summed with equal weight -- no tuning of that
weighting has been done, it's just the obvious starting point.

Usage:
    conda activate sts2
    python train.py dataset.pt --epochs 5 --out claws.pt
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # for `common`
from common.tokenizer import EntityVocab  # noqa: E402

from model import ClawsNet, GlobalVocab, TokenBatch  # noqa: E402


def loss_fn(
    logits: torch.Tensor, value: torch.Tensor, target_policy: torch.Tensor, target_value: float
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(logits, dim=-1)
    policy_loss = -(target_policy * log_probs).sum()
    value_loss = (value - target_value) ** 2
    return policy_loss, value_loss


def train(examples: list[dict], net: ClawsNet, epochs: int, lr: float, seed: int = 0) -> None:
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = random.Random(seed)

    for epoch in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        total_policy = total_value = 0.0

        for i in order:
            ex = examples[i]
            batch = TokenBatch.from_saved(ex)
            logits, value = net.score(batch, ex["options"])
            policy_loss, value_loss = loss_fn(logits, value, ex["target"], ex["value"])
            loss = policy_loss + value_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_policy += policy_loss.item()
            total_value += value_loss.item()

        n = len(examples)
        print(f"epoch {epoch + 1}/{epochs}: policy_loss={total_policy / n:.4f} "
              f"value_loss={total_value / n:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*",
                         help="materialize.py output(s), e.g. dataset.pt "
                              "(omit with --epochs 0 to just save an untrained init checkpoint)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="claws.pt")
    parser.add_argument("--init", default=None, help="resume from an existing checkpoint")
    args = parser.parse_args()

    examples: list[dict] = []
    for path in args.datasets:
        examples.extend(torch.load(path, weights_only=False))
    print(f"{len(examples)} training examples from {len(args.datasets)} dataset file(s)")

    net = ClawsNet(GlobalVocab(EntityVocab.load()))
    if args.init:
        net.load_state_dict(torch.load(args.init, weights_only=True))

    if args.epochs > 0:
        if not examples:
            raise SystemExit("no examples loaded -- pass a dataset, or --epochs 0 to just save "
                              "an untrained init checkpoint")
        train(examples, net, epochs=args.epochs, lr=args.lr)
    else:
        print("--epochs 0: skipping training, saving weights as-is")

    torch.save(net.state_dict(), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

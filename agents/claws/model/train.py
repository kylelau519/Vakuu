"""Trains ClawsNet on materialize.py's output.

Batch size 1, no padding: each example has a different token count and option count
(materialize.py's output isn't rectangular), the same constraint ClawsNet.encode/score already
assume -- see their docstrings, MCTS calls them one leaf at a time too. Revisit if batch-size-1
throughput actually becomes a bottleneck; padding/masking to batch examples is a real complexity
jump not worth taking preemptively.

Loss = policy cross-entropy against the recorded (possibly sparse -- see materialize.py's
_target_probs) distribution, plus value MSE, summed with equal weight -- no tuning of that
weighting has been done, it's just the obvious starting point.

--value-only skips the policy term. Recorded `policy` is only a real training signal when it
came from an actual MCTS search (visit counts); a one-hot uniform-random pick or a raw
untrained-network softmax is noise, and cross-entropy against noise actively teaches the network
to imitate whatever exploration randomness happened to land on (e.g. `end_turn` drawing a stray
high weight in a decision where it's clearly wrong) -- worse than no signal at all. Use this flag
for datasets built without CLAWS_MCTS (see training.sh); the value target stays legitimate either
way, since a combat's HP-loss outcome is real regardless of what policy generated the trajectory.

Usage:
    conda activate sts2
    python train.py dataset.pt --epochs 5 --out claws.pt
    python train.py dataset.pt --epochs 20 --out claws.pt --value-only  # non-MCTS data
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # see model.py's top -- two OpenMP
# runtimes (conda's llvm-openmp + torch's bundled copy) otherwise abort on import

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


def train(
    examples: list[dict], net: ClawsNet, epochs: int, lr: float, seed: int = 0,
    value_only: bool = False,
) -> None:
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
            loss = value_loss if value_only else policy_loss + value_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_policy += policy_loss.item()
            total_value += value_loss.item()

        n = len(examples)
        note = " (not trained on -- --value-only)" if value_only else ""
        print(f"epoch {epoch + 1}/{epochs}: policy_loss={total_policy / n:.4f}{note} "
              f"value_loss={total_value / n:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*",
                         help="materialize.py output(s), e.g. dataset.pt "
                              "(omit with --epochs 0 to just save an untrained init checkpoint)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="claws.pt")
    parser.add_argument("--init", default="claws.pt", help="resume from an existing checkpoint")
    parser.add_argument("--value-only", action="store_true",
                         help="skip policy cross-entropy -- for datasets built without "
                              "CLAWS_MCTS, where recorded `policy` is noise, not real search "
                              "signal (see module docstring)")
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
        train(examples, net, epochs=args.epochs, lr=args.lr, value_only=args.value_only)
    else:
        print("--epochs 0: skipping training, saving weights as-is")

    torch.save(net.state_dict(), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

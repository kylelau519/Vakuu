"""Trains ChoicesParadoxNet on materialize.py's output. Near-verbatim copy of
claws/model/train.py -- same loss formula, same reasoning -- pointed at this package's network.

Batch size 1, no padding: each example has a different token count and option count.

Policy loss is an advantage-weighted log-likelihood, not cross-entropy against a fitted target
distribution -- see materialize.py's module docstring for where `advantage` comes from.

`policy_loss = -advantage * log_prob(action_index).clamp(min=_LOGP_FLOOR)`: a positive advantage
pushes that action's probability up on minimization; a negative advantage pushes it down. The
log_prob floor exists for the same reason claws/model/train.py's does: a fixed dataset replayed
over many epochs has no equilibrium on the push-down side (log_prob -> -inf as prob -> 0), so
clamping it gives a floor past which the gradient zeroes out instead of chasing probability to
exactly 0.

Usage:
    conda activate sts2
    python train.py dataset.pt --epochs 5 --out choices_paradox.pt --init ""  # round 1
    python train.py dataset.pt --epochs 5 --out choices_paradox.pt --init choices_paradox.pt
    python train.py dataset.pt --epochs 5 --out choices_paradox.pt --init "" \\
        --warm-start-from ../../claws/model/claws.pt  # seed card embeddings from Claws
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

# Batch size 1, tiny per-example tensors -- see claws/model/train.py's docstring for the
# measured single-thread-vs-default speedup this is copied from.
torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # for `common`
from common.tokenizer import EntityVocab  # noqa: E402

from model import (  # noqa: E402
    ChoicesParadoxNet, GlobalVocab, TokenBatch, load_state_dict_allowing_growth,
)

_LOGP_FLOOR = -6.0  # ~0.25% probability -- see module docstring for why this exists


def _normalize_advantages(examples: list[dict]) -> None:
    """In-place, zero-mean/unit-std across every policy_real example's advantage."""
    advantages = [ex["advantage"] for ex in examples if ex.get("policy_real", False)]
    if not advantages:
        return
    mean = sum(advantages) / len(advantages)
    var = sum((a - mean) ** 2 for a in advantages) / len(advantages)
    std = var ** 0.5
    if std < 1e-8:
        return
    for ex in examples:
        if ex.get("policy_real", False):
            ex["advantage"] = (ex["advantage"] - mean) / std


def _snapshot(net: ChoicesParadoxNet) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def _count_changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> tuple[int, int]:
    """Exact (not near-) equality: how many scalars in the whole network moved at all this epoch.
    Sparse inputs (e.g. an embedding table) leave rows for unseen entities bit-for-bit unchanged,
    so this doubles as a per-epoch coverage signal, not just a "did training run" sanity check."""
    changed = total = 0
    for k, b in before.items():
        a = after[k]
        total += a.numel()
        changed += int((a != b).sum().item())
    return changed, total


def _l2_delta(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    """L2 norm of (after - before) across every parameter tensor, treating the whole network as
    one flat vector. A step-size signal, not an overfitting signal by itself: it tracks how far
    Adam moved the weights this epoch, which shrinks as the loss converges to a fixed point
    regardless of whether that fixed point generalizes. It's only suggestive of overfitting
    alongside a widening train/held-out loss gap -- large or non-decaying movement while train
    loss keeps dropping is consistent with the net still chasing (memorizing) individual examples
    rather than settling, but this alone can't distinguish that from a still-useful large update."""
    total_sq = 0.0
    for k, b in before.items():
        a = after[k]
        total_sq += ((a - b) ** 2).sum().item()
    return total_sq ** 0.5


def train(
    examples: list[dict], net: ChoicesParadoxNet, epochs: int, lr: float, seed: int = 0,
    value_only: bool = False,
) -> None:
    _normalize_advantages(examples)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = random.Random(seed)
    before = _snapshot(net)

    for epoch in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        total_policy = total_value = 0.0
        n_real = 0

        for i in order:
            ex = examples[i]
            batch = TokenBatch.from_saved(ex)
            logits, value = net.score(batch, ex["options"])
            value_loss = (value - ex["value"]) ** 2
            train_policy = ex.get("policy_real", False) and not value_only

            if train_policy:
                log_probs = F.log_softmax(logits, dim=-1)
                log_prob = log_probs[ex["action_index"]].clamp(min=_LOGP_FLOOR)
                policy_loss = -ex["advantage"] * log_prob
                loss = policy_loss + value_loss
            else:
                loss = value_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            if train_policy:
                total_policy += policy_loss.item()
                n_real += 1
            total_value += value_loss.item()

        n = len(examples)
        policy_avg = f"{total_policy / n_real:.4f} (n={n_real} real)" if n_real else "n/a"
        after = _snapshot(net)
        changed, total = _count_changed(before, after)
        l2_delta = _l2_delta(before, after)
        before = after
        print(f"epoch {epoch + 1}/{epochs}: policy_loss={policy_avg} "
              f"value_loss={total_value / n:.4f} "
              f"params_changed={changed}/{total} ({changed / total:.1%}) "
              f"param_delta_l2={l2_delta:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*",
                         help="materialize.py output(s), e.g. dataset.pt "
                              "(omit with --epochs 0 to just save an untrained init checkpoint)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--out", default="choices_paradox.pt")
    parser.add_argument("--init", default=None,
                         help="resume from an existing checkpoint, or pass \"\" to train from a "
                              "random. Required no implicit default")
    parser.add_argument("--warm-start-from", default=None,
                         help="only used with --init \"\": path to a claws.pt checkpoint to seed ")
    parser.add_argument("--value-only", action="store_true",
                         help="force-skip policy loss everywhere, even on policy_real examples")
    args = parser.parse_args()
    if args.init is None:
        raise SystemExit("--init is required -- pass a checkpoint path to resume from, or \"\" "
                          "to intentionally train from a random init (e.g. round 1).")
    if not args.init and os.path.exists(args.out):
        raise SystemExit(f"--init is empty but --out ({args.out}) already exists -- refusing to "
                          f"overwrite a trained checkpoint with a random init. If this really is "
                          f"a fresh start, remove or rename the existing {args.out} first.")

    examples: list[dict] = []
    for path in args.datasets:
        examples.extend(torch.load(path, weights_only=False))
    print(f"{len(examples)} training examples from {len(args.datasets)} dataset file(s)")

    if args.init:
        net = ChoicesParadoxNet(GlobalVocab(EntityVocab.load()))
        load_state_dict_allowing_growth(net, torch.load(args.init, weights_only=True))
        print(f"resumed from {args.init}")
    elif args.warm_start_from:
        net = ChoicesParadoxNet.warm_started_from(args.warm_start_from)
        print(f"random init, card embeddings warm-started from {args.warm_start_from}")
    else:
        net = ChoicesParadoxNet(GlobalVocab(EntityVocab.load()))
        print("--init \"\": training from a random init")

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

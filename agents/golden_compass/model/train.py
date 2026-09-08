"""Trains GoldenCompassNet on materialize.py's output. Near-verbatim copy of
choices_paradox/model/train.py -- same loss formula, same reasoning -- pointed at this package's
network. See materialize.py's module docstring for where `value`/`advantage` come from (this
trial's own terminal run outcome, no sibling-branch baseline).

Real padded batching (see claws/model/train.py's own module docstring for the padding/masking
reasoning): each example has a different token count and option count, padded up to the batch's
own max length.

Policy loss is an advantage-weighted log-likelihood, not cross-entropy against a fitted target
distribution: `policy_loss = -advantage * log_prob(action_index).clamp(min=_LOGP_FLOOR)`. A
positive advantage (won the run) pushes that map choice's probability up on minimization; a
negative one (died) pushes it down. The log_prob floor exists for the same reason
claws/model/train.py's does: a fixed dataset replayed over many epochs has no equilibrium on the
push-down side (log_prob -> -inf as prob -> 0), so clamping it gives a floor past which the
gradient zeroes out instead of chasing probability to exactly 0.

Usage:
    conda activate sts2
    python train.py dataset.pt --epochs 5 --out golden_compass.pt --init ""  # round 1
    python train.py dataset.pt --epochs 5 --out golden_compass.pt --init golden_compass.pt
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

# Tiny per-example tensors -- see claws/model/train.py's docstring for the measured
# single-thread-vs-default speedup this is copied from.
torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # for `common`
from common.tokenizer import EntityVocab  # noqa: E402

from model import GoldenCompassNet, GlobalVocab, FEATURE_DIM  # noqa: E402

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


def _collate(examples: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads a list of examples' token tensors up to the batch's own max token count -- see
    claws/model/train.py's own _collate, identical reasoning."""
    lens = [ex["type_ids"].shape[0] for ex in examples]
    max_len = max(lens)
    b = len(examples)
    type_ids = torch.zeros(b, max_len, dtype=torch.long)
    global_ids = torch.zeros(b, max_len, dtype=torch.long)
    features = torch.zeros(b, max_len, FEATURE_DIM)
    pad_mask = torch.ones(b, max_len, dtype=torch.bool)
    for i, (ex, n) in enumerate(zip(examples, lens)):
        type_ids[i, :n] = ex["type_ids"]
        global_ids[i, :n] = ex["global_ids"]
        features[i, :n] = ex["features"]
        pad_mask[i, :n] = False
    return type_ids, global_ids, features, pad_mask


def _batch_step(
    examples: list[dict], net: GoldenCompassNet, value_only: bool,
) -> tuple[torch.Tensor, float, float, int]:
    """One batch's loss -- see claws/model/train.py's own _batch_step, identical reasoning
    (one encode_batched() call for every example's tokens, one policy_mlp() call for every
    option across the whole batch)."""
    type_ids, global_ids, features, pad_mask = _collate(examples)
    h = net.encode_batched(type_ids, global_ids, features, pad_mask)  # [B, N, dim]

    queries = []
    spans: list[tuple[int, int]] = []
    for b, ex in enumerate(examples):
        start = len(queries)
        hb = h[b]
        for i in range(ex["n_options"]):
            queries.append(hb[ex["option_pos"][i]])
        spans.append((start, len(queries) - start))

    all_logits = net.policy_mlp(torch.stack(queries)).squeeze(-1)  # [total options in batch]
    player_h = torch.stack([h[b, ex["player_pos"]] for b, ex in enumerate(examples)])
    values = torch.tanh(net.value_mlp(player_h).squeeze(-1))  # [B]

    loss = torch.zeros(())
    total_policy = total_value = 0.0
    n_real = 0
    for b, ex in enumerate(examples):
        value_loss = (values[b] - ex["value"]) ** 2
        train_policy = ex.get("policy_real", False) and not value_only

        if train_policy:
            start, count = spans[b]
            log_probs = F.log_softmax(all_logits[start:start + count], dim=-1)
            log_prob = log_probs[ex["action_index"]].clamp(min=_LOGP_FLOOR)
            policy_loss = -ex["advantage"] * log_prob
            loss = loss + policy_loss + value_loss
            total_policy += policy_loss.item()
            n_real += 1
        else:
            loss = loss + value_loss
        total_value += value_loss.item()

    return loss / len(examples), total_policy, total_value, n_real


def _snapshot(net: GoldenCompassNet) -> dict[str, torch.Tensor]:
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
    examples: list[dict], net: GoldenCompassNet, epochs: int, lr: float, seed: int = 0,
    value_only: bool = False, batch_size: int = 32,
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

        for start in range(0, len(order), batch_size):
            batch = [examples[i] for i in order[start:start + batch_size]]
            loss, batch_policy, batch_value, batch_n_real = _batch_step(batch, net, value_only)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_policy += batch_policy
            total_value += batch_value
            n_real += batch_n_real

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
    parser.add_argument("--batch-size", type=int, default=32,
                         help="examples per padded batch (see module docstring)")
    parser.add_argument("--out", default="golden_compass.pt")
    parser.add_argument("--init", default=None,
                         help="resume from an existing checkpoint, or pass \"\" to train from a "
                              "random. Required no implicit default")
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
        net = GoldenCompassNet(GlobalVocab(EntityVocab.load()))
        net.load_state_dict(torch.load(args.init, weights_only=True))
        print(f"resumed from {args.init}")
    else:
        net = GoldenCompassNet(GlobalVocab(EntityVocab.load()))
        print("--init \"\": training from a random init")

    if args.epochs > 0:
        if not examples:
            raise SystemExit("no examples loaded -- pass a dataset, or --epochs 0 to just save "
                              "an untrained init checkpoint")
        train(examples, net, epochs=args.epochs, lr=args.lr, value_only=args.value_only,
              batch_size=args.batch_size)
    else:
        print("--epochs 0: skipping training, saving weights as-is")

    torch.save(net.state_dict(), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Trains ClawsNet on materialize.py's output.

Batch size 1, no padding: each example has a different token count and option count
(materialize.py's output isn't rectangular), the same constraint ClawsNet.encode/score already
assume -- see their docstrings, MCTS calls them one leaf at a time too. Revisit if batch-size-1
throughput actually becomes a bottleneck; padding/masking to batch examples is a real complexity
jump not worth taking preemptively.

Value loss is plain MSE against the recorded `value`, every example, always real (a combat's
HP-loss outcome is real no matter what policy generated the trajectory).

Policy loss is an advantage-weighted log-likelihood, not cross-entropy against a fitted target
distribution -- see materialize.py's module docstring for where `advantage` comes from
(whispering_earring's turn-level branch-and-adopt: a branch's own realized `value` minus the
mean value across every sibling branch that replayed the same turn-start). `policy_loss =
-advantage * log_prob(action_index)`: a positive advantage (this action did better than the
turn's own average) pushes that action's probability up on minimization; a negative advantage
pushes it down, directly, not just "less pull toward it than something else" the way a softmax
target would. Only examples with `policy_real = True` (materialize.py's `_build_baselines` found
>= 2 distinct first actions to compare) carry a meaningful `advantage`/`action_index` and ever
contribute this term -- everything else (`policy_real = False`) trains the value head only, same
reasoning as before: with nothing to compare against, there's no real signal to train on.

--value-only is a global override that skips the policy term entirely even on `policy_real`
examples -- for A/B comparison.

Note policy_loss can be negative here (when advantage < 0, minimizing it means driving that
action's probability *down*) -- expected for this signed objective, not a bug. This loss also
has no floor the way cross-entropy does (-advantage*log_prob keeps growing in magnitude as
log_prob -> -inf, it never saturates once a target is matched), so `_normalize_advantages`
rescales advantages to zero-mean/unit-std before training -- keeps step size independent of
that round's raw HP-loss scale, though it doesn't remove the underlying unboundedness. Pair
with a modest epoch count (this is closer to on-policy RL than supervised imitation of a fixed
label set -- grinding many epochs against the same static advantages risks the collapse
normalization alone won't fully prevent).

Usage:
    conda activate sts2
    python train.py dataset.pt --epochs 5 --out claws.pt
    python train.py dataset.pt --epochs 20 --out claws.pt --value-only  # force value-only
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

# Batch size 1, tiny per-example tensors (see this module's docstring) -- PyTorch's default
# intra-op thread count (one per core) spins up a parallel region for every matmul/attention op
# regardless of size, and thread-pool sync overhead dominates when there's this little real work
# per op. Measured ~1.6x faster with a single thread than the 8-thread default on this machine's
# actual dataset (2.46s -> 1.57s / 400 examples, identical loss values both ways).
torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # for `common`
from common.tokenizer import EntityVocab  # noqa: E402

from model import ClawsNet, GlobalVocab, TokenBatch  # noqa: E402


def _normalize_advantages(examples: list[dict]) -> None:
    """In-place, zero-mean/unit-std across every policy_real example's advantage. Raw
    advantages are HP-loss-scale (see materialize.py's _build_baselines) with no fixed
    reference -- without this, step size on the policy term drifts with whatever scale that
    round's combats happened to produce, and -advantage*log_prob(a) has no floor as
    log_prob->-inf (unlike bounded cross-entropy), so an ungoverned scale is what turns normal
    training into runaway collapse over many epochs. Standard practice for advantage-weighted
    policy gradients (e.g. PPO); doesn't remove that unboundedness, just keeps typical step
    size sane -- still pair with a modest epoch count, not 50."""
    advantages = [ex["advantage"] for ex in examples if ex.get("policy_real", False)]
    if not advantages:
        return
    mean = sum(advantages) / len(advantages)
    var = sum((a - mean) ** 2 for a in advantages) / len(advantages)
    std = var ** 0.5
    if std < 1e-8:
        return  # no spread to normalize -- leave as-is rather than divide by ~0
    for ex in examples:
        if ex.get("policy_real", False):
            ex["advantage"] = (ex["advantage"] - mean) / std


def train(
    examples: list[dict], net: ClawsNet, epochs: int, lr: float, seed: int = 0,
    value_only: bool = False,
) -> None:
    _normalize_advantages(examples)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = random.Random(seed)

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
                policy_loss = -ex["advantage"] * log_probs[ex["action_index"]]
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
        print(f"epoch {epoch + 1}/{epochs}: policy_loss={policy_avg} "
              f"value_loss={total_value / n:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*",
                         help="materialize.py output(s), e.g. dataset.pt "
                              "(omit with --epochs 0 to just save an untrained init checkpoint)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--out", default="claws.pt")
    parser.add_argument("--init", default="claws.pt", help="resume from an existing checkpoint")
    parser.add_argument("--value-only", action="store_true",
                         help="force-skip policy cross-entropy everywhere, even on policy_real "
                              "examples -- normally only non-policy_real (noisy placeholder) "
                              "examples skip it automatically (see module docstring)")
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

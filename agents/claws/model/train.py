"""Trains ClawsNet on materialize.py's output.

Examples are batched with padding: each example has a different token count and option count
(materialize.py's output isn't rectangular), so a batch pads every example's tokens up to the
batch's own max length (ClawsNet.encode_batched's key_padding_mask keeps padded rows out of
attention) and every option-scoring query across the whole batch is stacked into one
policy_mlp call (see _batch_step) rather than looping tiny ops one example at a time. Earlier
this ran batch size 1 with no padding at all, since per-example tensors were so tiny that
PyTorch's thread-pool sync overhead dominated any real batching gain -- see git history for that
version's measurements. Real batching now amortizes that overhead across --batch-size examples
per step instead of paying it per example, which matters most for the early-floor part of a
dataset (see whispering_earring's branch-and-adopt scheme): most runs pass through early floors,
so those depths dominate example count and are where batch-size-1 training time was worst.

Policy loss is an advantage-weighted log-likelihood, not cross-entropy against a fitted target
distribution -- see materialize.py's module docstring for where `advantage` comes from
 
`policy_loss = -advantage * log_prob(action_index).clamp(min=_LOGP_FLOOR)`: a positive advantage
pushes that action's probability up on minimization; a negative advantage pushes it down.

Note policy_loss can be negative here (when advantage < 0, minimizing it means driving that
action's probability *down*). The log_prob floor exists because that push-down direction is
otherwise unbounded: log_prob(action) -> -inf as prob(action) -> 0, so a fixed dataset replayed
over many epochs (this isn't on-policy REINFORCE -- the same (state, action, advantage) triples
recur every epoch) has no equilibrium and drives every negative-advantage action's probability to
exactly 0, collapsing the whole policy to one-hot. Clamping log_prob at _LOGP_FLOOR still lets a
bad action get actively penalized (same sign, same gradient direction) but gives that penalty a
floor -- once the action's probability is low enough, the clamp zeroes the gradient instead of
chasing it further. The positive-advantage side never needed this: pushing prob(action) -> 1
saturates naturally at log_prob = 0.

Usage:
    conda activate sts2
    python train.py dataset.pt --epochs 5 --out claws.pt --init ""  # round 1, random init
    python train.py dataset.pt --epochs 5 --out claws.pt --init claws.pt  # resume
    python train.py dataset.pt --epochs 20 --out claws.pt --init claws.pt --value-only
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

# Still single-threaded even with real batching now (see this module's docstring) -- dim=64
# keeps every matmul small enough that PyTorch's default intra-op thread count (one per core)
# spins up a parallel region whose sync overhead outweighs the extra work available to split.
# Measured on this machine's actual dataset (6616 examples, 1 epoch): batch 32 takes 3.5s single-
# threaded vs 4.0s at 8 threads, and single-threaded batch 32 is itself ~8x faster than the old
# batch-size-1 loop's 28.9s -- the real win is from batching, not the thread count.
torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # for `common`
from common.tokenizer import EntityVocab  # noqa: E402

from model import ClawsNet, FEATURE_DIM, GlobalVocab  # noqa: E402

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
        return  # no spread to normalize -- leave as-is rather than divide by ~0
    for ex in examples:
        if ex.get("policy_real", False):
            ex["advantage"] = (ex["advantage"] - mean) / std


def _collate(examples: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads a list of examples' token tensors up to the batch's own max token count. Returns
    (type_ids, global_ids, features, pad_mask), each with a leading batch dimension -- pad_mask
    is True at padded positions (see ClawsNet.encode_batched)."""
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
    examples: list[dict], net: ClawsNet, value_only: bool,
) -> tuple[torch.Tensor, float, float, int]:
    """One batch's loss, built with exactly two real matmul-bearing calls -- one
    encode_batched() for every example's tokens and one policy_mlp() call for every option
    across the whole batch -- instead of looping either per example. Per-example work here is
    just gathering which token rows feed each option's query and slicing the shared policy
    logits back apart, both cheap indexing with no thread-pool sync cost.

    Returns (loss, total_policy_loss, total_value_loss, n_real) -- the last three are floats/int
    for the caller's running epoch totals, not part of the backward graph."""
    type_ids, global_ids, features, pad_mask = _collate(examples)
    h = net.encode_batched(type_ids, global_ids, features, pad_mask)  # [B, N, dim]

    queries = []
    spans: list[tuple[int, int]] = []  # (start, count) into `queries` per example
    for b, ex in enumerate(examples):
        start = len(queries)
        hb = h[b]
        for action, args in ex["options"]:
            if action == "end_turn":
                q = net.end_turn_query + hb[ex["player_pos"]]
            elif action == "play_card":
                q = hb[ex["hand_pos"][args["card_index"]]]
                if "target_index" in args:
                    q = q + hb[ex["enemy_pos"][args["target_index"]]]
            elif action == "use_potion":
                q = hb[ex["potion_pos"][args["potion_index"]]]
                if "target_index" in args:
                    q = q + hb[ex["enemy_pos"][args["target_index"]]]
            else:
                raise ValueError(f"unhandled combat action {action!r}")
            queries.append(q)
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


def train(
    examples: list[dict], net: ClawsNet, epochs: int, lr: float, seed: int = 0,
    value_only: bool = False, batch_size: int = 32,
) -> None:
    _normalize_advantages(examples)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = random.Random(seed)

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
        print(f"epoch {epoch + 1}/{epochs}: policy_loss={policy_avg} "
              f"value_loss={total_value / n:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*",
                         help="materialize.py output(s), e.g. dataset.pt "
                              "(omit with --epochs 0 to just save an untrained init checkpoint)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32,
                         help="examples per padded batch (see module docstring)")
    parser.add_argument("--out", default="claws.pt")
    parser.add_argument("--init", default=None,
                         help="resume from an existing checkpoint, or pass \"\" to train from "
                              "a random init. Required no implicit default. ")
    parser.add_argument("--value-only", action="store_true",
                         help="force-skip policy cross-entropy everywhere, even on policy_real "
                              "examples -- normally only non-policy_real (noisy placeholder) "
                              "examples skip it automatically (see module docstring)")
    args = parser.parse_args()
    if args.init is None:
        raise SystemExit("--init is required -- pass a checkpoint path to resume from, or \"\" "
                          "to intentionally train from a random init (e.g. round 1). There is "
                          "no default: a mis-pasted command that drops this flag now fails here "
                          "instead of silently loading whatever \"claws.pt\" happens to resolve "
                          "to in the current directory.")
    if not args.init and os.path.exists(args.out):
        raise SystemExit(f"--init is empty but --out ({args.out}) already exists -- refusing to "
                          f"overwrite a trained checkpoint with a random init. If this really is "
                          f"a fresh start, remove or rename the existing {args.out} first.")

    examples: list[dict] = []
    for path in args.datasets:
        examples.extend(torch.load(path, weights_only=False))
    print(f"{len(examples)} training examples from {len(args.datasets)} dataset file(s)")

    net = ClawsNet(GlobalVocab(EntityVocab.load()))
    if args.init:
        net.load_state_dict(torch.load(args.init, weights_only=True))
        print(f"resumed from {args.init}")
    else:
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

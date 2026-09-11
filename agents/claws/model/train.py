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
distribution -- see materialize.py's module docstring for where `advantages` comes from.

Since the hard/stochastic-gating change (policy.py's _choose_combat_action_claws,
tri_head_design.md's Training section), each policy_real example also carries which head
(`ex["head"]`) the self-play sampler actually used for that decision -- a two-level pick, "which
strategy" then "which action under that strategy" -- so the loss is two REINFORCE terms, each off
its OWN advantage (`ex["advantages"]` is a dict, one entry per stream, not one shared number), not
one term through a differentiable probability-space blend:

`head_loss = -advantages[head] * log(P_head(action_index)).clamp(min=_LOGP_FLOOR)` (was picking
this action, given that head, a good idea, judged against THAT head's own reward stream?) and
`gate_loss = -advantages["win"] * log(gate_weights[head]).clamp(min=_LOGP_FLOOR)` (was picking
this head here a good idea, judged against the overall win outcome -- arbitrating between
strategies by outcome IS the gate's job, not any one head's). A positive advantage pushes the
matching term's probability up; negative pushes it down. Only the head that was actually sampled
gets its own policy logits touched by a given example -- the other two heads' logits get gradient
only from examples where THEY were the one sampled.

The gate also gets a THIRD, terminal-only term on top of the two above, on whichever examples
carry a real `ex["terminal_bonus"]` (materialize.py's TurnTargets -- the RAW graded win/lose
outcome, not baseline-relative): `-terminal_bonus * log(gate_weights[head])`. This is the
aggregator's own outcome-conditioned REINFORCE signal -- "did the strategy I picked here lead to
a good ending (a win with little health lost, or at least not a loss)" -- additive with, not a
replacement for, the gate's regular per-turn advantage term above.

Gate entropy bonus (`--gate-entropy-coef`, tri_head_design.md's Aggregator section): hard/
stochastic gating has no restoring force once softmax drives a head's weight to ~0 -- that head
then gets ~0 policy gradient forever regardless of whether its suggestions were actually good
(both REINFORCE terms above are gated by w_z). Observed in practice: the gate collapsed to
`defense=1.00, aggro=scale=0.00` within one epoch and never moved again across 8 more, partly
because `hp_term` (defense's own reward stream) is turn-level, not action-level -- `end_turn`
inherits whatever `hp_term` the whole turn produced (recorder.py's end_turn() backfill), so a
quiet "nothing happened" turn hands defense a free non-negative advantage that aggro/scale, whose
streams are 0 unless the agent actually acted, can't get. `loss -= gate_entropy_coef *
entropy(gate_weights)` per example counteracts the collapse directly: entropy is maximized (loss
lowered) by keeping gate weights spread out, so a head can't be driven all the way to a fixed
point at exactly 0 the way pure REINFORCE otherwise allows.

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
import math
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
#
# CPU vs. GPU crossover (measured on the 61783-example dataset, this machine's RTX 3060 Mobile):
# batch 32 CPU beats GPU (29.8s vs 38.5s -- kernel-launch/transfer overhead dominates at this
# size), but CPU cost grows ~quadratically with batch size (padding every example in a batch up
# to that batch's own max length inflates the attention cost) while GPU absorbs it -- batch 256
# is roughly a tie, batch 1024 favors GPU 2x (27.3s vs 61.3s), batch 4096 favors GPU 6x. Note the
# per-example Python loop in _batch_step that builds each option's query is a fixed ~27s/epoch
# cost regardless of device or batch size, so GPU plateaus there rather than keeps improving.
torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # for `common`
from common.tokenizer import EntityVocab  # noqa: E402

from model import ClawsNet, FEATURE_DIM, GlobalVocab, POLICY_STREAMS  # noqa: E402

_LOGP_FLOOR = -6.0  # ~0.25% probability -- see module docstring for why this exists
_PROB_FLOOR = math.exp(_LOGP_FLOOR)  # same floor, in probability space -- see _batch_step
_STREAM_IDX = {name: i for i, name in enumerate(POLICY_STREAMS)}  # head name -> gate column
_ALL_STREAMS = (*POLICY_STREAMS, "win")
_GATE_ENTROPY_COEF = 0.01  # see module docstring's Gate entropy bonus section


def _normalize_advantages(examples: list[dict]) -> None:
    """In-place, zero-mean/unit-std across every policy_real example's advantage -- independently
    per stream, since `ex["advantages"]` is now one number per head/gate stream rather than a
    single shared value (see module docstring): each head should be judged on a scale set by its
    OWN stream's spread, not one borrowed from a different stream that happens to vary more or
    less."""
    for name in _ALL_STREAMS:
        values = [ex["advantages"][name] for ex in examples if ex.get("policy_real", False)]
        if not values:
            continue
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = var ** 0.5
        if std < 1e-8:
            continue  # no spread to normalize -- leave as-is rather than divide by ~0
        for ex in examples:
            if ex.get("policy_real", False):
                ex["advantages"][name] = (ex["advantages"][name] - mean) / std


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


def _new_diag() -> dict[str, float]:
    """Per-epoch diagnostic accumulator (see _batch_step/train) -- first-run visibility into
    whether each head is actually learning something distinct, whether the gate still assigns
    every head some real weight (gate_*), and -- since hard/stochastic gating means self-play
    itself decides which head generates each training example -- whether the DATA actually
    contains a healthy mix of all three heads at all (head_used_*), not just what the current net
    would currently choose."""
    return (
        {f"value_loss_{name}": 0.0 for name in _ALL_STREAMS}
        | {f"gate_{name}": 0.0 for name in POLICY_STREAMS}
        | {"gate_n": 0.0}
        | {f"head_used_{name}": 0.0 for name in POLICY_STREAMS}
        | {"head_used_n": 0.0}
        | {"terminal_n": 0.0}
        | {"gate_entropy": 0.0}
    )


def _batch_step(
    examples: list[dict], net: ClawsNet, value_only: bool, device: torch.device,
    gate_entropy_coef: float = _GATE_ENTROPY_COEF,
) -> tuple[torch.Tensor, float, float, int, dict[str, float]]:
    """One batch's loss. Same query-per-option gathering as before encode_batched() runs (still
    exactly one matmul-bearing call per head across the whole batch's stacked queries, not looped
    per example), now fanned out across 3 policy heads + 3 value heads + win_value + the gate
    (tri_head_design.md's Architecture/Aggregator sections) instead of one policy_mlp/value_mlp
    pair. All 3 heads' logits are computed for the whole batch in that one matmul regardless of
    which head each example actually needs -- per-example work (gathering which token rows feed
    each option's query, slicing back apart, picking out just the recorded `ex["head"]`'s own
    slice) is cheap indexing, no thread-pool sync cost.

    Returns (loss, total_policy_loss, total_value_loss, n_real, diag) -- the last four are
    floats/int/dict for the caller's running epoch totals, not part of the backward graph. `diag`
    (see _new_diag) breaks total_value_loss down per head, sums this batch's gate weights, and
    counts how often each head was actually the one self-play sampled -- a first-run sanity check
    that isn't visible in the single combined loss number."""
    type_ids, global_ids, features, pad_mask = _collate(examples)
    type_ids, global_ids, features, pad_mask = (
        type_ids.to(device), global_ids.to(device), features.to(device), pad_mask.to(device)
    )
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

    query_batch = torch.stack(queries)  # [total options in batch, dim]
    all_logits = {
        name: net.policy_heads[name](query_batch).squeeze(-1) for name in POLICY_STREAMS
    }  # each [total options in batch]

    player_h = torch.stack([h[b, ex["player_pos"]] for b, ex in enumerate(examples)])  # [B, dim]
    stream_values = {name: net.value_heads[name](player_h).squeeze(-1) for name in POLICY_STREAMS}
    win_values = torch.tanh(net.win_value_mlp(player_h).squeeze(-1))  # [B]
    gate_in = torch.cat(
        [player_h, torch.stack([stream_values[n] for n in POLICY_STREAMS] + [win_values], dim=-1)],
        dim=-1,
    )
    gate_weights = torch.softmax(net.gate_mlp(gate_in), dim=-1)  # [B, 3]

    loss = torch.zeros((), device=device)
    total_policy = total_value = 0.0
    n_real = 0
    diag = _new_diag()
    for b, ex in enumerate(examples):
        targets = ex["value_targets"]
        per_head_loss = {
            name: (stream_values[name][b] - targets[name]) ** 2 for name in POLICY_STREAMS
        }
        per_head_loss["win"] = (win_values[b] - targets["win"]) ** 2
        for name, hl in per_head_loss.items():
            diag[f"value_loss_{name}"] += hl.item()
        value_loss = sum(per_head_loss.values())
        train_policy = ex.get("policy_real", False) and not value_only

        if train_policy:
            start, count = spans[b]
            head = ex["head"]
            head_idx = _STREAM_IDX[head]
            head_probs = F.softmax(all_logits[head][start:start + count], dim=-1)
            head_prob = head_probs[ex["action_index"]].clamp(min=_PROB_FLOOR)
            gate_prob = gate_weights[b, head_idx].clamp(min=_PROB_FLOOR)
            advantages = ex["advantages"]
            # Two REINFORCE terms off two DIFFERENT advantages, not one blend -- see module
            # docstring. Only `head`'s own policy logits (and the gate's own weight on `head`)
            # get gradient from this example; the other two heads are untouched here, and only
            # get trained by examples where THEY were the one self-play actually sampled.
            policy_loss = (
                -advantages[head] * torch.log(head_prob) - advantages["win"] * torch.log(gate_prob)
            )
            terminal_bonus = ex.get("terminal_bonus")
            if terminal_bonus is not None:
                # Third, terminal-only REINFORCE term for the gate alone -- see module docstring.
                policy_loss = policy_loss - terminal_bonus * torch.log(gate_prob)
                diag["terminal_n"] += 1
            # Gate entropy bonus -- see module docstring's Gate entropy bonus section. Kept
            # separate from policy_loss (not folded into the REINFORCE terms above) so
            # total_policy/policy_avg still reads as pure advantage-weighted log-likelihood, not
            # muddied by a regularizer with a different scale/sign convention.
            gate_entropy = -(gate_weights[b] * torch.log(gate_weights[b].clamp(min=_PROB_FLOOR))).sum()
            loss = loss + policy_loss + value_loss - gate_entropy_coef * gate_entropy
            diag["gate_entropy"] += gate_entropy.item()
            total_policy += policy_loss.item()
            n_real += 1
            diag[f"head_used_{head}"] += 1
            diag["head_used_n"] += 1
        else:
            loss = loss + value_loss
        total_value += value_loss.item()
        for i, name in enumerate(POLICY_STREAMS):
            diag[f"gate_{name}"] += gate_weights[b, i].item()
        diag["gate_n"] += 1

    return loss / len(examples), total_policy, total_value, n_real, diag


def _snapshot(net: ClawsNet) -> dict[str, torch.Tensor]:
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
    examples: list[dict], net: ClawsNet, epochs: int, lr: float, seed: int = 0,
    value_only: bool = False, batch_size: int = 64, device: torch.device = torch.device("cpu"),
    gate_entropy_coef: float = _GATE_ENTROPY_COEF,
) -> None:
    _normalize_advantages(examples)
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = random.Random(seed)
    before = _snapshot(net)

    for epoch in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        total_policy = total_value = 0.0
        n_real = 0
        epoch_diag = _new_diag()

        for start in range(0, len(order), batch_size):
            batch = [examples[i] for i in order[start:start + batch_size]]
            loss, batch_policy, batch_value, batch_n_real, batch_diag = _batch_step(
                batch, net, value_only, device, gate_entropy_coef
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_policy += batch_policy
            total_value += batch_value
            n_real += batch_n_real
            for k, v in batch_diag.items():
                epoch_diag[k] += v

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
        per_head = " ".join(
            f"{name}={epoch_diag[f'value_loss_{name}'] / n:.4f}" for name in _ALL_STREAMS
        )
        gate_n = epoch_diag["gate_n"] or 1.0
        gate_avg = " ".join(
            f"{name}={epoch_diag[f'gate_{name}'] / gate_n:.2f}" for name in POLICY_STREAMS
        )
        head_n = epoch_diag["head_used_n"] or 1.0
        head_avg = " ".join(
            f"{name}={epoch_diag[f'head_used_{name}'] / head_n:.2f}" for name in POLICY_STREAMS
        )
        print(f"  value_loss per head: {per_head}")
        print(f"  avg gate weights (aggro/defense/scale): {gate_avg}")
        print(f"  head usage in self-play data (aggro/defense/scale): {head_avg}")
        print(f"  terminal_bonus fired on {int(epoch_diag['terminal_n'])}/{n_real} policy_real examples")
        if n_real and not value_only:
            gate_entropy_avg = epoch_diag["gate_entropy"] / n_real
            print(f"  avg gate entropy: {gate_entropy_avg:.4f} (max ln(3)={math.log(3):.4f}, "
                  f"coef={gate_entropy_coef})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*",
                         help="materialize.py output(s), e.g. dataset.pt "
                              "(omit with --epochs 0 to just save an untrained init checkpoint)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32,
                         help="examples per padded batch (see module docstring). See module "
                              "docstring's CPU/GPU crossover note before raising this -- past "
                              "~256 it's only a win with --device cuda, and the learning rate "
                              "should scale roughly with sqrt(batch_size) for Adam")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                         help="default: cuda if available, else cpu")
    parser.add_argument("--out", default="claws.pt")
    parser.add_argument("--init", default=None,
                         help="resume from an existing checkpoint, or pass \"\" to train from "
                              "a random init. Required no implicit default. ")
    parser.add_argument("--value-only", action="store_true",
                         help="force-skip policy cross-entropy everywhere, even on policy_real "
                              "examples -- normally only non-policy_real (noisy placeholder) "
                              "examples skip it automatically (see module docstring)")
    parser.add_argument("--gate-entropy-coef", type=float, default=_GATE_ENTROPY_COEF,
                         help="weight on the gate's entropy bonus (see module docstring's Gate "
                              "entropy bonus section) -- counteracts hard-gating's tendency to "
                              "drive a head's weight to exactly 0 with no way back. 0 disables it.")
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"training on {device}")
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
              batch_size=args.batch_size, device=device,
              gate_entropy_coef=args.gate_entropy_coef)
    else:
        print("--epochs 0: skipping training, saving weights as-is")

    net.to("cpu")  # checkpoints stay device-agnostic regardless of what trained them
    torch.save(net.state_dict(), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

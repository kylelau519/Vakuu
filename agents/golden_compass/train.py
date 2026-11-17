"""Trains HazardNet on materialize.py's dataset.

    python train.py dataset.pt --epochs 30 --out golden_compass.pt --init <ckpt or "">

Loss = per-floor masked BCE on the death hazard (hazard_net.hazard_loss) + AUX_WEIGHT * the auxiliary
HP-cost MSE. Each run gets one vote however many states it logged (weight 1 / states in the run).
Runs, not samples, are split into train / held-out (crc32(run id) % HOLDOUT == 0), so a held-out
state's near-twins never sit in the training set. The best held-out epoch is kept.

`--epochs 0 --init ""` writes a random-init checkpoint (reset.sh).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402

import hazard_net as hn  # noqa: E402
from run_state import RunState  # noqa: E402


def _split(samples: list[dict], holdout: int) -> tuple[list[dict], list[dict]]:
    held = [s for s in samples if zlib.crc32(s["run"].encode()) % holdout == 0]
    train = [s for s in samples if zlib.crc32(s["run"].encode()) % holdout != 0]
    if not held or not train:  # too few runs to hold any out: train and watch the same data
        return samples, samples
    return train, held


def _batch(encoder: hn.Encoder, samples: list[dict], weights: dict[str, float]):
    b = encoder.collate([RunState.from_json(s["state"]) for s in samples])
    b["end"] = torch.tensor([s["end"] for s in samples])
    b["outcome"] = torch.tensor([s["outcome"] for s in samples])
    b["weight"] = torch.tensor([weights[s["run"]] for s in samples], dtype=torch.float)
    b["aux_target"] = torch.tensor([s["aux"] for s in samples], dtype=torch.float)
    b["aux_mask"] = torch.tensor([s["aux_mask"] for s in samples], dtype=torch.float)
    return b


def _loss(net: hn.HazardNet, b: dict, aux_weight: float) -> tuple[torch.Tensor, float, float]:
    logits, aux = net(b)
    hazard = hn.hazard_loss(logits, b["start"], b["end"], b["outcome"], b["weight"])
    aux_l = hn.aux_loss(aux, b["aux_target"], b["aux_mask"], b["weight"])
    return hazard + aux_weight * aux_l, hazard.item(), aux_l.item()


@torch.no_grad()
def _evaluate(net, encoder, samples, weights, batch_size, aux_weight) -> tuple[float, float]:
    net.eval()
    h = a = n = 0.0
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        b = _batch(encoder, chunk, weights)
        _, hz, ax = _loss(net, b, aux_weight)
        h, a, n = h + hz * len(chunk), a + ax * len(chunk), n + len(chunk)
    return h / max(n, 1), a / max(n, 1)


def _baseline_bias(samples: list[dict]) -> torch.Tensor:
    """logit of each floor's empirical death rate -- the hazard head starts out knowing only the floor."""
    b = {k: torch.tensor([s[k] for s in samples]) for k in ("end", "outcome")}
    start = torch.tensor([s["state"]["floor"] for s in samples])
    target, mask = hn.hazard_targets(start, b["end"], b["outcome"])
    rate = ((target * mask).sum(0) + 0.5e-3) / (mask.sum(0) + 1.0)
    rate = rate.clamp(1e-4, 0.5)
    return torch.log(rate / (1 - rate))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?")
    parser.add_argument("--out", required=True)
    parser.add_argument("--init", required=True, help='checkpoint to continue from, or "" for a fresh net')
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--holdout", type=int, default=5, help="1 run in N is held out")
    args = parser.parse_args()

    encoder = hn.Encoder()
    fresh = not args.init
    if fresh and os.path.exists(args.out):
        sys.exit(f"{args.out} exists and --init is empty: refusing to overwrite it with a random net")
    net = hn.HazardNet.build(encoder) if fresh else hn.HazardNet.from_checkpoint(args.init)

    samples = torch.load(args.dataset, weights_only=False) if args.dataset else []
    if fresh and samples:
        with torch.no_grad():
            net.hazard.bias.copy_(_baseline_bias(samples))
    if args.epochs == 0 or not samples:
        net.save(args.out)
        print(f"wrote {'random-init' if fresh else 'unchanged'} checkpoint to {args.out}")
        return

    per_run = Counter(s["run"] for s in samples)
    weights = {run: 1.0 / n for run, n in per_run.items()}
    train, held = _split(samples, args.holdout)
    print(f"{len(samples)} samples / {len(per_run)} runs; train {len(train)}, held-out {len(held)}")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    best, best_epoch = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        net.train()
        random.shuffle(train)
        total = 0.0
        for i in range(0, len(train), args.batch_size):
            b = _batch(encoder, train[i:i + args.batch_size], weights)
            loss, _, _ = _loss(net, b, args.aux_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(b["end"])
        hz, ax = _evaluate(net, encoder, held, weights, args.batch_size, args.aux_weight)
        score = hz + args.aux_weight * ax
        print(f"epoch {epoch:3d}  train {total / len(train):.4f}  held-out hazard {hz:.4f} aux {ax:.4f}")
        if score < best:
            best, best_epoch = score, epoch
            net.save(args.out)
        elif epoch - best_epoch >= args.patience:
            print(f"no held-out improvement for {args.patience} epochs; best was epoch {best_epoch}")
            break
    print(f"saved best (epoch {best_epoch}, held-out {best:.4f}) to {args.out}")


if __name__ == "__main__":
    main()

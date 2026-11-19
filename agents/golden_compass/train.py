"""Trains HazardNet on materialize.py's dataset.

    python train.py dataset.pt --epochs 30 --out golden_compass.pt --init <ckpt or "">

Loss = per-floor masked BCE on the death hazard (hazard_net.hazard_loss) + AUX_WEIGHT * the auxiliary
HP-cost MSE. Each run gets one vote however many states it logged (weight 1 / states in the run).
Branch data (--branch-dir, from whispering_earring/facility_branch.py): for each card-reward fork point, every option
(each offered card, and skip) was played K times for AUX combats with the draw pile reshuffled. The option's state (the
deck with that card; the base state for skip) is trained to predict its group-mean HP-cost share per combat (the aux
head, masked to the combats the continuations reached), plus a contrast loss: the predicted difference card - skip must
match the measured one -- the pairing carried into the loss. Held-out forks (same by-run split) report the contrast
error and how often the net picks the option the branches found best.

Runs, not samples, are split into train / held-out (crc32(run id) % HOLDOUT == 0), so a held-out
state's near-twins never sit in the training set. The best held-out epoch (hazard + aux-weight x aux, on the held-out runs) is kept; the branch metrics are printed
but do not pick the epoch -- a few dozen held-out forks are too noisy to select on.

`--epochs 0 --init ""` writes a random-init checkpoint (reset.sh).
"""
from __future__ import annotations

import argparse
import glob
import json
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
from run_state import RunState, with_card as rs_with_card  # noqa: E402


def _split(samples: list[dict], holdout: int) -> tuple[list[dict], list[dict]]:
    held = [s for s in samples if zlib.crc32(s["run"].encode()) % holdout == 0]
    train = [s for s in samples if zlib.crc32(s["run"].encode()) % holdout != 0]
    if not held or not train:  # too few runs to hold any out: train and watch the same data
        return samples, samples
    return train, held


def _load_branch(dirs: list[str]) -> list[dict]:
    """One dict per fork point: the options' states, group-mean per-combat HP share `t` [n_opt, AUX], weights `m`
    [n_opt, AUX] (share of the K continuations that reached that combat; 0 below 3), and the index of skip."""
    forks = []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            with open(path) as f:
                fk = json.load(f)
            base = RunState.from_json(fk["state"])
            is_map = fk.get("kind") == "map"  # a route fork: every option is a node, there is no skip
            cards = {} if is_map else {str(i): (cid, up) for i, cid, up in fk["cards"]}
            states, t, m, skip = [], [], [], None
            for key, opt in fk["options"].items():
                runs = [r for r in opt["runs"] if "cum_lost" in r]
                if len(runs) < 3:
                    continue
                cums = [(r["cum_lost"] + [r["cum_lost"][-1] if r["cum_lost"] else 0.0] * hn.AUX)[:hn.AUX] for r in runs]  # tolerate short lists
                share = [[c[i] - (c[i - 1] if i else 0.0) for i in range(hn.AUX)] for c in cums]
                ti, mi = [], []
                for i in range(hn.AUX):
                    got = [share[j][i] for j, r in enumerate(runs) if r["ended_at"] > i]
                    ti.append(sum(got) / len(got) if len(got) >= 3 else 0.0)
                    mi.append(len(got) / len(runs) if len(got) >= 3 else 0.0)
                if is_map:
                    state = RunState.from_json(fk["afterstates"][key])
                elif key == "skip":
                    skip, state = len(states), base
                else:
                    cid, up = cards[key]
                    state = rs_with_card(base, cid, up)
                states.append(state)
                t.append(ti)
                m.append(mi)
            if (skip is not None or is_map) and len(states) >= 2:
                forks.append({"run": fk["main_id"], "fork": fk["fork_id"], "states": states, "t": t, "m": m, "skip": skip})
    return forks


def _branch_loss(net, encoder, forks: list[dict], contrast_weight: float) -> tuple[torch.Tensor, dict]:
    """Aux-head regression on the options' group means + contrast (card - skip) loss, over a list of forks."""
    states = [s for f in forks for s in f["states"]]
    _logits, aux = net(encoder.collate(states))
    t = torch.tensor([row for f in forks for row in f["t"]], dtype=torch.float)
    m = torch.tensor([row for f in forks for row in f["m"]], dtype=torch.float)
    absolute = (m * (aux - t) ** 2).sum() / m.sum().clamp(min=1e-6)
    num = den = 0.0
    off = 0
    for f in forks:
        n = len(f["states"])
        if f["skip"] is None:  # a route fork: each node against the mean of the fork's nodes
            seen = (m[off:off + n] > 0).float()
            ref_aux = aux[off:off + n].mean(0)
            ref_t = (t[off:off + n] * seen).sum(0) / seen.sum(0).clamp(min=1.0)
            for j in range(n):
                w = m[off + j]
                num = num + (w * ((aux[off + j] - ref_aux) - (t[off + j] - ref_t)) ** 2).sum()
                den = den + w.sum()
            off += n
            continue
        sk = off + f["skip"]
        for j in range(n):
            if j == f["skip"]:
                continue
            w = torch.minimum(m[off + j], m[sk])
            num = num + (w * ((aux[off + j] - aux[sk]) - (t[off + j] - t[sk])) ** 2).sum()
            den = den + w.sum()
        off += n
    contrast = num / den if torch.is_tensor(den) and den > 0 else torch.zeros(())
    return absolute + contrast_weight * contrast, {"abs": absolute.item(), "contrast": float(contrast.detach())}


@torch.no_grad()
def _branch_eval(net, encoder, forks: list[dict], contrast_weight: float) -> dict:
    """Held-out forks: the loss, plus how often the net's lowest predicted total HP cost is the option the branches
    found cheapest (top-1), over forks with at least three options."""
    net.eval()
    loss, parts = _branch_loss(net, encoder, forks, contrast_weight)
    hit = n = 0
    for f in forks:
        if len(f["states"]) < 3:
            continue
        _l, aux = net(encoder.collate(f["states"]))
        t, m = torch.tensor(f["t"]), torch.tensor(f["m"])
        measured = (t * (m > 0)).sum(1)
        n += 1
        hit += int(aux.sum(1).argmin() == measured.argmin())
    return {"loss": loss.item(), **parts, "top1": hit / n if n else float("nan"), "forks": len(forks)}


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
    start = torch.tensor([hn.head_floor(s["state"].get("act") or 1, s["state"]["floor"]) for s in samples])
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
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW decoupled weight decay (0 = plain Adam, as before)")
    parser.add_argument("--holdout", type=int, default=5, help="1 run in N is held out")
    parser.add_argument("--branch-dir", nargs="*", default=[], help="facility_branch.py output directories (card-reward forks)")
    parser.add_argument("--branch-weight", type=float, default=1.0, help="weight of the branch loss next to the hazard loss")
    parser.add_argument("--branch-contrast", type=float, default=1.0, help="weight of the card-minus-skip contrast inside it")
    parser.add_argument("--branch-forks", type=int, default=16, help="fork points per branch step")
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

    branch = _load_branch(args.branch_dir) if args.branch_dir else []
    b_train = [f for f in branch if zlib.crc32(f["run"].encode()) % args.holdout != 0]
    b_held = [f for f in branch if zlib.crc32(f["run"].encode()) % args.holdout == 0]
    if branch:
        print(f"branch data: {len(branch)} fork points ({len(b_train)} train, {len(b_held)} held-out)")
        if not b_train or not b_held:  # too few forks to hold any out
            b_train = b_held = branch

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, best_epoch = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        net.train()
        random.shuffle(train)
        random.shuffle(b_train)
        total = 0.0
        steps = (len(train) + args.batch_size - 1) // args.batch_size
        b_batches = [b_train[j:j + args.branch_forks] for j in range(0, len(b_train), args.branch_forks)]
        b_at = {round(k * steps / len(b_batches)): b_batches[k] for k in range(len(b_batches))} if b_batches else {}
        for step, i in enumerate(range(0, len(train), args.batch_size)):
            b = _batch(encoder, train[i:i + args.batch_size], weights)
            loss, _, _ = _loss(net, b, args.aux_weight)
            if step in b_at:  # each branch batch once per epoch, spread over the epoch's steps
                b_loss, _ = _branch_loss(net, encoder, b_at[step], args.branch_contrast)
                loss = loss + args.branch_weight * b_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(b["end"])
        hz, ax = _evaluate(net, encoder, held, weights, args.batch_size, args.aux_weight)
        score = hz + args.aux_weight * ax
        line = f"epoch {epoch:3d}  train {total / len(train):.4f}  held-out hazard {hz:.4f} aux {ax:.4f}"
        if b_held:
            be = _branch_eval(net, encoder, b_held, args.branch_contrast)
            line += f"  branch loss {be['loss']:.4f} (abs {be['abs']:.4f} contrast {be['contrast']:.4f}) top1 {be['top1']:.2f} on {be['forks']} forks"
        print(line)
        if score < best:
            best, best_epoch = score, epoch
            net.save(args.out)
        elif epoch - best_epoch >= args.patience:
            print(f"no held-out improvement for {args.patience} epochs; best was epoch {best_epoch}")
            break
    print(f"saved best (epoch {best_epoch}, held-out {best:.4f}) to {args.out}")


if __name__ == "__main__":
    main()

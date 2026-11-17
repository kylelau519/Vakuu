"""Gate: is the candidate compass a more accurate hazard model than the accepted one?

    python compare.py CANDIDATE.pt ACCEPTED.pt [--runs 40] [--workers 8]   # exit 0 = candidate passes, 1 = it does not

Plays fresh random-seed runs with facility_compass.py (Claws greedy, half the runs with each compass choosing the
route, so neither net's own trajectories are favoured), written to a temp dir so nothing trains on them, and scores
both nets' death-hazard BCE on every run (the per-run mean over its logged states). The candidate wins a run when its
loss is lower. PASS: it wins >= 65% of the non-tied runs and its mean loss is lower. Floors reached play no part.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
HERE = Path(__file__).resolve().parent
WE = HERE.parent / "whispering_earring"
CLAWS = HERE.parent / "claws" / "model" / "claws.pt"
THRESHOLD = 0.65


def _play(ckpt: str, runs: int, workers: int, raw: str, ckpt_dir: str) -> None:
    env = dict(os.environ, OMP_NUM_THREADS="1", DOTNET_gcServer="0", CLAWS_NOISE_EPS="0", GOLDEN_COMPASS_CHECKPOINT=ckpt)
    if CLAWS.exists():
        env["CLAWS_CHECKPOINT"] = str(CLAWS)
    subprocess.run([sys.executable, str(WE / "facility_compass.py"), "--trials", str(runs), "--branch-workers", str(workers),
                    "--ascension", "1", "--raw-dir", raw, "--ckpt-dir", ckpt_dir],
                   cwd=WE, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("accepted")
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cand, acc = os.path.abspath(a.candidate), os.path.abspath(a.accepted)
    tmp = tempfile.mkdtemp(prefix="compass_gate_")
    try:
        dirs = []
        for i, ck in enumerate((cand, acc)):
            raw = f"{tmp}/raw{i}"
            os.makedirs(raw)
            dirs.append(raw)
            _play(ck, a.runs // 2 + (a.runs % 2 if i == 0 else 0), a.workers, raw, f"{tmp}/ckpt{i}")
        ds = f"{tmp}/ds.pt"
        subprocess.run([sys.executable, str(HERE / "materialize.py"), *dirs, "--out", ds], cwd=HERE,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        sys.path.insert(0, str(HERE))
        import torch
        import hazard_net as hn
        import train as tr
        samples = torch.load(ds, weights_only=False)
        runs = sorted({s["run"] for s in samples})
        enc = hn.Encoder()

        def per_run_loss(path: str) -> dict[str, float]:
            net = hn.HazardNet.from_checkpoint(path)
            out: dict[str, list[float]] = {}
            with torch.no_grad():
                for i in range(0, len(samples), 512):
                    chunk = samples[i:i + 512]
                    b = tr._batch(enc, chunk, {s["run"]: 1.0 for s in chunk})
                    logits, _ = net(b)
                    t, m = hn.hazard_targets(b["start"], b["end"], b["outcome"], logits.shape[1])
                    loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, t, reduction="none") * m).sum(1)
                    for s, v in zip(chunk, loss.tolist()):
                        out.setdefault(s["run"], []).append(v)
            return {r: sum(v) / len(v) for r, v in out.items()}

        new, old = per_run_loss(cand), per_run_loss(acc)
        diff = [old[r] - new[r] for r in runs]  # positive: the candidate is better
        won, lost = sum(x > 1e-9 for x in diff), sum(x < -1e-9 for x in diff)
        frac = won / max(won + lost, 1)
        mean_new, mean_old = sum(new.values()) / len(new), sum(old.values()) / len(old)
        print(f"gate: {len(runs)} fresh runs, {len(samples)} states; mean hazard loss candidate {mean_new:.4f} accepted {mean_old:.4f}")
        print(f"gate: candidate better on {won}, worse on {lost}, tied {len(diff) - won - lost}; {frac:.0%} of non-tied (need {THRESHOLD:.0%})")
        passed = frac >= THRESHOLD and mean_new < mean_old
        print("gate: PASS -- promote the candidate" if passed else "gate: no improvement -- keep the accepted compass")
        return 0 if passed else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

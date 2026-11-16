"""Is Claws getting better at combat? Plays the same fixed seeds with each checkpoint and one fixed
non-combat policy (random map, tier-list cards: the hazard net is switched off), so the ONLY thing that
differs between rows is how Claws fights. Training's own `value_rmse` can't answer this -- it is error on
the training data -- and the compass loop's table mixes in route and deck changes.

    python eval_claws.py --ckpt random --ckpt ../claws/model/claws.pt          # one comparison
    python eval_claws.py --backups 6 --ckpt ../claws/model/claws.pt --ckpt random   # a curve over past rounds
    python eval_claws.py --show                                                # just print the log

Per checkpoint: runs, mean floor reached (+- standard error), HP lost per combat (share of max HP, deaths
count in full), the share of combats won, turns per combat. Higher floor / win rate and lower HP lost mean
better. A difference smaller than about two standard errors is noise: raise --runs.

Rows append to data/claws_eval.jsonl. `random` is the untrained baseline (uniform-random legal actions).
By default Claws plays sampled with noise, as in data generation; --greedy plays its top choice instead."""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import sts2_path  # noqa: F401
from facility_common import FacilityConfig
from facility_compass import FacilityCompass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "golden_compass"))
import progress  # noqa: E402

LOG = os.path.join(HERE, "data", "claws_eval.jsonl")
BACKUPS = os.path.normpath(os.path.join(HERE, "..", "claws", "model", "backups"))


def evaluate(ckpt: str, label: str, runs: int, seeds: list[str], workers: int, greedy: bool, ascension: int) -> dict:
    name = "random" if ckpt == "random" else os.path.basename(ckpt)
    # Spawned workers read these at import, so set them before the facility starts. The hazard net is off
    # so map / card choices are the same for every checkpoint.
    os.environ.pop("GOLDEN_COMPASS_CHECKPOINT", None)
    if ckpt == "random":
        os.environ.pop("CLAWS_CHECKPOINT", None)
    else:
        os.environ["CLAWS_CHECKPOINT"] = os.path.abspath(ckpt)
    if greedy:
        os.environ["CLAWS_GREEDY"] = "1"
    else:
        os.environ.pop("CLAWS_GREEDY", None)

    tmp = os.path.join(HERE, "data", "eval_tmp", name.replace(".pt", ""))
    shutil.rmtree(tmp, ignore_errors=True)
    FacilityCompass(FacilityConfig(
        characters=["Ironclad"], ascension=ascension, seeds=seeds, n_branch_workers=workers,
        n_main_trials=runs, raw_dir=tmp,
    )).run()
    row = progress.summarize(tmp, label)
    shutil.rmtree(tmp, ignore_errors=True)
    if row is None:
        raise SystemExit(f"{name}: no finished runs")
    row.update(ckpt=name, greedy=greedy)
    return row


def show(rows: list[dict], last: int = 14) -> None:
    print(f"{'label':>8} {'checkpoint':<28} {'runs':>4} {'floor':>11} {'hp lost/combat':>14} {'combat win':>10} {'turns':>6}")
    for r in rows[-last:]:
        floor = f"{r['mean_end_floor']:.2f}±{r['floor_se']}" if r.get("floor_se") is not None else f"{r['mean_end_floor']:.2f}"
        print(f"{str(r['label']):>8} {r['ckpt']:<28} {r['runs']:4d} {floor:>11} {str(r['mean_hp_lost_per_combat']):>14} "
              f"{str(r.get('combat_win_rate')):>10} {str(r.get('mean_turns')):>6}" + ("  (greedy)" if r.get("greedy") else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", action="append", default=[], help='a claws .pt, or "random"; repeatable')
    parser.add_argument("--backups", type=int, default=0, help="also evaluate N evenly spaced checkpoints from claws/model/backups/")
    parser.add_argument("--label", default=None, help="tag for the rows (e.g. the training round); default: the checkpoint's timestamp")
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--seeds", nargs="+", default=[f"eval-{i}" for i in range(1, 9)])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ascension", type=int, default=1)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--log", default=LOG)
    parser.add_argument("--show", action="store_true", help="print the log and exit")
    args = parser.parse_args()
    LOG_PATH = args.log

    if not args.show:
        ckpts = list(args.ckpt)
        files = sorted(glob.glob(os.path.join(BACKUPS, "claws_*.pt")))
        if args.backups and files:
            step = max(1, len(files) // args.backups)
            ckpts = files[::step][-args.backups:] + ckpts
        if not ckpts:
            parser.error("give --ckpt and/or --backups (or --show)")
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        for ckpt in ckpts:
            label = args.label or ("-" if ckpt == "random" else os.path.basename(ckpt).replace("claws_", "").replace(".pt", ""))
            row = evaluate(ckpt, label, args.runs, args.seeds, args.workers, args.greedy, args.ascension)
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(row) + "\n")
            show([row], 1)

    if os.path.exists(LOG_PATH):
        print("\n--- log ---")
        show([json.loads(line) for line in open(LOG_PATH) if line.strip()])


if __name__ == "__main__":
    main()

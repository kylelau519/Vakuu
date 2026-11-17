"""How well does the whole agent play? Plays fresh random seeds (a new map every run) with each Claws checkpoint, with
the golden compass choosing map / card rewards and NO exploration noise: Claws plays its top action (no
Dirichlet noise) and the compass its top-scored option (COMPASS_EPS=0, COMPASS_TAU~0). The compass checkpoint
is the same in every row of one invocation, so rows still differ only in how Claws fights, but the route and
deck are the compass's. Training's own `value_rmse` can't answer this -- it is error on the training data.

    python eval_claws.py --ckpt random --ckpt ../claws/model/claws.pt          # one comparison
    python eval_claws.py --backups 6 --ckpt ../claws/model/claws.pt --ckpt random   # a curve over past rounds
    python eval_claws.py --no-compass --sample ...   # the old eval: random map, tier-list cards, noisy Claws
    python eval_claws.py --show                                                # just print the log

Per checkpoint: runs, mean floor reached (+- standard error), HP lost per combat (share of max HP, deaths
count in full), the share of combats won, turns per combat. Higher floor / win rate and lower HP lost mean
better. A difference smaller than about two standard errors is noise: raise --runs.

Rows append to data/claws_eval.jsonl; each records `greedy` and `compass` (the compass checkpoint's file name
and mtime, or false). Rows from before this change have no `compass` key: they ran with the compass off and
Claws sampled with noise, so they are not comparable with new rows. `random` is the untrained baseline
(uniform-random legal actions). The compass checkpoint is --compass-ckpt, else $GOLDEN_COMPASS_CHECKPOINT, else
golden_compass/golden_compass.pt; if none exists the compass is off and the row says so."""
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


DEFAULT_COMPASS = os.path.normpath(os.path.join(HERE, "..", "golden_compass", "golden_compass.pt"))


def evaluate(ckpt: str, label: str, runs: int, seeds: list[str] | None, workers: int, greedy: bool, ascension: int,
             compass: str | None = None) -> dict:
    name = "random" if ckpt == "random" else os.path.basename(ckpt)
    # Spawned workers read these at import, so set them before the facility starts. The compass checkpoint is
    # the same for every checkpoint of one invocation; with it off, map / card choices are random / tier list.
    compass_row: str | bool = False
    if compass and os.path.exists(compass):
        os.environ["GOLDEN_COMPASS_CHECKPOINT"] = os.path.abspath(compass)
        compass_row = f"{os.path.basename(compass)}@{int(os.path.getmtime(compass))}"
        if greedy:  # no exploration: always the top-scored option
            os.environ["COMPASS_EPS"] = "0"
            os.environ["COMPASS_TAU"] = "0"  # compass.py floors the temperature at 1e-6
        else:
            os.environ.pop("COMPASS_EPS", None)
            os.environ.pop("COMPASS_TAU", None)
    else:
        if compass:
            print(f"warning: no compass checkpoint at {compass}; evaluating with the compass off")
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
    row.update(ckpt=name, greedy=greedy, compass=compass_row)
    return row


def show(rows: list[dict], last: int = 14) -> None:
    print(f"{'label':>8} {'checkpoint':<28} {'runs':>4} {'floor (total)':>13} {'boss':>9} {'act2':>5} {'hp lost/combat':>14} {'combat win':>10} {'turns':>6}")
    for r in rows[-last:]:
        floor = f"{r['mean_end_floor']:.2f}±{r['floor_se']}" if r.get("floor_se") is not None else f"{r['mean_end_floor']:.2f}"
        print(f"{str(r['label']):>8} {r['ckpt']:<28} {r['runs']:4d} {floor:>13} {str(r.get('reached_boss', '?')) + '/' + str(r['runs']):>9} {r.get('act2_rate', 0):>5.0%} {str(r['mean_hp_lost_per_combat']):>14} "
              f"{str(r.get('combat_win_rate')):>10} {str(r.get('mean_turns')):>6}" + ("  (greedy)" if r.get("greedy") else "")
              + ("  +compass" if r.get("compass") else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", action="append", default=[], help='a claws .pt, or "random"; repeatable')
    parser.add_argument("--backups", type=int, default=0, help="also evaluate N evenly spaced checkpoints from claws/model/backups/")
    parser.add_argument("--label", default=None, help="tag for the rows (e.g. the training round); default: the checkpoint's timestamp")
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--seeds", nargs="+", default=None, help="default: a fresh random seed per run")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ascension", type=int, default=1)
    parser.add_argument("--sample", action="store_true",
                        help="play sampled with exploration noise (Claws' Dirichlet, the compass's eps/tau) "
                             "instead of the top choice; the old default")
    parser.add_argument("--no-compass", action="store_true", help="random map, tier-list cards (the old eval)")
    parser.add_argument("--compass-ckpt", default=os.environ.get("GOLDEN_COMPASS_CHECKPOINT") or DEFAULT_COMPASS)
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
            row = evaluate(ckpt, label, args.runs, args.seeds, args.workers, not args.sample, args.ascension,
                           None if args.no_compass else args.compass_ckpt)
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(row) + "\n")
            show([row], 1)

    if os.path.exists(LOG_PATH):
        print("\n--- log ---")
        show([json.loads(line) for line in open(LOG_PATH) if line.strip()])


if __name__ == "__main__":
    main()

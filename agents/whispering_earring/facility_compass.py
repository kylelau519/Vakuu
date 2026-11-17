"""Golden Compass data generation: plain whole runs at real HP (jobs.RunJob), one file per run, no
forks. Each run teaches the hazard net which states died on which floor. The macro policy is the
current hazard net with exploration (golden_compass/compass.py), or uniform random / the tier list
while no GOLDEN_COMPASS_CHECKPOINT exists.

The combat agent is whatever CLAWS_CHECKPOINT points at, so this data is only valid for that agent:
retrain the hazard net after Claws changes (training_compass.sh does this each round)."""
from __future__ import annotations

import glob
import os
import random
from queue import Empty

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)

from checkpoint import Checkpoint
from facility_common import FacilityConfig, NewFacility, add_common_cli_args
from jobs import Job, RunJob, load_act_checkpoint
from utils import new_run


class FacilityCompass(NewFacility):
    def __init__(self, cfg: FacilityConfig):
        super().__init__(cfg)
        self._started = 0
        self._jobs: dict[str, Job] = {}
        self._floors: list[int] = []
        self._tally = {"lost": 0, "won": 0, "censored": 0, "error": 0}
        self._capped = 0
        self.resume_pool: list[str] = []
        if cfg.resume_act:
            self.resume_pool = sorted(glob.glob(os.path.join(cfg.ckpt_dir, f"*_act{cfg.resume_act}.ckpt.json")))
            if not self.resume_pool:
                raise SystemExit(f"--resume-act {cfg.resume_act}: no checkpoints in {cfg.ckpt_dir} yet "
                                 "(they are written as runs reach that act; run without --resume-act first)")

    def _more_runs(self) -> bool:
        return self.cfg.n_main_trials is None or self._started < self.cfg.n_main_trials

    def _top_up(self) -> None:
        """Keeps two jobs per worker queued so no worker idles between runs."""
        while self._more_runs() and len(self._jobs) < 2 * self.cfg.n_branch_workers:
            self._started += 1
            if self.resume_pool:  # skip the acts a past run already beat: start from one of its act-boundary saves
                sidecar = random.choice(self.resume_pool)
                src = os.path.basename(sidecar)[:-len(".ckpt.json")]
                cp, offset, acts = load_act_checkpoint(sidecar, f"{src}_r{os.urandom(3).hex()}")
                job = RunJob(cp, self.cfg.raw_dir, self.cfg.ckpt_dir, offset, acts)
            else:
                character = random.choice(self.cfg.characters)
                seed, main_id = new_run(character, self.cfg.seeds, self._started - 1)
                job = RunJob(Checkpoint.fresh(main_id, character, seed, self.cfg.ascension),
                             self.cfg.raw_dir, self.cfg.ckpt_dir)
            self._jobs[job.uid()] = job
            self.job_queue.put(job)

    def _record(self, uid: str, result) -> None:
        """Folds one finished run into the running tally on the status row. (A failed job's own crash line is
        already printed by the worker, so it only counts here.)"""
        aux = result.auxiliary_data
        if aux.get("error"):
            self._tally["error"] += 1
            last = f"{uid} FAILED"
        else:
            self._tally[result.outcome] = self._tally.get(result.outcome, 0) + 1
            self._floors.append(aux["total_floor"])
            last = f"{uid} {result.outcome} @floor {aux['total_floor']} (act {aux['acts_reached']})"
            if aux.get("capped"):
                self._capped += 1
                last += f" CAPPED ({aux['capped']})"
        done = sum(self._tally.values())
        total = self.cfg.n_main_trials if self.cfg.n_main_trials is not None else "inf"
        mean = sum(self._floors) / len(self._floors) if self._floors else 0.0
        self.status(
            f"runs {done}/{total} | died {self._tally['lost']} won {self._tally['won']} "
            f"stalled {self._tally['censored']} (capped {self._capped}) err {self._tally['error']} | "
            f"mean floor {mean:.1f} best {max(self._floors, default=0)} | last: {last}"
        )

    def run(self) -> None:
        self.start()
        try:
            self._top_up()
            while self._jobs:
                try:
                    uid, result = self.result_queue.get(timeout=self.cfg.stall_timeout_s)
                except Empty:
                    self.live.scroll(f"No result for {self.cfg.stall_timeout_s:.0f}s -- treating the pool as wedged.")
                    break
                self._jobs.pop(uid, None)
                self._record(uid, result)
                self._top_up()
        except KeyboardInterrupt:
            self.live.scroll("Stopping facility (waiting for in-flight jobs to finish)...")
        finally:
            self.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Whispering Earring golden_compass data facility (real-HP runs)")
    add_common_cli_args(parser)
    parser.add_argument("--ckpt-dir", default="data/act_checkpoints",
                        help="where each run's first-map-node-of-act-N saves go (N >= 2), and where --resume-act reads them")
    parser.add_argument("--resume-act", type=int, default=0,
                        help="start every run from a random saved checkpoint of this act (2 = just past the act-1 boss) "
                             "instead of from floor 0 (default: 0 = fresh runs)")
    args = parser.parse_args()

    FacilityCompass(
        FacilityConfig(
            characters=args.characters,
            ascension=args.ascension,
            seeds=args.seeds,
            n_branch_workers=args.branch_workers,
            n_main_trials=args.trials,
            raw_dir=args.raw_dir,
            ckpt_dir=args.ckpt_dir,
            resume_act=args.resume_act,
            verbose=args.verbose,
        )
    ).run()

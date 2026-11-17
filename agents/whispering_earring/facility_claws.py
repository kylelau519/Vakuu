"""Claws data generation: a mainline plays a run straight from the policy, then every turn of every
combat it played is forked `forks_per_turn` times, each fork playing that combat out from the same
checkpoint. The mainline never adopts a fork, so nothing waits on anything: a mainline job runs
first, its forks are queued when it returns, and every job writes its own file (see jobs.py)."""
from __future__ import annotations

import glob
import json
import multiprocessing as mp
import os
import random
from queue import Empty

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)

from checkpoint import Checkpoint
from dedup import ForkDedup
from facility_common import FacilityConfig, NewFacility, add_common_cli_args
from jobs import ForkJob, Job, JobResult, MainlineJob
from utils import new_run


class FacilityClaws(NewFacility):
    def __init__(self, cfg: FacilityConfig):
        super().__init__(cfg)
        self._save_dir = os.path.join(cfg.raw_dir, "checkpoints")
        self._started = 0
        self._jobs: dict[str, Job] = {}  # uid -> job, for every job still out
        self._forks_left: dict[str, int] = {}  # main_id -> forks still to report (a "pipeline" is open until 0)
        self._mainlines_done = 0
        self._forks_queued = 0
        self._forks_done = 0
        self._errors = 0
        self._last = ""
        self._manager = None  # mp.Manager, only with cfg.dedup_forks: owns each open run's shared key registry
        self._dedups: dict[str, ForkDedup] = {}  # main_id -> its registry, while the run is open
        self._dup_stats: dict[str, dict[str, dict]] = {}  # main_id -> {"c001t01": {"forks", "discarded", "multiplicity"}}
        self._dups_discarded = 0

    def _more_runs(self) -> bool:
        return self.cfg.n_main_trials is None or self._started < self.cfg.n_main_trials

    def _submit(self, job, priority: bool = False) -> None:
        self._jobs[job.uid()] = job
        (self.priority_job_queue if priority else self.job_queue).put(job)

    def _start_run(self) -> None:
        character = random.choice(self.cfg.characters)
        seed, main_id = new_run(character, self.cfg.seeds, self._started)
        self._started += 1
        self._forks_left[main_id] = 1  # the mainline itself; forks are added when it returns
        dedup = None
        if self.cfg.dedup_forks:
            dedup = self._dedups[main_id] = ForkDedup(self._manager.dict())
            self._dup_stats[main_id] = {}
        self._submit(MainlineJob(
            Checkpoint.fresh(main_id, character, seed, self.cfg.ascension), self._save_dir, self.cfg.raw_dir,
            dedup=dedup,
        ), priority=True)

    def _fork_all(self, mainline: MainlineJob, result: JobResult) -> int:
        """Queues forks_per_turn forks of every turn of every combat the mainline played."""
        cp = mainline.checkpoint
        n = 0
        for combat in result.auxiliary_data["combats"]:
            if combat["outcome"] == "stalled":
                continue
            for turn_index, turn in enumerate(combat["turns"], start=1):
                at_turn = Checkpoint(
                    cp.main_id, cp.character, cp.seed, cp.ascension,
                    actions=result.action_log[:turn["start_index"]],
                    save_path=combat["save_path"], save_index=combat["save_index"],
                )
                for fork_index in range(self.cfg.forks_per_turn):
                    self._submit(ForkJob(at_turn, combat["combat_index"], turn_index, fork_index, self.cfg.raw_dir,
                                         dedup=self._dedups.get(cp.main_id)))
                    n += 1
                if cp.main_id in self._dup_stats:
                    self._dup_stats[cp.main_id][f"c{combat['combat_index']:03d}t{turn_index:02d}"] = {
                        "forks": self.cfg.forks_per_turn, "discarded": 0, "multiplicity": {}}
        return n

    def _report(self) -> None:
        total = self.cfg.n_main_trials if self.cfg.n_main_trials is not None else "inf"
        self.status(
            f"mainlines {self._mainlines_done}/{total} | forks {self._forks_done}/{self._forks_queued} "
            f"| dup {self._dups_discarded} | err {self._errors} | last: {self._last or '-'}"
        )

    def _close_if_done(self, main_id: str) -> None:
        if self._forks_left[main_id] > 0:
            return
        del self._forks_left[main_id]
        self._dedups.pop(main_id, None)
        self._write_dedup_stats(main_id)
        for path in glob.glob(os.path.join(self._save_dir, f"{main_id}_a*.json")):
            os.remove(path)  # nothing can still need this run's saves

    def _write_dedup_stats(self, main_id: str) -> None:
        """raw_dir/dedup/{main_id}.json: per group, how many forks were discarded as duplicates, and how many
        each surviving member stands for. Its own folder, because materialize.py reads every *.json directly
        under raw_dir as a trial file."""
        stats = self._dup_stats.pop(main_id, None)
        if stats is None:
            return
        os.makedirs(os.path.join(self.cfg.raw_dir, "dedup"), exist_ok=True)
        with open(os.path.join(self.cfg.raw_dir, "dedup", f"{main_id}.json"), "w") as f:
            json.dump({"main_id": main_id, "groups": stats}, f, indent=1)

    def run(self) -> None:
        """Keeps n_main_workers runs (a mainline plus its forks) open until n_main_trials have started."""
        self.start()
        if self.cfg.dedup_forks:
            self._manager = mp.Manager()
        try:
            while len(self._forks_left) < self.cfg.n_main_workers and self._more_runs():
                self._start_run()
            while self._forks_left:
                try:
                    uid, result = self.result_queue.get(timeout=self.cfg.stall_timeout_s)
                except Empty:
                    self.live.scroll(f"No result for {self.cfg.stall_timeout_s:.0f}s -- treating the pool as wedged.")
                    break
                job = self._jobs.pop(uid, None)
                error = result.auxiliary_data.get("error")
                self._errors += bool(error)  # the failing worker already printed its own crash line
                if isinstance(job, MainlineJob):
                    n_forks = 0 if error else self._fork_all(job, result)
                    self._forks_left[job.main_id] += n_forks - 1
                    self._mainlines_done += 1
                    self._forks_queued += n_forks
                    if not error:
                        combats = result.auxiliary_data["combats"]
                        self._last = (f"{job.main_id} {result.outcome}, {len(combats)} combats, "
                                      f"{sum(c['loss'] for c in combats)} HP lost")
                elif job is not None:
                    self._forks_left[job.main_id] -= 1
                    self._forks_done += 1
                    owner = result.auxiliary_data.get("duplicate_of")
                    if owner is not None and job.main_id in self._dup_stats:
                        g = self._dup_stats[job.main_id][f"c{job.combat_index:03d}t{job.turn_index:02d}"]
                        g["discarded"] += 1
                        g["multiplicity"][owner] = g["multiplicity"].get(owner, 1) + 1  # owner's own member counts 1
                        self._dups_discarded += 1
                self._report()
                if job is not None:
                    self._close_if_done(job.main_id)
                while len(self._forks_left) < self.cfg.n_main_workers and self._more_runs():
                    self._start_run()
        except KeyboardInterrupt:
            self.live.scroll("Stopping facility (waiting for in-flight jobs to finish)...")
        finally:
            self.stop()
            if self._manager is not None:
                self._manager.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Whispering Earring claws data facility (mainline + forks)")
    add_common_cli_args(parser)
    parser.add_argument("--forks-per-turn", type=int, default=4,
                        help="forks of every mainline combat turn (default: 4; the design calls for ~50)")
    parser.add_argument("--dedup-forks", action="store_true",
                        help="drop a fork the moment its compared turn ends in a state (visible + hidden: RNG "
                             "streams, draw order, monster queue) an earlier member of its group already "
                             "reached, instead of playing out the combat; discards are counted per group in "
                             "raw-dir/dedup/ (default: off)")
    args = parser.parse_args()

    FacilityClaws(
        FacilityConfig(
            characters=args.characters,
            ascension=args.ascension,
            seeds=args.seeds,
            n_branch_workers=args.branch_workers,
            forks_per_turn=args.forks_per_turn,
            dedup_forks=args.dedup_forks,
            n_main_workers=args.main_workers,
            n_main_trials=args.trials,
            raw_dir=args.raw_dir,
            verbose=args.verbose,
        )
    ).run()

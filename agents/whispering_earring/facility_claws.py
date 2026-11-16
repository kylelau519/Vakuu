"""Claws data generation: a mainline plays a run straight from the policy, then every turn of every
combat it played is forked `forks_per_turn` times, each fork playing that combat out from the same
checkpoint. The mainline never adopts a fork, so nothing waits on anything: a mainline job runs
first, its forks are queued when it returns, and every job writes its own file (see jobs.py)."""
from __future__ import annotations

import glob
import os
import random
from queue import Empty

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)

from checkpoint import Checkpoint
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
        self._submit(MainlineJob(
            Checkpoint.fresh(main_id, character, seed, self.cfg.ascension), self._save_dir, self.cfg.raw_dir,
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
                    self._submit(ForkJob(at_turn, combat["combat_index"], turn_index, fork_index, self.cfg.raw_dir))
                    n += 1
        return n

    def _close_if_done(self, main_id: str) -> None:
        if self._forks_left[main_id] > 0:
            return
        del self._forks_left[main_id]
        for path in glob.glob(os.path.join(self._save_dir, f"{main_id}_a*.json")):
            os.remove(path)  # nothing can still need this run's saves

    def run(self) -> None:
        """Keeps n_main_workers runs (a mainline plus its forks) open until n_main_trials have started."""
        self.start()
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
                if error:
                    self.live.scroll(f"{uid} failed: {error}")
                if isinstance(job, MainlineJob):
                    n_forks = 0 if error else self._fork_all(job, result)
                    self._forks_left[job.main_id] += n_forks - 1
                    if not error:
                        combats = result.auxiliary_data["combats"]
                        self.live.scroll(
                            f"{job.main_id}: {result.outcome}, {len(combats)} combats "
                            f"({sum(c['loss'] for c in combats)} HP lost), {n_forks} forks queued"
                        )
                elif job is not None:
                    self._forks_left[job.main_id] -= 1
                if job is not None:
                    self._close_if_done(job.main_id)
                while len(self._forks_left) < self.cfg.n_main_workers and self._more_runs():
                    self._start_run()
        except KeyboardInterrupt:
            self.live.scroll("Stopping facility (waiting for in-flight jobs to finish)...")
        finally:
            self.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Whispering Earring claws data facility (mainline + forks)")
    add_common_cli_args(parser)
    parser.add_argument("--forks-per-turn", type=int, default=4,
                        help="forks of every mainline combat turn (default: 4; the design calls for ~50)")
    args = parser.parse_args()

    FacilityClaws(
        FacilityConfig(
            characters=args.characters,
            ascension=args.ascension,
            seeds=args.seeds,
            n_branch_workers=args.branch_workers,
            forks_per_turn=args.forks_per_turn,
            n_main_workers=args.main_workers,
            n_main_trials=args.trials,
            raw_dir=args.raw_dir,
            verbose=args.verbose,
        )
    ).run()

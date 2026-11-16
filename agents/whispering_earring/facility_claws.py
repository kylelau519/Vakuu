"""Combat turn-branching: the Whispering Earring self-play facility generating training data for
`claws`, built on jobs (see jobs.py) run by a pool of BufferWorkers (see facility_common.NewFacility)."""
from __future__ import annotations

import random
from queue import Empty

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)

from checkpoint import Checkpoint
from facility_common import FacilityConfig, NewFacility, add_common_cli_args
from jobs import TrialJob
from utils import random_seed


class FacilityClaws(NewFacility):
    def __init__(self, cfg: FacilityConfig):
        super().__init__(cfg)
        self._started = 0

    def _more_trials(self) -> bool:
        return self.cfg.n_main_trials is None or self._started < self.cfg.n_main_trials

    def _submit_trial(self) -> None:
        character = random.choice(self.cfg.characters)
        seed = random_seed()
        self._started += 1
        self.submit_job(TrialJob(
            Checkpoint.fresh(f"{character}_{seed}", character, seed, self.cfg.ascension),
            self.cfg.raw_dir,
        ))

    def submit_job(self, job, priority: bool = False) -> None:
        (self.priority_job_queue if priority else self.job_queue).put(job)

    def run(self) -> None:
        """Keeps n_main_workers trials in flight until n_main_trials have been started (None = forever)."""
        self.start()
        try:
            in_flight = 0
            while in_flight < self.cfg.n_main_workers and self._more_trials():
                self._submit_trial()
                in_flight += 1
            while in_flight:
                try:
                    uid, result = self.result_queue.get(timeout=self.cfg.stall_timeout_s)
                except Empty:
                    self.live.scroll(f"No result for {self.cfg.stall_timeout_s:.0f}s -- treating the pool as wedged.")
                    break
                in_flight -= 1
                error = result.auxiliary_data.get("error")
                self.live.scroll(
                    f"{uid}: {result.outcome}, {len(result.action_log)} actions" + (f" ({error})" if error else "")
                )
                if self._more_trials():
                    self._submit_trial()
                    in_flight += 1
        except KeyboardInterrupt:
            self.live.scroll("Stopping facility (waiting for in-flight jobs to finish)...")
        finally:
            self.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Whispering Earring self-play facility (combat turn-branching)")
    add_common_cli_args(parser)
    parser.add_argument("--branches-per-turn", type=int, default=3)
    args = parser.parse_args()

    FacilityClaws(
        FacilityConfig(
            characters=args.characters,
            ascension=args.ascension,
            n_branch_workers=args.branch_workers,
            branches_per_turn=args.branches_per_turn,
            n_main_workers=args.main_workers,
            n_main_trials=args.trials,
            raw_dir=args.raw_dir,
            verbose=args.verbose,
        )
    ).run()

"""Combat turn-branching: the Whispering Earring self-play facility generating training data for
`claws`, built on jobs (see jobs.py) run by a pool of BufferWorkers (see facility_common.NewFacility)."""
from __future__ import annotations

import random

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)

from checkpoint import Checkpoint
from facility_common import FacilityConfig, NewFacility, add_common_cli_args
from jobs import TrialJob
from utils import random_seed


class FacilityClaws(NewFacility):
    def __init__(self, cfg: FacilityConfig):
        super().__init__(cfg)

    def run(self) -> None:
        self.start()
        for _ in range(self.cfg.n_main_trials):
            self.create_main_job()

    def create_main_job(self):
        character = random.choice(self.cfg.characters)
        seed = random_seed()
        trial_job = TrialJob(
            Checkpoint.fresh(f"{character}_{seed}", character, seed, self.cfg.ascension)
        )
        self.priority_job_queue.put(trial_job)


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

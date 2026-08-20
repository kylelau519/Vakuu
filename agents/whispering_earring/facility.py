"""The facility: one main worker running the main branch of a run, N sub-workers each
idle until a branch job appears in the queue.

Only the main branch creates branches: at the first combat_play decision of a level, it
writes `branches_per_combat` checkpoint .json files (clone of its action log so far, plus
an "explore" tag) into the queue dir and pushes their paths onto the job queue. An idle
sub-worker picks one up, replays a fresh subprocess to that checkpoint, and plays out
*only* that one combat with the exploration policy -- then the branch is discarded,
win or lose. The main branch is never interrupted by this; it keeps playing forward on
its own subprocess.

If the main branch dies (game_over), it's saved like everything else, and the same
worker process immediately starts a new run with a fresh seed. FacilityConfig.n_main_trials
caps how many main-line trials to run before stopping (None runs indefinitely, Ctrl-C to
stop); either way, once the main worker is done, branch workers drain whatever's still
queued before exiting -- no branch job gets abandoned just because the main line stopped.
"""
from __future__ import annotations

import multiprocessing as mp
import random
import time
from dataclasses import dataclass, field
from queue import Empty

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)
from env import Sts2Session
from installation import ensure_setup

from checkpoint import BranchJob, read_branch_job, write_branch_job
from policy import choose_action
from recorder import TrialRecorder

_SEED_WORDS = ["falcon", "ember", "wolf", "raven", "thorn", "comet", "storm", "harbor"]


def _random_seed() -> str:
    return f"{random.choice(_SEED_WORDS)}-{random.randint(0, 999)}"


@dataclass
class FacilityConfig:
    characters: list[str] = field(default_factory=lambda: ["Ironclad"])
    ascension: int = 0
    branches_per_combat: int = 3  # side-copies spawned per combat, on top of the main line
    n_branch_workers: int = 4
    n_main_trials: int | None = None  # stop after this many main-line trials; None = forever
    raw_dir: str = "data/raw"
    queue_dir: str = "data/queue"


def _play_combat_only(session: Sts2Session, decision) -> str:
    """Drives one combat to resolution (win -> next decision, or death -> game_over),
    from wherever `decision` currently sits. Used by branch workers, which only ever
    care about a single combat and stop the moment it resolves. session.step() logs to
    session.logger itself, so this loop doesn't touch the recorder directly."""
    while decision.decision == "combat_play":
        name, kwargs = choose_action(decision, session)
        decision = session.step(name, **kwargs)
    return "died" if decision.decision == "game_over" else "won"


def _spawn_branches(job_queue, cfg: FacilityConfig, main_id: str, floor: int, combat_index: int,
                     character: str, seed: str, actions_so_far: list[dict]) -> None:
    for b in range(cfg.branches_per_combat):
        job = BranchJob(
            main_id=main_id, floor=floor, combat_index=combat_index, branch_index=b,
            character=character, seed=seed, ascension=cfg.ascension,
            actions=list(actions_so_far),
        )
        path = write_branch_job(job, cfg.queue_dir)
        job_queue.put(path)


def main_worker_loop(cfg: FacilityConfig, job_queue: mp.Queue, stop_event) -> None:
    completed = 0
    while not stop_event.is_set() and (cfg.n_main_trials is None or completed < cfg.n_main_trials):
        character = random.choice(cfg.characters)
        seed = _random_seed()
        main_id = f"{character}_{seed}"

        recorder = TrialRecorder(main_id, cfg.raw_dir, {
            "character": character, "seed": seed, "main_id": main_id, "role": "main",
        })
        session = Sts2Session(logger=recorder)
        try:
            try:
                decision = session.start(character, seed, cfg.ascension)
                recorder.log_state(decision)
                prev_type = None
                combat_index = 0

                while decision.decision != "game_over":
                    if decision.decision == "combat_play" and prev_type != "combat_play":
                        combat_index += 1
                        _spawn_branches(job_queue, cfg, main_id, decision.context.floor, combat_index,
                                         character, seed, session.action_log)

                    name, kwargs = choose_action(decision, session)
                    prev_type = decision.decision
                    decision = session.step(name, **kwargs)
            except Exception as e:
                # a crashed trial (Sts2ActionError from an invalid action, or anything
                # else) is still a dead main branch -- log it, save what we have, and
                # let the outer while restart with a fresh seed, same as a normal
                # game_over. Catching broadly on purpose: no failure here should be able
                # to take the whole worker process down with it.
                recorder.log_error(f"{type(e).__name__}: {e}")
            try:
                session.save(f"{cfg.raw_dir}/{main_id}.json")
            except Exception as e:
                recorder.log_error(f"save failed: {type(e).__name__}: {e}")
        finally:
            recorder.close()
            session.close()
        completed += 1
        # main branch is dead -- loop restarts with a fresh seed unless the target
        # trial count is reached or stop_event is set, checked at the top of the while

    stop_event.set()  # no more branch jobs coming; branch workers drain the queue and exit


def branch_worker_loop(cfg: FacilityConfig, job_queue: mp.Queue, stop_event) -> None:
    while True:
        try:
            path = job_queue.get(timeout=1.0)
        except Empty:
            if stop_event.is_set():
                return
            continue
        if path is None:  # shutdown sentinel
            return

        job = read_branch_job(path)
        recorder = TrialRecorder(job.branch_id, cfg.raw_dir, {
            "character": job.character, "seed": job.seed, "main_id": job.main_id,
            "floor": job.floor, "combat_index": job.combat_index,
            "branch_index": job.branch_index, "role": "branch",
        })
        session = Sts2Session(logger=recorder)
        try:
            try:
                decision = session.open_from_checkpoint(job.character, job.seed, job.ascension, job.actions)
                recorder.log_state(decision)
                _play_combat_only(session, decision)
            except Exception as e:
                recorder.log_error(f"{type(e).__name__}: {e}")
            try:
                session.save(f"{cfg.raw_dir}/{job.branch_id}.json")
            except Exception as e:
                recorder.log_error(f"save failed: {type(e).__name__}: {e}")
        finally:
            recorder.close()
            session.close()


class Facility:
    def __init__(self, cfg: FacilityConfig):
        self.cfg = cfg
        self.job_queue: mp.Queue = mp.Queue()
        self.stop_event = mp.Event()
        self.main_proc: mp.Process | None = None
        self.branch_procs: list[mp.Process] = []

    def start(self) -> None:
        ensure_setup()  # once, here -- not per-session, which would repeat the same checks

        self.main_proc = mp.Process(
            target=main_worker_loop, args=(self.cfg, self.job_queue, self.stop_event), daemon=True,
        )
        self.main_proc.start()

        for _ in range(self.cfg.n_branch_workers):
            p = mp.Process(
                target=branch_worker_loop, args=(self.cfg, self.job_queue, self.stop_event), daemon=True,
            )
            p.start()
            self.branch_procs.append(p)

    def stop(self, timeout: float = 60) -> None:
        self.stop_event.set()
        for _ in self.branch_procs:
            self.job_queue.put(None)
        if self.main_proc:
            self.main_proc.join(timeout)
        for p in self.branch_procs:
            p.join(timeout)

    def run(self) -> None:
        """Blocks until done: forever (Ctrl-C to stop) if cfg.n_main_trials is None,
        otherwise until the main worker hits that count and the branch queue drains."""
        self.start()
        try:
            if self.cfg.n_main_trials is None:
                while True:
                    time.sleep(60)
            else:
                self.main_proc.join()
                print(f"Main worker finished {self.cfg.n_main_trials} trial(s) -- "
                      f"draining remaining branch queue...")
        except KeyboardInterrupt:
            print("\nStopping facility (waiting for in-flight trials to finish)...")
        finally:
            self.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Whispering Earring self-play facility")
    parser.add_argument("--trials", type=int, default=None,
                         help="stop after this many main-line trials complete (default: run forever, Ctrl-C to stop)")
    parser.add_argument("--characters", nargs="+", default=["Ironclad"])
    parser.add_argument("--branch-workers", type=int, default=4)
    parser.add_argument("--branches-per-combat", type=int, default=3)
    args = parser.parse_args()

    facility = Facility(FacilityConfig(
        characters=args.characters,
        n_branch_workers=args.branch_workers,
        branches_per_combat=args.branches_per_combat,
        n_main_trials=args.trials,
    ))
    facility.run()

"""The facility: one main worker running the main branch of a run, N sub-workers each
idle until a branch job appears in the queue.

Combat is delegated entirely to branches, per model/README.md's MCTS strategy section: "among
the K attempts, if any won, use that attempt's resulting state... as the main branch. If all K
attempts lose, accept the loss and end that lineage there." At every combat_play decision, the
main worker writes `branches_per_combat` checkpoint .json files (clone of its action log so far)
into the queue dir, pushes their paths onto the job queue, and *blocks* waiting for all K to
finish (see _resolve_combat) -- it does not play that combat itself. Once results are in: if any
branch won, its full action log (checkpoint prefix + the combat it just played) is adopted as the
main line's own continuation -- the main worker's session is torn down and reopened from that
action log, and the run proceeds from there. If none won, one attempt's action log is still
adopted the same way (so the fatal combat gets recorded -- a loss is valid signal, see
recorder.py), and the lineage ends.

This makes the main line's throughput gated by branch-worker capacity -- every combat is now a
blocking round-trip through the job queue, not a background side-effect the main line ignores.
Branch workers themselves are unaffected: branch_worker_loop still just replays a checkpoint and
plays exactly one combat to resolution, same as before.

If the main branch dies (game_over) outside of combat (rare, but the loop doesn't assume
otherwise), it's saved like everything else, and the same worker process immediately starts a new
run with a fresh seed. FacilityConfig.n_main_trials caps how many main-line trials to run before
stopping (None runs indefinitely, Ctrl-C to stop); either way, once the main worker is done,
branch workers drain whatever's still queued before exiting -- no branch job gets abandoned just
because the main line stopped.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass, field
from queue import Empty

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)
from env import Sts2Session
from installation import ensure_setup

from checkpoint import BranchJob, read_branch_job, write_branch_job
from policy import choose_action
from recorder import CompactRecorder

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
    care about a single combat and stop the moment it resolves. session.step() logs
    actions/state to session.logger itself, so this loop only reaches into the recorder
    directly for attach_policy -- there's no other hook to pass a per-step policy through."""
    while decision.decision == "combat_play":
        name, kwargs, policy = choose_action(decision, session)
        if policy is not None:
            session.logger.attach_policy(policy)
        decision = session.step(name, **kwargs)
    # Must match _OUTCOME_VALUE's keys (recorder.py) and main_worker_loop's convention below, or
    # CompactRecorder.save() silently backfills no value at all.
    return "lost" if decision.decision == "game_over" else "won"


def _spawn_branches(job_queue, cfg: FacilityConfig, main_id: str, floor: int, combat_index: int,
                     character: str, seed: str, actions_so_far: list[dict]) -> list[str]:
    branch_ids = []
    for b in range(cfg.branches_per_combat):
        job = BranchJob(
            main_id=main_id, floor=floor, combat_index=combat_index, branch_index=b,
            character=character, seed=seed, ascension=cfg.ascension,
            actions=list(actions_so_far),
        )
        path = write_branch_job(job, cfg.queue_dir)
        job_queue.put(path)
        branch_ids.append(job.branch_id)
    return branch_ids


def _wait_for_branch_results(
    raw_dir: str, branch_ids: list[str], stop_event, poll_interval: float = 1.0, timeout: float = 600
) -> dict[str, dict]:
    """Polls for each branch's saved trial file (branch_worker_loop always writes one, win, lose,
    or error -- see its own try/finally). Returns whatever's readable once every id is accounted
    for, `stop_event` fires, or `timeout` elapses -- a branch that never finishes (a hung/crashed
    worker) is simply missing from the result, not an error raised here; the caller decides what
    "fewer than K results" means."""
    results: dict[str, dict] = {}
    pending = set(branch_ids)
    deadline = time.time() + timeout
    while pending and time.time() < deadline and not stop_event.is_set():
        for branch_id in list(pending):
            path = os.path.join(raw_dir, f"{branch_id}.json")
            if not os.path.exists(path):
                continue
            try:
                with open(path) as f:
                    results[branch_id] = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue  # still being written -- retry next poll
            pending.discard(branch_id)
        if pending:
            time.sleep(poll_interval)
    if pending:
        print(f"warning: {len(pending)}/{len(branch_ids)} branch(es) didn't finish within "
              f"{timeout}s: {sorted(pending)}")
    return results


def _clean_actions(actions: list[dict]) -> list[dict]:
    """Strips CompactRecorder's added `policy`/`value` keys back to bare {cmd, action, args?} --
    what Sts2Session.open_from_checkpoint replays over the wire."""
    return [
        {"cmd": e["cmd"], "action": e["action"], **({"args": e["args"]} if "args" in e else {})}
        for e in actions
    ]


def _absorb_combat(recorder: CompactRecorder, full_actions: list[dict], prefix_len: int) -> None:
    """Feeds an adopted branch's combat actions (with their recorded `policy`) through
    `recorder` as though the main line had played them directly -- the same attach_policy/
    log_action calls main_worker_loop's own loop would have made, just sourced from the branch's
    saved trial instead of a live session.step(). full_actions[:prefix_len] is the checkpoint
    prefix the branch was handed -- already accounted for by the main recorder, skipped here."""
    for entry in full_actions[prefix_len:]:
        if "policy" in entry:
            recorder.attach_policy(entry["policy"])
        recorder.log_action(entry)


def _resolve_combat(
    job_queue, cfg: FacilityConfig, recorder: CompactRecorder, session: Sts2Session,
    character: str, seed: str, main_id: str, floor: int, combat_index: int, stop_event,
):
    """Delegates one combat to cfg.branches_per_combat parallel attempts and adopts a result --
    see the module docstring. Returns (session, decision, outcome): outcome is None if a combat
    was resolved and the run should continue with the returned (freshly reopened) session/
    decision; otherwise outcome is "lost" and the caller should stop -- the returned
    session/decision are then just the unchanged inputs, since there was nothing to adopt."""
    prefix = list(session.action_log)
    branch_ids = _spawn_branches(job_queue, cfg, main_id, floor, combat_index, character, seed, prefix)
    results = _wait_for_branch_results(cfg.raw_dir, branch_ids, stop_event)

    # Guards against adopting a branch whose replay of the *prefix itself* failed (actions
    # shorter than what it was handed) -- see _resolve_combat's docstring; adopting that would
    # silently discard the run's actual history instead of continuing from it.
    usable = {bid: r for bid, r in results.items() if len(r.get("actions") or []) >= len(prefix)}
    winner = next((r for r in usable.values() if r.get("outcome") == "won"), None)
    adopted = winner or next(iter(usable.values()), None)

    if adopted is None:
        recorder.log_error(
            f"combat {combat_index}: no usable branch results ({len(usable)}/{len(branch_ids)} usable)"
        )
        return session, None, "lost"

    _absorb_combat(recorder, adopted["actions"], len(prefix))
    session.close()
    new_session = Sts2Session(logger=recorder)
    new_decision = new_session.open_from_checkpoint(
        character, seed, cfg.ascension, _clean_actions(adopted["actions"])
    )
    return new_session, new_decision, (None if winner is not None else "lost")


def main_worker_loop(cfg: FacilityConfig, job_queue: mp.Queue, stop_event) -> None:
    completed = 0
    while not stop_event.is_set() and (cfg.n_main_trials is None or completed < cfg.n_main_trials):
        character = random.choice(cfg.characters)
        seed = _random_seed()
        main_id = f"{character}_{seed}"

        recorder = CompactRecorder({
            "character": character, "seed": seed, "ascension": cfg.ascension,
            "main_id": main_id, "role": "main",
        })
        session = Sts2Session(logger=recorder)
        outcome = None
        try:
            try:
                decision = session.start(character, seed, cfg.ascension)
                combat_index = 0

                while decision.decision != "game_over":
                    if decision.decision == "combat_play":
                        combat_index += 1
                        session, decision, combat_outcome = _resolve_combat(
                            job_queue, cfg, recorder, session, character, seed, main_id,
                            decision.context.floor, combat_index, stop_event,
                        )
                        if combat_outcome is not None:
                            outcome = combat_outcome
                            break
                        continue

                    name, kwargs, policy = choose_action(decision, session)
                    if policy is not None:
                        recorder.attach_policy(policy)
                    decision = session.step(name, **kwargs)

                if outcome is None:  # loop ended via game_over, not a combat lineage-ending loss
                    outcome = "won" if decision.victory else "lost"
            except Exception as e:
                # a crashed trial (Sts2ActionError from an invalid action, or anything
                # else) is still a dead main branch -- log it, save what we have, and
                # let the outer while restart with a fresh seed, same as a normal
                # game_over. Catching broadly on purpose: no failure here should be able
                # to take the whole worker process down with it.
                recorder.log_error(f"{type(e).__name__}: {e}")
                outcome = "error"
            try:
                recorder.save(f"{cfg.raw_dir}/{main_id}.json", session.action_log, outcome=outcome)
            except Exception as e:
                recorder.log_error(f"save failed: {type(e).__name__}: {e}")
        finally:
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
        recorder = CompactRecorder({
            "character": job.character, "seed": job.seed, "ascension": job.ascension,
            "main_id": job.main_id, "floor": job.floor, "combat_index": job.combat_index,
            "branch_index": job.branch_index, "role": "branch",
        }, start_step=len(job.actions))
        session = Sts2Session(logger=recorder)
        outcome = None
        try:
            try:
                decision = session.open_from_checkpoint(job.character, job.seed, job.ascension, job.actions)
                outcome = _play_combat_only(session, decision)
            except Exception as e:
                recorder.log_error(f"{type(e).__name__}: {e}")
                outcome = "error"
            try:
                recorder.save(f"{cfg.raw_dir}/{job.branch_id}.json", session.action_log, outcome=outcome)
            except Exception as e:
                recorder.log_error(f"save failed: {type(e).__name__}: {e}")
        finally:
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

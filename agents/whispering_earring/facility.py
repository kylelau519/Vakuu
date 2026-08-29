from __future__ import annotations

import multiprocessing as mp
import os
import random
import threading
import time
from dataclasses import dataclass, field
from queue import Empty

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)
from env import Sts2ActionError, Sts2Session
from buffer_env import Sts2BufferSession
from installation import ensure_setup

from checkpoint import BranchJob
from policy import CombatSimBudget, choose_action
from recorder import CompactRecorder

_SEED_WORDS = ["falcon", "ember", "wolf", "raven", "thorn", "comet", "storm", "harbor"]

# README.md's Value head: "HP loss is used as the metric. Every combat allows mall hp loss as
# chip damage to avoid strongly discouraging values."
_CHIP_DAMAGE_ALLOWANCE = 3
_DEATH_VALUE = -1.0


def _random_seed() -> str:
    return f"{random.choice(_SEED_WORDS)}-{random.randint(0, 999_999)}"


_WORKER_TAG_COLORS = [31, 32, 33, 34, 35, 36, 91, 92, 93, 94, 95, 96]  # ANSI red..bright-cyan


def _tag_worker_stderr(label: str, worker_index: int) -> None:
    """Sts2BufferSession loads Sts2Headless's CLR in-process (clr_bridge.load_clr), and its
    [SIM]/[INFO] logging writes unconditionally to this process's real fd 2 (Console.Error
    dups it once on first access). With many branch workers as separate processes sharing one
    inherited terminal, those writes interleave with no way to tell workers apart. Redirect fd 2
    to a pipe *before* the CLR loads and relay it through a background thread that prefixes every
    line with a colored worker tag -- this must run before the first Sts2BufferSession() call in
    this process, since re-pointing fd 2 after the CLR has dup'd it is a no-op."""
    color = _WORKER_TAG_COLORS[worker_index % len(_WORKER_TAG_COLORS)]
    tag = f"\x1b[{color}m[{label} {worker_index}]\x1b[0m"

    real_stderr = os.fdopen(os.dup(2), "w")
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def relay() -> None:
        with os.fdopen(read_fd) as pipe_reader:
            for line in pipe_reader:
                real_stderr.write(f"{tag} {line}")
                real_stderr.flush()

    threading.Thread(target=relay, daemon=True, name=f"stderr-relay-{label}-{worker_index}").start()


def _combat_value(hp_before: int, hp_after: int, max_hp: int, died: bool) -> float:
    """Per-combat value target for recorder.py's end_combat()"""
    if died:
        return _DEATH_VALUE
    if not max_hp:
        return 1.0
    excess_loss = max(0, (hp_before - hp_after) - _CHIP_DAMAGE_ALLOWANCE)
    return 1.0 - excess_loss / max_hp


@dataclass
class FacilityConfig:
    characters: list[str] = field(default_factory=lambda: ["Ironclad"])
    ascension: int = 0
    branches_per_combat: int = 3  # side-copies spawned per combat, on top of the main line
    n_branch_workers: int = 4
    n_main_workers: int = 1  # concurrent main lines, sharing one branch-worker pool
    n_main_trials: int | None = None  # stop after this many main-line trials *total*; None = forever
    raw_dir: str = "data/raw"


def _play_combat_only(session: Sts2BufferSession, decision) -> str:
    hp_before, max_hp = decision.player_hp, decision.player_max_hp
    session.logger.begin_combat()
    budget = CombatSimBudget()
    try:
        while decision.decision in ("combat_play", "card_select"):
            name, kwargs, policy = choose_action(decision, session, budget)
            if policy is not None:
                session.logger.attach_policy(policy)
            decision = session.step(name, **kwargs)
        assert decision.decision in ("card_reward", "game_over"), (
            f"combat ended on unexpected decision {decision.decision!r} instead of card_reward/"
            "game_over -- would silently score this combat wrong if treated as resolved"
        )
        if decision.decision == "card_reward":
            # A branch's own reward pick is discarded either way -- only the main line's adopted
            # actions matter, and it makes its own pick itself, via choose_action's generic
            # card_reward handling, once _resolve_combat's replay of this branch's adopted
            # actions reaches this same point (see main_worker_loop). So this must NOT go through
            # session.step(), which would append skip_card_reward to session.action_log --
            # _absorb_combat/_resolve_combat replay that log verbatim onto the main line when this
            # branch is adopted, which would permanently bake "skip the reward" into the main
            # line's own history and pre-empt its own pick before it ever runs. Calling the
            # buffer session directly (bypassing step()'s action_log/logger side effects) still
            # does what's actually needed here: RunSimulator's card-reward prompt blocks a real
            # background thread (HeadlessCardSelector.GetSelectedCardReward's
            # ManualResetEventSlim.Wait) until skip_card_reward/select_card_reward resolves it,
            # and branch_worker_loop reuses this OS process for many jobs in sequence -- every
            # branch that wins without releasing that thread permanently strands one more
            # ThreadPool thread in the process. Discovered via later jobs on the same worker
            # increasingly failing with "Sts2ActionError: No pending card reward".
            raw = session._session.Action("skip_card_reward", None)
            if session._decision(raw).decision == "error":
                raise Sts2ActionError(session._session.LastError or "unknown error")
    finally:
        budget.close()
    died = decision.decision == "game_over"
    session.logger.end_combat(_combat_value(hp_before, decision.player_hp, max_hp, died))
    return "lost" if died else "won"


def _spawn_branches(job_queue, main_worker_index: int, cfg: FacilityConfig, main_id: str, floor: int,
                     combat_index: int, character: str, seed: str,
                     actions_so_far: list[dict]) -> list[str]:
    branch_ids = []
    for b in range(cfg.branches_per_combat):
        job = BranchJob(
            main_id=main_id, floor=floor, combat_index=combat_index, branch_index=b,
            character=character, seed=seed, ascension=cfg.ascension,
            actions=list(actions_so_far),
        )
        job_queue.put((job, main_worker_index))
        branch_ids.append(job.branch_id)
    return branch_ids


def _wait_for_combat_result(
    results_queue, branch_ids: list[str], prefix_len: int, stop_event, timeout: float = 600
) -> dict | None:
    """Blocks on results_queue (branch_id, result) pairs pushed by branch_worker_loop the
    moment its trial is saved, see its own docstring -- and returns as soon as one of
    `branch_ids` reports a win, without waiting on the rest """
    usable: dict[str, dict] = {}
    pending = set(branch_ids)
    deadline = time.time() + timeout
    while pending and time.time() < deadline and not stop_event.is_set():
        try:
            branch_id, result = results_queue.get(timeout=1.0)
        except Empty:
            continue
        if branch_id not in pending:
            continue  # stale result from an already-resolved earlier combat
        pending.discard(branch_id)
        if result is None or len(result.get("actions") or []) < prefix_len:
            continue  # branch errored before saving, or its prefix replay itself failed
        usable[branch_id] = result
        if result.get("outcome") == "won":
            return result
    if pending:
        print(f"warning: {len(pending)}/{len(branch_ids)} branch(es) didn't finish within "
              f"{timeout}s: {sorted(pending)}")
    return next(iter(usable.values()), None)


def _clean_actions(actions: list[dict]) -> list[dict]:
    """Strips CompactRecorder's added `policy`/`value` keys back to bare {cmd, action, args?}"""
    return [
        {"cmd": e["cmd"], "action": e["action"], **({"args": e["args"]} if "args" in e else {})}
        for e in actions
    ]


def _absorb_combat(recorder: CompactRecorder, full_actions: list[dict], prefix_len: int) -> None:
    """Record the combat through logger"""
    for entry in full_actions[prefix_len:]:
        if "policy" in entry:
            recorder.attach_policy(entry["policy"])
        if "value" in entry:
            recorder.attach_value(entry["value"])
        recorder.log_action(entry)


def _resolve_combat(
    job_queue, results_queue, main_worker_index: int, cfg: FacilityConfig, recorder: CompactRecorder,
    session: Sts2Session, character: str, seed: str, main_id: str, floor: int, combat_index: int, stop_event,
):
    """Delegates one combat to cfg.branches_per_combat parallel attempts and adopts a result --
    see the module docstring. Returns (session, decision, outcome): outcome is None if a combat
    was resolved and the run should continue with the returned (freshly reopened) session/
    decision; otherwise outcome is "lost" (an adopted branch actually died) or "error" (no
    branch produced a usable result -- infra failure, not a death) and the caller should stop
    -- the returned session/decision are then just the unchanged inputs, since there was
    nothing to adopt."""
    prefix = list(session.action_log)
    branch_ids = _spawn_branches(
        job_queue, main_worker_index, cfg, main_id, floor, combat_index, character, seed, prefix
    )
    adopted = _wait_for_combat_result(results_queue, branch_ids, len(prefix), stop_event)

    if adopted is None:
        recorder.log_error(
            f"combat {combat_index}: no usable branch results (0/{len(branch_ids)} usable)"
        )
        return session, None, "error"

    _absorb_combat(recorder, adopted["actions"], len(prefix))
    session.close()
    new_session = Sts2Session(logger=recorder)
    new_decision = new_session.open_from_checkpoint(
        character, seed, cfg.ascension, _clean_actions(adopted["actions"])
    )
    return new_session, new_decision, (None if adopted.get("outcome") == "won" else "lost")


def main_worker_loop(
    cfg: FacilityConfig, job_queue: mp.Queue, results_queue: mp.Queue, main_worker_index: int,
    stop_event, trial_counter=None,
) -> None:
    """job_queue is shared with every other main worker, but results_queue is this worker's own private queue, created and owned by
    Facility, one per main worker. It must not be shared."""
    # Each main worker is its own process; reseed explicitly so concurrent workers started via
    # fork (which would otherwise inherit identical RNG state) don't all draw the same
    # character/seed. spawn (macOS default) doesn't need this, but it's cheap insurance.
    random.seed(os.urandom(16))

    while not stop_event.is_set():
        if cfg.n_main_trials is not None:
            assert trial_counter is not None
            with trial_counter.get_lock():
                if trial_counter.value >= cfg.n_main_trials:
                    break
                trial_counter.value += 1

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
                            job_queue, results_queue, main_worker_index, cfg, recorder, session,
                            character, seed, main_id, decision.context.floor, combat_index, stop_event,
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
                recorder.log_error(f"{type(e).__name__}: {e}")
                outcome = "error"
            try:
                recorder.save(f"{cfg.raw_dir}/{main_id}.json", session.action_log, outcome=outcome)
            except Exception as e:
                recorder.log_error(f"save failed: {type(e).__name__}: {e}")
        finally:
            session.close()
        # main branch is dead -- loop restarts with a fresh seed unless the target
        # trial count is reached or stop_event is set, checked at the top of the while


def branch_worker_loop(
    cfg: FacilityConfig, job_queue: mp.Queue, result_queues: list[mp.Queue], stop_event,
    worker_index: int = 0,
) -> None:
    """result_queues: every main worker's private results_queue, indexed by main_worker_index."""
    _tag_worker_stderr("branch", worker_index)
    while True:
        try:
            item = job_queue.get(timeout=1.0)
        except Empty:
            if stop_event.is_set():
                return
            continue
        if item is None:  # shutdown sentinel
            return
        job, main_worker_index = item
        results_queue = result_queues[main_worker_index]

        recorder = CompactRecorder({
            "character": job.character, "seed": job.seed, "ascension": job.ascension,
            "main_id": job.main_id, "floor": job.floor, "combat_index": job.combat_index,
            "branch_index": job.branch_index, "role": "branch",
        }, start_step=len(job.actions))
        session = Sts2BufferSession(logger=recorder)
        outcome = None
        result = None
        try:
            try:
                decision = session.open_from_checkpoint(job.character, job.seed, job.ascension, job.actions)
                outcome = _play_combat_only(session, decision)
            except Exception as e:
                recorder.log_error(f"{type(e).__name__}: {e}")
                outcome = "error"
            try:
                result = recorder.save(f"{cfg.raw_dir}/{job.branch_id}.json", session.action_log, outcome=outcome)
            except Exception as e:
                recorder.log_error(f"save failed: {type(e).__name__}: {e}")
        finally:
            session.close()
            # Always notify, even on failure (result stays None) -- the main line's
            # pending set would otherwise wait out the full timeout for this branch_id.
            results_queue.put((job.branch_id, result))


class Facility:
    def __init__(self, cfg: FacilityConfig):
        self.cfg = cfg
        self.job_queue: mp.Queue = mp.Queue()  # shared dispatch pool -- every main worker feeds it
        self.stop_event = mp.Event()
        self.trial_counter = mp.Value("i", 0) if cfg.n_main_trials is not None else None
        self.main_procs: list[mp.Process] = []
        self.main_result_queues: list[mp.Queue] = []
        self.branch_procs: list[mp.Process] = []

    def start(self) -> None:
        ensure_setup()  # once, here -- not per-session, which would repeat the same checks

        self.main_result_queues.extend(mp.Queue() for _ in range(self.cfg.n_main_workers))

        for i, results_queue in enumerate(self.main_result_queues):
            p = mp.Process(
                target=main_worker_loop,
                args=(self.cfg, self.job_queue, results_queue, i, self.stop_event, self.trial_counter),
                daemon=True,
            )
            p.start()
            self.main_procs.append(p)

        for i in range(self.cfg.n_branch_workers):
            p = mp.Process(
                target=branch_worker_loop,
                args=(self.cfg, self.job_queue, self.main_result_queues, self.stop_event, i), daemon=True,
            )
            p.start()
            self.branch_procs.append(p)

    def stop(self, timeout: float = 60) -> None:
        self.stop_event.set()
        for _ in self.branch_procs:
            self.job_queue.put(None)
        for p in self.main_procs:
            p.join(timeout)
        for p in self.branch_procs:
            p.join(timeout)

    def run(self) -> None:
        self.start()
        try:
            if self.cfg.n_main_trials is None:
                while True:
                    time.sleep(60)
            else:
                for p in self.main_procs:
                    p.join()
                print(f"Main workers finished {self.cfg.n_main_trials} trial(s) total -- "
                      f"draining remaining branch queue...")
        except KeyboardInterrupt:
            print("\nStopping facility (waiting for in-flight trials to finish)...")
        finally:
            self.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Whispering Earring self-play facility")
    parser.add_argument("--trials", type=int, default=None,
                         help="stop after this many main-line trials complete in total (default: run forever, Ctrl-C to stop)")
    parser.add_argument("--characters", nargs="+", default=["Ironclad"])
    parser.add_argument("--branch-workers", type=int, default=4,
                         help="shared pool size -- serves branch jobs from every main worker, not "
                              "one dedicated set per main line (default: 4)")
    parser.add_argument("--branches-per-combat", type=int, default=3)
    parser.add_argument("--main-workers", type=int, default=1,
                         help="concurrent main lines sharing the branch-worker pool above -- raise "
                              "this to keep branch workers fed between combats (default: 1)")
    parser.add_argument("--raw-dir", default="data/raw",
                         help="where trial JSON gets written -- resolved against the process's "
                              "cwd if relative (default: 'data/raw'), so pass an absolute path "
                              "when invoking this script from outside agents/whispering_earring/")
    args = parser.parse_args()

    facility = Facility(FacilityConfig(
        characters=args.characters,
        n_branch_workers=args.branch_workers,
        branches_per_combat=args.branches_per_combat,
        n_main_workers=args.main_workers,
        n_main_trials=args.trials,
        raw_dir=args.raw_dir,
    ))
    facility.run()

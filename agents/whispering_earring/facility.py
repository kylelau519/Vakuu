from __future__ import annotations

import multiprocessing as mp
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from queue import Empty

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)
from env import Sts2ActionError, Sts2Session
from buffer_env import Sts2BufferSession
from installation import ensure_setup

from checkpoint import BranchJob
from policy import choose_action, get_claws_net
from recorder import CompactRecorder

_SEED_WORDS = ["falcon", "ember", "wolf", "raven", "thorn", "comet", "storm", "harbor"]

_HEALTHY_HP_FRACTION = 0.5
_LOW_HP_SLOPE = 3.0


def _random_seed() -> str:
    return f"{random.choice(_SEED_WORDS)}-{random.randint(0, 999_999)}"


_WORKER_TAG_COLORS = [31, 32, 33, 34, 35, 36, 91, 92, 93, 94, 95, 96]  # ANSI red..bright-cyan

# Strips ANSI/terminal control sequences (CSI, OSC, lone ESC) and other C0 control bytes (except
# \t/\n) out of relayed worker output before it reaches the real terminal -- see _tag_worker_stderr.
_CONTROL_SEQUENCE_RE = re.compile(
    r"\x1b(?:\[[0-9;?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])"
    r"|[\x00-\x08\x0b-\x1f\x7f]"
)

# Sts2Headless's [SIM]/[INFO] logging is all free text through one generic Log() call in the
# engine -- there's no separate "milestone" vs. "per-action" log level to hook. In non-verbose
# mode (the default), only lines matching this are relayed -- the per-turn/per-room events that
# tell you a branch is actually advancing, not every card play, potion use, or RNG roll inside it.
_MILESTONE_RE = re.compile(
    r"Creating RunState|Rooms generated|Run launched|relics finalized|Entered Act|"
    r"EnterRoom:|Post-combat:|Moving to map coord|Card reward pending|Selected card reward|"
    r"Bundle selection pending|has won against encounter|has lost"
)


def _tag_worker_stderr(label: str, worker_index: int, log_queue: mp.Queue, verbose: bool) -> None:
    """Sts2BufferSession loads Sts2Headless's CLR in-process (clr_bridge.load_clr), and its
    [SIM]/[INFO] logging writes unconditionally to this process's real fd 2 (Console.Error
    dups it once on first access)."""
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def relay() -> None:
        with os.fdopen(read_fd) as pipe_reader:
            for line in pipe_reader:
                clean = _CONTROL_SEQUENCE_RE.sub("", line)
                if verbose or _MILESTONE_RE.search(clean):
                    log_queue.put((label, worker_index, clean))

    threading.Thread(target=relay, daemon=True, name=f"stderr-relay-{label}-{worker_index}").start()


def _log_writer_loop(log_queue: mp.Queue) -> None:
    """Runs in a single background thread in the parent (Facility) process -- the only writer to
    the real terminal for every branch worker's relayed stderr."""
    while True:
        item = log_queue.get()
        if item is None:
            return
        label, worker_index, line = item
        color = _WORKER_TAG_COLORS[worker_index % len(_WORKER_TAG_COLORS)]
        tag = f"\x1b[{color}m[{label} {worker_index}]\x1b[0m"
        sys.stderr.write(f"{tag} {line}")
        sys.stderr.flush()


def _combat_value(hp_after: int, max_hp: int, died: bool) -> float:
    """Terminal (combat-ending) value target for recorder.py's end_turn() -- used by
    _play_turn_only for whichever turn actually ends the combat. Continuous in the resulting
    HP fraction across the death boundary (hp_after is forced to 0 when died, not handled as a
    separate case) -- see this module's _HEALTHY_HP_FRACTION/_LOW_HP_SLOPE comment for why: a
    branch that barely died and a sibling that barely survived, from the same branch-and-adopt
    turn-start, shouldn't be scored a full cliff apart for what may have been a marginal
    difference."""
    if not max_hp:
        return 1.0
    hp_frac = max(0.0, hp_after) / max_hp if not died else 0.0
    if hp_frac >= _HEALTHY_HP_FRACTION:
        return hp_frac
    shortfall = _HEALTHY_HP_FRACTION - hp_frac
    return _HEALTHY_HP_FRACTION - shortfall * _LOW_HP_SLOPE


@dataclass
class FacilityConfig:
    characters: list[str] = field(default_factory=lambda: ["Ironclad"])
    ascension: int = 0
    branches_per_turn: int = 3  # side-copies spawned per turn, on top of the main line
    n_branch_workers: int = 4
    n_main_workers: int = 1  # concurrent main lines, sharing one branch-worker pool
    n_main_trials: int | None = None  # stop after this many main-line trials *total*; None = forever
    raw_dir: str = "data/raw"
    verbose: bool = False  # relay every branch worker's [SIM]/[INFO] line, not just milestones


def _bypass_card_reward(session: Sts2BufferSession) -> None:
    # A branch's own reward pick is discarded either way -- only the main line's adopted
    # actions matter, and it makes its own pick itself. Discovered via later jobs on the same worker
    # increasingly failing with "Sts2ActionError: No pending card reward".
    raw = session._session.Action("skip_card_reward", None)
    if session._decision(raw).decision == "error":
        raise Sts2ActionError(session._session.LastError or "unknown error")


def _play_turn_only(
    session: Sts2BufferSession, decision, combat_hp_before: int, combat_max_hp: int
) -> str:
    """Plays exactly one turn (through the mid-turn card_select edge case, e.g. Power Potion/
    Brand, the same way the old whole-combat loop did) and scores it before returning --
    "continue" if the combat carries on into another round, "won"/"lost" if this turn ended
    it. See facility.py module context / the turn-level branch-and-adopt plan for why turns,
    not whole combats, are the branch unit now."""
    hp_before_turn = decision.player_hp
    session.logger.begin_turn()
    ended_turn = False
    while decision.decision in ("combat_play", "card_select"):
        name, kwargs, policy = choose_action(decision, session)
        if policy is not None:
            session.logger.attach_policy(policy)
        decision = session.step(name, **kwargs)
        if name == "end_turn":
            ended_turn = True
        if ended_turn and decision.decision == "combat_play":
            break  # new round started -- this turn is fully resolved, combat continues
    assert decision.decision in ("combat_play", "card_reward", "game_over"), (
        f"turn ended on unexpected decision {decision.decision!r} -- would silently score "
        "this turn wrong if treated as resolved"
    )

    if decision.decision == "card_reward":
        _bypass_card_reward(session)
        outcome = "won"
        turn_value = _combat_value(decision.player_hp, combat_max_hp, died=False)
    elif decision.decision == "game_over":
        outcome = "lost"
        turn_value = _combat_value(decision.player_hp, combat_max_hp, died=True)
    else:
        outcome = "continue"
        v_net = 0.0
        net = get_claws_net()
        if net is not None:
            _, _, value_t = net.forward(decision)
            v_net = float(value_t.item())
        hp_term = (
            -(hp_before_turn - decision.player_hp) / combat_max_hp if combat_max_hp else 0.0
        )
        turn_value = 0.75 * hp_term + 1 * v_net 

    session.logger.end_turn(turn_value)
    return outcome


def _spawn_turn_branches(
    job_queue, main_worker_index: int, cfg: FacilityConfig, main_id: str, floor: int,
    combat_index: int, turn_index: int, character: str, seed: str, actions_so_far: list[dict],
    combat_hp_before: int, combat_max_hp: int,
) -> list[str]:
    branch_ids = []
    for b in range(cfg.branches_per_turn):
        job = BranchJob(
            main_id=main_id, floor=floor, combat_index=combat_index, turn_index=turn_index,
            branch_index=b, character=character, seed=seed, ascension=cfg.ascension,
            combat_hp_before=combat_hp_before, combat_max_hp=combat_max_hp,
            actions=list(actions_so_far),
        )
        job_queue.put((job, main_worker_index))
        branch_ids.append(job.branch_id)
    return branch_ids


def _turn_score_key(result: dict) -> tuple[int, float]:
    """Ranks a turn-branch result for adoption: any non-"lost" outcome beats a "lost" one,
    then maximize the branch's own end_turn()-backfilled value -- identical across every one
    of its new actions (end_turn() backfills the same scalar onto the whole turn's range), so
    the last new action's `value` is representative."""
    outcome = result.get("outcome")
    actions = result.get("actions") or []
    value = actions[-1].get("value", float("-inf")) if actions else float("-inf")
    return (0 if outcome == "lost" else 1, value)


def _wait_for_turn_result(
    results_queue, branch_ids: list[str], prefix_len: int, stop_event, timeout: float = 120
) -> dict | None:
    """Blocks on results_queue (branch_id, result) pairs pushed by branch_worker_loop the
    moment its trial is saved. Unlike the old whole-combat version, most branches here report
    "continue" (combat still ongoing) rather than a terminal outcome, so they can't just be
    accepted on first success -- every branch's turn_value (see _turn_score_key) is compared
    once all have reported (or timed out). A branch that itself won the combat this turn is
    still an immediate, unambiguous best pick."""
    usable: dict[str, dict] = {}
    pending = set(branch_ids)
    deadline = time.time() + timeout
    while pending and time.time() < deadline and not stop_event.is_set():
        try:
            branch_id, result = results_queue.get(timeout=1.0)
        except Empty:
            continue
        if branch_id not in pending:
            continue  # stale result from an already-resolved earlier turn
        pending.discard(branch_id)
        if result is None or len(result.get("actions") or []) < prefix_len:
            continue  # branch errored before saving, or its prefix replay itself failed
        usable[branch_id] = result
        if result.get("outcome") == "won":
            return result
    if pending:
        print(f"warning: {len(pending)}/{len(branch_ids)} turn branch(es) didn't finish within "
              f"{timeout}s: {sorted(pending)}")
    return max(usable.values(), key=_turn_score_key) if usable else None


def _clean_actions(actions: list[dict]) -> list[dict]:
    """Strips CompactRecorder's added `policy`/`value` keys back to bare {cmd, action, args?}"""
    return [
        {"cmd": e["cmd"], "action": e["action"], **({"args": e["args"]} if "args" in e else {})}
        for e in actions
    ]


def _absorb_actions(recorder: CompactRecorder, full_actions: list[dict], prefix_len: int) -> None:
    """Record an adopted branch's newly-taken actions (one turn's worth, or a combat-ending
    turn's) through logger."""
    for entry in full_actions[prefix_len:]:
        if "policy" in entry:
            recorder.attach_policy(entry["policy"])
        if "value" in entry:
            recorder.attach_value(entry["value"])
        recorder.log_action(entry)


def _resolve_turn(
    job_queue, results_queue, main_worker_index: int, cfg: FacilityConfig, recorder: CompactRecorder,
    session: Sts2Session, character: str, seed: str, main_id: str, floor: int, combat_index: int,
    turn_index: int, combat_hp_before: int, combat_max_hp: int, stop_event,
):
    """Delegates one turn to cfg.branches_per_turn parallel attempts and adopts the
    best-scoring result """
    prefix = list(session.action_log)
    branch_ids = _spawn_turn_branches(
        job_queue, main_worker_index, cfg, main_id, floor, combat_index, turn_index,
        character, seed, prefix, combat_hp_before, combat_max_hp,
    )
    adopted = _wait_for_turn_result(results_queue, branch_ids, len(prefix), stop_event)

    if adopted is None:
        recorder.log_error(
            f"combat {combat_index} turn {turn_index}: no usable branch results "
            f"(0/{len(branch_ids)} usable)"
        )
        return session, None, "error"

    _absorb_actions(recorder, adopted["actions"], len(prefix))
    session.close()
    new_session = Sts2Session(logger=recorder)
    new_decision = new_session.open_from_checkpoint(
        character, seed, cfg.ascension, _clean_actions(adopted["actions"])
    )
    return new_session, new_decision, adopted.get("outcome")  # "continue" | "won" | "lost"


def _resolve_combat(
    job_queue, results_queue, main_worker_index: int, cfg: FacilityConfig, recorder: CompactRecorder,
    session: Sts2Session, decision, character: str, seed: str, main_id: str, combat_index: int, stop_event,
):
    """Drives _resolve_turn turn-by-turn until the combat actually ends."""
    act = decision.context.act
    floor = decision.context.floor
    combat_hp_before, combat_max_hp = decision.player.hp, decision.player.max_hp
    turn_index = 0
    while True:
        turn_index += 1
        session, decision, turn_outcome = _resolve_turn(
            job_queue, results_queue, main_worker_index, cfg, recorder, session,
            character, seed, main_id, floor, combat_index, turn_index,
            combat_hp_before, combat_max_hp, stop_event,
        )
        if turn_outcome == "continue":
            continue
        if turn_outcome in ("won", "lost"):
            recorder.record_combat(act, floor, combat_index, turn_outcome)
        if turn_outcome == "won":
            # NOTE (2026-08-29): this is where the planned HealAndEnterCombat-based combat-variant
            # fan-out (see whispering_earring_heal_reenter_branching_plan memory) will hook in once
            # engine-integrity is confirmed. A won combat leaves `decision` on a pending
            # `card_reward` -- HealAndEnterCombat called while that's still pending does NOT
            # re-enter combat, it silently no-ops back onto card_reward (only healing HP as a side
            # effect) instead of erroring, so a caller not checking decision.decision=="combat_play"
            # would think it worked. Must call session.step("skip_card_reward") (or
            # select_card_reward) first, THEN heal_and_enter_combat, whenever combat_outcome=="won".
            # (No such gotcha on a loss -- heal_and_enter_combat works immediately from game_over.)
            return session, decision, None
        return session, decision, turn_outcome  # "lost" or "error"


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
                            decision, character, seed, main_id, combat_index, stop_event,
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
    log_queue: mp.Queue, worker_index: int = 0,
) -> None:
    """result_queues: every main worker's private results_queue, indexed by main_worker_index."""
    _tag_worker_stderr("branch", worker_index, log_queue, cfg.verbose)
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
            "turn_index": job.turn_index, "branch_index": job.branch_index, "role": "branch",
            "prefix_len": len(job.actions),
        }, start_step=len(job.actions))
        session = Sts2BufferSession(logger=recorder)
        outcome = None
        result = None
        try:
            try:
                decision = session.open_from_checkpoint(job.character, job.seed, job.ascension, job.actions)
                outcome = _play_turn_only(session, decision, job.combat_hp_before, job.combat_max_hp)
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
        self.log_queue: mp.Queue = mp.Queue()  # every branch worker's relayed stderr -- see _tag_worker_stderr
        self.stop_event = mp.Event()
        self.trial_counter = mp.Value("i", 0) if cfg.n_main_trials is not None else None
        self.main_procs: list[mp.Process] = []
        self.main_result_queues: list[mp.Queue] = []
        self.branch_procs: list[mp.Process] = []
        self.log_thread: threading.Thread | None = None

    def start(self) -> None:
        ensure_setup()  # once, here -- not per-session, which would repeat the same checks

        self.log_thread = threading.Thread(
            target=_log_writer_loop, args=(self.log_queue,), daemon=True, name="branch-log-writer",
        )
        self.log_thread.start()

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
                args=(self.cfg, self.job_queue, self.main_result_queues, self.stop_event, self.log_queue, i),
                daemon=True,
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
        # Every branch process (the only producers) has exited by now -- safe to shut the single
        # terminal-writer thread down.
        self.log_queue.put(None)
        if self.log_thread is not None:
            self.log_thread.join(timeout)

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
    parser.add_argument("--branches-per-turn", type=int, default=3)
    parser.add_argument("--main-workers", type=int, default=1,
                         help="concurrent main lines sharing the branch-worker pool above -- raise "
                              "this to keep branch workers fed between combats (default: 1)")
    parser.add_argument("--raw-dir", default="data/raw",
                         help="where trial JSON gets written -- resolved against the process's "
                              "cwd if relative (default: 'data/raw'), so pass an absolute path "
                              "when invoking this script from outside agents/whispering_earring/")
    parser.add_argument("--verbose", action="store_true",
                         help="relay every branch worker's [SIM]/[INFO] line (card plays, RNG "
                              "rolls, etc.) instead of just per-turn/per-room milestones "
                              "(default: milestones only)")
    args = parser.parse_args()

    facility = Facility(FacilityConfig(
        characters=args.characters,
        n_branch_workers=args.branch_workers,
        branches_per_turn=args.branches_per_turn,
        n_main_workers=args.main_workers,
        n_main_trials=args.trials,
        raw_dir=args.raw_dir,
        verbose=args.verbose,
    ))
    facility.run()

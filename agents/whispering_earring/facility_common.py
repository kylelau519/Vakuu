"""Shared self-play-facility infrastructure: worker-process orchestration, stderr relay/logging,
FacilityConfig, and the pieces of combat scoring/turn-playing that more than one facility needs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import multiprocessing as mp
import os
import random
import shutil
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from queue import Empty
from typing import Any, Callable

from agents.whispering_earring.workers import BufferWorker
import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)
import buffer_layout as bl
from env import Sts2ActionError, Sts2EngineTimeoutError, Sts2Session
from buffer_env import Sts2BufferSession, decode_token, BufferDecision
from installation import ensure_setup
from utils import *
from policy import choose_action, get_claws_net
from recorder import CompactRecorder



def _tag_worker_stderr(label: str, worker_index: int, log_queue: mp.Queue, verbose: bool) -> None:
    """Sts2BufferSession loads Sts2Headless's CLR in-process (clr_bridge.load_clr),  Spwans a background thread to relay the worker's stderr through log_queue, tagging each line
 """
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def relay() -> None:
        crash_seen = False  # sticky once _CRASH_RE fires -- see _CRASH_RE's own comment
        with os.fdopen(read_fd) as pipe_reader:
            for line in pipe_reader:
                clean = CONTROL_SEQUENCE_RE.sub("", line)
                if not crash_seen and CRASH_RE.search(clean):
                    crash_seen = True
                if verbose or crash_seen or MILESTONE_RE.search(clean):
                    log_queue.put((label, worker_index, clean, crash_seen))

    threading.Thread(target=relay, daemon=True, name=f"stderr-relay-{label}-{worker_index}").start()

class _LiveStatus:
    """Fixed one-line-per-worker live status board for _log_writer_loop, 
    each worker gets a permanent row, redrawn in place.  """

    def __init__(self, labels: list[tuple[str, int]]):
        self.rows = {key: i for i, key in enumerate(labels)}
        self.n = len(labels)
        self.active = sys.stderr.isatty() and self.n > 0
        self._lock = threading.Lock()
        if self.active:
            for label, idx in labels:
                sys.stderr.write(f"{self._tag(label, idx)} (waiting...)\n")
            sys.stderr.flush()

    def update(self, label: str, worker_index: int, line: str) -> None:
        key = (label, worker_index)
        if not self.active or key not in self.rows:
            self.scroll(f"{self._tag(label, worker_index)} {line}")
            return
        with self._lock:
            up = self.n - self.rows[key]  # rows numbered top-down; cursor idles just below row n-1
            width = shutil.get_terminal_size(fallback=(120, 24)).columns
            avail = max(10, width - len(f"[{label} {worker_index}] ") - 1)
            msg = line.rstrip("\n")
            if len(msg) > avail:
                msg = msg[: avail - 1] + "…"
            # \x1b[{up}A/B: move up/down `up` rows without touching the terminal's scrollback --
            # \r + \x1b[2K then clears and rewrites just that row before returning to where every
            # scrolling write (this class's own `scroll()` included) expects the cursor to be.
            sys.stderr.write(f"\x1b[{up}A\r\x1b[2K{self._tag(label, worker_index)} {msg}\x1b[{up}B\r")
            sys.stderr.flush()

    def scroll(self, text: str) -> None:
        with self._lock:
            self.active = False
            sys.stderr.write(text if text.endswith("\n") else f"{text}\n")
            sys.stderr.flush()
            
    def loop(self, log_queue: mp.Queue, on_activity: Callable[[], None] | None = None) -> None:
        """Runs in a single background thread in the parent (Facility) process -- the only writer to
        the real terminal for every branch worker's relayed stderr.  """
        while True:
            item = log_queue.get()
            if item is None:
                return
            label, worker_index, line, is_crash = item
            if on_activity is not None:
                on_activity()
            if is_crash:
                self.scroll(f"{self._tag(label, worker_index)} {line.rstrip()}")
            else:
                self.update(label, worker_index, line)

    @staticmethod
    def _tag(label: str, worker_index: int) -> str:
        color = WORKER_TAG_COLORS[worker_index % len(WORKER_TAG_COLORS)]
        return f"\x1b[{color}m[{label} {worker_index}]\x1b[0m"

@dataclass
class CombatOutcome:
    """Bundles _combat_value's inputs"""
    player_hp_after: int
    player_max_hp: int
    is_last_of_act: bool = False


# can move to BufferDecision.
def _map_point_type_at(decision, col: int, row: int) -> str | None:
    if hasattr(decision, "arr"):
        slot = bl.map_boss_point(decision.arr) if row == -1 else bl.map_point_slot(decision.arr, row, col)
        return bl.map_point_type(slot)
    chosen = next((c for c in decision.choices if (c.col, c.row) == (col, row)), None)
    return chosen.type if chosen is not None else None


# This can be defined in terms of the public BufferDecision API, remove later
# def _tracked_enemy_hp(arr, enemy_ids: frozenset[str]) -> int:
#     if not enemy_ids:
#         return 0
#     return sum(
#         bl.enemy_hp(slot) for _, slot in bl.enemies(arr)
#         if decode_token(bl.enemy_token(slot)) in enemy_ids
#     )


@dataclass
class FacilityConfig:
    n_branch_workers: int = 4
    n_main_workers: int = 1  # concurrent main lines, sharing one branch-worker pool
    n_main_trials: int = 1  # stop after this many main-line trials *total*; None = forever
    raw_dir: str = "data/raw"
    verbose: bool = False  
    stall_timeout_s: float = 300.0


def _play_turn_only(
    session, decision, combat_hp_before: int, combat_max_hp: int,
    is_last_of_act: bool = False, enemy_ids: frozenset[str] = frozenset(), enemy_max_hp: int = 0,
    bypass_reward_on_win: bool = True,
):
    hp_before_turn = decision.player_hp
    enemy_hp_before_turn = _tracked_enemy_hp(decision.arr, enemy_ids)
    session.logger.begin_turn()
    ended_turn = False
    while decision.decision in ("combat_play", "card_select"):
        # purpose is never threaded here -- a mid-turn card_select (Power Potion, Brand, ...)
        # never follows a shop/rest_site pick, so it's never "remove"/"upgrade" (see
        # choose_action's docstring); the 4th return value is discarded for the same reason.
        name, kwargs, policy, _ = choose_action(decision, session)
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
        if bypass_reward_on_win:
            _bypass_card_reward(session)
        outcome = "won"
        turn_value = min(1.0, _combat_value(
            CombatOutcome(decision.player_hp, combat_max_hp, is_last_of_act)
        ) + 1.0)
    elif decision.decision == "game_over":
        outcome = "lost"
        turn_value = -1.0
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
        enemy_term = 0.0
        if enemy_max_hp:
            enemy_hp_after_turn = _tracked_enemy_hp(decision.arr, enemy_ids)
            enemy_term = (enemy_hp_before_turn - enemy_hp_after_turn) / enemy_max_hp
        hp_weight = _BOSS_HP_WEIGHT if is_last_of_act else _HP_WEIGHT
        turn_value = hp_weight * (hp_term + enemy_term) + v_net

    session.logger.end_turn(turn_value)
    return outcome, decision


def _clean_actions(actions: list[dict]) -> list[dict]:
    """Strips CompactRecorder's added `policy`/`value` keys back to bare {cmd, action, args?}"""
    return [
        {"cmd": e["cmd"], "action": e["action"], **({"args": e["args"]} if "args" in e else {})}
        for e in actions
    ]



# def _apply_meta_result(
#     meta_id: str, value: float | None, pending_meta_jobs: dict, checkpoint_refcount: dict,
#     recorder: CompactRecorder,
# ) -> None:
#     """In the queue if it sees result, take the result and record it, and decrement the checkpoint refcount. 
#     If the refcount reaches 0, delete the checkpoint file."""
#     entry = pending_meta_jobs.pop(meta_id, None)
#     if entry is None:
#         return  # stale -- not among this trial's currently-tracked jobs (already applied, or from
#         # a trial that's since moved on)
#     checkpoint_path, is_adopted, commit_step = entry
#     if is_adopted and value is not None:
#         recorder.set_value(commit_step, value)
#     if checkpoint_path is None:
#         return  # this divergence's jobs reconstructed via full replay, not a checkpoint -- nothing to refcount
#     remaining = checkpoint_refcount.get(checkpoint_path)
#     if remaining is None:
#         return
#     remaining -= 1
#     if remaining <= 0:
#         checkpoint_refcount.pop(checkpoint_path, None)
#         try:
#             os.remove(checkpoint_path)
#         except OSError:
#             pass
#     else:
#         checkpoint_refcount[checkpoint_path] = remaining


# def _drain_meta_results_nowait(
#     meta_results_queue, pending_meta_jobs: dict, checkpoint_refcount: dict, recorder: CompactRecorder,
# ) -> None:
#     """Non-blocking: Drain the queue results immeidately, useful for the main line to catch up with the branch workers' results."""
#     while True:
#         try:
#             meta_id, value = meta_results_queue.get_nowait()
#         except Empty:
#             return
#         _apply_meta_result(meta_id, value, pending_meta_jobs, checkpoint_refcount, recorder)


# def _drain_meta_results_blocking(
#     meta_results_queue, pending_meta_jobs: dict, checkpoint_refcount: dict, recorder: CompactRecorder,
#     timeout: float,
# ) -> None:
#     """If the main line is done, wait for any remaining meta results to arrive."""
#     deadline = time.time() + timeout
#     while pending_meta_jobs and time.time() < deadline:
#         try:
#             meta_id, value = meta_results_queue.get(timeout=1.0)
#         except Empty:
#             continue
#         _apply_meta_result(meta_id, value, pending_meta_jobs, checkpoint_refcount, recorder)


@dataclass
class _TrialState:
    """Mutable per-trial bookkeeping threaded through _play_trial (see main_worker_loop).
    `session` lives here rather than as a plain local so a mid-combat exception still leaves the
    caller's finally-block session.close() pointed at the live session -- combat_resolver rebinds
    it in place whenever it enters a fresh session."""
    session: Sts2Session | Sts2BufferSession
    checkpoint_dir: str
    meta_budget: int | None
    combat_checkpoint_path: str | None = None
    # Native checkpoint (and its entry action's position in action_log) taken before the most
    # recently entered room, of ANY type -- not just combat. Meta-decision reconstruction
    # (_resolve_meta_decision/_run_meta_option_job) needs this fast-path replay base for
    # rest_site/event_choice divergences exactly as much as it does for a card_reward one right
    # after a combat; see room_index below for why the checkpoint file itself can't be keyed off
    # combat_index the way it used to be.
    last_room_checkpoint_path: str | None = None
    last_room_entry_index: int | None = None
    # Increments once per select_map_node, regardless of what room it leads into -- combat_index
    # (a _play_trial local, not tracked here) only increments for actual combats, so keying the
    # checkpoint filename off it collided whenever two non-combat rooms (e.g. a rest_site then an
    # event_choice) were entered before the next real combat: both would compute the same
    # "combat_index + 1" and silently overwrite each other's checkpoint file out from under any
    # branch job still reading the first one.
    room_index: int = 0
    pending_purpose: str | None = None  # threaded through choose_action -- see its docstring
    divergence_index: int = 0
    checkpoint_paths_this_trial: list = field(default_factory=list)

    pending_map_steps: list = field(default_factory=list)
    pending_map_step_types: list = field(default_factory=list)
    local_map_rewards: dict = field(default_factory=dict) 
    
    pending_meta_jobs: dict = field(default_factory=dict)
    checkpoint_refcount: dict = field(default_factory=dict)


def _play_trial(
    cfg: FacilityConfig, job_queue_priority: mp.Queue, job_queue: mp.Queue, results_queue,
    meta_results_queue, main_worker_index: int, recorder, character: str, seed: str, main_id: str,
    this_trial_resume: dict | None, stop_event, queue_manager, combat_resolver: Callable,
    meta_resolver: Callable | None, state: _TrialState,
) -> tuple[str, object]:
    """Plays one trial's decisions from the first through game_over, or through an ending combat
    outcome ("lost"/"error"). Returns (outcome, final decision); any exception propagates to the
    caller, which records it as an "error" outcome."""
    if this_trial_resume is not None:
        decision = state.session.open_from_checkpoint(
            character, seed, cfg.ascension, this_trial_resume["actions"]
        )
        combat_index = this_trial_resume["combat_index"]
    else:
        decision = state.session.start(character, seed, cfg.ascension)
        combat_index = 0

    while decision.decision != "game_over":
        if decision.decision == "combat_play":
            combat_index += 1
            state.session, decision, combat_outcome = combat_resolver(
                job_queue_priority, results_queue, main_worker_index, cfg, recorder,
                state.session, decision, character, seed, main_id, combat_index, stop_event,
                state.combat_checkpoint_path,
            )
            state.combat_checkpoint_path = None  # consumed -- not valid for the next combat
            # combat_outcome is None exactly on a win (see facility_claws._resolve_combat) --
            # attribute an Elite/Boss shaping bonus to the select_map_node choice that led into
            # it (always 1:1 with the combat it starts).
            if (combat_outcome is None and state.pending_map_step_types
                    and state.pending_map_step_types[-1] in ("Elite", "Boss")):
                bonus = (_GOLDEN_COMPASS_ELITE_BONUS
                         if state.pending_map_step_types[-1] == "Elite"
                         else _GOLDEN_COMPASS_BOSS_BONUS)
                step = state.pending_map_steps[-1]
                state.local_map_rewards[step] = state.local_map_rewards.get(step, 0.0) + bonus
            if combat_outcome is not None:
                return combat_outcome, decision
            continue

        if (meta_resolver is not None and state.meta_budget != 0
                and decision.decision in META_ELIGIBLE_DECISIONS):
            state.divergence_index += 1
            if state.meta_budget is not None:
                state.meta_budget -= 1
            decision, new_pending = meta_resolver(
                job_queue, results_queue, meta_results_queue, main_worker_index, cfg,
                recorder, state.session, decision, character, seed, main_id,
                decision.decision, state.divergence_index, stop_event, queue_manager,
                state.last_room_checkpoint_path, state.last_room_entry_index,
            )
            for meta_id, ckpt_path, is_adopted, commit_step in new_pending:
                state.pending_meta_jobs[meta_id] = (ckpt_path, is_adopted, commit_step)
                if ckpt_path is not None:
                    state.checkpoint_refcount[ckpt_path] = (
                        state.checkpoint_refcount.get(ckpt_path, 0) + 1
                    )
            # Cheap -- returns immediately once meta_results_queue is empty. Catches most
            # completions naturally as the trial keeps playing; _drain_meta_results_blocking is
            # the bounded catch-all once the trial ends (see main_worker_loop).
            _drain_meta_results_nowait(
                meta_results_queue, state.pending_meta_jobs, state.checkpoint_refcount, recorder
            )
            continue

        name, kwargs, policy, state.pending_purpose = choose_action(
            decision, state.session, state.pending_purpose
        )
        if policy is not None:
            recorder.attach_policy(policy)
        if name == "select_map_node":
            state.pending_map_steps.append(recorder.mark_step())
            state.pending_map_step_types.append(
                _map_point_type_at(decision, kwargs["col"], kwargs["row"])
            )
            state.room_index += 1
            checkpoint_path = os.path.join(
                state.checkpoint_dir, f"{main_id}_r{state.room_index:03d}.json"
            )
            os.makedirs(state.checkpoint_dir, exist_ok=True)
            state.session.save(checkpoint_path)
            state.checkpoint_paths_this_trial.append(checkpoint_path)
        decision = state.session.step(name, **kwargs)
        if name == "select_map_node":
            state.last_room_checkpoint_path = checkpoint_path
            state.last_room_entry_index = len(state.session.action_log) - 1
            if decision.decision == "combat_play":
                state.combat_checkpoint_path = checkpoint_path

    return ("won" if decision.victory else "lost"), decision


def _apply_golden_compass_rewards(recorder, outcome: str, state: _TrialState) -> None:
    """Backfills each select_map_node step's training value: this trial's terminal +1/-1 plus a
    backward-discounted Elite/Boss-win shaping bonus (state.local_map_rewards), decayed by
    _GOLDEN_COMPASS_GAMMA per step of distan."""
    if outcome not in ("won", "lost") or not state.pending_map_steps:
        return
    map_reward = 1.0 if outcome == "won" else -1.0
    shaping = 0.0
    discounted_shaping = [0.0] * len(state.pending_map_steps)
    for i in range(len(state.pending_map_steps) - 1, -1, -1):
        shaping = (state.local_map_rewards.get(state.pending_map_steps[i], 0.0)
                   + _GOLDEN_COMPASS_GAMMA * shaping)
        discounted_shaping[i] = shaping
    for step, bonus in zip(state.pending_map_steps, discounted_shaping):
        recorder.set_value(step, map_reward + bonus)


def main_worker_loop(
    cfg: FacilityConfig, job_queue: mp.Queue, job_queue_priority: mp.Queue, log_queue: mp.Queue,
    results_queue, main_worker_index: int, stop_event, trial_counter=None,
    resume: dict | None = None, manager_address=None, manager_authkey: bytes | None = None,
    meta_results_queue=None, *,
    combat_resolver: Callable, meta_resolver: Callable | None = None,
    session_factory: Callable = Sts2Session,
) -> None:
    # Spwan 
    _tag_worker_stderr("main", main_worker_index, log_queue, cfg.verbose)
    os.setpgid(0, 0)
    random.seed(os.urandom(16))  # avoid fork-inherited identical RNG across concurrent workers

    queue_manager = None
    if meta_resolver is not None and manager_address is not None:
        from multiprocessing.managers import SyncManager
        queue_manager = SyncManager(address=manager_address, authkey=manager_authkey)
        queue_manager.connect()

    while not stop_event.is_set():
        if cfg.n_main_trials is not None:
            assert trial_counter is not None
            with trial_counter.get_lock():
                if trial_counter.value >= cfg.n_main_trials:
                    break
                trial_counter.value += 1

        this_trial_resume, resume = resume, None  # consumed at most once, on the first trial
        if this_trial_resume is not None:
            character = this_trial_resume["character"]
            seed = this_trial_resume["seed"]
            main_id = this_trial_resume.get("main_id") or f"{character}_{seed}"
        else:
            character = random.choice(cfg.characters)
            seed = _random_seed()
            main_id = f"{character}_{seed}"

        recorder = CompactRecorder({
            "character": character, "seed": seed, "ascension": cfg.ascension,
            "main_id": main_id, "role": "main",
        })
        state = _TrialState(
            session=session_factory(logger=recorder),
            checkpoint_dir=os.path.join(cfg.raw_dir, "checkpoints"),
            meta_budget=cfg.meta_branches_per_trial,
        )
        outcome = None
        poisoned = False
        try:
            try:
                outcome, _ = _play_trial(
                    cfg, job_queue_priority, job_queue, results_queue, meta_results_queue,
                    main_worker_index, recorder, character, seed, main_id, this_trial_resume,
                    stop_event, queue_manager, combat_resolver, meta_resolver, state,
                )
            except Sts2EngineTimeoutError as e:
                recorder.log_error(f"{type(e).__name__}: {e}")
                outcome = "error"
                # buffer_env._reap_abandoned_workers() permanently poisons this whole OS process
                # once a call is abandoned to a timeout -- the stuck CLR call keeps running
                # forever on an abandoned thread, and every Sts2BufferSession() constructed in
                # this process from now on (i.e. every future trial this worker would attempt)
                # immediately re-raises this same error (see that function's docstring). Looping
                # to the next trial the way the generic except below does would just fail
                # identically, forever, silently burning through cfg.n_main_trials' shared budget
                # with empty "error" trials. A subprocess-based Sts2Session has no such
                # process-wide poisoning -- only its own subprocess needs replacing, which the
                # next trial's fresh session_factory() call already does -- so this only forces an
                # exit when the poisoned session was actually a buffer one.
                poisoned = isinstance(state.session, Sts2BufferSession)
            except Exception as e:
                recorder.log_error(f"{type(e).__name__}: {e}")
                outcome = "error"
            _apply_golden_compass_rewards(recorder, outcome, state)
            # Bounded wait for this trial's still-outstanding MetaOptionJobs (no-op whenever
            # meta_resolver is None) -- must run BEFORE recorder.save() below, the only chance for
            # a late-arriving adopted-option value to land via recorder.set_value. Whatever
            # doesn't resolve within _META_DRAIN_TIMEOUT is abandoned; its checkpoint still gets
            # deleted below regardless.
            _drain_meta_results_blocking(
                meta_results_queue, state.pending_meta_jobs, state.checkpoint_refcount, recorder,
                timeout=_META_DRAIN_TIMEOUT,
            )
            try:
                recorder.save(
                    f"{cfg.raw_dir}/{main_id}.json", state.session.action_log, outcome=outcome
                )
            except Exception as e:
                recorder.log_error(f"save failed: {type(e).__name__}: {e}")
        finally:
            state.session.close()
            # Every combat branch has resolved by now (combat's own _wait_for_turn_result blocks
            # the main line synchronously); meta-decision branches got a fair chance via the
            # bounded drain above. Anything still outstanding is a straggler: safe to reclaim its
            # checkpoint unconditionally here rather than let these accumulate for the facility's
            # life (a job reading one late just gets a clean, already-handled error -- see
            # _run_meta_option_job's try/except).
            for path in state.checkpoint_paths_this_trial:
                try:
                    os.remove(path)
                except OSError:
                    pass
        if poisoned:
            # Exit this worker process outright rather than looping to the top of the while --
            # see the Sts2EngineTimeoutError handling above for why looping here can't recover.
            # Facility doesn't respawn a dead main worker mid-round, so this trades "one fewer
            # active main worker for the rest of this round" for "an endless stream of instant
            # empty error trials burning the shared n_main_trials budget" -- the former is a
            # bounded, visible loss of throughput; the latter can silently eat the whole round.
            return
        # main line is dead -- loop restarts with a fresh seed unless the target trial count is
        # reached or stop_event is set, checked at the top of the while


def branch_worker_loop(
    cfg: FacilityConfig, job_queue: mp.Queue, job_queue_priority: mp.Queue, stop_event,
    log_queue: mp.Queue, worker_index: int = 0, *, handlers: dict[type, Callable],
) -> None:
    """Each job_queue/job_queue_priority item is (job, results_queue) -- the queue to report back
    to travels WITH the job, rather than being looked up from a fixed main_worker_index -> queue
    table, so a caller other than the issuing main worker can dispatch and collect its own results
    independently.

    handlers: job type -> handler(cfg, job, results_queue). Dispatches purely on type(job), so a
    single shared pool can serve BranchJob (combat-turn branching, facility_claws.py) and
    MetaOptionJob (card_reward branching, facility_choices_paradox.py) side by side.
    facility_claws.py registers only BranchJob's handler; facility_choices_paradox.py registers
    both (reusing facility_claws's BranchJob handler as-is, since its own main line still plays
    combat turn-branched exactly like facility_claws.py's does)."""
    # See main_worker_loop's matching call for why -- keeps Facility.stop()'s force-kill uniform
    # (os.killpg on main_procs AND branch_procs alike).
    os.setpgid(0, 0)
    _tag_worker_stderr("branch", worker_index, log_queue, cfg.verbose)
    while True:
        try:
            item = job_queue_priority.get_nowait()
        except Empty:
            try:
                item = job_queue.get(timeout=0.2)
            except Empty:
                if stop_event.is_set():
                    return
                continue
        if item is None:  # shutdown sentinel
            return
        job, results_queue = item
        handler = handlers[type(job)]
        handler(cfg, job, results_queue)


""" New design:
Each worker has its own sts2bufferSession work station, and they are responsible to run job.execute()
Worker will look into job queue and pick up a job, and they will return something to result_queue

Facility on the otherhand, is responsible to manage the workers, and the job queue. It will spawn workers, and it will feed them jobs. It will also collect results from the result queue and do something with them.
It is possible a new job should be created when a job is not completed, this should be handled by sending result to result_queue, and facility will look into the result and decide if a new job should be created and sent to job queue.
"""



class Facility:
    """Pure process orchestration -- knows nothing about combat-turn or meta-decision branching
    specifically. main_worker_fn/branch_worker_fn are fully-bound callables matching
    main_worker_loop's/branch_worker_loop's positional signature (the resolver/handler kwargs
    each concrete facility needs are pre-bound via functools.partial by its own CLI entrypoint --
    see facility_claws.py's or facility_choices_paradox.py's `if __name__` block)."""

    def __init__(
        self, cfg: FacilityConfig, main_worker_fn: Callable, branch_worker_fn: Callable,
        resume: dict | None = None,
    ):
        self.cfg = cfg
        self.main_worker_fn = main_worker_fn
        self.branch_worker_fn = branch_worker_fn
        self.resume = resume  # see main_worker_loop's docstring -- consumed by worker 0's first trial only
        self.job_queue: mp.Queue = mp.Queue()  # shared dispatch pool -- every main worker feeds it
        # High-priority twin of job_queue -- see main_worker_loop's docstring for why. Always
        # drained first by branch_worker_loop.
        self.job_queue_priority: mp.Queue = mp.Queue()
        self.log_queue: mp.Queue = mp.Queue()  # every branch worker's relayed stderr -- see _tag_worker_stderr
        self.stop_event = mp.Event()
        self.trial_counter = mp.Value("i", 0) if cfg.n_main_trials is not None else None
        self.main_procs: list[mp.Process] = []
        self.main_result_queues: list = []
        # Per-main-worker MetaOptionJob results -- see start()'s comment for why this is separate
        # from main_result_queues.
        self.main_meta_result_queues: list = []
        self.branch_procs: list[mp.Process] = []
        self.log_thread: threading.Thread | None = None
        self.live: _LiveStatus | None = None  # built in start() -- see _LiveStatus
        self.manager = None  # multiprocessing.managers.SyncManager, created in start()
        self._last_activity = time.monotonic()

    def _mark_activity(self) -> None:
        self._last_activity = time.monotonic()

    def start(self) -> None:
        ensure_setup()  # once, here -- not per-session, which would repeat the same checks
        self._last_activity = time.monotonic()  # don't count ensure_setup()'s own time against it

        labels = [("main", i) for i in range(self.cfg.n_main_workers)]
        labels += [("branch", i) for i in range(self.cfg.n_branch_workers)]
        self.live = _LiveStatus(labels)
        self.log_thread = threading.Thread(
            target=self.live.loop, args=(self.log_queue, self._mark_activity),
            daemon=True, name="branch-log-writer",
        )
        self.log_thread.start()

        # A manager is better than the Queue because it allows multiple processes to communicate and share the same job pool
        self.manager = mp.Manager()
        manager_address = self.manager.address
        manager_authkey = bytes(self.manager._authkey)
        self.main_result_queues.extend(self.manager.Queue() for _ in range(self.cfg.n_main_workers))
        # Separate from main_result_queues 
        self.main_meta_result_queues.extend(
            self.manager.Queue() for _ in range(self.cfg.n_main_workers)
        )

        for i, results_queue in enumerate(self.main_result_queues):
            p = mp.Process(
                target=self.main_worker_fn,
                args=(self.cfg, self.job_queue, self.job_queue_priority, self.log_queue,
                      results_queue, i, self.stop_event, self.trial_counter,
                      self.resume if i == 0 else None, manager_address, manager_authkey,
                      self.main_meta_result_queues[i]),
                daemon=True,
            )
            p.start()
            self.main_procs.append(p)

        for i in range(self.cfg.n_branch_workers):
            p = mp.Process(
                target=self.branch_worker_fn,
                args=(self.cfg, self.job_queue, self.job_queue_priority, self.stop_event,
                      self.log_queue, i),
                daemon=True,
            )
            p.start()
            self.branch_procs.append(p)

    def stop(self, timeout: float = 90) -> None:
        self.stop_event.set()
        for _ in self.branch_procs:
            self.job_queue.put(None)

        deadline = time.monotonic() + timeout
        for p in self.main_procs + self.branch_procs:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                p.join(remaining)
        for p in self.main_procs + self.branch_procs:
            if p.is_alive():
                self.live.scroll(
                    f"{p.name} didn't stop within the {timeout}s shutdown window -- killing its "
                    f"process group (SIGKILL)"
                )
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    p.kill()
                p.join()
        self.log_queue.put(None)
        if self.log_thread is not None:
            self.log_thread.join(timeout)
        if self.manager is not None:
            self.manager.shutdown()

    def run(self) -> None:
        self.start()
        def _on_sigterm(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, _on_sigterm)
        try:
            if self.cfg.n_main_trials is None:
                while True:
                    time.sleep(60)
            else:
                # NOT a plain `for p in self.main_procs: p.join()` -- that would block forever if
                # a main worker wedged. Polls instead, and treats "no relayed log line for
                # stall_timeout_s while a worker is still alive" as wedged rather than legitimately
                # busy -- falls through to stop() below either way, which does the actual kill.
                while any(p.is_alive() for p in self.main_procs):
                    for p in self.main_procs:
                        p.join(timeout=5)
                    stalled_for = time.monotonic() - self._last_activity
                    if stalled_for > self.cfg.stall_timeout_s and any(
                        p.is_alive() for p in self.main_procs
                    ):
                        # self.live.scroll(), not print() -- see stop()'s matching comment.
                        self.live.scroll(
                            f"No worker activity for {stalled_for:.0f}s (> "
                            f"{self.cfg.stall_timeout_s:.0f}s) -- treating the remaining main "
                            f"worker(s) as wedged and abandoning them (see stop() for the kill)."
                        )
                        break
                self.live.scroll(
                    f"Main workers finished {self.cfg.n_main_trials} trial(s) total -- "
                    f"draining remaining branch queue..."
                )
        except KeyboardInterrupt:
            self.live.scroll("Stopping facility (waiting for in-flight trials to finish)...")
        finally:
            self.stop()
    

class NewFacility(ABC):
    def __init__(self, cfg: FacilityConfig):
        self.cfg = cfg
        self.job_queue: mp.Queue = mp.Queue()
        self.priority_job_queue: mp.Queue = mp.Queue()
        self.log_queue: mp.Queue = mp.Queue()
        self.result_queue: mp.Queue = mp.Queue()
        self.main_workers_processes: list[mp.Process] = []
        self.branch_workers_processes: list[mp.Process] = []
        self.stop_event = mp.Event()
        self.log_thread: threading.Thread | None = None
        self.live: _LiveStatus | None = None  # built in start() -- see _LiveStatus
        self.trial_counter = mp.Value(
            "i", 0) if cfg.n_main_trials is not None else None
        self._manager = None
        self._last_activity = time.monotonic()
        self._trial_counter = mp.Value("i", 0) 
        
        
    def start(self):
        labels = [("main", i) for i in range(self.cfg.n_main_workers)]
        labels += [("branch", i) for i in range(self.cfg.n_branch_workers)]
        self.live = _LiveStatus(labels)
        self.log_thread = threading.Thread(
            target=self.live.loop, args=(self.log_queue, self._mark_activity()),
            daemon=True, name="branch-log-writer",
        )
        self.log_thread.start()
        # something to start the first few mainline
        self.spawn_workers()
        
        
        
    def _mark_activity(self) -> None:
        self._last_activity = time.monotonic()

    def stop(self):
        pass
    
    def run(self) -> None:
        pass
    
    def spawn_workers(self):
        for i in range(self.cfg.n_main_workers):
            worker = BufferWorker(
                self.job_queue, self.priority_job_queue, self.log_queue, self.result_queue, i
            )
            p = mp.Process(target=worker.run, name=f"main_worker_{i}")
            p.start()
            self.main_workers_processes.append(p)
        for i in range(self.cfg.n_branch_workers):
            worker = BufferWorker(
                self.job_queue, self.priority_job_queue, self.log_queue, self.result_queue, i
            )
            p = mp.Process(target=worker.run, name=f"branch_worker_{i}")
            p.start()
            self.branch_workers_processes.append(p)

    def spawn_job(self):
        pass
    
    @abstractmethod
    def drain_workers(self):
        pass
    
    def drain_results(self):
        pass
    
    

def add_common_cli_args(parser) -> None:
    """Flags shared by facility_claws.py's and facility_choices_paradox.py's CLIs."""
    parser.add_argument("--trials", type=int, default=None,
                         help="stop after this many main-line trials complete in total (default: run forever, Ctrl-C to stop)")
    parser.add_argument("--characters", nargs="+", default=["Ironclad"])
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--branch-workers", type=int, default=4,
                         help="shared pool size -- serves branch jobs from every main worker, not "
                              "one dedicated set per main line (default: 4)")
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
    parser.add_argument("--resume-branch-file", default=None,
                         help="recover a trial that was progressing normally (had already won "
                              "combats) when the facility died before writing its own output ")


def load_resume(path: str) -> dict:
    import json
    with open(path) as f:
        r = json.load(f)
    return {
        "character": r["character"], "seed": r["seed"], "main_id": r["main_id"],
        "combat_index": r["combat_index"], "actions": _clean_actions(r["actions"]),
    }

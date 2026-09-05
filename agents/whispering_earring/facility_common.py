"""Shared self-play-facility infrastructure: worker-process orchestration, stderr relay/logging,
FacilityConfig, and the pieces of combat scoring/turn-playing that more than one facility needs.

`Facility`, `main_worker_loop`, and `branch_worker_loop` are pure orchestration here -- they take
`combat_resolver`/`meta_resolver` (main line) and `handlers` (branch dispatch, keyed by job type)
as callables/mappings, and know nothing about combat turn-branching or meta-decision branching
specifically. Each concrete facility module binds its own domain logic to these via
`functools.partial` before constructing a `Facility`
"""
from __future__ import annotations

import multiprocessing as mp
import os
import re
import random
import shutil
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from queue import Empty
from typing import Callable

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)
import buffer_layout as bl
from env import Sts2ActionError, Sts2Session
from buffer_env import decode_token
from installation import ensure_setup

from policy import choose_action, get_claws_net
from recorder import CompactRecorder

_SEED_WORDS = ["falcon", "ember", "wolf", "raven", "thorn", "comet", "storm", "harbor"]

_HEALTHY_HP_FRACTION = 0.5
_LOW_HP_SLOPE = 3.0

# Boss-fight (act-final combat) reward shaping. At ascension 2+, winning an act heals 80% of
# missing HP, so leftover HP is worth much less there than in a normal fight  
_BOSS_HP_WEIGHT = 0.2
_BOSS_WIN_FLOOR = 0.8
_BOSS_ENEMY_WEIGHT = 0.2

# Golden Compass reward shaping (see main_worker_loop's pending_map_steps machinery)
# the old scheme did.
_GOLDEN_COMPASS_GAMMA = 0.9
_GOLDEN_COMPASS_ELITE_BONUS = 2.0
_GOLDEN_COMPASS_BOSS_BONUS = 3.0


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
# engine 
_MILESTONE_RE = re.compile(
    r"Creating RunState|Rooms generated|Run launched|relics finalized|Entered Act|"
    r"EnterRoom:|Post-combat:|Moving to map coord|Card reward pending|Selected card reward|"
    r"Bundle selection pending|has won against encounter|has lost"
)

# CoreCLR's own unhandled-exception/fatal-error dump. Without this, a CLR crash was silently dropped
# by _MILESTONE_RE in non-verbose mode, leaving a worker that just silently died or wedged with zero visible output.
_CRASH_RE = re.compile(
    r"Unhandled [Ee]xception|Fatal error\.|Process terminating|Stack overflow\.|"
    r"Assertion failed|Segmentation fault|SIGABRT|SIGSEGV"
)


def _tag_worker_stderr(label: str, worker_index: int, log_queue: mp.Queue, verbose: bool) -> None:
    """Sts2BufferSession loads Sts2Headless's CLR in-process (clr_bridge.load_clr) """
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def relay() -> None:
        crash_seen = False  # sticky once _CRASH_RE fires -- see _CRASH_RE's own comment
        with os.fdopen(read_fd) as pipe_reader:
            for line in pipe_reader:
                clean = _CONTROL_SEQUENCE_RE.sub("", line)
                if not crash_seen and _CRASH_RE.search(clean):
                    crash_seen = True
                if verbose or crash_seen or _MILESTONE_RE.search(clean):
                    log_queue.put((label, worker_index, clean, crash_seen))

    threading.Thread(target=relay, daemon=True, name=f"stderr-relay-{label}-{worker_index}").start()


def _tag(label: str, worker_index: int) -> str:
    color = _WORKER_TAG_COLORS[worker_index % len(_WORKER_TAG_COLORS)]
    return f"\x1b[{color}m[{label} {worker_index}]\x1b[0m"


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
                sys.stderr.write(f"{_tag(label, idx)} (waiting...)\n")
            sys.stderr.flush()

    def update(self, label: str, worker_index: int, line: str) -> None:
        key = (label, worker_index)
        if not self.active or key not in self.rows:
            self.scroll(f"{_tag(label, worker_index)} {line}")
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
            sys.stderr.write(f"\x1b[{up}A\r\x1b[2K{_tag(label, worker_index)} {msg}\x1b[{up}B\r")
            sys.stderr.flush()

    def scroll(self, text: str) -> None:
        with self._lock:
            self.active = False
            sys.stderr.write(text if text.endswith("\n") else f"{text}\n")
            sys.stderr.flush()


def _log_writer_loop(
    log_queue: mp.Queue, live: _LiveStatus, on_activity: Callable[[], None] | None = None
) -> None:
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
            live.scroll(f"{_tag(label, worker_index)} {line.rstrip()}")
        else:
            live.update(label, worker_index, line)


@dataclass
class CombatOutcome:
    """Bundles _combat_value's inputs"""
    player_hp_after: int
    player_max_hp: int
    is_last_of_act: bool = False


def _combat_value(outcome: CombatOutcome) -> float:
    if not outcome.player_max_hp:
        return 1.0
    hp_frac = max(0.0, outcome.player_hp_after) / outcome.player_max_hp
    if outcome.is_last_of_act:
        return _BOSS_WIN_FLOOR + _BOSS_HP_WEIGHT * hp_frac
    if hp_frac >= _HEALTHY_HP_FRACTION:
        return hp_frac
    shortfall = _HEALTHY_HP_FRACTION - hp_frac
    return _HEALTHY_HP_FRACTION - shortfall * _LOW_HP_SLOPE


def _tracked_enemy_hp(arr, boss_ids: frozenset[str]) -> int:
    """Sums current HP across only the enemy slots whose decoded token is in boss_ids for aggressive mode"""
    if not boss_ids:
        return 0
    return sum(
        bl.enemy_hp(slot) for _, slot in bl.enemies(arr)
        if decode_token(bl.enemy_token(slot)) in boss_ids
    )


@dataclass
class FacilityConfig:
    characters: list[str] = field(default_factory=lambda: ["Ironclad"])
    ascension: int = 0
    branches_per_turn: int = 3  # side-copies spawned per combat turn, on top of the main line
    n_branch_workers: int = 4
    n_main_workers: int = 1  # concurrent main lines, sharing one branch-worker pool
    n_main_trials: int | None = None  # stop after this many main-line trials *total*; None = forever
    raw_dir: str = "data/raw"
    verbose: bool = False  
    stall_timeout_s: float = 300.0
    # Meta-decision (card_reward, eventually card_select) branching 
    meta_branches_per_trial: int | None = 0  # 0 = disabled; None = unlimited


def _bypass_card_reward(session) -> None:
    # A branch's own reward pick is discarded either way -- only the main line's adopted actions
    # matter.
    raw = session._session.Action("skip_card_reward", None)
    if session._decision(raw).decision == "error":
        raise Sts2ActionError(session._session.LastError or "unknown error")


def _play_turn_only(
    session, decision, combat_hp_before: int, combat_max_hp: int,
    is_last_of_act: bool = False, boss_ids: frozenset[str] = frozenset(), boss_max_hp: int = 0,
    bypass_reward_on_win: bool = True,
):
    """Plays exactly one turn (through the mid-turn card_select edge case, e.g. Power Potion/
    Brand) and scores it before returning `(outcome, decision)` -- outcome is "continue" if the
    combat carries on, "won"/"lost" if this turn ended it; decision is whatever the game shows
    after (a fresh combat_play round, the pending card_reward on a win, or game_over on a loss).

    For boss fight reward are shaped differently.

    bypass_reward_on_win: For branch job to skip reward because they are irrelevant."""
    hp_before_turn = decision.player_hp
    enemy_hp_before_turn = _tracked_enemy_hp(decision.arr, boss_ids) if is_last_of_act else 0
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
        turn_value = _combat_value(
            CombatOutcome(decision.player_hp, combat_max_hp, is_last_of_act)
        )
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
        if is_last_of_act and boss_max_hp:
            enemy_hp_after_turn = _tracked_enemy_hp(decision.arr, boss_ids)
            enemy_term = (enemy_hp_before_turn - enemy_hp_after_turn) / boss_max_hp
        hp_weight = _BOSS_HP_WEIGHT if is_last_of_act else 0.75
        turn_value = hp_weight * hp_term + _BOSS_ENEMY_WEIGHT * enemy_term + 1 * v_net

    session.logger.end_turn(turn_value)
    return outcome, decision


def _clean_actions(actions: list[dict]) -> list[dict]:
    """Strips CompactRecorder's added `policy`/`value` keys back to bare {cmd, action, args?}"""
    return [
        {"cmd": e["cmd"], "action": e["action"], **({"args": e["args"]} if "args" in e else {})}
        for e in actions
    ]


META_ELIGIBLE_DECISIONS = frozenset({"card_reward"})
_META_DRAIN_TIMEOUT = 30.0


def _apply_meta_result(
    meta_id: str, value: float | None, pending_meta_jobs: dict, checkpoint_refcount: dict,
    recorder: CompactRecorder,
) -> None:
    """In the queue if it sees result, take the result and record it, and decrement the checkpoint refcount. 
    If the refcount reaches 0, delete the checkpoint file."""
    entry = pending_meta_jobs.pop(meta_id, None)
    if entry is None:
        return  # stale -- not among this trial's currently-tracked jobs (already applied, or from
        # a trial that's since moved on)
    checkpoint_path, is_adopted, commit_step = entry
    if is_adopted and value is not None:
        recorder.set_value(commit_step, value)
    if checkpoint_path is None:
        return  # this divergence's jobs reconstructed via full replay, not a checkpoint -- nothing to refcount
    remaining = checkpoint_refcount.get(checkpoint_path)
    if remaining is None:
        return
    remaining -= 1
    if remaining <= 0:
        checkpoint_refcount.pop(checkpoint_path, None)
        try:
            os.remove(checkpoint_path)
        except OSError:
            pass
    else:
        checkpoint_refcount[checkpoint_path] = remaining


def _drain_meta_results_nowait(
    meta_results_queue, pending_meta_jobs: dict, checkpoint_refcount: dict, recorder: CompactRecorder,
) -> None:
    """Non-blocking: Drain the queue results immeidately, useful for the main line to catch up with the branch workers' results."""
    while True:
        try:
            meta_id, value = meta_results_queue.get_nowait()
        except Empty:
            return
        _apply_meta_result(meta_id, value, pending_meta_jobs, checkpoint_refcount, recorder)


def _drain_meta_results_blocking(
    meta_results_queue, pending_meta_jobs: dict, checkpoint_refcount: dict, recorder: CompactRecorder,
    timeout: float,
) -> None:
    """If the main line is done, wait for any remaining meta results to arrive."""
    deadline = time.time() + timeout
    while pending_meta_jobs and time.time() < deadline:
        try:
            meta_id, value = meta_results_queue.get(timeout=1.0)
        except Empty:
            continue
        _apply_meta_result(meta_id, value, pending_meta_jobs, checkpoint_refcount, recorder)


@dataclass
class _TrialState:
    """Mutable per-trial bookkeeping threaded through _play_trial (see main_worker_loop).
    `session` lives here rather than as a plain local so a mid-combat exception still leaves the
    caller's finally-block session.close() pointed at the live session -- combat_resolver rebinds
    it in place whenever it enters a fresh Sts2Session."""
    session: Sts2Session
    checkpoint_dir: str
    meta_budget: int | None
    combat_checkpoint_path: str | None = None
    # Native checkpoint (and its entry action's position in action_log)
    last_combat_checkpoint_path: str | None = None
    last_combat_entry_index: int | None = None
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
            state.last_combat_checkpoint_path = state.combat_checkpoint_path
            state.last_combat_entry_index = len(state.session.action_log) - 1
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
                state.last_combat_checkpoint_path, state.last_combat_entry_index,
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
            chosen = next(
                (c for c in decision.choices
                 if (c.col, c.row) == (kwargs["col"], kwargs["row"])),
                None,
            )
            state.pending_map_step_types.append(chosen.type if chosen is not None else None)
            checkpoint_path = os.path.join(
                state.checkpoint_dir, f"{main_id}_c{combat_index + 1:03d}.json"
            )
            os.makedirs(state.checkpoint_dir, exist_ok=True)
            state.session.save(checkpoint_path)
            state.checkpoint_paths_this_trial.append(checkpoint_path)
        decision = state.session.step(name, **kwargs)
        if name == "select_map_node" and decision.decision == "combat_play":
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
) -> None:
    """Runs main-line trials (via _play_trial) until stop_event fires or cfg.n_main_trials is
    reached.

    job_queue/job_queue_priority/log_queue are shared with every other main worker; results_queue
    and meta_results_queue are this worker's own private queues (Facility-owned) and must not be
    shared -- meta_results_queue is kept separate from results_queue so an asynchronously-
    dispatched MetaOptionJob result can never be mistaken for a stale turn-branch result by
    combat's own _wait_for_turn_result. job_queue_priority carries this worker's own combat turn-
    branches so they're never stuck behind another main line's meta-chain jobs on the plain
    job_queue; branch_worker_loop always drains it first.

    combat_resolver(job_queue_priority, results_queue, main_worker_index, cfg, recorder, session,
    decision, character, seed, main_id, combat_index, stop_event, checkpoint_path) ->
    (session, decision, outcome): drives one combat to resolution. outcome is None on a win, else
    "lost"/"error" ends the trial.

    meta_resolver(job_queue, results_queue, meta_results_queue, main_worker_index, cfg, recorder,
    session, decision, character, seed, main_id, decision_kind, divergence_index, stop_event,
    queue_manager, last_combat_checkpoint_path, last_combat_entry_index) -> (decision,
    pending_meta_jobs): called whenever decision.decision is META_ELIGIBLE and this trial's
    meta-branch budget (cfg.meta_branches_per_trial) allows it. pending_meta_jobs is a list of
    (meta_id, checkpoint_path, is_adopted, commit_step) tuples for jobs just dispatched onto
    job_queue. None (facility_claws.py's case) means meta-eligible decisions just fall through to
    choose_action(). last_combat_checkpoint_path/entry_index (may be None pre-first-combat) locate
    the most recently entered combat's checkpoint and its entry point in action_log -- a
    checkpoint taken directly at the pending decision does NOT work for reconstructing "right
    after that combat" (confirmed by hand: save/load_save doesn't preserve a pending decision
    screen).

    manager_address/manager_authkey: (address, authkey) to reconnect to the Facility's own
    multiprocessing.Manager() via SyncManager(...).connect() -- this worker is daemonic and can't
    own a Manager itself, and a live SyncManager isn't picklable through Process(args=...) at all
    (confirmed by hand). Unused by the one shipped meta_resolver; kept for a future one that needs
    manager-backed queues at runtime. Skipped when meta_resolver is None.

    resume: if given, this worker's FIRST trial only reconstructs an in-progress run instead of
    starting fresh -- for recovering a trial that was progressing normally when the facility was
    killed before it reached game_over. Keys: character, seed, main_id, combat_index (combats
    already resolved), actions (full action log up to the resume point, same shape a branch/trial
    JSON's own "actions" field has)."""
    _tag_worker_stderr("main", main_worker_index, log_queue, cfg.verbose)
    # New process group -- Facility.stop()'s force-kill uses os.killpg on it (see there for why:
    # combat_resolver's dotnet-run subprocesses need to die with it, not be orphaned).
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
            session=Sts2Session(logger=recorder),
            checkpoint_dir=os.path.join(cfg.raw_dir, "checkpoints"),
            meta_budget=cfg.meta_branches_per_trial,
        )
        outcome = None
        try:
            try:
                outcome, _ = _play_trial(
                    cfg, job_queue_priority, job_queue, results_queue, meta_results_queue,
                    main_worker_index, recorder, character, seed, main_id, this_trial_resume,
                    stop_event, queue_manager, combat_resolver, meta_resolver, state,
                )
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
            target=_log_writer_loop, args=(self.log_queue, self.live, self._mark_activity),
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

    def stop(self, timeout: float = 60) -> None:
        self.stop_event.set()
        for _ in self.branch_procs:
            self.job_queue.put(None)

        deadline = time.monotonic() + timeout
        for p in self.main_procs + self.branch_procs:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                p.join(remaining)
        # A process still alive here didn't respond to stop_event within the shared deadline --
        # most likely wedged inside a native CLR call that never returns. SIGKILL can't be
        # deferred, so the join() right after is bounded even without its own timeout.
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
        # Every branch process (the only producers) has exited by now -- safe to shut the single
        # terminal-writer thread down.
        self.log_queue.put(None)
        if self.log_thread is not None:
            self.log_thread.join(timeout)
        # Every process that could still be holding a proxy into this manager has exited by now --
        # safe to tear its server process down too.
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

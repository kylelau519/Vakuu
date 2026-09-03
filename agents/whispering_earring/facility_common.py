"""Shared self-play-facility infrastructure: worker-process orchestration, stderr relay/logging,
FacilityConfig, and the pieces of combat scoring/turn-playing that more than one facility needs.

Split out of what used to be a single facility.py once a second facility (choices_paradox's
card_reward meta-branching, see agents/choices_paradox/model/README.md's Design section) needed
the same worker-pool machinery: `facility_claws.py` drives combat-turn branching (Claws's own
training data, the original whispering_earring scheme) and `facility_choices_paradox.py` adds
card_reward branching on top -- both build a `Facility` from this module and supply their own
main-line/branch-worker logic.

`Facility`, `main_worker_loop`, and `branch_worker_loop` are pure orchestration here -- they take
`combat_resolver`/`meta_resolver` (main line) and `handlers` (branch dispatch, keyed by job type)
as callables/mappings, and know nothing about combat turn-branching or meta-decision branching
specifically. Each concrete facility module binds its own domain logic to these via
`functools.partial` before constructing a `Facility` -- see either concrete module's CLI
entrypoint for the wiring.
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
# missing HP, so leftover HP is worth much less there than in a normal fight -- player HP loss is
# weighted down (_BOSS_HP_WEIGHT) and win/loss dominates (_BOSS_WIN_FLOOR) instead.
# _BOSS_ENEMY_WEIGHT rewards damage dealt to the boss this turn, which normal fights deliberately
# don't (no clock forcing urgency there); starts at the same magnitude as _BOSS_HP_WEIGHT until
# there's training signal to tune it.
_BOSS_HP_WEIGHT = 0.2
_BOSS_WIN_FLOOR = 0.8
_BOSS_ENEMY_WEIGHT = 0.2


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

# CoreCLR's own unhandled-exception/fatal-error dump. Without this, a CLR crash (confirmed
# happening for real -- see buffer_env.py's _ClrWorker) was silently dropped by _MILESTONE_RE in
# non-verbose mode, leaving a worker that just silently died or wedged with zero visible output.
# Once a line matches, relay() below switches that stream to always-relay for the rest of its life
# (relay()'s crash_seen), since the managed stack trace that follows won't itself match
# _MILESTONE_RE.
_CRASH_RE = re.compile(
    r"Unhandled [Ee]xception|Fatal error\.|Process terminating|Stack overflow\.|"
    r"Assertion failed|Segmentation fault|SIGABRT|SIGSEGV"
)


def _tag_worker_stderr(label: str, worker_index: int, log_queue: mp.Queue, verbose: bool) -> None:
    """Sts2BufferSession loads Sts2Headless's CLR in-process (clr_bridge.load_clr), and its
    [SIM]/[INFO] logging writes unconditionally to this process's real fd 2 (Console.Error
    dups it once on first access)."""
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
    """Fixed one-line-per-worker status board for _log_writer_loop, in place of the old scrolling
    relay -- each worker gets a permanent row, redrawn in place, instead of milestone lines
    scrolling out of view faster than they can be read.

    `labels` fixes the row order up front (every worker Facility is about to spawn, known before
    any of them can log anything) so a row never needs to be created mid-run. Falls back to plain
    scrolling (`scroll()`, used for every subsequent line once triggered) the first time a crash
    is relayed, or immediately if stderr isn't a real terminal: a multi-line stack trace collapsed
    into one row would lose everything but its last line, and the fixed rows' cursor-position math
    only holds if nothing else scrolls the terminal in between updates -- both irreversible for the
    life of this object rather than something to patch around mid-display."""

    def __init__(self, labels: list[tuple[str, int]]):
        self.rows = {key: i for i, key in enumerate(labels)}
        self.n = len(labels)
        self.active = sys.stderr.isatty() and self.n > 0
        # update() runs on _log_writer_loop's own thread; scroll() is also called directly from
        # Facility.run()/stop() on the main thread (see their call sites) while that thread may
        # still be alive -- guards both against interleaving mid-escape-sequence.
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
        """Permanent fallback to plain scrolling output -- crash dumps (multi-line, must not be
        collapsed into one row) and Facility's own status prints (stall/shutdown notices, which
        would otherwise land below the fixed rows and desync their cursor-position math) both go
        through this instead of a bare print()."""
        with self._lock:
            self.active = False
            sys.stderr.write(text if text.endswith("\n") else f"{text}\n")
            sys.stderr.flush()


def _log_writer_loop(
    log_queue: mp.Queue, live: _LiveStatus, on_activity: Callable[[], None] | None = None
) -> None:
    """Runs in a single background thread in the parent (Facility) process -- the only writer to
    the real terminal for every branch worker's relayed stderr.

    on_activity: called on every real (non-sentinel) line -- Facility.run()'s stall watchdog uses
    this as its liveness signal (a worker wedged inside a native CLR call can still show up as
    "alive" without ever making Python-visible progress again)."""
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
    """Bundles _combat_value's inputs -- one struct instead of a positional-argument list that
    grows every time a new reward signal comes in."""
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
    """Sums current HP across only the enemy slots whose decoded token is in boss_ids -- the
    entities present when this boss combat was entered (see facility_claws._resolve_combat),
    fixed for the whole fight. A mid-fight summon never matches boss_ids so it never contributes,
    and a tracked entity that has since died simply contributes 0 rather than needing special-
    casing."""
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
    verbose: bool = False  # relay every branch worker's [SIM]/[INFO] line, not just milestones
    # Facility.run()'s watchdog: if no relayed log line has landed for longer than this while a
    # main worker is still alive, it's treated as wedged and killed rather than joined forever --
    # added after a live CoreCLR crash got stuck mid-abort instead of terminating the process,
    # which otherwise hangs the whole facility (Python's atexit multiprocessing cleanup does an
    # equally untimed join). 1800s (30 min) is generous -- a legitimate long walk still produces a
    # milestone line at least once per combat/room.
    stall_timeout_s: float = 1800.0

    # Meta-decision (card_reward, eventually card_select) branching -- see
    # agents/choices_paradox/model/README.md's Design section. facility_claws.py's CLI never
    # touches this, so its default of 0 (disabled) keeps that facility's behavior unchanged.
    # facility_choices_paradox.py's CLI sets it to None (branch every eligible decision) by
    # default.
    meta_branches_per_trial: int | None = 0  # 0 = disabled; None = unlimited; else a decrementing
    # per-trial budget (not a sampling scheme -- see the README's FacilityConfig knobs section)


def _bypass_card_reward(session) -> None:
    # A branch's own reward pick is discarded either way -- only the main line's adopted actions
    # matter. Only used when a combat-turn branch's job ends the instant it wins (see
    # _play_turn_only's bypass_reward_on_win); a meta-branch horizon walk that keeps playing past
    # the win instead resolves the same card_reward for real and must NOT go through this bypass.
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

    is_last_of_act/boss_ids/boss_max_hp come from the matching BranchJob fields (see
    checkpoint.py's BranchJob docstring) and drive boss-fight reward shaping; boss_ids/boss_max_hp
    are only ever non-empty when is_last_of_act is True.

    bypass_reward_on_win: combat-turn branch workers (facility_claws) always want the default --
    a branch's job ends the instant it wins, and a dangling pending card_reward would otherwise
    leak into the worker's next job. facility_choices_paradox's horizon walk passes False instead,
    since it keeps playing past the win and needs the reward resolved for real."""
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


# Phase 1 scope (agents/choices_paradox/model/README.md's Phasing section): only card_reward is
# meta-branchable. Phase 2 widens this to card_select's remove/upgrade purposes.
META_ELIGIBLE_DECISIONS = frozenset({"card_reward"})


def main_worker_loop(
    cfg: FacilityConfig, job_queue: mp.Queue, job_queue_priority: mp.Queue, log_queue: mp.Queue,
    results_queue, main_worker_index: int, stop_event, trial_counter=None,
    resume: dict | None = None, manager_address=None, manager_authkey: bytes | None = None, *,
    combat_resolver: Callable, meta_resolver: Callable | None = None,
) -> None:
    """job_queue/job_queue_priority/log_queue are shared with every other main worker;
    results_queue is this worker's own private queue, created and owned by Facility, and must not
    be shared.

    job_queue_priority exists so this worker's OWN combat turn-branches are never stuck behind
    facility_choices_paradox.py's meta-chain jobs (which land on the plain job_queue and can vastly
    outnumber the main line's own); branch_worker_loop always drains job_queue_priority first.
    Called below with job_queue_priority whenever it's the main line driving its own combat
    directly; job_queue is what gets threaded into meta_resolver.

    combat_resolver(job_queue, results_queue, main_worker_index, cfg, recorder, session, decision,
    character, seed, main_id, combat_index, stop_event, checkpoint_path) -> (session, decision,
    outcome) drives one combat to resolution -- both concrete facilities pass
    facility_claws._resolve_combat.

    meta_resolver(job_queue, results_queue, main_worker_index, cfg, recorder, session, decision,
    character, seed, main_id, decision_kind, divergence_index, stop_event, queue_manager,
    last_combat_checkpoint_path, last_combat_entry_index) -> decision, called whenever
    decision.decision is in META_ELIGIBLE_DECISIONS and this trial's meta-branch budget
    (cfg.meta_branches_per_trial, decremented below) allows it. None (facility_claws.py's case)
    means meta-eligible decisions just fall through to the ordinary choose_action() path below.

    last_combat_checkpoint_path/last_combat_entry_index: the native checkpoint (and that combat's
    entry action's position in action_log) for whichever combat was entered most recently -- e.g.
    the one that just produced the card_reward decision meta_resolver is about to be called with.
    A meta_resolver needing to reconstruct "the state right after that combat" can replay just
    that combat's own tail of actions via load_and_resume_combat(last_combat_checkpoint_path,
    action_log[last_combat_entry_index], action_log[last_combat_entry_index + 1:]) instead of
    replaying from StartRun -- a checkpoint taken directly AT the pending card_reward does NOT
    work for this (confirmed by hand: RunSimulator's save/load_save doesn't preserve a pending
    decision screen -- reloading lands on the FOLLOWING decision, past the reward). May be None if
    no checkpoint was available yet (this run's very first combat).

    manager_address/manager_authkey: how to reach the Facility's own multiprocessing.Manager()
    (built and owned by the Facility process, which is not itself daemonic). Not the SyncManager
    object itself -- a live SyncManager holds its own server Process handle, which isn't picklable
    through Process(args=...) at all (confirmed by hand: "cannot pickle 'weakref.ReferenceType'
    object") -- so only its plain (address, authkey) values are handed to this worker at spawn,
    and reconnected here via SyncManager(...).connect() into `queue_manager`, threaded into every
    meta_resolver call. Currently unused by the one shipped meta_resolver; kept as plumbing in
    case a future one needs to mint manager-backed queues at runtime (a meta_resolver can't build
    its own mp.Manager() locally, since this worker process is itself daemonic and "daemonic
    processes are not allowed to have children" -- .connect() doesn't spawn anything, so it's fine
    here). Skipped entirely when meta_resolver is None.

    resume: if given, this worker's FIRST trial (only) reconstructs an in-progress run instead of
    starting fresh -- for recovering a trial that was progressing normally when the facility was
    killed before it reached game_over. Keys: character, seed, main_id, combat_index (combats
    already resolved), actions (full action log up to the resume point, same shape a branch/trial
    JSON's own "actions" field has)."""
    _tag_worker_stderr("main", main_worker_index, log_queue, cfg.verbose)
    # New process group, with this worker as its leader -- Facility.stop()'s force-kill path uses
    # os.killpg on this group instead of killing just this one PID. Needed because combat_resolver
    # shells out to a real `dotnet run` OS subprocess (Sts2Session) and re-spawns a fresh one every
    # combat turn -- confirmed by hand that one of these can sit around for DAYS, fully orphaned,
    # once its owning Python process dies without running Sts2Session.close(). Must happen before
    # anything below can spawn a child: a process can only change its OWN group before it has any
    # children of its own to leave behind in the old one.
    os.setpgid(0, 0)
    # Each main worker is its own process; reseed explicitly so concurrent workers started via
    # fork (which would otherwise inherit identical RNG state) don't all draw the same
    # character/seed. spawn (macOS default) doesn't need this, but it's cheap insurance.
    random.seed(os.urandom(16))

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
        session = Sts2Session(logger=recorder)
        outcome = None
        combat_checkpoint_path = None
        # The checkpoint (and its entry action's position in action_log) for whichever combat was
        # MOST RECENTLY entered -- unlike combat_checkpoint_path (cleared once handed to
        # combat_resolver), these survive past that combat's resolution so a meta_resolver call on
        # the card_reward decision it just produced can reconstruct "right after that win" via
        # load_and_resume_combat instead of replaying the whole run. last_combat_entry_index stays
        # valid after the combat resolves since actions are only ever appended, never reordered.
        last_combat_checkpoint_path = None
        last_combat_entry_index = None
        pending_purpose = None  # threaded through choose_action -- see its docstring
        meta_budget = cfg.meta_branches_per_trial
        divergence_index = 0
        # One file per combat (not reused/overwritten across the trial) -- see the checkpoint-
        # taking comment below for why reuse is unsafe. Deleted once this trial concludes.
        checkpoint_dir = os.path.join(cfg.raw_dir, "checkpoints")
        checkpoint_paths_this_trial: list[str] = []
        try:
            try:
                if this_trial_resume is not None:
                    decision = session.open_from_checkpoint(
                        character, seed, cfg.ascension, this_trial_resume["actions"]
                    )
                    combat_index = this_trial_resume["combat_index"]
                else:
                    decision = session.start(character, seed, cfg.ascension)
                    combat_index = 0

                while decision.decision != "game_over":
                    if decision.decision == "combat_play":
                        combat_index += 1
                        last_combat_checkpoint_path = combat_checkpoint_path
                        # Fixed for the rest of this combat's life, including after it resolves --
                        # see the comment above on why this stays a valid index into action_log.
                        last_combat_entry_index = len(session.action_log) - 1
                        session, decision, combat_outcome = combat_resolver(
                            job_queue_priority, results_queue, main_worker_index, cfg, recorder,
                            session, decision, character, seed, main_id, combat_index, stop_event,
                            combat_checkpoint_path,
                        )
                        combat_checkpoint_path = None  # consumed -- not valid for the next combat
                        if combat_outcome is not None:
                            outcome = combat_outcome
                            break
                        continue

                    if (meta_resolver is not None and meta_budget != 0
                            and decision.decision in META_ELIGIBLE_DECISIONS):
                        divergence_index += 1
                        if meta_budget is not None:
                            meta_budget -= 1
                        decision = meta_resolver(
                            job_queue, results_queue, main_worker_index, cfg, recorder, session,
                            decision, character, seed, main_id, decision.decision,
                            divergence_index, stop_event, queue_manager,
                            last_combat_checkpoint_path, last_combat_entry_index,
                        )
                        continue

                    name, kwargs, policy, pending_purpose = choose_action(
                        decision, session, pending_purpose
                    )
                    if policy is not None:
                        recorder.attach_policy(policy)
                    if name == "select_map_node":
                        # select_map_node is the only action that consumes RNG for encounter
                        # selection, at room-entry time -- the checkpoint MUST be taken before
                        # it, not after, or a branch reloading it and replaying this same action
                        # would draw a different encounter (see checkpoint.py's docstring and
                        # the whispering_earring_combat_checkpoint_branching memory). Whether
                        # this node turns out to be a combat is only known after stepping into
                        # it below; if it isn't, this save is simply never consumed (harmless --
                        # cleaned up with the rest of this trial's checkpoints below).
                        #
                        # Filename is unique per (trial, prospective combat index), NOT reused
                        # across the trial: a straggler branch worker (the shared pool can be
                        # oversubscribed) can still be reading this file long after the main
                        # line has moved on to a LATER select_map_node, and a reused filename
                        # would let it silently load the wrong state.
                        checkpoint_path = os.path.join(
                            checkpoint_dir, f"{main_id}_c{combat_index + 1:03d}.json"
                        )
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        session.save(checkpoint_path)
                        checkpoint_paths_this_trial.append(checkpoint_path)
                    decision = session.step(name, **kwargs)
                    if name == "select_map_node" and decision.decision == "combat_play":
                        combat_checkpoint_path = checkpoint_path

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
            # By the time a trial concludes, every combat/meta-decision it contained has already
            # resolved (or given up on) all its own branches -- any straggler still unprocessed
            # at this point is far enough past its own wait-for-results timeout that its result
            # would be discarded as stale anyway, so it's safe to reclaim the disk here rather
            # than let these accumulate for the life of the facility.
            for path in checkpoint_paths_this_trial:
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
    MetaBranchJob (card_reward branching, facility_choices_paradox.py) side by side.
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
        self.branch_procs: list[mp.Process] = []
        self.log_thread: threading.Thread | None = None
        self.live: _LiveStatus | None = None  # built in start() -- see _LiveStatus
        self.manager = None  # multiprocessing.managers.SyncManager, created in start()
        # Updated from _log_writer_loop's thread on every relayed log line -- run()'s stall
        # watchdog reads it from the main thread. Plain float, not Lock-guarded: each side only
        # ever does a whole-object read or write (atomic under the GIL), never a read-modify-write.
        self._last_activity = time.monotonic()

    def _mark_activity(self) -> None:
        self._last_activity = time.monotonic()

    def start(self) -> None:
        ensure_setup()  # once, here -- not per-session, which would repeat the same checks
        self._last_activity = time.monotonic()  # don't count ensure_setup()'s own time against it

        # Row order fixed up front -- every worker Facility is about to spawn is already known
        # here, before any of them can log anything, so _LiveStatus never needs to add a row
        # mid-run.
        labels = [("main", i) for i in range(self.cfg.n_main_workers)]
        labels += [("branch", i) for i in range(self.cfg.n_branch_workers)]
        self.live = _LiveStatus(labels)
        self.log_thread = threading.Thread(
            target=_log_writer_loop, args=(self.log_queue, self.live, self._mark_activity),
            daemon=True, name="branch-log-writer",
        )
        self.log_thread.start()

        # Manager()-backed, not plain mp.Queue() -- branch_worker_loop routes a result back to
        # whichever queue rode along with the job (see its docstring), and a raw Queue can only
        # cross into an already-running process via spawn-time inheritance -- pickling one through
        # ANOTHER queue at runtime raises "Queue objects should only be shared between processes
        # through inheritance" (confirmed by hand on spawn-based multiprocessing, macOS's
        # default). A SyncManager's Queue() is a proxy, not a raw pipe wrapper, so it stays
        # shippable at any time. Built here (Facility's own process, not itself daemonic) since a
        # main worker process IS daemonic and can't build its own Manager (a Manager.start()'s its
        # own server process, and "daemonic processes are not allowed to have children"). Each
        # main worker gets this manager's (address, authkey) at spawn, not the SyncManager object
        # itself -- a live SyncManager holds its own server Process handle, which isn't picklable
        # at all (confirmed by hand: "cannot pickle 'weakref.ReferenceType' object"). Address+
        # authkey are plain picklable values; main_worker_loop reconnects to this SAME server via
        # SyncManager(...).connect(), which -- unlike .start() -- doesn't spawn anything, so it's
        # fine from a daemonic process.
        self.manager = mp.Manager()
        manager_address = self.manager.address
        manager_authkey = bytes(self.manager._authkey)
        self.main_result_queues.extend(self.manager.Queue() for _ in range(self.cfg.n_main_workers))

        for i, results_queue in enumerate(self.main_result_queues):
            p = mp.Process(
                target=self.main_worker_fn,
                args=(self.cfg, self.job_queue, self.job_queue_priority, self.log_queue,
                      results_queue, i, self.stop_event, self.trial_counter,
                      self.resume if i == 0 else None, manager_address, manager_authkey),
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
        for p in self.main_procs:
            p.join(timeout)
        for p in self.branch_procs:
            p.join(timeout)
        # A process still alive here didn't respond to stop_event within `timeout` -- most likely
        # wedged inside a native CLR call that never returns. SIGKILL can't be deferred, so the
        # join() right after is bounded even without its own timeout.
        #
        # os.killpg, not p.kill() -- a plain p.kill() only signals this ONE pid, orphaning any
        # live `dotnet run` Sts2Session subprocess a wedged main worker holds (SIGKILL gives it no
        # chance to run Sts2Session.close()). Confirmed by hand: exactly one of these orphans
        # survived for 11+ days, invisible to training_choices.sh's own `pkill` cleanup (which
        # only matches the Python side). Each worker made itself its own process group leader at
        # startup for exactly this -- os.killpg(pid, SIGKILL) with the worker's own pid (its group
        # id) takes out the worker plus every subprocess it spawned in one shot.
        # ProcessLookupError means it already exited between is_alive() and here -- falls back to
        # p.kill() purely so the join() below still has a well-defined target.
        for p in self.main_procs + self.branch_procs:
            if p.is_alive():
                # self.live.scroll(), not print() -- log_thread (writing self.live) is still alive
                # here (joined below), and a bare print() to stdout could still visually land
                # inside the live board's rows on the same terminal even though it's a separate fd.
                self.live.scroll(
                    f"{p.name} didn't stop within {timeout}s -- killing its process group (SIGKILL)"
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
        # SIGINT (Ctrl-C) already unwinds via KeyboardInterrupt below. Without this handler a
        # plain `kill <pid>` (SIGTERM) falls through to the default action and skips stop()
        # entirely, leaving every daemon worker orphaned and idling on the job queue forever.
        # SIGKILL still can't be caught, so this doesn't cover a hard `kill -9`.
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
                              "combats) when the facility died before writing its own output "
                              "(e.g. killed mid-run). Pass the path to any surviving branch (or "
                              "main-trial) JSON with a \"won\" or \"continue\" outcome from that "
                              "trial's last resolved turn -- its own \"actions\" field already "
                              "has the full history, and its character/seed/main_id/combat_index "
                              "fields identify where to pick up. Worker 0's first trial resumes "
                              "from it (character/seed forced, ignoring --characters); every "
                              "other worker, and worker 0's later trials if --trials allows more "
                              "than one, run normally.")


def load_resume(path: str) -> dict:
    import json
    with open(path) as f:
        r = json.load(f)
    return {
        "character": r["character"], "seed": r["seed"], "main_id": r["main_id"],
        "combat_index": r["combat_index"], "actions": _clean_actions(r["actions"]),
    }

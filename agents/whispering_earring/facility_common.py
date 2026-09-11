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
from env import Sts2ActionError, Sts2EngineTimeoutError, Sts2Session
from buffer_env import Sts2BufferSession, BufferDecision, decode_token
from installation import ensure_setup

from policy import choose_action, get_claws_net, golden_compass, _gc_model
from recorder import CompactRecorder
from checkpoint import MapOptionJob

_SEED_WORDS = ["falcon", "ember", "wolf", "raven", "thorn", "comet", "storm", "harbor"]

_HEALTHY_HP_FRACTION = 0.5
_LOW_HP_SLOPE = 3.0

# Per-turn shaping weight, applied equally to hp_term and enemy_term below -- both are
# potential-differences (telescope to a fixed total over the whole combat regardless of which
# turn the damage happened on), so weighting them the same doesn't bake in a fixed HP-for-damage
# trade rate; it only densifies credit assignment turn-by-turn.
_HP_WEIGHT = 0.75

# Boss-fight (act-final combat) reward shaping. At ascension 2+, winning an act heals 80% of
# missing HP, so leftover HP is worth much less there than in a normal fight
_BOSS_HP_WEIGHT = 0.2
_BOSS_WIN_FLOOR = 0.8

# Golden Compass reward shaping (see main_worker_loop's pending_map_steps machinery)
# the old scheme did.
_GOLDEN_COMPASS_GAMMA = 0.9
_GOLDEN_COMPASS_ELITE_BONUS = 2.0
_GOLDEN_COMPASS_BOSS_BONUS = 3.0


def _random_seed() -> str:
    return f"{random.choice(_SEED_WORDS)}-{random.randint(0, 999_999)}"


_DEBUG_REWARDS = bool(os.environ.get("CLAWS_DEBUG_REWARDS"))  # per-turn tri-head reward trace,
# see _play_turn_only -- opt-in (can be noisy at any real branch fan-out) rather than gated on
# FacilityConfig.verbose, since threading a new flag through every _play_turn_only caller (branch
# jobs, map-option jobs, meta-option jobs, eval) just to reach this one print isn't worth it.

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


def _map_point_type_at(decision, col: int, row: int) -> str | None:
    """The MapPointType (as a string, e.g. "Elite"/"Boss") of the map node at (col, row) --
    used only for Golden Compass's Elite/Boss shaping bonus (see _play_trial's select_map_node
    branch). BufferDecision has no `.choices` list to match against like the pydantic Decision
    does, but the same data is directly readable off the map grid via buffer_layout, keyed by
    (row, col) -- row/col -1 is map_points'/GetFullMap()'s shared sentinel for the boss node,
    which lives outside the row/col grid on both sides."""
    if hasattr(decision, "arr"):
        slot = bl.map_boss_point(decision.arr) if row == -1 else bl.map_point_slot(decision.arr, row, col)
        return bl.map_point_type(slot)
    chosen = next((c for c in decision.choices if (c.col, c.row) == (col, row)), None)
    return chosen.type if chosen is not None else None


def _tracked_enemy_hp(arr, enemy_ids: frozenset[str]) -> int:
    """Sums current HP across only the enemy slots whose decoded token is in enemy_ids -- anything
    summoned mid-fight (not in the set captured at combat entry) is silently not counted."""
    if not enemy_ids:
        return 0
    return sum(
        bl.enemy_hp(slot) for _, slot in bl.enemies(arr)
        if decode_token(bl.enemy_token(slot)) in enemy_ids
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
    # Meta-decision (card_reward, rest_site, eventually card_select) branching
    meta_branches_per_trial: int | None = 0  # 0 = disabled; None = unlimited
    # None = play each option-chain to natural death (original behavior); a positive int caps
    # how many combats a chain plays past its own commit before it's scored on whatever
    # total_value it's accumulated so far -- see _run_meta_option_job. No terminal penalty either
    # way, same reasoning as the unbounded case: cutting a chain off early isn't a loss.
    meta_horizon_combats: int | None = None
    # select_map_node branching (agents/golden_compass) -- see _resolve_map_decision/
    # _run_map_option_job. 0 = disabled (main line still commits via golden_compass's own DP
    # heuristic, just with no branch-derived training signal); None = unlimited.
    map_branches_per_trial: int | None = 0
    # How many further select_map_node decisions a MapOptionJob walk plays past its own commit
    # before scoring -- see _run_map_option_job's docstring for why this is rooms, not combats,
    # and why the default is 1 (score the committed sibling's own room, stop the instant the walk
    # is back at the map for a further junction).
    map_room_horizon: int = 1
    # Root Dirichlet noise on claws combat-action sampling (policy.choose_action's
    # combat_dirichlet_noise) -- see _play_turn_only. Off lets a run sample straight from the
    # net's own softmax, e.g. to eval/compare a checkpoint without exploration noise skewing it.
    combat_dirichlet_noise: bool = True


def _bypass_card_reward(session) -> None:
    # A branch's own reward pick is discarded either way -- only the main line's adopted actions
    # matter.
    raw = session._session.Action("skip_card_reward", None)
    if session._decision(raw).decision == "error":
        raise Sts2ActionError(session._session.LastError or "unknown error")


_warned_no_claws_net = False  # print the fallback note below at most once per worker process


def _win_value_of(decision) -> float:
    """win_value estimate for the state a turn is about to be played from. Every turn but a
    combat's first reuses the previous turn's own recorded win_value_after (see _play_turn_only's
    win_value_before param); a combat's first turn has no previous turn to reuse, so callers get
    one extra net call from here at combat entry -- see tri_head_design.md's win_value threading."""
    net = get_claws_net()
    if net is None:
        global _warned_no_claws_net
        if not _warned_no_claws_net:
            _warned_no_claws_net = True
            print(
                "[claws] no CLAWS_CHECKPOINT loaded in this worker -- win_value/scale_term will "
                "be a flat 0.0 for every continuing (non-combat-ending) turn this round, so "
                "sibling comparisons over non-terminal turns will get 0 REINFORCE advantage "
                "(materialize.py's _group_baseline degrades to baseline==value exactly). This is "
                "expected for a checkpoint-less round 1; set CLAWS_CHECKPOINT once a checkpoint "
                "exists so win_value/scale_term carry real signal.",
                file=sys.stderr,
            )
        return 0.0
    _, _, values = net.forward(decision)
    return float(values["win"].item())


def _play_turn_only(
    session, decision, combat_hp_before: int, combat_max_hp: int,
    is_last_of_act: bool = False, enemy_ids: frozenset[str] = frozenset(), enemy_max_hp: int = 0,
    bypass_reward_on_win: bool = True, combat_dirichlet_noise: bool = True,
    win_value_before: float = 0.0,
):
    """Plays exactly one turn (through the mid-turn card_select edge case, e.g. Power Potion/
    Brand) and scores it before returning `(outcome, decision, reward)` -- outcome is "continue" if
    the combat carries on, "won"/"lost" if this turn ended it; decision is whatever the game shows
    after (a fresh combat_play round, the pending card_reward on a win, or game_over on a loss);
    reward is this turn's tri-head reward streams (tri_head_design.md):
    `{hp_term, enemy_term, scale_term, win_value_after}`.

    For boss fight reward are shaped differently.

    bypass_reward_on_win: For branch job to skip reward because they are irrelevant.
    combat_dirichlet_noise: forwarded to policy.choose_action -- see FacilityConfig.combat_dirichlet_noise.
    win_value_before: this turn's own win_value_before -- the previous turn's win_value_after, or
    (for a combat's first turn) a fresh _win_value_of(decision) call the caller makes at combat
    entry. Feeds scale_term's residual below."""
    hp_before_turn = decision.player_hp
    enemy_hp_before_turn = _tracked_enemy_hp(decision.arr, enemy_ids)
    session.logger.begin_turn()
    ended_turn = False
    while decision.decision in ("combat_play", "card_select"):
        # purpose is never threaded here -- a mid-turn card_select (Power Potion, Brand, ...)
        # never follows a shop/rest_site pick, so it's never "remove"/"upgrade" (see
        # choose_action's docstring); the 4th return value is discarded for the same reason.
        name, kwargs, policy, _ = choose_action(
            decision, session, combat_dirichlet_noise=combat_dirichlet_noise
        )
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

    # hp_term/enemy_term are computed for every turn, not just ones that don't end the combat --
    # value_aggro/value_defense need real reward on the turn that finishes a fight too (a killing
    # blow is exactly the kind of thing "aggro" should get credit for).
    hp_term = -(hp_before_turn - decision.player_hp) / combat_max_hp if combat_max_hp else 0.0
    enemy_term = 0.0
    if enemy_max_hp and decision.decision != "game_over":  # enemy state isn't meaningful on death
        enemy_hp_after_turn = _tracked_enemy_hp(decision.arr, enemy_ids)
        enemy_term = (enemy_hp_before_turn - enemy_hp_after_turn) / enemy_max_hp

    if decision.decision == "card_reward":
        if bypass_reward_on_win:
            _bypass_card_reward(session)
        outcome = "won"
        win_value_after = _combat_value(
            CombatOutcome(decision.player_hp, combat_max_hp, is_last_of_act)
        )
    elif decision.decision == "game_over":
        outcome = "lost"
        win_value_after = -1.0
    else:
        outcome = "continue"
        win_value_after = _win_value_of(decision)

    # scale_term: how much the graded outcome moved this turn, for reasons other than the direct
    # HP/enemy swings already credited to the other two streams -- see tri_head_design.md's Reward
    # stream definitions.
    scale_term = (win_value_after - win_value_before) - hp_term - enemy_term
    reward = {
        "hp_term": hp_term, "enemy_term": enemy_term, "scale_term": scale_term,
        "win_value_after": win_value_after,
    }
    if _DEBUG_REWARDS:
        print(
            f"[claws] turn reward: outcome={outcome} hp_term={hp_term:+.3f} "
            f"enemy_term={enemy_term:+.3f} scale_term={scale_term:+.3f} "
            f"win_value_before={win_value_before:+.3f} win_value_after={win_value_after:+.3f}",
            file=sys.stderr,
        )
    session.logger.end_turn(reward)
    return outcome, decision, reward


def _clean_actions(actions: list[dict]) -> list[dict]:
    """Strips CompactRecorder's added `policy`/`value` keys back to bare {cmd, action, args?}"""
    return [
        {"cmd": e["cmd"], "action": e["action"], **({"args": e["args"]} if "args" in e else {})}
        for e in actions
    ]


# NOTE: on a BufferDecision, a blessing/event-triggered card reward reports decision.decision ==
# "event_card_reward", not "card_reward" (buffer_layout.decision_name distinguishes the two
# sections; the pydantic side unifies both under "card_reward" -- see policy.py's
# _choose_card_reward_from_buffer). Since that string isn't in this set, an event-triggered card
# reward on a buffer-session main line falls through to choose_action() same as any other
# unbranched decision, rather than getting the meta-option branching an ordinary post-combat
# card_reward gets -- a real but narrow coverage gap from running the main line on
# Sts2BufferSession, not a bug (gameplay still proceeds correctly either way).
META_ELIGIBLE_DECISIONS = frozenset({"card_reward", "rest_site", "event_choice"})
# Raised from 30.0: MetaOptionJobs land on the low-priority half of the shared branch-worker
# queue (combat turn-branches always go first, see facility_choices_paradox.py's dispatch), so
# with few branch workers relative to many concurrent main lines (e.g. 3 workers / 10 main
# workers), a meta job can sit queued -- not even started -- well past 30s. Confirmed live: every
# sibling of a divergence failing with the same FileNotFoundException (not just a straggler among
# otherwise-successful siblings) means the whole divergence missed the window, not a per-job race.
_META_DRAIN_TIMEOUT = 120.0

# Combat BranchJobs (facility_claws._spawn_turn_branches) have no per-checkpoint refcount the way
# MetaOptionJobs do (see _apply_meta_result) -- facility_claws._wait_for_turn_result adopts
# branch_ids[0] and returns as soon as it's usable, WITHOUT waiting for the other
# branches_per_turn-1 siblings, which can still be queued behind job_queue_priority's backlog or
# mid-flight reading the SAME room checkpoint (unchanged across every turn of one combat -- see
# _resolve_combat). Deleting a trial's checkpoints the moment that trial ends (the old behavior)
# races those stragglers: confirmed live via a cluster of Sts2Headless FileNotFoundExceptions all
# pointing at one now-deleted checkpoint, from many turn/branch ids of the SAME combat. Aging
# checkpoints out by mtime instead (see _sweep_stale_checkpoints) sidesteps the race without
# needing to plumb a refcount through combat_resolver's generic signature -- comfortably longer
# than _META_DRAIN_TIMEOUT since combat branches sit on the higher-priority queue and should clear
# faster than meta jobs do.
_CHECKPOINT_GRACE_PERIOD = 300.0


def _sweep_stale_checkpoints(checkpoint_dir: str) -> None:
    """Deletes room checkpoints older than _CHECKPOINT_GRACE_PERIOD -- called once per trial
    (main_worker_loop) instead of deleting a trial's own checkpoints the instant it ends, so a
    still-running straggler BranchJob (see _CHECKPOINT_GRACE_PERIOD's comment) gets a generous
    window to finish reading its checkpoint before the file disappears from under it."""
    try:
        entries = os.scandir(checkpoint_dir)
    except OSError:
        return
    cutoff = time.time() - _CHECKPOINT_GRACE_PERIOD
    with entries:
        for entry in entries:
            try:
                if entry.stat().st_mtime < cutoff:
                    os.remove(entry.path)
            except OSError:
                pass


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

    pending_map_steps: list = field(default_factory=list)
    pending_map_step_types: list = field(default_factory=list)
    local_map_rewards: dict = field(default_factory=dict) 
    
    pending_meta_jobs: dict = field(default_factory=dict)
    checkpoint_refcount: dict = field(default_factory=dict)

    map_budget: int | None = 0  # see FacilityConfig.map_branches_per_trial


def _play_trial(
    cfg: FacilityConfig, job_queue_priority: mp.Queue, job_queue: mp.Queue, results_queue,
    meta_results_queue, main_worker_index: int, recorder, character: str, seed: str, main_id: str,
    this_trial_resume: dict | None, stop_event, queue_manager, combat_resolver: Callable,
    meta_resolver: Callable | None, map_resolver: Callable | None, state: _TrialState,
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

        if (map_resolver is not None and state.map_budget != 0
                and decision.decision == "map_select"):
            state.divergence_index += 1
            if state.map_budget is not None:
                state.map_budget -= 1
            name, kwargs, new_pending = map_resolver(
                job_queue, meta_results_queue, cfg, recorder, state.session, decision,
                character, seed, main_id, state.divergence_index,
                state.last_room_checkpoint_path, state.last_room_entry_index,
            )
            for meta_id, ckpt_path, is_adopted, commit_step in new_pending:
                state.pending_meta_jobs[meta_id] = (ckpt_path, is_adopted, commit_step)
                if ckpt_path is not None:
                    state.checkpoint_refcount[ckpt_path] = (
                        state.checkpoint_refcount.get(ckpt_path, 0) + 1
                    )
            _drain_meta_results_nowait(
                meta_results_queue, state.pending_meta_jobs, state.checkpoint_refcount, recorder
            )
        else:
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


def _run_map_option_job(cfg: FacilityConfig, job: MapOptionJob, results_queue) -> None:
    """branch_worker_loop's MapOptionJob handler -- reconstructs to job.commit_args's sibling,
    plays forward up to job.room_horizon further select_map_node decisions (fighting whatever
    combat that leads into via _play_turn_only, same unbranched single-line playout
    _run_meta_option_job uses for its own chains -- no further turn-branching), then scores the
    walk's TERMINAL state: -1.0 if it died along the way, else _combat_value's hp-fraction-shaped
    read of however healthy the walk arrived at its own cutoff.

    Deliberately NOT a flat per-combat win bonus (_run_meta_option_job's own `_WIN_BONUS` scheme,
    which this is otherwise structurally identical to): that formula never penalizes a chain for
    seeking a fight out, only for losing one it started, which is a fine incentive for a
    card_reward pick that can't affect whether combat happens at all -- but a map choice's whole
    job IS deciding whether to walk into one, so a flat bonus per win rewards packing in
    additional risky fights the same walk didn't need to take. Scoring by terminal HP fraction
    instead means an extra fight only pays off if it's still net-positive for survivability, and a
    route through a Rest Site (or around an optional Monster) scores on its own merits -- no
    special-cased "avoid combat" term needed, since _combat_value's own >=50%-HP-vs-below-50%-HP
    shaping already treats "won, but limped out at 10% HP" as nearly as bad as losing outright."""
    import torch

    recorder = CompactRecorder({
        "character": job.character, "seed": job.seed, "ascension": job.ascension,
        "main_id": job.main_id, "role": "branch", "branch_kind": "map",
        "divergence_index": job.divergence_index, "option_index": job.option_index,
        "prefix_len": len(job.prefix),
    }, start_step=len(job.prefix))
    session = Sts2BufferSession(logger=recorder)
    value = None
    outcome = None
    pending_purpose = None
    try:
        try:
            if job.last_room_checkpoint_path is not None and job.last_room_entry_index is not None:
                entry_action = job.prefix[job.last_room_entry_index]
                room_actions = job.prefix[job.last_room_entry_index + 1:]
                decision = session.load_and_resume_combat(
                    job.last_room_checkpoint_path, entry_action, room_actions
                )
            else:
                decision = session.open_from_checkpoint(
                    job.character, job.seed, job.ascension, job.prefix
                )
            session.action_log = list(job.prefix)
            commit_step = recorder.mark_step()
            decision = session.step(job.commit_name, **job.commit_args)
            # Reseed AFTER committing, same reasoning as _run_meta_option_job's own walk_seed --
            # every sibling option's post-commit policy draws start from the identical stream.
            random.seed(job.walk_seed)
            torch.manual_seed(job.walk_seed)

            rooms_walked = 0
            while decision.decision != "game_over":
                if decision.decision == "combat_play":
                    combat_hp_before, combat_max_hp = decision.player_hp, decision.player_max_hp
                    win_value_before = _win_value_of(decision)
                    turn_outcome = "continue"
                    while turn_outcome == "continue":
                        turn_outcome, decision, reward = _play_turn_only(
                            session, decision, combat_hp_before, combat_max_hp,
                            bypass_reward_on_win=False, win_value_before=win_value_before,
                        )
                        win_value_before = reward["win_value_after"]
                    if turn_outcome == "lost":
                        outcome = "lost"
                        break
                    continue

                # card_reward_dirichlet_noise=False: any card_reward this walk hits past its own
                # commit would otherwise pick up extra sampling variance on top of the sibling
                # comparison this whole job exists to measure -- same reasoning as
                # _run_meta_option_job's own chains.
                name, kwargs, policy, pending_purpose = choose_action(
                    decision, session, pending_purpose, card_reward_dirichlet_noise=False,
                )
                if policy is not None:
                    recorder.attach_policy(policy)
                decision = session.step(name, **kwargs)
                if name == "select_map_node":
                    rooms_walked += 1
                    if rooms_walked >= job.room_horizon:
                        outcome = "horizon"
                        break

            if outcome is None:
                outcome = "won" if decision.victory else "lost"

            if outcome == "lost":
                value = -1.0
            else:
                value = _combat_value(CombatOutcome(decision.player_hp, decision.player_max_hp))
        except Exception as e:
            recorder.log_error(f"{type(e).__name__}: {e}")
            outcome = "error"

        if outcome != "error":
            recorder.set_value(commit_step, value)
        try:
            recorder.save(f"{cfg.raw_dir}/{job.map_id}.json", session.action_log, outcome=outcome)
        except Exception as e:
            recorder.log_error(f"save failed: {type(e).__name__}: {e}")
    finally:
        session.close()
        random.seed(os.urandom(16))
        torch.seed()
        results_queue.put((job.map_id, None if outcome == "error" else value))


def _resolve_map_decision(
    job_queue, meta_results_queue, cfg: FacilityConfig, recorder: CompactRecorder, session, decision,
    character: str, seed: str, main_id: str, divergence_index: int,
    last_room_checkpoint_path: str | None, last_room_entry_index: int | None,
) -> tuple[str, dict, list]:
    """_play_trial's map_resolver: picks which sibling the main line commits to, then dispatches a
    MapOptionJob for every sibling (including the adopted one) to the shared branch-worker pool
    for real per-choice scoring (see _run_map_option_job) -- golden_compass/model/materialize.py
    reads these branch files directly, grouped by (main_id, divergence_index) and baselined
    against their own sibling group's mean, not any value tagged onto the main line's own trial
    file (this resolver never calls recorder.set_value itself -- unlike _resolve_meta_decision,
    there's no "adopted option's value" to backfill into the main line, since materialize.py reads
    every sibling equally either way).

    Returns (name, kwargs, pending) -- NOT a stepped decision, unlike _resolve_meta_decision:
    select_map_node's own checkpoint-take/step already has one shared home in _play_trial's own
    generic tail (used for every select_map_node regardless of which chooser picked it), so this
    only decides what to commit to and what to branch, leaving the actual session.step() there.

    Adopts via golden_compass's own DP heuristic for now, same "reasonable, not trained" bootstrap
    every other subsystem here starts from -- no trained net is wired into this live path yet (see
    golden_compass/model/README.md); once MapOptionJob data trains one under the simplified
    one-hop action space (golden_compass/model/model.py), this is the one place that needs to
    change to sample from it instead."""
    if isinstance(decision, BufferDecision):
        options = _gc_model.legal_options_from_buffer(decision)
    else:
        options = _gc_model.legal_options(decision)
    if len(options) < 2:
        name, args = options[0]
        return name, args, []

    if isinstance(decision, BufferDecision):
        hp_frac = decision.player_hp / decision.player_max_hp if decision.player_max_hp else 1.0
        adopted_coord, _children = golden_compass.choose_next_node_from_buffer(decision.arr, hp_frac)
    else:
        hp_frac = decision.player.hp / decision.player.max_hp if decision.player.max_hp else 1.0
        adopted_coord = golden_compass.choose_next_node(session.get_map(), hp_frac)

    adopted_index = next(
        (i for i, (_, a) in enumerate(options) if (a["col"], a["row"]) == adopted_coord), 0
    )
    name, args = options[adopted_index]

    prefix = list(session.action_log)
    walk_seed = random.getrandbits(64)

    pending: list[tuple[str, str | None, bool, int | None]] = []
    for option_index, (opt_name, opt_args) in enumerate(options):
        job = MapOptionJob(
            main_id=main_id, character=character, seed=seed, ascension=cfg.ascension,
            divergence_index=divergence_index, option_index=option_index, prefix=prefix,
            last_room_checkpoint_path=last_room_checkpoint_path,
            last_room_entry_index=last_room_entry_index,
            commit_name=opt_name, commit_args=opt_args, walk_seed=walk_seed,
            room_horizon=cfg.map_room_horizon,
        )
        # is_adopted is always False here -- see this function's own docstring on why no value
        # ever gets backfilled into the main line's recorder for a map decision.
        job_queue.put((job, meta_results_queue))
        pending.append((job.map_id, last_room_checkpoint_path, False, None))

    return name, args, pending


def main_worker_loop(
    cfg: FacilityConfig, job_queue: mp.Queue, job_queue_priority: mp.Queue, log_queue: mp.Queue,
    results_queue, main_worker_index: int, stop_event, trial_counter=None,
    resume: dict | None = None, manager_address=None, manager_authkey: bytes | None = None,
    meta_results_queue=None, *,
    combat_resolver: Callable, meta_resolver: Callable | None = None,
    map_resolver: Callable | None = None, session_factory: Callable = Sts2Session,
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
    queue_manager, last_room_checkpoint_path, last_room_entry_index) -> (decision,
    pending_meta_jobs): called whenever decision.decision is META_ELIGIBLE and this trial's
    meta-branch budget (cfg.meta_branches_per_trial) allows it. pending_meta_jobs is a list of
    (meta_id, checkpoint_path, is_adopted, commit_step) tuples for jobs just dispatched onto
    job_queue. None (facility_claws.py's case) means meta-eligible decisions just fall through to
    choose_action(). last_room_checkpoint_path/entry_index (may be None pre-first-room) locate the
    most recently entered room's checkpoint (of ANY room type, not just combat -- a card_reward
    right after combat, a rest_site, an event_choice, ... may all share the same one if no fresh
    select_map_node happened in between) and its entry action's position in action_log -- a
    checkpoint taken directly at the pending decision does NOT work for reconstructing "right
    after entering this room" (confirmed by hand: save/load_save doesn't preserve a pending
    decision screen).

    manager_address/manager_authkey: (address, authkey) to reconnect to the Facility's own
    multiprocessing.Manager() via SyncManager(...).connect() -- this worker is daemonic and can't
    own a Manager itself, and a live SyncManager isn't picklable through Process(args=...) at all
    (confirmed by hand). Unused by the one shipped meta_resolver; kept for a future one that needs
    manager-backed queues at runtime. Skipped when meta_resolver is None.

    resume: if given, this worker's FIRST trial only reconstructs an in-progress run instead of
    starting fresh -- for recovering a trial that was progressing normally when the facility was
    killed before it reached game_over. Keys: character, seed, main_id, combat_index (combats
    already resolved), actions (full action log up to the resume point, same shape a branch/trial
    JSON's own "actions" field has).

    session_factory(logger=recorder) -> a fresh main-line session: Sts2Session (subprocess, the
    default) or Sts2BufferSession (in-process CLR, faster but see the Sts2EngineTimeoutError
    handling below). combat_resolver/meta_resolver must tolerate whichever one this worker was
    built with -- facility_claws._resolve_combat and this module's _map_point_type_at both branch
    on hasattr(decision, "arr") for exactly this reason.

    map_resolver(job_queue, meta_results_queue, cfg, recorder, session, decision, character, seed,
    main_id, divergence_index, last_room_checkpoint_path, last_room_entry_index) -> (name, kwargs,
    pending_map_jobs): called whenever decision.decision == "map_select" and this trial's map-
    branch budget (cfg.map_branches_per_trial) allows it, instead of choose_action(). Returns the
    action to commit to (not a stepped decision -- select_map_node's own checkpoint-take/step
    stays centralized in _play_trial's own generic tail) plus a (map_id, checkpoint_path,
    is_adopted, commit_step) tuple per MapOptionJob just dispatched, same shape meta_resolver's own
    pending_meta_jobs uses (see _resolve_map_decision) -- shares pending_meta_jobs/
    checkpoint_refcount bookkeeping with meta_resolver's jobs rather than tracking its own. None
    (facility_claws.py's/facility_choices_paradox.py's case when map branching is off) means
    map_select falls through to choose_action() same as before."""
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
        checkpoint_dir = os.path.join(cfg.raw_dir, "checkpoints")
        _sweep_stale_checkpoints(checkpoint_dir)  # opportunistic, once per trial -- see its docstring
        state = _TrialState(
            session=session_factory(logger=recorder),
            checkpoint_dir=checkpoint_dir,
            meta_budget=cfg.meta_branches_per_trial,
            map_budget=cfg.map_branches_per_trial,
        )
        outcome = None
        poisoned = False
        try:
            try:
                outcome, _ = _play_trial(
                    cfg, job_queue_priority, job_queue, results_queue, meta_results_queue,
                    main_worker_index, recorder, character, seed, main_id, this_trial_resume,
                    stop_event, queue_manager, combat_resolver, meta_resolver, map_resolver, state,
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
            # doesn't resolve within _META_DRAIN_TIMEOUT is abandoned; its checkpoint ages out via
            # _sweep_stale_checkpoints on a later trial instead of being force-deleted here.
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
            # No per-trial checkpoint cleanup here -- combat BranchJobs have no completion
            # tracking (unlike MetaOptionJobs' checkpoint_refcount), so a turn's un-adopted or
            # still-queued siblings (facility_claws._wait_for_turn_result returns as soon as the
            # PREFERRED branch reports, not once every branch has) can still be reading this
            # trial's checkpoints well after _play_trial returns. _sweep_stale_checkpoints
            # (called once per trial, above) reclaims them by age instead once they're safely
            # unreferenced -- see _CHECKPOINT_GRACE_PERIOD's comment for why deleting immediately
            # here used to race those stragglers.
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

    def stop(self, timeout: float = 120) -> None:
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
    parser.add_argument("--no-combat-dirichlet-noise", action="store_true",
                         help="sample claws combat actions straight from the net's own softmax, "
                              "with no root Dirichlet noise mixed in (default: noise enabled -- "
                              "see policy._COMBAT_DIRICHLET_EPS_PER_OPTION)")


def load_resume(path: str) -> dict:
    import json
    with open(path) as f:
        r = json.load(f)
    return {
        "character": r["character"], "seed": r["seed"], "main_id": r["main_id"],
        "combat_index": r["combat_index"], "actions": _clean_actions(r["actions"]),
    }

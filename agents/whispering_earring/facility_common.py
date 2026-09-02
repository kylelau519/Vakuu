"""Shared self-play-facility infrastructure: worker-process orchestration, stderr relay/logging,
FacilityConfig, and the pieces of combat scoring/turn-playing that more than one facility needs.

Split out of what used to be a single facility.py once a second facility (choices_paradox's
card_reward meta-branching, see agents/choices_paradox/model/README.md's Design section) needed
the same worker-pool machinery: `facility_claws.py` drives combat-turn branching (Claws's own
training data, the original whispering_earring scheme) and `facility_choices_paradox.py` adds
card_reward branching on top -- both build a `Facility` from this module and supply their own
main-line/branch-worker logic.

The split point: `Facility` itself, `main_worker_loop`, and `branch_worker_loop` are pure
orchestration here -- they take `combat_resolver`/`meta_resolver` (main line) and `handlers`
(branch dispatch, keyed by job type) as callables/mappings, and know nothing about combat
turn-branching or meta-decision branching specifically. Each concrete facility module binds its
own domain logic to these via `functools.partial` before constructing a `Facility` -- see either
concrete module's CLI entrypoint for the wiring.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import re
import random
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
# missing HP, so HP left over at the end of a boss fight is worth roughly 1/5 of what it's worth
# in a normal fight, and turtling to preserve it just gets outscaled by the boss instead of
# banking anything real -- so player HP loss is weighted down (_BOSS_HP_WEIGHT) and win/loss
# dominates (_BOSS_WIN_FLOOR). _BOSS_ENEMY_WEIGHT is the mirror-image signal: damage dealt to the
# boss this turn, which normal fights deliberately don't reward (it'd just encourage reckless
# damage races with no clock forcing urgency) but boss fights should, since scaling makes racing
# the boss's HP down before it out-scales you the actually-correct play. Starts at the same
# magnitude as _BOSS_HP_WEIGHT (taking and dealing damage weighted symmetrically) until there's
# training signal to tune it.
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


@dataclass
class CombatOutcome:
    """Bundles _combat_value's inputs -- one struct instead of a positional-argument list that
    grows every time a new reward signal comes in (e.g. a boss-fight enemy-HP-race term: how
    close the boss was to dying, not just the player, still being designed -- see the module's
    _BOSS_HP_WEIGHT/_BOSS_WIN_FLOOR comment)."""
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
    fixed for the whole fight. Deliberately ignores anything else in the room: a mid-fight summon
    never matches boss_ids so it never contributes, and a tracked entity that has since died
    simply contributes 0 (its slot is gone, not matched) rather than needing special-casing."""
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

    # Meta-decision (card_reward, eventually card_select) branching -- see
    # agents/choices_paradox/model/README.md's Design section. Shared here (not split into a
    # separate config) because both facilities' main_worker_loop is the same generic function;
    # facility_claws.py's CLI never touches these, so its default of 0 (disabled) is what keeps
    # its behavior identical to before this module existed. facility_choices_paradox.py's CLI
    # sets meta_branches_per_trial to None (branch every eligible decision, matching combat's own
    # always-branch default) unless told otherwise.
    meta_branches_per_trial: int | None = 0  # 0 = disabled; None = unlimited; caps how many
    # eligible meta-decisions per trial actually spawn branches (a decrementing per-trial budget,
    # not a sampling scheme -- see the README's FacilityConfig knobs section for why)
    meta_options_replicates: int = 3  # M: independent rollouts per legal option, averaged before
    # comparing across options (variance reduction, not just more data -- see the README)
    meta_horizon_combats: int = 1  # play forward this many *won* combats (or a loss) before
    # bootstrapping the remainder off Claws's own value head
    meta_branch_timeout_s: float = 600.0  # generous vs. combat's own 120s turn-branch timeout,
    # since a meta-branch job is many combats long, not one turn


def _bypass_card_reward(session) -> None:
    # A branch's own reward pick is discarded either way -- only the main line's adopted
    # actions matter, and it makes its own pick itself. Discovered via later jobs on the same worker
    # increasingly failing with "Sts2ActionError: No pending card reward". Only used when a
    # combat-turn branch's job ends the instant it wins (see _play_turn_only's
    # bypass_reward_on_win) -- a meta-branch horizon walk that keeps playing past the win instead
    # resolves the same card_reward for real, through choose_action/session.step, and must NOT
    # go through this bypass (see facility_choices_paradox._play_meta_forward).
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
    combat carries on into another round, "won"/"lost" if this turn ended it; decision is
    whatever the game is showing after this turn (a fresh combat_play round, the pending
    card_reward on a win, or game_over on a loss).

    is_last_of_act, boss_ids, boss_max_hp all come from the matching BranchJob fields (decided
    once at combat entry from the main line's live context.room_type/context.enemies -- the
    buffer protocol used here has no room_type of its own, see checkpoint.py's BranchJob
    docstring) and drive the boss-fight reward shaping in both the terminal (_combat_value) and
    non-terminal branches below. boss_ids/boss_max_hp are only ever non-empty when
    is_last_of_act is True.

    bypass_reward_on_win: combat-turn branch workers (facility_claws) always want the default --
    a branch's job ends the instant it wins, and the pending card_reward left dangling would
    otherwise leak into the next job the worker process picks up (see _bypass_card_reward).
    facility_choices_paradox's horizon walk passes False instead: it keeps playing past the win,
    so the pending card_reward must stay real (and get resolved through the normal
    choose_action/session.step path by the caller) rather than silently discarded."""
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
    cfg: FacilityConfig, job_queue: mp.Queue, results_queue: mp.Queue, main_worker_index: int,
    stop_event, trial_counter=None, resume: dict | None = None, *,
    combat_resolver: Callable, meta_resolver: Callable | None = None,
) -> None:
    """job_queue is shared with every other main worker, but results_queue is this worker's own
    private queue, created and owned by Facility, one per main worker. It must not be shared.

    combat_resolver(job_queue, results_queue, main_worker_index, cfg, recorder, session, decision,
    character, seed, main_id, combat_index, stop_event, checkpoint_path) -> (session, decision,
    outcome) drives one combat to resolution -- both concrete facilities pass
    facility_claws._resolve_combat.

    meta_resolver(job_queue, results_queue, main_worker_index, cfg, recorder, session, decision,
    character, seed, main_id, decision_kind, divergence_index, stop_event) -> decision, called
    whenever decision.decision is in META_ELIGIBLE_DECISIONS and this trial's meta-branch budget
    (cfg.meta_branches_per_trial, decremented below) allows it. None (facility_claws.py's case)
    means meta-eligible decisions just fall through to the ordinary choose_action() path below,
    same as every other decision type.

    resume: if given, this worker's FIRST trial (only) reconstructs an in-progress run instead
    of starting fresh -- for recovering a trial that was progressing normally (e.g. had already
    won several combats) when the facility was killed before it reached game_over and got a
    chance to write its own output. Keys: character, seed, main_id, combat_index (how many
    combats are already resolved -- the next one encountered is numbered combat_index+1, same
    convention the fresh-start path uses), actions (the full action log up to the resume point,
    same shape a branch/trial JSON's own "actions" field already has -- e.g. lift it straight
    from a "won"/"continue" branch file's "actions")."""
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
                        session, decision, combat_outcome = combat_resolver(
                            job_queue, results_queue, main_worker_index, cfg, recorder, session,
                            decision, character, seed, main_id, combat_index, stop_event,
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
                            divergence_index, stop_event,
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
    cfg: FacilityConfig, job_queue: mp.Queue, result_queues: list[mp.Queue], stop_event,
    log_queue: mp.Queue, worker_index: int = 0, *, handlers: dict[type, Callable],
) -> None:
    """result_queues: every main worker's private results_queue, indexed by main_worker_index.

    handlers: job type -> handler(cfg, job, results_queue). Dispatches purely on type(job), so a
    single shared pool can serve BranchJob (combat-turn branching, facility_claws.py) and
    MetaBranchJob (card_reward branching, facility_choices_paradox.py) side by side -- see
    agents/choices_paradox/model/README.md's "Shared worker pool, not a dedicated one" for why
    that sharing is deliberate. facility_claws.py registers only BranchJob's handler;
    facility_choices_paradox.py registers both (reusing facility_claws's BranchJob handler as-is,
    since its own main line still plays combat turn-branched exactly like facility_claws.py's
    does)."""
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
                target=self.main_worker_fn,
                args=(self.cfg, self.job_queue, results_queue, i, self.stop_event, self.trial_counter,
                      self.resume if i == 0 else None),
                daemon=True,
            )
            p.start()
            self.main_procs.append(p)

        for i in range(self.cfg.n_branch_workers):
            p = mp.Process(
                target=self.branch_worker_fn,
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
        # SIGINT (Ctrl-C) already unwinds via KeyboardInterrupt below. Without this handler
        # a plain `kill <pid>` (SIGTERM -- what most process managers/IDE stop buttons send)
        # falls through to the default action and skips stop() entirely, leaving every daemon
        # branch/main worker orphaned (reparented to pid 1) and idling on the job queue forever
        # -- that's how these pile up across crashed/killed runs. SIGKILL still can't be caught,
        # so this doesn't cover a hard `kill -9`.
        def _on_sigterm(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, _on_sigterm)
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

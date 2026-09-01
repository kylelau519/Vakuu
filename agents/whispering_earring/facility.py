from __future__ import annotations

import multiprocessing as mp
import os
import random
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from queue import Empty

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)
import buffer_layout as bl
from env import Sts2ActionError, Sts2Session
from buffer_env import Sts2BufferSession, decode_token
from installation import ensure_setup

from checkpoint import BranchJob
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
    entities present when this boss combat was entered (see _resolve_combat), fixed for the
    whole fight. Deliberately ignores anything else in the room: a mid-fight summon never
    matches boss_ids so it never contributes, and a tracked entity that has since died simply
    contributes 0 (its slot is gone, not matched) rather than needing special-casing."""
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
    session: Sts2BufferSession, decision, combat_hp_before: int, combat_max_hp: int,
    is_last_of_act: bool = False, boss_ids: frozenset[str] = frozenset(), boss_max_hp: int = 0,
) -> str:
    """Plays exactly one turn (through the mid-turn card_select edge case, e.g. Power Potion/
    Brand, the same way the old whole-combat loop did) and scores it before returning --
    "continue" if the combat carries on into another round, "won"/"lost" if this turn ended
    it. See facility.py module context / the turn-level branch-and-adopt plan for why turns,
    not whole combats, are the branch unit now.

    is_last_of_act, boss_ids, boss_max_hp all come from the matching BranchJob fields (decided
    once at combat entry from the main line's live context.room_type/context.enemies -- the
    buffer protocol used here has no room_type of its own, see checkpoint.py's BranchJob
    docstring) and drive the boss-fight reward shaping in both the terminal (_combat_value) and
    non-terminal branches below. boss_ids/boss_max_hp are only ever non-empty when
    is_last_of_act is True."""
    hp_before_turn = decision.player_hp
    enemy_hp_before_turn = _tracked_enemy_hp(decision.arr, boss_ids) if is_last_of_act else 0
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
        # hp_term and v_net are independently bounded but their sum isn't -- floor it so a live,
        # still-recoverable turn can never score worse than an outright loss (-1.0, set above).
        turn_value = max(turn_value, -1.0)

    session.logger.end_turn(turn_value)
    return outcome


def _spawn_turn_branches(
    job_queue, main_worker_index: int, cfg: FacilityConfig, main_id: str, act: int, floor: int,
    combat_index: int, turn_index: int, character: str, seed: str, actions_so_far: list[dict],
    combat_hp_before: int, combat_max_hp: int, checkpoint_path: str | None, combat_start_index: int,
    is_last_of_act: bool = False, boss_ids: frozenset[str] = frozenset(), boss_max_hp: int = 0,
) -> list[str]:
    branch_ids = []
    for b in range(cfg.branches_per_turn):
        job = BranchJob(
            main_id=main_id, act=act, floor=floor, combat_index=combat_index, turn_index=turn_index,
            branch_index=b, character=character, seed=seed, ascension=cfg.ascension,
            combat_hp_before=combat_hp_before, combat_max_hp=combat_max_hp, is_last_of_act=is_last_of_act,
            boss_ids=boss_ids, boss_max_hp=boss_max_hp,
            actions=list(actions_so_far), checkpoint_path=checkpoint_path,
            combat_start_index=combat_start_index,
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
        if (result is None or result.get("outcome") == "error"
                or len(result.get("actions") or []) < prefix_len):
            # branch errored before saving (or reported its own error outcome), or its prefix
            # replay itself failed -- _turn_score_key only ranks "lost" as worse than everything
            # else, so an unfiltered "error" branch would otherwise be adopted like a legitimate
            # "won"/"continue" result, silently making this turn (and the main-line trial) fail
            # without ever calling recorder.log_error on the main recorder (that only happens
            # below when NO branch is usable at all) -- see the "why does outcome=='error' have
            # no error message" investigation this fixes.
            continue
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
    session: Sts2Session, character: str, seed: str, main_id: str, act: int, floor: int, combat_index: int,
    turn_index: int, combat_hp_before: int, combat_max_hp: int, stop_event,
    checkpoint_path: str | None, combat_start_index: int, is_last_of_act: bool = False,
    boss_ids: frozenset[str] = frozenset(), boss_max_hp: int = 0,
):
    """Delegates one turn to cfg.branches_per_turn parallel attempts and adopts the
    best-scoring result """
    prefix = list(session.action_log)
    branch_ids = _spawn_turn_branches(
        job_queue, main_worker_index, cfg, main_id, act, floor, combat_index, turn_index,
        character, seed, prefix, combat_hp_before, combat_max_hp,
        checkpoint_path, combat_start_index, is_last_of_act, boss_ids, boss_max_hp,
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
    cleaned = _clean_actions(adopted["actions"])
    if checkpoint_path is not None:
        # Same fast path branch_worker_loop uses (see checkpoint.py's docstring) -- resync the
        # main line to the winning branch's state via load_save + this combat's own actions,
        # instead of open_from_checkpoint's full replay from StartRun. That full replay used to
        # be paid here on EVERY turn regardless of the branch-side fix, growing with total run
        # depth (not just combat depth) since this is the main line's own history, not a
        # branch's -- measured turn-to-turn latency climbing from ~1.7s early in a run to ~7s+
        # by ~200 actions in. load_and_resume_combat has no memory of anything before the
        # checkpoint, so the pre-combat prefix is spliced back onto action_log afterward to keep
        # the main line's own history complete for the trial's final output JSON.
        entry_action = cleaned[combat_start_index]
        in_combat_actions = cleaned[combat_start_index + 1:]
        new_decision = new_session.load_and_resume_combat(
            checkpoint_path, entry_action, in_combat_actions
        )
        new_session.action_log = cleaned[:combat_start_index] + new_session.action_log
    else:
        new_decision = new_session.open_from_checkpoint(character, seed, cfg.ascension, cleaned)
    return new_session, new_decision, adopted.get("outcome")  # "continue" | "won" | "lost"


def _resolve_combat(
    job_queue, results_queue, main_worker_index: int, cfg: FacilityConfig, recorder: CompactRecorder,
    session: Sts2Session, decision, character: str, seed: str, main_id: str, combat_index: int, stop_event,
    checkpoint_path: str | None,
):
    """Drives _resolve_turn turn-by-turn until the combat actually ends.

    checkpoint_path is the native save the main line took right before the select_map_node
    action that entered this combat (see main_worker_loop) -- None if no checkpoint was
    available (e.g. this run's very first combat). combat_start_index is that entry action's
    position in session.action_log, captured once here: session.action_log already has the
    entry action as its last element the moment this is called, and every turn-branch spawned
    for this combat replays from the same fixed point regardless of which session object later
    comes to represent the main line's turn-by-turn state (see _resolve_turn's new_session).
    is_last_of_act is decided once here, from context.room_type -- a live per-room field read fresh by
    the engine every decision (unlike the buffer protocol's boss1_token/boss2_token, which are
    per-act constants and stay set through every combat of the act, boss or not) -- and carried
    for the whole combat via BranchJob.is_last_of_act (see checkpoint.py's docstring). boss_ids/
    boss_max_hp are captured the same way, from context.enemies -- the entities actually in the
    room at boss-combat entry, ignoring anything summoned later (see facility.py's
    _tracked_enemy_hp and checkpoint.py's docstring)."""
    act = decision.context.act
    floor = decision.context.floor
    is_last_of_act = decision.context.room_type == "Boss"
    boss_ids = frozenset(e.id for e in decision.enemies if e.id) if is_last_of_act else frozenset()
    boss_max_hp = sum(e.max_hp for e in decision.enemies if e.id in boss_ids)
    combat_hp_before, combat_max_hp = decision.player.hp, decision.player.max_hp
    combat_start_index = len(session.action_log) - 1
    turn_index = 0
    while True:
        turn_index += 1
        session, decision, turn_outcome = _resolve_turn(
            job_queue, results_queue, main_worker_index, cfg, recorder, session,
            character, seed, main_id, act, floor, combat_index, turn_index,
            combat_hp_before, combat_max_hp, stop_event,
            checkpoint_path, combat_start_index, is_last_of_act, boss_ids, boss_max_hp,
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
            if checkpoint_path is not None:
                # `session` here was reconstructed by _resolve_turn's fast load_and_resume_combat
                # path. That path is faithful for everything checked in-combat (hand/piles/hp/
                # potions/enemies), including the terminal card_reward's own offer -- confirmed by
                # direct comparison against continuous play. But it does NOT preserve whatever RNG
                # cursor governs FUTURE map/encounter generation: a session rebuilt this way can
                # land on a different room type than continuous play at the very next
                # select_map_node, even though nothing observable up to that point differs
                # (confirmed: a session that merely calls session.save() and keeps playing
                # continuously matches continuous play exactly, so the desync is specific to
                # session.load_save() into a fresh process, not to taking the checkpoint itself).
                # Harmless while still inside the combat (nothing here reads map RNG), but this
                # session is about to become the ongoing main-line session past the combat's end,
                # driving every subsequent map/event/shop room -- so rebuild it once, safely, via
                # a full StartRun replay (mechanically identical to continuous play) before handing
                # it back. Only combat-internal turns pay the fast path's cost; the combat-ending
                # turn pays open_from_checkpoint's cost once, same as it always did before this
                # combat's fast-path resync was introduced.
                full_action_log = list(session.action_log)
                session.close()
                safe_session = Sts2Session(logger=recorder)
                decision = safe_session.open_from_checkpoint(
                    character, seed, cfg.ascension, full_action_log
                )
                session = safe_session
            return session, decision, None
        return session, decision, turn_outcome  # "lost" or "error"


def main_worker_loop(
    cfg: FacilityConfig, job_queue: mp.Queue, results_queue: mp.Queue, main_worker_index: int,
    stop_event, trial_counter=None, resume: dict | None = None,
) -> None:
    """job_queue is shared with every other main worker, but results_queue is this worker's own private queue, created and owned by
    Facility, one per main worker. It must not be shared.

    resume: if given, this worker's FIRST trial (only) reconstructs an in-progress run instead
    of starting fresh -- for recovering a trial that was progressing normally (e.g. had already
    won several combats) when the facility was killed before it reached game_over and got a
    chance to write its own output. Keys: character, seed, main_id, combat_index (how many
    combats are already resolved -- the next one encountered is numbered combat_index+1, same
    convention the fresh-start path uses), actions (the full action log up to the resume point,
    same shape a branch/trial JSON's own "actions" field already has -- e.g. lift it straight
    from a "won"/"continue" branch file's "actions"). See resume_trial.py for the CLI that
    builds this dict from such a file and drives a whole one-worker Facility with it."""
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
                        session, decision, combat_outcome = _resolve_combat(
                            job_queue, results_queue, main_worker_index, cfg, recorder, session,
                            decision, character, seed, main_id, combat_index, stop_event,
                            combat_checkpoint_path,
                        )
                        combat_checkpoint_path = None  # consumed -- not valid for the next combat
                        if combat_outcome is not None:
                            outcome = combat_outcome
                            break
                        continue

                    name, kwargs, policy = choose_action(decision, session)
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
                        # across the trial like an earlier version of this did: _wait_for_turn_
                        # result gives up on straggler branch workers after a timeout (the shared
                        # branch-worker pool can be oversubscribed -- training.sh runs 3 main
                        # workers x up to 30 branches/turn against a 15-worker pool), so a
                        # straggler can still be reading this file long after the main line has
                        # moved on to a LATER select_map_node. A reused filename meant that
                        # straggler would silently load a wrong, unrelated state and replay this
                        # combat's entry action into it -- landing on whatever room that
                        # coordinate happened to resolve to there instead of erroring, which is
                        # exactly what produced the "unexpected decision" AssertionErrors in
                        # data/raw (e.g. f15c07t01_b29 landing on event_choice). A unique filename
                        # means a late straggler still loads the CORRECT state for its own combat
                        # (its result is simply ignored as stale by _wait_for_turn_result if it's
                        # that late -- same as today -- instead of corrupting the branch).
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
            # By the time a trial concludes, every combat it contained has already resolved (or
            # given up on) all its own turn-branches -- any straggler still unprocessed at this
            # point is far enough past _wait_for_turn_result's timeout that its result would be
            # discarded as stale anyway, so it's safe to reclaim the disk here rather than let
            # these accumulate for the life of the facility.
            for path in checkpoint_paths_this_trial:
                try:
                    os.remove(path)
                except OSError:
                    pass
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
            "main_id": job.main_id, "act": job.act, "floor": job.floor, "combat_index": job.combat_index,
            "turn_index": job.turn_index, "branch_index": job.branch_index, "role": "branch",
            "prefix_len": len(job.actions),
        }, start_step=len(job.actions))
        session = Sts2BufferSession(logger=recorder)
        outcome = None
        result = None
        try:
            try:
                if job.checkpoint_path is not None:
                    # Fast path: resume from the native save taken right before this combat's
                    # room-entry action, replaying only this combat's own actions, instead of
                    # replaying the entire run from StartRun (see checkpoint.py's docstring).
                    # load_and_resume_combat has no memory of anything before the checkpoint --
                    # action_log comes back as just [entry_action, *in_combat_actions], so the
                    # pre-combat prefix is prepended here to keep session.action_log identical
                    # in content to what open_from_checkpoint would have produced (this recorder
                    # was built with start_step=len(job.actions), and _resolve_turn's caller-side
                    # indexing all assumes that same full-history shape on the way out).
                    entry_action = job.actions[job.combat_start_index]
                    in_combat_actions = job.actions[job.combat_start_index + 1:]
                    decision = session.load_and_resume_combat(
                        job.checkpoint_path, entry_action, in_combat_actions
                    )
                    session.action_log = job.actions[:job.combat_start_index] + session.action_log
                else:
                    decision = session.open_from_checkpoint(
                        job.character, job.seed, job.ascension, job.actions
                    )
                outcome = _play_turn_only(
                    session, decision, job.combat_hp_before, job.combat_max_hp, job.is_last_of_act,
                    job.boss_ids, job.boss_max_hp,
                )
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
    def __init__(self, cfg: FacilityConfig, resume: dict | None = None):
        self.cfg = cfg
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
                target=main_worker_loop,
                args=(self.cfg, self.job_queue, results_queue, i, self.stop_event, self.trial_counter,
                      self.resume if i == 0 else None),
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
    args = parser.parse_args()

    resume = None
    if args.resume_branch_file:
        import json
        with open(args.resume_branch_file) as f:
            _r = json.load(f)
        resume = {
            "character": _r["character"], "seed": _r["seed"], "main_id": _r["main_id"],
            "combat_index": _r["combat_index"], "actions": _clean_actions(_r["actions"]),
        }

    facility = Facility(FacilityConfig(
        characters=args.characters,
        n_branch_workers=args.branch_workers,
        branches_per_turn=args.branches_per_turn,
        n_main_workers=args.main_workers,
        n_main_trials=args.trials,
        raw_dir=args.raw_dir,
        verbose=args.verbose,
    ), resume=resume)
    facility.run()

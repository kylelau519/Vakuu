"""Combat turn-branching: the original whispering_earring self-play facility, generating training
data for `claws` (see agents/claws/model/). Split out of a single facility.py once
facility_choices_paradox.py needed the same worker-pool machinery for a second, independent kind
of branching (card_reward meta-decisions) -- see facility_common.py's module docstring for the
split, and this package's README.md for the branching scheme this file implements (N concurrent
main-line workers, each forking off a checkpoint at every *turn* to explore alternate lines via a
shared pool of branch workers, discarded after that one turn resolves -- a main line adopts
whichever of its branches scores best, per `_turn_score_key`, once all have reported or timed
out).

Branch workers drive their own turn via sts2-cli's buffer protocol (`buffer_env.Sts2BufferSession`,
in-process, no subprocess/JSON/pydantic decode) instead of the JSON `Sts2Session` -- see sts2-cli's
`src/Sts2Headless/README.md` for the wire format and `policy.py`'s combat/card-select choosers for
how they branch on the decision type (`isinstance(decision, BufferDecision)`) to read fields via
`buffer_layout.py` instead of a pydantic attribute tree. The main self-play line still uses
`Sts2Session`/pydantic `Decision` throughout, since it never plays `combat_play` directly (see
`_resolve_turn`) -- only branch workers do.
"""
from __future__ import annotations

import functools
import multiprocessing as mp
import time
from queue import Empty

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)
from env import Sts2Session
from buffer_env import Sts2BufferSession

from checkpoint import BranchJob
from recorder import CompactRecorder
from facility_common import (
    Facility, FacilityConfig, _clean_actions, _play_turn_only,
    add_common_cli_args, branch_worker_loop, load_resume, main_worker_loop,
)


def _spawn_turn_branches(
    job_queue, results_queue, cfg: FacilityConfig, main_id: str, act: int, floor: int,
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
        # The dispatched-to queue travels with the job itself (not looked up from a
        # main-worker-index table) -- see branch_worker_loop's docstring. Here it's always the
        # caller's own results_queue, but that queue is no longer necessarily "the issuing main
        # worker's one shared queue" -- facility_choices_paradox.py's option-chains each pass
        # their own private one through _resolve_turn/_resolve_combat.
        job_queue.put((job, results_queue))
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
    moment its trial is saved. Most branches here report "continue" (combat still ongoing)
    rather than a terminal outcome, so they can't just be accepted on first success -- every
    branch's turn_value (see _turn_score_key) is compared once all have reported (or timed
    out). A branch that itself won the combat this turn is still an immediate, unambiguous
    best pick."""
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
            # below when NO branch is usable at all).
            continue
        usable[branch_id] = result
        if result.get("outcome") == "won":
            return result
    if pending:
        print(f"warning: {len(pending)}/{len(branch_ids)} turn branch(es) didn't finish within "
              f"{timeout}s: {sorted(pending)}")
    return max(usable.values(), key=_turn_score_key) if usable else None


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
    best-scoring result."""
    prefix = list(session.action_log)
    branch_ids = _spawn_turn_branches(
        job_queue, results_queue, cfg, main_id, act, floor, combat_index, turn_index,
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
    cleaned = _clean_actions(adopted["actions"])
    if checkpoint_path is not None:
        # Resync the main line to the winning branch's state IN PLACE, on this SAME live
        # session, via load_save + this combat's own actions -- instead of closing this
        # session's subprocess and spawning a brand-new one first, which is what this used to do
        # (close() + Sts2Session(logger=recorder), one dotnet subprocess fork/exec + CoreCLR
        # cold start PER TURN). Confirmed safe by debug_verify_session_reuse.py (4 seeds, up to 4
        # rooms past combat, zero divergence): the map-RNG-desync class of bug this respawn used
        # to look like it might be guarding against (see _resolve_combat's post-win
        # "safe_session" comment) is a property of what load_save()/SaveCheckpoint itself does
        # and doesn't restore, not of whether the process calling it is fresh -- reusing one live
        # session across repeated load_save() calls reproduces continuous play identically to
        # respawning a fresh one every time. (That same script's "no repair" variant also matched
        # continuous play on every seed tried, suggesting _resolve_combat's own post-win full
        # replay below may no longer be necessary at all -- not acted on here, since that's a
        # separate, higher-stakes claim than this one and deserves its own dedicated
        # verification before removing an existing safety net.) load_and_resume_combat has no
        # memory of anything before the checkpoint, so the pre-combat prefix is spliced back onto
        # action_log afterward to keep the main line's own history complete for the trial's final
        # output JSON.
        entry_action = cleaned[combat_start_index]
        in_combat_actions = cleaned[combat_start_index + 1:]
        new_decision = session.load_and_resume_combat(
            checkpoint_path, entry_action, in_combat_actions
        )
        session.action_log = cleaned[:combat_start_index] + session.action_log
    else:
        # Rare (only the very first combat of a run lacks a checkpoint) and NOT covered by
        # debug_verify_session_reuse.py's testing (that only exercised load_save reuse, not
        # repeated start_run/StartRun on one live session) -- kept as a fresh respawn rather than
        # extending the reuse optimization somewhere it hasn't actually been verified.
        session.close()
        session = Sts2Session(logger=recorder)
        new_decision = session.open_from_checkpoint(character, seed, cfg.ascension, cleaned)
    return session, new_decision, adopted.get("outcome")  # "continue" | "won" | "lost"


def _resolve_combat(
    job_queue, results_queue, main_worker_index: int, cfg: FacilityConfig, recorder: CompactRecorder,
    session: Sts2Session, decision, character: str, seed: str, main_id: str, combat_index: int, stop_event,
    checkpoint_path: str | None,
):
    """Drives _resolve_turn turn-by-turn until the combat actually ends.

    checkpoint_path is the native save the main line took right before the select_map_node
    action that entered this combat (see facility_common.main_worker_loop) -- None if no
    checkpoint was available (e.g. this run's very first combat). combat_start_index is that
    entry action's position in session.action_log, captured once here: session.action_log
    already has the entry action as its last element the moment this is called, and every
    turn-branch spawned for this combat replays from the same fixed point regardless of which
    session object later comes to represent the main line's turn-by-turn state (see
    _resolve_turn's new_session). is_last_of_act is decided once here, from context.room_type --
    a live per-room field read fresh by the engine every decision (unlike the buffer protocol's
    boss1_token/boss2_token, which are per-act constants and stay set through every combat of the
    act, boss or not) -- and carried for the whole combat via BranchJob.is_last_of_act (see
    checkpoint.py's docstring). boss_ids/boss_max_hp are captured the same way, from
    context.enemies -- the entities actually in the room at boss-combat entry, ignoring anything
    summoned later (see facility_common._tracked_enemy_hp and checkpoint.py's docstring)."""
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
            # A won combat leaves `decision` on a pending card_reward -- see
            # facility_common._bypass_card_reward for why that's already been discarded on every
            # branch that reported "won" (each branch's own reward pick is thrown away; only the
            # main line's adopted actions matter, and it makes its own pick below).
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


def _run_branch_job(cfg: FacilityConfig, job: BranchJob, results_queue) -> None:
    """branch_worker_loop's BranchJob handler -- one combat turn attempt, played unbranched via
    facility_common._play_turn_only and scored, then saved to its own trial file."""
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
            outcome, _decision = _play_turn_only(
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Whispering Earring self-play facility (combat turn-branching)")
    add_common_cli_args(parser)
    parser.add_argument("--branches-per-turn", type=int, default=3)
    args = parser.parse_args()

    resume = load_resume(args.resume_branch_file) if args.resume_branch_file else None

    facility = Facility(
        FacilityConfig(
            characters=args.characters,
            n_branch_workers=args.branch_workers,
            branches_per_turn=args.branches_per_turn,
            n_main_workers=args.main_workers,
            n_main_trials=args.trials,
            raw_dir=args.raw_dir,
            verbose=args.verbose,
        ),
        main_worker_fn=functools.partial(main_worker_loop, combat_resolver=_resolve_combat),
        branch_worker_fn=functools.partial(branch_worker_loop, handlers={BranchJob: _run_branch_job}),
        resume=resume,
    )
    facility.run()

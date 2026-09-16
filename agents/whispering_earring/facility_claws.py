"""Combat turn-branching: the original whispering_earring self-play facility, generating training
data for `claws` """
from __future__ import annotations

import functools
import multiprocessing as mp
import time
from queue import Empty

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)
import buffer_layout as bl
from env import Sts2Session
from buffer_env import Sts2BufferSession, decode_token
from checkpoint import BranchJob
from recorder import CompactRecorder
from utils import random_seed
from facility_common import (
    Facility, FacilityConfig, NewFacility, _clean_actions, _play_turn_only,
    add_common_cli_args, branch_worker_loop, load_resume, main_worker_loop,
)
from jobs import *

def _spawn_turn_branches(
    job_queue, results_queue, cfg: FacilityConfig, main_id: str, act: int, floor: int,
    combat_index: int, turn_index: int, character: str, seed: str, actions_so_far: list[dict],
    combat_hp_before: int, combat_max_hp: int, checkpoint_path: str | None, combat_start_index: int,
    is_last_of_act: bool = False, enemy_ids: frozenset[str] = frozenset(), enemy_max_hp: int = 0,
) -> list[str]:
    branch_ids = []
    for b in range(cfg.branches_per_turn):
        job = BranchJob(
            main_id=main_id, act=act, floor=floor, combat_index=combat_index, turn_index=turn_index,
            branch_index=b, character=character, seed=seed, ascension=cfg.ascension,
            combat_hp_before=combat_hp_before, combat_max_hp=combat_max_hp, is_last_of_act=is_last_of_act,
            enemy_ids=enemy_ids, enemy_max_hp=enemy_max_hp,
            actions=list(actions_so_far), checkpoint_path=checkpoint_path,
            combat_start_index=combat_start_index,
        )
        job_queue.put((job, results_queue))
        branch_ids.append(job.branch_id)
    return branch_ids


def _wait_for_turn_result(
    results_queue, branch_ids: list[str], prefix_len: int, stop_event, timeout: float = 120
) -> dict | None:
    """Blocks on results_queue (branch_id, result) pairs, adopting whichever branch's result is
    returned -- in strict branch_ids order, NOT by comparing outcomes or values. branch_ids[0] is
    the preferred continuation; a later index is only used when an earlier one itself failed to
    produce a usable result (crashed/errored/timed out), never because its outcome was worse.

    This is deliberate: adopting by outcome ("any survivor beats a loss") or by value ("take the
    highest-scoring branch") biases the main line's future toward states that either dodged death
    via N parallel attempts (survivorship bias -- real single-pass deployment gets no such
    do-over) or that the network's OWN current value estimate already rates highly (a "continue"
    branch's value is ClawsNet's own unverified guess, not ground truth -- greedily maximizing a
    noisy estimate across many candidates is upward-biased by construction, the "winner's curse",
    and self-reinforces whatever the net already believes). Picking a fixed index sidesteps both:
    the main line's continuation no longer depends on any branch's outcome or value at all.

    Every branch is still recorded to its own file by _run_branch_job regardless of what's
    returned here, so branches that lose out on adoption (including outright losses) still
    contribute their own (state, action, value) to materialize.py's REINFORCE baseline -- only
    which branch's *future* gets explored is affected by this function."""
    results: dict[str, dict] = {}
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
                or len(result.get("actions") or []) <= prefix_len):
            continue  # unusable -- the next index in branch_ids takes over as preferred
        results[branch_id] = result
        for candidate_id in branch_ids:
            if candidate_id in results:
                return results[candidate_id]
            if candidate_id in pending:
                break  # an earlier index hasn't reported yet -- don't skip ahead of it
    if pending:
        print(f"warning: {len(pending)}/{len(branch_ids)} turn branch(es) didn't finish within "
              f"{timeout}s: {sorted(pending)}")
    for candidate_id in branch_ids:
        if candidate_id in results:
            return results[candidate_id]
    return None


# NOT USED 
def _turn_score_key(result: dict) -> tuple[int, float]:
    outcome = result.get("outcome")
    actions = result.get("actions") or []
    value = actions[-1].get("value", float("-inf")) if actions else float("-inf")
    return (0 if outcome == "lost" else 1, value)


def _wait_for_turn_result_by_value(
    results_queue, branch_ids: list[str], prefix_len: int, stop_event, timeout: float = 120
) -> dict | None:
    """NOT USED  """
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
                or len(result.get("actions") or []) <= prefix_len):
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
    session: Sts2Session | Sts2BufferSession, character: str, seed: str, main_id: str, act: int, floor: int, combat_index: int,
    turn_index: int, combat_hp_before: int, combat_max_hp: int, stop_event,
    checkpoint_path: str | None, combat_start_index: int, is_last_of_act: bool = False,
    enemy_ids: frozenset[str] = frozenset(), enemy_max_hp: int = 0,
):
    """Delegates one turn to cfg.branches_per_turn parallel attempts and adopts a fixed-index
    branch's result (see _wait_for_turn_result for why not the best-scoring one)."""
    prefix = list(session.action_log)
    branch_ids = _spawn_turn_branches(
        job_queue, results_queue, cfg, main_id, act, floor, combat_index, turn_index,
        character, seed, prefix, combat_hp_before, combat_max_hp,
        checkpoint_path, combat_start_index, is_last_of_act, enemy_ids, enemy_max_hp,
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
        # Resync the main line to the winning branch's state IN PLACE, on this SAME live session,
        # via load_save + this combat's own actions
        entry_action = cleaned[combat_start_index]
        in_combat_actions = cleaned[combat_start_index + 1:]
        new_decision = session.load_and_resume_combat(
            checkpoint_path, entry_action, in_combat_actions
        )
        session.action_log = cleaned[:combat_start_index] + session.action_log
    else:
        session.close()
        session = type(session)(logger=recorder)
        new_decision = session.open_from_checkpoint(character, seed, cfg.ascension, cleaned)
    return session, new_decision, adopted.get("outcome")  # "continue" | "won" | "lost"


def _combat_context(decision):
    """(act, floor, is_last_of_act, enemy_ids, enemy_max_hp, combat_hp_before, combat_max_hp) at
    combat entry, from either decision type -- _resolve_combat is shared by facility_claws.py's
    own main line (Sts2BufferSession) and facility_choices_paradox.py's (still Sts2Session), which
    can run on different session types, so this can't assume one shape.

    is_last_of_act is derived by checking whether any of this combat's enemies IS the act's boss
    (run_context's boss1/boss2 tokens), rather than comparing a "room_type" string -- the buffer
    protocol's RunContext section has no room_type field to compare against (see
    buffer_layout.py's RUN_CONTEXT_* comment), and this is the more direct mechanistic check
    either way: a "Boss" room_type is itself just a label for this fight's enemies including the
    act's designated boss."""
    if hasattr(decision, "arr"):  # BufferDecision
        arr = decision.arr
        ctx = bl.run_context(arr)
        boss_ids = {decode_token(ctx["boss1_token"]), decode_token(ctx["boss2_token"])} - {None}
        enemy_slots = list(bl.enemies(arr))
        enemy_ids = frozenset(
            tok for _, slot in enemy_slots for tok in [decode_token(bl.enemy_token(slot))] if tok
        )
        enemy_max_hp = sum(bl.enemy_max_hp(slot) for _, slot in enemy_slots)
        return (
            ctx["act_index"], ctx["floor"], bool(enemy_ids & boss_ids),
            enemy_ids, enemy_max_hp, decision.player_hp, decision.player_max_hp,
        )
    enemy_ids = frozenset(e.id for e in decision.enemies if e.id)
    enemy_max_hp = sum(e.max_hp for e in decision.enemies if e.id in enemy_ids)
    return (
        decision.context.act, decision.context.floor, decision.context.room_type == "Boss",
        enemy_ids, enemy_max_hp, decision.player.hp, decision.player.max_hp,
    )


def _resolve_combat(
    job_queue, results_queue, main_worker_index: int, cfg: FacilityConfig, recorder: CompactRecorder,
    session: Sts2Session | Sts2BufferSession, decision, character: str, seed: str, main_id: str,
    combat_index: int, stop_event, checkpoint_path: str | None,
):
    """Drives _resolve_turn turn-by-turn until the combat actually ends.  """
    (act, floor, is_last_of_act, enemy_ids, enemy_max_hp,
     combat_hp_before, combat_max_hp) = _combat_context(decision)
    combat_start_index = len(session.action_log) - 1
    turn_index = 0
    while True:
        turn_index += 1
        session, decision, turn_outcome = _resolve_turn(
            job_queue, results_queue, main_worker_index, cfg, recorder, session,
            character, seed, main_id, act, floor, combat_index, turn_index,
            combat_hp_before, combat_max_hp, stop_event,
            checkpoint_path, combat_start_index, is_last_of_act, enemy_ids, enemy_max_hp,
        )
        if turn_outcome == "continue":
            continue
        if turn_outcome in ("won", "lost"):
            recorder.record_combat(act, floor, combat_index, turn_outcome)
        if turn_outcome == "won":
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
            # fast load from checkpoint
            if job.checkpoint_path is not None:
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
                job.enemy_ids, job.enemy_max_hp,
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





class FacilityClaws(NewFacility):
    """The original Whispering Earring self-play facility, generating training data for `claws` via
    turn-branching.  """
    def __init__(self, cfg: FacilityConfig):
        super().__init__(cfg)
        
        
    def run(self) -> None:
        self.start()
        for _ in range(self.cfg.n_main_trials):
            self.create_main_job()

    def create_main_job(self, character: str = "Ironclad", ascension: int = 1):
        seed = random_seed()
        trial_job = TrialJob(
            BasicRunInfo(
                main_id=f"{character}_{seed}", seed=seed, character=character, ascension=ascension
            ),
            self.result_queue,
        )
        self.priority_job_queue.put(trial_job)
        

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
            ascension=args.ascension,
            n_branch_workers=args.branch_workers,
            branches_per_turn=args.branches_per_turn,
            n_main_workers=args.main_workers,
            n_main_trials=args.trials,
            raw_dir=args.raw_dir,
            verbose=args.verbose,
        ),
        main_worker_fn=functools.partial(
            main_worker_loop, combat_resolver=_resolve_combat, session_factory=Sts2BufferSession
        ),
        branch_worker_fn=functools.partial(branch_worker_loop, handlers={BranchJob: _run_branch_job}),
        resume=resume,
    )
    facility.run()

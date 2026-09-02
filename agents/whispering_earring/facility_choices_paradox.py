"""Card-reward meta-decision branching, generating training data for `choices_paradox` (see
agents/choices_paradox/model/README.md's Design section for the full scheme this file
implements). Phase 1 scope only (see that README's Phasing section): branches every eligible
card_reward decision along the main line, one legal option at a time, each option averaged across
`meta_options_replicates` independent replicate rollouts; card_select's remove/upgrade purposes
still resolve via choices_paradox/policy.py's existing heuristic, unbranched, until Phase 2.

Shares its worker pool and combat turn-branching with facility_claws.py rather than duplicating
either -- the main line here still plays combat exactly like facility_claws.py's main line does
(reuses facility_claws._resolve_combat unchanged), and the shared branch_worker_loop dispatches
BranchJob (combat) and MetaBranchJob (card_reward) jobs to their own handlers side by side. See
agents/choices_paradox/model/README.md's "Shared worker pool, not a dedicated one" for why that
sharing is deliberate rather than a missing optimization.

Unlike combat's turn-branch adoption, a meta-branch's winning option is adopted by simply
stepping it on the live, never-torn-down main-line session (`session.step(name, **args)`) --  no
`load_and_resume_combat`/`open_from_checkpoint` reconstruction. This is safe *because* no
replicate's own downstream play is ever grafted onto the main line -- every replicate's only job
is producing a value estimate for the option it committed to -- and it sidesteps
`_resolve_combat`'s documented `load_save`-into-fresh-process RNG desync class entirely, by
construction. Re-derive that argument explicitly before any future edit introduces
`load_save`-based reconstruction into this module's own adoption step.
"""
from __future__ import annotations

import functools
import importlib.util
import sys
import time
from pathlib import Path
from queue import Empty

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)
from buffer_env import Sts2BufferSession

from checkpoint import BranchJob, MetaBranchJob
from recorder import CompactRecorder
from policy import choose_action, get_claws_net
from facility_common import (
    Facility, FacilityConfig, _play_turn_only, add_common_cli_args, branch_worker_loop,
    load_resume, main_worker_loop,
)
from facility_claws import _resolve_combat, _run_branch_job

# Loaded by file path (not sys.path + `import model`) for the same reason policy.py loads
# choices_paradox/policy.py and claws/model's ClawsNet this way: this package's model.py lives
# under a sibling package with no __init__.py-based package structure to import through
# normally, and doing it by path avoids any risk of colliding with another module also named
# `model` elsewhere on sys.path (e.g. claws/model/model.py, if that ever gets imported in the
# same process -- policy.py's get_claws_net can load ClawsNet in this very process).
_model_path = Path(__file__).resolve().parents[1] / "choices_paradox" / "model" / "model.py"
_spec = importlib.util.spec_from_file_location("choices_paradox_model", _model_path)
assert _spec is not None and _spec.loader is not None
_cp_model = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _cp_model  # dataclasses' own class-processing looks itself up via
# sys.modules[cls.__module__] during exec_module -- module_from_spec alone doesn't register it
_spec.loader.exec_module(_cp_model)
legal_options = _cp_model.legal_options
legal_options_from_buffer = _cp_model.legal_options_from_buffer


def _wait_for_meta_results(
    results_queue, branch_ids: list[str], prefix_len: int, n_options: int, stop_event,
    timeout: float,
) -> list[float | None] | None:
    """Averages each option's realized value (the commit action's own recorder-attached value,
    at actions[prefix_len] -- see _play_meta_forward) across its usable replicates. A replicate
    that errored (engine exception) or never finished within the timeout is dropped; a replicate
    that simply lost within its horizon is usable and contributes its floored -1.0 like any other
    real outcome -- see this module's docstring. Returns one average per option (None for an
    option with zero usable replicates), or None outright if nothing came back usable at all."""
    sums = [0.0] * n_options
    counts = [0] * n_options
    pending = set(branch_ids)
    deadline = time.time() + timeout
    while pending and time.time() < deadline and not stop_event.is_set():
        try:
            branch_id, result = results_queue.get(timeout=1.0)
        except Empty:
            continue
        if branch_id not in pending:
            continue  # stale result from an already-resolved earlier meta-decision
        pending.discard(branch_id)
        if result is None or result.get("outcome") == "error":
            continue
        actions = result.get("actions") or []
        if len(actions) <= prefix_len or "value" not in actions[prefix_len]:
            continue
        option_index = result.get("option_index")
        if option_index is None or not (0 <= option_index < n_options):
            continue
        sums[option_index] += actions[prefix_len]["value"]
        counts[option_index] += 1
    if pending:
        print(f"warning: {len(pending)}/{len(branch_ids)} meta-branch(es) didn't finish within "
              f"{timeout}s")
    if not any(counts):
        return None
    return [(sums[i] / counts[i]) if counts[i] else None for i in range(n_options)]


def _resolve_meta_decision(
    job_queue, results_queue, main_worker_index: int, cfg: FacilityConfig, recorder: CompactRecorder,
    session, decision, character: str, seed: str, main_id: str, decision_kind: str,
    divergence_index: int, stop_event,
):
    """Branches every legal option of `decision` (a live, pydantic-typed card_reward decision on
    the main Sts2Session) across cfg.meta_options_replicates replicate rollouts each, then adopts
    the best-averaged option by stepping it directly on `session` -- see this module's docstring
    for why no session reconstruction is needed here. Falls back to the plain heuristic pick
    (this is exactly what choose_action would have done anyway) if fewer than 2 options exist to
    compare, or if every spawned replicate came back unusable -- either way, forward progress on
    the main line matters more than a missed training signal for this one decision."""
    options = legal_options(decision, decision_kind)
    if len(options) < 2:
        if not options:
            return decision  # defensive -- card_reward always offers skip_card_reward at least
        name, args = options[0]
        return session.step(name, **args)

    prefix = list(session.action_log)
    branch_ids = []
    for option_index in range(len(options)):
        for replicate_index in range(cfg.meta_options_replicates):
            job = MetaBranchJob(
                main_id=main_id, decision_kind=decision_kind, divergence_index=divergence_index,
                option_index=option_index, replicate_index=replicate_index,
                character=character, seed=seed, ascension=cfg.ascension,
                actions=list(prefix), horizon_combats=cfg.meta_horizon_combats,
            )
            job_queue.put((job, main_worker_index))
            branch_ids.append(job.branch_id)

    values = _wait_for_meta_results(
        results_queue, branch_ids, len(prefix), len(options), stop_event, cfg.meta_branch_timeout_s,
    )
    if values is None:
        recorder.log_error(
            f"meta-decision {divergence_index} ({decision_kind}): no usable option results "
            f"(0/{len(branch_ids)} usable) -- falling back to the heuristic pick"
        )
        name, args, policy, _ = choose_action(decision, session)
        if policy is not None:
            recorder.attach_policy(policy)
        return session.step(name, **args)

    best_option = max(range(len(options)), key=lambda i: values[i] if values[i] is not None else float("-inf"))
    name, args = options[best_option]
    # choices_paradox/model/materialize.py's materialize_main_trial pulls the main line's own
    # adopted meta-decision back out as a value-regression example, the same way facility_claws's
    # _absorb_actions attaches an adopted turn-branch's policy/value onto the main line -- it only
    # checks "policy" in entry as a presence marker (never reads its content), so this doesn't
    # need to be a real probability distribution, just informative for debugging.
    recorder.attach_policy([
        {"action": a, "args": kw, "value": v} for (a, kw), v in zip(options, values)
    ])
    recorder.attach_value(values[best_option])
    return session.step(name, **args)


def _play_meta_forward(session, decision, job: MetaBranchJob, recorder: CompactRecorder) -> str:
    """Commits to job's chosen legal option, then plays forward unbranched (reusing
    facility_common._play_turn_only per combat, purpose threaded exactly like the main line does
    via choose_action) until job.horizon_combats wins have been reached and the next combat_play
    decision is in view (never mid-combat -- the cutoff always lands on a clean decision), or a
    loss/game-over occurs first.

    Scores the whole walk as a single value, attached retroactively (recorder.set_value) to the
    option-commit action's own step -- unlike combat's per-turn attach_value/end_turn scoring,
    only that one action is ever assigned a value here, since it's the only one materialize.py's
    grouping actually reads (see that module's _scan_one/materialize_group). Every action taken
    after the commit still gets its own per-turn value via _play_turn_only's normal
    begin_turn/end_turn bracketing where combat is played, purely for human debugging -- nothing
    downstream consumes it.

    Buffer-protocol decisions carry no context.room_type/enemies (unlike the main line -- see
    facility_claws._resolve_combat's docstring), so a branch worker here has no reliable way to
    detect a boss fight. Every combat encountered inside the horizon walk is scored with the
    plain (non-boss) formula as a result -- an accepted v1 simplification (see this package's
    README's Open questions section), revisit only if boss combats landing inside a horizon walk
    turn out to bias training."""
    options = legal_options_from_buffer(decision, job.decision_kind)
    name, args = options[job.option_index]
    commit_step = recorder.mark_step()
    decision = session.step(name, **args)

    wins = 0
    pending_purpose: str | None = None
    outcome = "continue"  # "continue" here means "reached the horizon still alive"
    while True:
        if decision.decision == "game_over":
            outcome = "won" if decision.victory else "lost"
            break
        if wins >= job.horizon_combats and decision.decision == "combat_play":
            break  # horizon reached -- cutoff lands cleanly at the next combat's start
        if decision.decision == "combat_play":
            combat_hp_before, combat_max_hp = decision.player_hp, decision.player_max_hp
            combat_outcome = "continue"
            while combat_outcome == "continue":
                combat_outcome, decision = _play_turn_only(
                    session, decision, combat_hp_before, combat_max_hp,
                    bypass_reward_on_win=False,
                )
            if combat_outcome == "lost":
                outcome = "lost"
                break
            wins += 1
            continue
        name, kwargs, policy, pending_purpose = choose_action(decision, session, pending_purpose)
        if policy is not None:
            recorder.attach_policy(policy)
        decision = session.step(name, **kwargs)

    if outcome == "lost":
        value = -1.0
    elif outcome == "won":
        value = 1.0  # beat the whole run within the horizon -- best case, above anything
        # _combat_value's own HP-fraction formula reaches
    else:
        value = 0.0
        net = get_claws_net()
        if net is not None:
            _, _, value_t = net.forward(decision)
            value = float(value_t.item())

    recorder.set_value(commit_step, value)
    return outcome


def _run_meta_job(cfg: FacilityConfig, job: MetaBranchJob, results_queue) -> None:
    """branch_worker_loop's MetaBranchJob handler -- one (option, replicate) rollout, played
    forward via _play_meta_forward and saved to its own trial file. `option_index` is included in
    the recorded meta dict specifically so _wait_for_meta_results can group usable replicates by
    option without re-deriving it from the trial's actions."""
    recorder = CompactRecorder({
        "character": job.character, "seed": job.seed, "ascension": job.ascension,
        "main_id": job.main_id, "role": "branch", "branch_kind": "meta",
        "decision_kind": job.decision_kind, "divergence_index": job.divergence_index,
        "option_index": job.option_index, "replicate_index": job.replicate_index,
        "prefix_len": len(job.actions),
    }, start_step=len(job.actions))
    session = Sts2BufferSession(logger=recorder)
    outcome = None
    result = None
    try:
        try:
            decision = session.open_from_checkpoint(job.character, job.seed, job.ascension, job.actions)
            outcome = _play_meta_forward(session, decision, job, recorder)
        except Exception as e:
            recorder.log_error(f"{type(e).__name__}: {e}")
            outcome = "error"
        try:
            result = recorder.save(f"{cfg.raw_dir}/{job.branch_id}.json", session.action_log, outcome=outcome)
        except Exception as e:
            recorder.log_error(f"save failed: {type(e).__name__}: {e}")
    finally:
        session.close()
        results_queue.put((job.branch_id, result))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Whispering Earring self-play facility (card_reward meta-branching, "
                     "for choices_paradox) -- combat is still turn-branched exactly like "
                     "facility_claws.py's main line, sharing the same branch-worker pool"
    )
    add_common_cli_args(parser)
    parser.add_argument("--branches-per-turn", type=int, default=3,
                         help="combat turn-branching, same knob and default as facility_claws.py "
                              "-- the main line here still plays combat this way between "
                              "card_reward decisions")
    parser.add_argument("--meta-branches-per-trial", type=int, default=None,
                         help="cap on how many card_reward decisions per trial actually spawn "
                              "branches (default: unlimited -- branch every one, matching "
                              "combat's own always-branch default)")
    parser.add_argument("--meta-options-replicates", type=int, default=3,
                         help="M: independent replicate rollouts per legal option, averaged "
                              "before comparing across options (default: 3)")
    parser.add_argument("--meta-horizon-combats", type=int, default=1,
                         help="play forward this many *won* combats (or a loss) past the "
                              "card_reward before bootstrapping off Claws's own value head "
                              "(default: 1 -- a starting guess, see this package's README)")
    parser.add_argument("--meta-branch-timeout", type=float, default=600.0,
                         help="seconds to wait for a meta-branch's replicates before giving up "
                              "on stragglers (default: 600 -- generous vs. combat's own 120s "
                              "turn-branch timeout, since a meta-branch is many combats long)")
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
            meta_branches_per_trial=args.meta_branches_per_trial,
            meta_options_replicates=args.meta_options_replicates,
            meta_horizon_combats=args.meta_horizon_combats,
            meta_branch_timeout_s=args.meta_branch_timeout,
        ),
        main_worker_fn=functools.partial(
            main_worker_loop, combat_resolver=_resolve_combat, meta_resolver=_resolve_meta_decision,
        ),
        branch_worker_fn=functools.partial(
            branch_worker_loop, handlers={BranchJob: _run_branch_job, MetaBranchJob: _run_meta_job},
        ),
        resume=resume,
    )
    facility.run()

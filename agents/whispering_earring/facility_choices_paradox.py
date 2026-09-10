from __future__ import annotations

import functools
import importlib.util
import os
import random
import sys
from pathlib import Path

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)
import buffer_layout as bl
from buffer_env import BufferDecision, Sts2BufferSession, decode_token

from checkpoint import BranchJob, MapOptionJob, MetaOptionJob
from recorder import CompactRecorder
from policy import _BufferCard, choose_action
from facility_common import (
    CombatOutcome, Facility, FacilityConfig, _combat_value, _play_turn_only,
    _resolve_map_decision, _run_map_option_job,
    add_common_cli_args, branch_worker_loop, load_resume, main_worker_loop,
)
from facility_claws import _resolve_combat, _run_branch_job


_model_path = Path(__file__).resolve().parents[1] / "choices_paradox" / "model" / "model.py"
_spec = importlib.util.spec_from_file_location("choices_paradox_model", _model_path)
assert _spec is not None and _spec.loader is not None
_cp_model = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _cp_model
_spec.loader.exec_module(_cp_model)
legal_options = _cp_model.legal_options
legal_options_from_buffer = _cp_model.legal_options_from_buffer

_choices_paradox_policy_path = Path(__file__).resolve().parents[1] / "choices_paradox" / "policy.py"
_spec = importlib.util.spec_from_file_location("choices_paradox_policy_module", _choices_paradox_policy_path)
assert _spec is not None and _spec.loader is not None
_choices_paradox_policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_choices_paradox_policy)

_DIRICHLET_EPS = 0.15
_DIRICHLET_ALPHA_SCALE = 10.0

# Flat per-combat-won contribution to an option-chain's accumulated value
_WIN_BONUS = 0.5

_CHOICES_PARADOX_CHECKPOINT = os.environ.get("CHOICES_PARADOX_CHECKPOINT")
_choices_paradox_net = None  # lazy-loaded once per main-worker process, on first divergence


def _load_choices_paradox_net():
    global _choices_paradox_net
    if _choices_paradox_net is not None or not _CHOICES_PARADOX_CHECKPOINT:
        return _choices_paradox_net
    if not os.path.exists(_CHOICES_PARADOX_CHECKPOINT):
        return None  # round 1, or CHOICES_PARADOX_CHECKPOINT unset by training_choices.sh yet
    _choices_paradox_net = _cp_model.ChoicesParadoxNet.from_checkpoint(_CHOICES_PARADOX_CHECKPOINT)
    return _choices_paradox_net


def _choose_meta_option_claws(decision, decision_kind: str, options: list[tuple[str, dict]]) -> int:
    """Picks which option the REAL trial actually commits to -- ChoicesParadoxNet's own softmax
    over `options`, mixed with root Dirichlet noise"""
    import torch

    net = _load_choices_paradox_net()
    if net is None:
        if decision_kind != "card_reward":
            # No pre-net heuristic wired up for rest_site yet (its buffer-side option slots carry
            # no identity to prefer HEAL over SMITH by -- see model.py's tokenize_from_buffer),
            # and no card_select one either -- uniform random until a checkpoint exists, same
            # "reasonable, not trained" spirit as choices_paradox/policy.py's own placeholders.
            return random.randrange(len(options))
        # Mirrors policy.py's _choose_card_reward_from_buffer -- same event_card_reward-vs-
        # ordinary-card_reward buffer split, same _BufferCard/decode_token construction.
        if isinstance(decision, BufferDecision):
            arr = decision.arr
            if bl.decision_name(arr) == "event_card_reward":
                cards = [
                    _BufferCard(i, decode_token(bl.card_token(slot)))
                    for i, slot in bl.event_card_reward_cards(arr)
                ]
                can_skip = True
            else:
                cards = [
                    _BufferCard(i, decode_token(bl.card_token(slot)))
                    for i, slot in bl.reward_cards(arr)
                ]
                can_skip = bl.can_skip_reward(arr)
            deck_size = bl.player_deck_size(arr)
        else:
            cards, can_skip, deck_size = decision.cards, decision.can_skip, decision.player.deck_size or 0
        index = _choices_paradox_policy.choose_card_reward(cards, can_skip, deck_size)
        for i, (opt_name, opt_args) in enumerate(options):
            if index is None:
                if opt_name == "skip_card_reward":
                    return i
            elif opt_name == "select_card_reward" and opt_args.get("card_index") == index:
                return i
        return 0  # defensive -- the heuristic's pick should always be one of `options`

    _, logits, _ = net.forward(decision, decision_kind)
    probs = torch.softmax(logits, dim=-1)
    alpha = _DIRICHLET_ALPHA_SCALE / len(options)
    noise = torch.distributions.Dirichlet(torch.full((len(options),), alpha)).sample()
    probs = (1 - _DIRICHLET_EPS) * probs + _DIRICHLET_EPS * noise
    return int(torch.multinomial(probs, 1).item())


def _run_meta_option_job(cfg: FacilityConfig, job: MetaOptionJob, results_queue) -> None:
    """branch_worker_loop's MetaOptionJob handler -- plays one card_reward option all the way to
    natural death, combat by combat, on ONE in-process Sts2BufferSession held open for the chain's
    entire walk """
    import torch

    recorder = CompactRecorder({
        "character": job.character, "seed": job.seed, "ascension": job.ascension,
        "main_id": job.main_id, "role": "branch", "branch_kind": "meta",
        "decision_kind": job.decision_kind, "divergence_index": job.divergence_index,
        "option_index": job.option_index, "prefix_len": len(job.prefix),
    }, start_step=len(job.prefix))
    session = Sts2BufferSession(logger=recorder)
    total_value = 0.0
    pending_purpose = None
    outcome = None
    try:
        try:
            # Fast load from checkpoint -- the checkpoint taken before whichever room was most
            # recently entered (combat, rest_site, event_choice, ...; see _TrialState.room_index),
            # not necessarily a combat's own.
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
            # Reseed AFTER committing (the commit itself makes no random.*/torch.* draws) so
            # every sibling option's own post-commit policy draws start from the identical
            # stream -- see this function's docstring on walk_seed.
            random.seed(job.walk_seed)
            torch.manual_seed(job.walk_seed)

            combats_resolved = 0
            while decision.decision != "game_over":
                if decision.decision == "combat_play":
                    combat_hp_before, combat_max_hp = decision.player_hp, decision.player_max_hp
                    turn_outcome = "continue"
                    while turn_outcome == "continue":
                        turn_outcome, decision = _play_turn_only(
                            session, decision, combat_hp_before, combat_max_hp,
                            bypass_reward_on_win=False,
                            combat_dirichlet_noise=cfg.combat_dirichlet_noise,
                        )
                    if turn_outcome == "lost":
                        outcome = "lost"
                        break
                    total_value += _WIN_BONUS + _combat_value(
                        CombatOutcome(decision.player_hp, decision.player_max_hp)
                    )
                    combats_resolved += 1
                    if job.horizon_combats is not None and combats_resolved >= job.horizon_combats:
                        # Bounded-horizon cutoff -- not a loss, just stop measuring here. Same "no
                        # terminal penalty" reasoning as the natural-death case: total_value is
                        # whatever this chain accumulated up to the cutoff, no bonus or penalty
                        # for what would have happened past it.
                        outcome = "horizon"
                        break
                    continue

                # card_reward_dirichlet_noise=False: any card_reward this chain hits past its own
                # initial commit would otherwise pick up extra sampling variance on top of the
                # option-chain comparison this whole job exists to measure.
                name, kwargs, policy, pending_purpose = choose_action(
                    decision, session, pending_purpose, card_reward_dirichlet_noise=False,
                )
                if policy is not None:
                    recorder.attach_policy(policy)
                decision = session.step(name, **kwargs)

            if outcome is None:
                outcome = "won" if decision.victory else "lost"
        except Exception as e:
            recorder.log_error(f"{type(e).__name__}: {e}")
            outcome = "error"

        if outcome != "error":
            recorder.set_value(commit_step, total_value)
        try:
            recorder.save(f"{cfg.raw_dir}/{job.meta_id}.json", session.action_log, outcome=outcome)
        except Exception as e:
            recorder.log_error(f"save failed: {type(e).__name__}: {e}")
    finally:
        session.close()
        # See this function's docstring on job.walk_seed -- must run regardless of outcome, so a
        # crashed/errored chain doesn't leave this worker process walk_seed-correlated either.
        random.seed(os.urandom(16))
        torch.seed()
        results_queue.put((job.meta_id, None if outcome == "error" else total_value))


def _resolve_meta_decision(
    job_queue, results_queue, meta_results_queue, main_worker_index: int, cfg: FacilityConfig,
    recorder: CompactRecorder, session, decision, character: str, seed: str, main_id: str,
    decision_kind: str, divergence_index: int, stop_event, queue_manager,
    last_room_checkpoint_path: str | None, last_room_entry_index: int | None,
):
    """"""
    options = (
        legal_options_from_buffer(decision, decision_kind)
        if isinstance(decision, BufferDecision)
        else legal_options(decision, decision_kind)
    )
    if len(options) < 2:
        if not options:
            return decision, []  # defensive -- card_reward always offers skip_card_reward at least
        name, args = options[0]
        return session.step(name, **args), []

    prefix = list(session.action_log)
    # One shared walk_seed for every sibling option of THIS divergence only siblings of the SAME divergence should share one.
    walk_seed = random.getrandbits(64)
    # The mainline choose a combat option
    adopted_index = _choose_meta_option_claws(decision, decision_kind, options)
    commit_step = recorder.mark_step()
    recorder.attach_policy([{"action": a, "args": kw, "value": None} for a, kw in options])
    name, args = options[adopted_index]
    decision = session.step(name, **args)

    pending_meta_jobs = []
    # Spawn a job for each option, including the one the mainline actually adopted
    for option_index, (opt_name, opt_args) in enumerate(options):
        job = MetaOptionJob(
            main_id=main_id, character=character, seed=seed, ascension=cfg.ascension,
            decision_kind=decision_kind, divergence_index=divergence_index,
            option_index=option_index, prefix=prefix,
            last_room_checkpoint_path=last_room_checkpoint_path,
            last_room_entry_index=last_room_entry_index,
            commit_name=opt_name, commit_args=opt_args, walk_seed=walk_seed,
            horizon_combats=cfg.meta_horizon_combats,
        )
        # The dispatched-to queue travels with the job itself, not looked up from a table -- see
        # branch_worker_loop's docstring. Lands on the plain (low-priority) job_queue so this
        # trial's own combat turn-branches (job_queue_priority) are never stuck behind it.
        job_queue.put((job, meta_results_queue))
        pending_meta_jobs.append(
            (job.meta_id, last_room_checkpoint_path, option_index == adopted_index, commit_step)
        )

    return decision, pending_meta_jobs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Whispering Earring self-play facility (card_reward meta-branching, "
                     "for choices_paradox) -- the main line's own combat is still turn-branched "
                     "exactly like facility_claws.py's main line, sharing its branch-worker pool; "
                     "each card_reward option's own evaluation chain plays unbranched and is "
                     "dispatched to that same pool asynchronously, without the main line waiting "
                     "on it (see this module's docstring)"
    )
    add_common_cli_args(parser)
    parser.add_argument("--branches-per-turn", type=int, default=3,
                         help="combat turn-branching, same knob and default as facility_claws.py ")
    parser.add_argument("--meta-branches-per-trial", type=int, default=None,
                         help="cap on how many card_reward decisions per trial actually spawn "
                              "chains (default: unlimited)")
    parser.add_argument("--meta-horizon-combats", type=int, default=6,
                         help="cap each option-chain at this many combats past its own commit "
                              "instead of playing to natural death (default: unbounded)")
    parser.add_argument("--map-branches-per-trial", type=int, default=0,
                         help="cap on how many select_map_node decisions per trial spawn "
                              "MapOptionJob branches for golden_compass (default: 0, disabled; "
                              "pass -1 for unlimited)")
    parser.add_argument("--map-room-horizon", type=int, default=1,
                         help="rooms a MapOptionJob branch walks past its own commit before "
                              "scoring (default: 1 -- see facility_common._run_map_option_job)")
    args = parser.parse_args()

    resume = load_resume(args.resume_branch_file) if args.resume_branch_file else None
    map_branches_per_trial = None if args.map_branches_per_trial < 0 else args.map_branches_per_trial

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
            meta_branches_per_trial=args.meta_branches_per_trial,
            meta_horizon_combats=args.meta_horizon_combats,
            map_branches_per_trial=map_branches_per_trial,
            map_room_horizon=args.map_room_horizon,
            combat_dirichlet_noise=not args.no_combat_dirichlet_noise,
        ),
        main_worker_fn=functools.partial(
            main_worker_loop, combat_resolver=_resolve_combat, meta_resolver=_resolve_meta_decision,
            map_resolver=_resolve_map_decision, session_factory=Sts2BufferSession,
        ),
        branch_worker_fn=functools.partial(
            branch_worker_loop,
            handlers={
                BranchJob: _run_branch_job, MetaOptionJob: _run_meta_option_job,
                MapOptionJob: _run_map_option_job,
            },
        ),
        resume=resume,
    )
    facility.run()

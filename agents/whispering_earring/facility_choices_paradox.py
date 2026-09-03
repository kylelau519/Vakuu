"""Card-reward meta-decision branching, generating training data for `choices_paradox` (see
agents/choices_paradox/model/README.md's Design section). Every eligible card_reward decision
along the main line branches into one chain per legal option, each played forward to natural death
(a real loss, not a fixed horizon) -- not a single noisy rollout, but not the main line's own
turn-branch-and-adopt combat quality either (see below on why). card_select's remove/upgrade
purposes still resolve via choices_paradox/policy.py's existing heuristic, unbranched, until
Phase 2.

Each option-chain runs UNBRANCHED, SEQUENTIALLY (one at a time, not concurrent), on its own
`Sts2BufferSession` (buffer_env.py -- in-process CLR calls via pythonnet) held open for the
chain's entire natural-death walk. This replaced an earlier design that turn-branched every
chain's combat through facility_claws's shared branch-worker pool the same way the main line
does, with sibling options run concurrently on threads:

- Compute cost: facility_claws.py's `_resolve_turn` closes and respawns a real `dotnet run` OS
  subprocess on EVERY combat turn, replaying the whole combat's actions into it over JSON. The
  main line pays that once per turn, total; the old design paid it once per turn PER CHAIN, and a
  single divergence spawns ~3-4 chains, each playing to natural death (so replay cost grows with
  how many actions a chain has already accumulated) -- measured contributor, not hypothetical: one
  option-chain two combats and ten turns in had already accumulated 123 actions to replay on every
  one of those resyncs. In-process BufferSession, held open for the whole walk, pays neither the
  subprocess spawn nor the resync at all.
- Thread safety: the old design's concurrency was safe only because each chain's Sts2Session was
  its own OS subprocess with its own results_queue -- provably independent. Sts2BufferSession has
  no such isolation: every session in a process shares the one CoreCLR clr_bridge.load_clr() loads
  into it, and debug_repro_branch_divergence.py already caught RunSimulator's save/profile state
  racing across merely-concurrent *processes* sharing only an unlocked on-disk profile path
  (confirmed to cause real RNG/state divergence from the first replayed decision) -- threads
  sharing the same in-process CLR directly would be at least as exposed. So chains here run one at
  a time, trading away the old design's wall-clock parallelism for not needing an OS subprocess (or
  the shared branch-worker pool) per chain at all -- by far the bigger cost removed.

The combat-quality trade this makes explicit: each chain's own play is exactly what
`choose_action`'s Dirichlet-noised policy picks (combat included, via Claws's own net over the
buffer protocol, the same path branch workers already use), on-policy but unbranched -- noisier
per chain than the main line's turn-branch-and-adopt quality. Accepted because the option
comparison this module trains on only needs to be unbiased on average, and chain volume should
average out per-chain noise the same way materialize.py's group-baseline subtraction already
discounts other per-chain variance -- trading combat-quality-per-chain for far more chains is the
better use of the same compute budget here (unlike the main line, whose own trajectory and Claws's
turn-branch training data both still want the higher-quality adopted play).

Design:
- One chain per legal option, no replicate averaging (M=1 by design -- see the README's "Original
  meaning of replicate"; the noise that used to justify M replicates on a raw single-turn rollout
  is instead absorbed by chain volume).
- Each option-branch commits its card, then plays forward using the ordinary self-play policy --
  nothing else is allowed to diverge between siblings, so any outcome difference is attributable
  to the one choice being compared. Enforced via a shared walk_seed (drawn once per divergence in
  _resolve_meta_decision, common to every sibling): reseeds Python's global `random` and torch's
  global RNG identically right after each sibling commits, so policy draws (map routing, Dirichlet-
  sampled cards, tie-breaks) are bit-identical for as long as game state hasn't actually diverged --
  "common random numbers," standard variance reduction for a paired counterfactual comparison. This
  can't and doesn't force identical downstream content (reward offers, monster picks) once real
  divergence begins -- that's the engine's own internal RNG drifting once siblings take different
  actions, and that drift IS the causal signal being measured. Reseeding the GLOBAL state (rather
  than threading an explicit Generator) is safe only because chains run sequentially -- no other
  chain's RNG state to clobber mid-walk.
- The divergence is reconstructed via the pre-combat native checkpoint main_worker_loop already
  took before the combat that produced this card_reward (last_combat_checkpoint_path), replaying
  just that one combat's tail of actions via load_and_resume_combat to land back on the reward --
  NOT a fresh checkpoint taken at the pending card_reward decision itself: confirmed by hand that
  doesn't work (RunSimulator.SaveCheckpoint rolls back to a pre-room map state whenever the current
  room isn't a MapRoom -- TryRollbackSerializedSaveToPreRoom, sts2-cli's RunSimulator.SaveLoad.cs --
  so a checkpoint taken mid-card_reward reloads PAST it, at map_select, with no pending-reward
  signal left; the save format has no representation of "mid-room" state outside MapRoom at all).
  This is the chain's ONLY reconstruction -- it never re-checkpoints itself again, just keeps
  stepping its live session forward to natural death.
- No fixed horizon, no TD-bootstrap off Claws's value head -- a chain plays until it actually
  loses. Value = _WIN_BONUS + that combat's own _combat_value (facility_common.py's HP-fraction/
  boss-shaped quality-of-win term), summed per combat won -- except is_last_of_act is always False
  for a chain's own combats: that flag comes from the main line's live pydantic
  `context.room_type` in the original design (see checkpoint.py's docstring on why the buffer
  protocol alone can't recover it), and a chain has no pydantic Decision to read that from. Accepted
  simplification: a chain-internal boss win scores with the ordinary HP-fraction formula instead of
  the boss-shaped one.
- _WIN_BONUS started at 1.0, dropped to 0.5 after the first live smoke test: at 1.0, combat-count
  survival dominated pick-quality (_combat_value, typically ~0.1-0.5) -- three genuinely different
  picks landed within ~0.2 of each other while skipping landed ~1.1 lower. Retune again if trained
  output still looks survival-count-dominated. No terminal -1 for the eventual loss -- every chain
  loses eventually by construction, so a flat penalty would just be a constant materialize.py's own
  group-baseline subtraction already cancels out.
- No recursive meta-branching for any card_reward a chain encounters past its own initial commit --
  resolved by a single self-play pick like any other decision. Combinatorial cost (options^depth) is
  why, and it's a deliberate choice, not a placeholder.
- The trial that continues to the next divergence does NOT adopt whichever option scored best. It
  rewinds to the pre-divergence state (the live session never left it -- each option's chain runs on
  its OWN forked session) and commits via ChoicesParadoxNet's own softmax-plus-Dirichlet-noise
  prediction instead (_choose_meta_option_claws) -- keeps the real trial on-policy rather than
  coupled to whichever option an under-trained net happened to explore best this round.

Output format is unchanged from Phase 1 (role="branch", branch_kind="meta", one value attached to
the commit action) -- one file per option instead of per (option, replicate) is exactly the M=1
degenerate case _group_baseline's math already handles, so materialize.py needs no changes.
"""
from __future__ import annotations

import functools
import importlib.util
import os
import random
import sys
from pathlib import Path

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)
from buffer_env import Sts2BufferSession

from checkpoint import BranchJob
from recorder import CompactRecorder
from policy import choose_action
from facility_common import (
    CombatOutcome, Facility, FacilityConfig, _combat_value, _play_turn_only,
    add_common_cli_args, branch_worker_loop, load_resume, main_worker_loop,
)
from facility_claws import _resolve_combat, _run_branch_job

# Loaded by file path (not sys.path + `import model`) for the same reason policy.py loads
# choices_paradox/policy.py and claws/model's ClawsNet this way: this package's model.py lives
# under a sibling package with no __init__.py-based package structure to import through
# normally, and doing it by path avoids colliding with another module also named `model`
# elsewhere on sys.path (e.g. claws/model/model.py).
_model_path = Path(__file__).resolve().parents[1] / "choices_paradox" / "model" / "model.py"
_spec = importlib.util.spec_from_file_location("choices_paradox_model", _model_path)
assert _spec is not None and _spec.loader is not None
_cp_model = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _cp_model  # dataclasses' own class-processing looks itself up via
# sys.modules[cls.__module__] during exec_module -- module_from_spec alone doesn't register it
_spec.loader.exec_module(_cp_model)
legal_options = _cp_model.legal_options

_choices_paradox_policy_path = Path(__file__).resolve().parents[1] / "choices_paradox" / "policy.py"
_spec = importlib.util.spec_from_file_location("choices_paradox_policy_module", _choices_paradox_policy_path)
assert _spec is not None and _spec.loader is not None
_choices_paradox_policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_choices_paradox_policy)

# Same Dirichlet convention as policy.py's own _choose_combat_action_claws (root noise mixed into
# the softmax before sampling, concentration adapted to how many options are on offer) -- kept as
# a local copy rather than importing policy.py's underscore-prefixed constants across modules.
_DIRICHLET_EPS = 0.15
_DIRICHLET_ALPHA_SCALE = 10.0

# Flat per-combat-won contribution to an option-chain's accumulated value, on top of that combat's
# own _combat_value -- see this module's docstring for why 0.5, not the original 1.0.
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
    over `options`, mixed with root Dirichlet noise, sampled (not argmax). Deliberately NOT the
    best-scoring option from this divergence's own chain evaluations (see this module's docstring)
    -- falls back to the existing tier-list heuristic (ignoring the chain values entirely) when no
    checkpoint is loaded yet, same "noisy but harmless" round-1 fallback Claws's own combat picker
    already follows."""
    import torch

    net = _load_choices_paradox_net()
    if net is None:
        # Round 1 (no checkpoint yet) -- the same tier-list heuristic policy.py's own
        # _choose_generic_action delegates to for card_reward, called directly here rather than
        # through choose_action (which would need a live session this function doesn't have one
        # to spare -- decision.cards/.can_skip/.player are all this heuristic actually reads).
        index = _choices_paradox_policy.choose_card_reward(
            decision.cards, decision.can_skip, decision.player.deck_size or 0
        )
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


def _evaluate_option_chain(
    cfg: FacilityConfig, character: str, seed: str, ascension: int, main_id: str,
    decision_kind: str, divergence_index: int, option_index: int, prefix: list[dict],
    last_combat_checkpoint_path: str | None, last_combat_entry_index: int | None,
    commit_name: str, commit_args: dict, walk_seed: int,
) -> float | None:
    """Plays one card_reward option all the way to natural death, combat by combat, on ONE
    in-process Sts2BufferSession held open for the chain's entire walk (see this module's docstring
    for the unbranched/sequential design and its rationale). Terminates on the first loss; returns
    the accumulated value, or None if this chain errored before reaching a terminal state.

    walk_seed: the SAME value for every sibling option of this divergence (drawn once by
    _resolve_meta_decision) -- reseeds Python's global `random` and torch's global RNG right after
    committing (see below), giving common-random-numbers variance reduction (see this module's
    docstring). Only removes noise from wherever siblings haven't actually diverged yet; once a
    chain's own state differs (different cards change what's drawn/playable), the streams desync
    from there, which is the causal signal being measured, not something to fight.

    Runs on its own Sts2BufferSession -- never touches or is grafted onto the main-line session the
    caller holds. Reconstructed via load_and_resume_combat(last_combat_checkpoint_path, ...) --
    replaying just the tail of actions for the ONE combat that produced this card_reward -- rather
    than a full open_from_checkpoint(prefix) replay from StartRun (see this module's docstring on
    why a fresh checkpoint at the card_reward itself doesn't work). Falls back to the full replay
    when no combat checkpoint was available (last_combat_checkpoint_path is None -- vanishingly
    rare). Either way the reconstructed session has no memory of anything before the checkpoint/
    replay point, so `prefix` is spliced back onto session.action_log afterward, needed purely so
    CompactRecorder's start_step=len(prefix) indexing lines up with the action_log this chain's own
    recorder.save() is eventually handed. This is the chain's only reconstruction -- nothing
    re-checkpoints it again after this point.

    No recursive meta-branching for any card_reward this chain encounters past its own initial
    commit -- resolved via a single choose_action pick, same as main_worker_loop's own behavior
    whenever meta_resolver isn't invoked."""
    recorder = CompactRecorder({
        "character": character, "seed": seed, "ascension": ascension, "main_id": main_id,
        "role": "branch", "branch_kind": "meta", "decision_kind": decision_kind,
        "divergence_index": divergence_index, "option_index": option_index,
        "prefix_len": len(prefix),
    }, start_step=len(prefix))
    branch_main_id = f"{main_id}_m{divergence_index:03d}_o{option_index:02d}"
    session = Sts2BufferSession(logger=recorder)
    total_value = 0.0
    pending_purpose = None
    outcome = None
    try:
        try:
            if last_combat_checkpoint_path is not None and last_combat_entry_index is not None:
                entry_action = prefix[last_combat_entry_index]
                in_combat_actions = prefix[last_combat_entry_index + 1:]
                decision = session.load_and_resume_combat(
                    last_combat_checkpoint_path, entry_action, in_combat_actions
                )
            else:
                decision = session.open_from_checkpoint(character, seed, ascension, prefix)
            session.action_log = list(prefix)
            commit_step = recorder.mark_step()
            decision = session.step(commit_name, **commit_args)
            # Reseed AFTER committing (the commit itself makes no random.*/torch.* draws) so
            # every sibling option's own post-commit policy draws start from the identical
            # stream -- see this function's docstring on walk_seed.
            import torch
            random.seed(walk_seed)
            torch.manual_seed(walk_seed)

            while decision.decision != "game_over":
                if decision.decision == "combat_play":
                    # Unbranched: one turn at a time on this same session via
                    # facility_common._play_turn_only, looped until the combat itself resolves.
                    # bypass_reward_on_win=False keeps the terminal card_reward real so it can be
                    # resolved via choose_action below (see _play_turn_only's own docstring).
                    combat_hp_before, combat_max_hp = decision.player_hp, decision.player_max_hp
                    turn_outcome = "continue"
                    while turn_outcome == "continue":
                        turn_outcome, decision = _play_turn_only(
                            session, decision, combat_hp_before, combat_max_hp,
                            bypass_reward_on_win=False,
                        )
                    if turn_outcome == "lost":
                        outcome = "lost"
                        break
                    total_value += _WIN_BONUS + _combat_value(
                        CombatOutcome(decision.player_hp, decision.player_max_hp)
                    )
                    continue

                name, kwargs, policy, pending_purpose = choose_action(decision, session, pending_purpose)
                if policy is not None:
                    recorder.attach_policy(policy)
                decision = session.step(name, **kwargs)

            if outcome is None:
                # Reached game_over directly (e.g. beat the final boss) rather than via a combat
                # "lost" -- rare but not impossible; total_value has already accumulated every
                # combat this chain actually won.
                outcome = "won" if decision.victory else "lost"
        except Exception as e:
            recorder.log_error(f"{type(e).__name__}: {e}")
            outcome = "error"

        if outcome != "error":
            recorder.set_value(commit_step, total_value)
        try:
            recorder.save(f"{cfg.raw_dir}/{branch_main_id}.json", session.action_log, outcome=outcome)
        except Exception as e:
            recorder.log_error(f"save failed: {type(e).__name__}: {e}")
    finally:
        session.close()

    return None if outcome == "error" else total_value


def _resolve_meta_decision(
    job_queue, results_queue, main_worker_index: int, cfg: FacilityConfig, recorder: CompactRecorder,
    session, decision, character: str, seed: str, main_id: str, decision_kind: str,
    divergence_index: int, stop_event, queue_manager,
    last_combat_checkpoint_path: str | None, last_combat_entry_index: int | None,
):
    """Evaluates every legal option of `decision` (a live, pydantic-typed card_reward decision on
    the main Sts2Session) by playing each to natural death via _evaluate_option_chain, one option
    at a time (see this module's docstring for why sequential), then commits for real via
    ChoicesParadoxNet's own prediction (_choose_meta_option_claws) -- not necessarily whichever
    option scored best. Falls back to the plain heuristic pick (what choose_action would have done
    anyway) if fewer than 2 options exist to compare, or if every chain errored -- forward progress
    on the main line matters more than a missed training signal for this one decision.

    job_queue/results_queue/main_worker_index/stop_event/queue_manager are unused by this
    function's own body -- kept only because main_worker_loop's meta_resolver contract calls every
    meta_resolver with this same positional signature (see its docstring); facility_claws.py's
    combat_resolver still uses job_queue/results_queue/stop_event for the branch-worker pool the
    main line's own combat uses.

    last_combat_checkpoint_path/last_combat_entry_index (from main_worker_loop, see its docstring):
    passed straight through to every option-chain, which reconstructs its own starting state from
    it (_evaluate_option_chain's own load_and_resume_combat call, see its docstring and this
    module's docstring for why a fresh checkpoint at the card_reward itself doesn't work)."""
    options = legal_options(decision, decision_kind)
    if len(options) < 2:
        if not options:
            return decision  # defensive -- card_reward always offers skip_card_reward at least
        name, args = options[0]
        return session.step(name, **args)

    prefix = list(session.action_log)
    # One shared walk_seed for every sibling option of THIS divergence (common random numbers --
    # see _evaluate_option_chain's docstring). Drawn fresh per divergence, not per trial: a fixed
    # cross-divergence seed would make every divergence's chains replay identical "random" choices
    # too, which isn't wanted -- only siblings of the SAME divergence should share one.
    walk_seed = random.getrandbits(64)
    values: list[float | None] = [
        _evaluate_option_chain(
            cfg, character, seed, cfg.ascension, main_id, decision_kind, divergence_index,
            option_index, prefix, last_combat_checkpoint_path, last_combat_entry_index, name, args,
            walk_seed,
        )
        for option_index, (name, args) in enumerate(options)
    ]
    # Every chain above deliberately reseeded random/torch to the SAME walk_seed -- restore
    # unpredictable global state now, before _choose_meta_option_claws's own Dirichlet sampling
    # below (the REAL adoption draw) and before control returns to the main line's own subsequent
    # play, both of which should NOT stay correlated with walk_seed.
    import torch
    random.seed(os.urandom(16))
    torch.seed()

    # choices_paradox/model/materialize.py's materialize_main_trial pulls the main line's own
    # adopted meta-decision back out as a value-regression example, the same way facility_claws's
    # _absorb_actions attaches an adopted turn-branch's policy/value onto the main line -- it only
    # checks "policy" in entry as a presence marker (never reads its per-option content), so a None
    # value for an errored option's chain is harmless here.
    recorder.attach_policy([
        {"action": a, "args": kw, "value": v} for (a, kw), v in zip(options, values)
    ])

    if not any(v is not None for v in values):
        recorder.log_error(
            f"meta-decision {divergence_index} ({decision_kind}): every option's chain errored "
            f"-- falling back to the heuristic pick"
        )
        name, args, policy, _ = choose_action(decision, session)
        if policy is not None:
            recorder.attach_policy(policy)
        return session.step(name, **args)

    adopted_index = _choose_meta_option_claws(decision, decision_kind, options)
    adopted_value = values[adopted_index]
    if adopted_value is not None:
        recorder.attach_value(adopted_value)
    name, args = options[adopted_index]
    return session.step(name, **args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Whispering Earring self-play facility (card_reward meta-branching, "
                     "for choices_paradox) -- the main line's own combat is still turn-branched "
                     "exactly like facility_claws.py's main line, sharing its branch-worker pool; "
                     "each card_reward option's own evaluation chain plays unbranched instead (see "
                     "this module's docstring)"
    )
    add_common_cli_args(parser)
    parser.add_argument("--branches-per-turn", type=int, default=3,
                         help="combat turn-branching, same knob and default as facility_claws.py "
                              "-- only the main line's own combat uses it; option-chains play "
                              "unbranched (see this module's docstring)")
    parser.add_argument("--meta-branches-per-trial", type=int, default=None,
                         help="cap on how many card_reward decisions per trial actually spawn "
                              "chains (default: unlimited -- branch every one, matching combat's "
                              "own always-branch default)")
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
        ),
        main_worker_fn=functools.partial(
            main_worker_loop, combat_resolver=_resolve_combat, meta_resolver=_resolve_meta_decision,
        ),
        branch_worker_fn=functools.partial(
            branch_worker_loop, handlers={BranchJob: _run_branch_job},
        ),
        resume=resume,
    )
    facility.run()

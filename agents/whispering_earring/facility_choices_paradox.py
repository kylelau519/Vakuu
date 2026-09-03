"""Card-reward meta-decision branching, generating training data for `choices_paradox` (see
agents/choices_paradox/model/README.md's Design section). Every eligible
card_reward decision along the main line branches into one chain per legal option, each played
forward to natural death (a real loss, not a fixed horizon) -- not a single noisy rollout, but
NOT the main line's own turn-branch-and-adopt combat quality either (see below on why that
changed). card_select's remove/upgrade purposes still resolve via choices_paradox/policy.py's
existing heuristic, unbranched, until Phase 2.

UNBRANCHED, in-process combat, one option-chain at a time -- deliberately NOT what an earlier
version of this module did (turn-branching every combat through facility_claws's shared
branch-worker pool, same as the main line, with sibling options run concurrently on threads).
That version was correct but badly compute-bound: `_resolve_combat`'s per-turn resync
(facility_claws.py's `_resolve_turn`) closes and respawns a brand-new `Sts2Session` -- a real
`dotnet run` OS subprocess -- on EVERY combat turn, then replays every action taken in that
combat so far into it over the JSON protocol. The main line only ever pays that once per turn,
total. This module's option-chains paid it once per turn PER CHAIN, and a single card_reward
divergence spawns one chain per legal option (~3-4), with `meta_branches_per_trial` such
divergences per trial (default unlimited) -- each chain playing to natural death, so the deeper a
divergence happens, the more turns its sibling chains tend to survive before dying (a built-up
deck outlasts more combats), and the replay-into-a-fresh-subprocess cost of each resync grows
with how many actions that combat has already accumulated. Measured contributor, not
hypothetical: a single option-chain branch file two combats and ten turns into its own walk had
already accumulated 123 actions to replay on every one of those ten per-turn resyncs. Combined
with `branches_per_turn` BranchJobs dispatched to the shared pool by EVERY combat turn of EVERY
concurrently-running chain, this both multiplied the expensive part of combat resolution many
times over and contended with the main line's own combat for the same fixed-size worker pool
(see facility_common.py's job_queue_priority split, added to blunt that second half -- kept, and
still correct, even though option-chains no longer use the shared pool at all after this
rewrite).

Each option-chain now runs on its own `Sts2BufferSession` (buffer_env.py -- in-process CLR calls
via pythonnet, no subprocess, no JSON encode/decode) held open for the chain's ENTIRE natural-
death walk: one session, stepped forward turn by turn and decision by decision with no
respawning, no per-turn resync, and no turn-branching at all -- `choose_action` (policy.py)
already drives every decision kind (combat_play included, via Claws's own net) through the
buffer protocol generically, since branch workers have used it that way since the buffer-mode
rewrite; this reuses that same path for a chain's ENTIRE run instead of one isolated turn. The
combat-quality trade this makes explicit: each chain's own play is now literally what
`choose_action`'s Dirichlet-noised policy picks, on-policy but unbranched -- noisier per chain
than the main line's turn-branch-and-adopt quality, same category of trade Phase 1's
fixed-horizon scheme already made and this module's own predecessor design deliberately moved
away from. Accepted here for the opposite reason Phase 1 was replaced: the option comparison
this module trains on only needs to be unbiased on average, and the chain volume this now
affords should average out per-chain noise the same way materialize.py's group-baseline
subtraction already discounts other per-chain variance -- not a considered claim that the noise
is negligible, just that trading combat-quality-per-chain for far more chains is the better use
of the same compute budget here specifically (unlike the main line, whose own trajectory and
Claws's own turn-branch training data both still want the higher-quality adopted play).

Options run SEQUENTIALLY now (_resolve_meta_decision), NOT concurrently on threads the way an
earlier version of this module did. That earlier concurrency was safe specifically because each
chain's Sts2Session was its own OS subprocess with its own private results_queue -- provably
independent. Sts2BufferSession has no such isolation: every session in a process shares the one
CoreCLR clr_bridge.load_clr() loads into it (see buffer_env.py), and
debug_repro_branch_divergence.py already caught RunSimulator's underlying save/profile state
racing across merely-concurrent *processes* (each with its own independent CoreCLR, sharing only
an unlocked on-disk profile path) -- confirmed there to cause real RNG/state divergence from the
very first decision replayed. Multiple BufferSessions concurrently alive on threads INSIDE one
process, sharing that same loaded CoreCLR directly, would be at least as exposed to that same
class of bug, not less -- so chains here are evaluated one at a time, trading away the
wall-clock parallelism the old concurrent design had in exchange for a combat layer that no
longer needs an OS subprocess (or the shared branch-worker pool) per chain at all, which is by
far the bigger cost this rewrite removes.

Design (Change of strategy from Phase 1's M-replicate, fixed-horizon-then-bootstrap scheme):
- One chain per legal option, no replicate averaging (M=1, by design -- see the README's
  "Original meaning of replicate" discussion; the noise that used to justify M replicates on a raw
  single-turn rollout is instead absorbed by chain VOLUME now -- see this module's docstring on
  why unbranched-but-many beats branched-but-few for this specific training signal).
- Each option-branch commits its card, then plays forward using the ordinary self-play policy
  (map routing, Claws' Dirichlet-noised policy for combat, the existing choices_paradox tier-list
  heuristic for anything else) -- nothing else is allowed to diverge between sibling
  option-branches, so any difference in outcome is attributable to the one choice being compared,
  same principle Phase 1 already established. Enforced concretely via a shared walk_seed
  (_resolve_meta_decision draws one per divergence, common to every sibling): reseeds Python's
  global `random` and torch's global RNG identically at the start of every sibling's post-commit
  walk, so their policy draws (which map node, which Dirichlet-sampled card, tie-breaks) are
  bit-identical for as long as their game state hasn't actually diverged yet -- "common random
  numbers," a standard variance-reduction technique for exactly this kind of paired counterfactual
  comparison. This does NOT and cannot force identical downstream content (which cards a reward
  offers, monster picks) once real divergence begins -- that's the engine's OWN internal RNG,
  which necessarily drifts once sibling chains have taken different actions (different cards
  played consume it differently) -- and that drift IS the causal signal this module measures, not
  noise to eliminate. See _evaluate_option_chain's docstring for the full mechanism and why
  reseeding the global state (rather than threading an explicit Generator) is safe here.
- The divergence itself is reconstructed via the pre-combat native checkpoint main_worker_loop
  already took before the combat that produced this card_reward (last_combat_checkpoint_path, see
  its docstring), replaying just that ONE combat's own tail of actions via
  load_and_resume_combat to land back on the reward -- NOT a fresh checkpoint taken at the
  pending card_reward decision itself: confirmed by hand that doesn't work
  (RunSimulator.SaveCheckpoint rolls back to a pre-room map state whenever the current room isn't
  a MapRoom -- TryRollbackSerializedSaveToPreRoom, sts2-cli's RunSimulator.SaveLoad.cs -- so a
  checkpoint taken mid-card_reward reloads PAST it, at map_select, with no pending-reward signal
  left to recover; the save format has no representation of "mid-room" state outside MapRoom at
  all). This is now the ONLY checkpoint/reconstruction paid per chain -- unlike the branched
  predecessor design, a chain never re-checkpoints or reconstructs itself again after this one
  reconstruction, since there's no per-turn resync or branch-worker fast-load path left to feed;
  it just keeps stepping its own live Sts2BufferSession forward to natural death.
- No fixed horizon, no TD-bootstrap off Claws's value head -- a chain plays until it actually
  loses. Value formula: _WIN_BONUS, plus that combat's own _combat_value (the same HP-fraction/
  boss-shaped quality-of-win term facility_common.py already uses elsewhere), for every combat WON
  along the chain, summed -- EXCEPT is_last_of_act is always False for a chain's own combats now:
  that flag came from the main line's live pydantic `context.room_type` in the branched
  predecessor design (see checkpoint.py's docstring on why the buffer protocol alone can't
  recover it -- boss1_token/boss2_token are per-act constants, not "is THIS room the boss"), and a
  chain has no pydantic Decision anywhere in its walk anymore to read that from. Accepted
  simplification, not a considered decision to keep it that way forever: a chain-internal boss
  win is scored with the ordinary HP-fraction formula instead of the boss-shaped one, a smaller
  value-shaping error than the noise already accepted above.
- _WIN_BONUS started at 1.0 and was dropped to 0.5 after the first live smoke test: at 1.0, "how
  many combats did this chain survive" (an integer count) dominated "how well did this pick's
  presence shape each fight" (_combat_value, typically ~0.1-0.5) -- three genuinely different
  card picks landed within ~0.2 of each other while skipping landed ~1.1 lower, i.e. the signal
  that's actually supposed to differentiate options was getting drowned out by combat-count
  noise. Retune again if trained output still looks survival-count-dominated rather than
  pick-quality-dominated. No terminal -1 for the eventual loss -- every chain loses eventually by
  construction, so a flat penalty would just be a constant materialize.py's own group-baseline
  subtraction (_group_baseline) already cancels out; a chain that dies after 2 combats naturally
  scores lower than one that dies after 8 without it.
- No recursive meta-branching for any card_reward a chain encounters past its own initial commit
  -- resolved by a single self-play pick like any other decision, same as main_worker_loop's own
  behavior whenever meta_resolver isn't invoked. Combinatorial cost is exactly why (options^depth
  branching), and this was a deliberate choice, not a placeholder.
- The trial that actually continues to the next divergence does NOT adopt whichever option scored
  best. It rewinds to the pre-divergence state (the live session never left it -- every option's
  chain runs on its OWN forked session, never grafted onto the main line) and commits via
  ChoicesParadoxNet's own softmax-plus-Dirichlet-noise prediction instead (see
  _choose_meta_option_claws) -- keeps the real trial's trajectory on-policy rather than either
  blind-random or coupled to whichever option an under-trained net happened to explore best this
  round, same "own predict + Dirichlet noise, to avoid a blind spot" reasoning combat's own
  adoption already follows.

Output format is deliberately unchanged from Phase 1 (role="branch", branch_kind="meta", one
value attached to the commit action via recorder.mark_step()/set_value()) -- one file per option
instead of per (option, replicate) is exactly the M=1 degenerate case _group_baseline's own "grand
mean equals per-option-mean-of-equal-size-groups" math already handles, so materialize.py needs no
changes to consume this scheme's output.
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
# own _combat_value -- see this module's docstring for why 0.5 (not the original 1.0): at 1.0,
# raw survival depth swamped the actual pick-quality signal _combat_value is meant to carry.
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
    -- falls back to the existing tier-list heuristic (via choose_action, ignoring the chain
    values entirely) when no checkpoint is loaded yet, same "noisy but harmless" round-1 fallback
    Claws's own combat picker already follows."""
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
    in-process Sts2BufferSession held open for the chain's entire walk -- no turn-branching, no
    branch-worker dispatch, no per-turn session resync (see this module's docstring for why that
    changed from an earlier turn-branch-and-adopt design: that version's per-turn
    close-and-respawn-a-subprocess resync, multiplied by every concurrently-running chain, was the
    actual bottleneck). Terminates on the first loss; returns the accumulated value (see this
    module's docstring for the formula), or None if this chain errored before ever reaching a
    terminal state.

    walk_seed: the SAME value for every sibling option of this divergence (drawn once by
    _resolve_meta_decision) -- reseeds Python's global `random` and torch's global RNG right after
    committing (see below), so every sibling chain's post-commit policy draws (map routing via
    policy.py's random.choice, Claws's Dirichlet-noised combat sampling, choices_paradox's own
    tie-breaking) start from an identical stream. This is "common random numbers" variance
    reduction, not a correctness requirement: without it, sibling chains diverge from arbitrary
    RNG noise (which map node got picked, which of two equal-strength cards got sampled) that has
    nothing to do with the option actually being compared, inflating the value difference's
    variance and partly evening out the very effect this module exists to measure. Only removes
    THAT kind of noise, not genuine divergence -- once a chain's own state actually differs from
    its siblings' (a different card in the deck changes what's drawn/playable, which consumes the
    game engine's OWN internal RNG differently), the two streams necessarily desync from there;
    that's the causal signal being measured, not something to fight. Deliberately reseeds the
    GLOBAL random/torch state rather than threading an explicit Generator/random.Random through
    every choose_action call site -- safe specifically because _resolve_meta_decision now runs
    every option-chain SEQUENTIALLY (see that function's docstring on why concurrent
    Sts2BufferSession chains aren't safe to begin with), so there's no other chain's RNG state to
    clobber mid-walk.

    Runs on its OWN Sts2BufferSession -- never touches or is grafted onto the live main-line
    session the caller holds, mirroring Phase 1's own "no replicate's downstream play is ever
    grafted onto the main line" invariant. Reconstructed via
    load_and_resume_combat(last_combat_checkpoint_path, ...) -- replaying just the tail of actions
    for the ONE combat that produced this card_reward decision -- rather than open_from_checkpoint
    replaying the entire run from StartRun, or a fresh checkpoint taken directly at the pending
    card_reward decision (confirmed NOT to work: RunSimulator's save/load_save doesn't preserve a
    pending decision screen -- see main_worker_loop's last_combat_checkpoint_path docstring for
    both). Falls back to the full open_from_checkpoint(prefix) replay when no combat checkpoint
    was available (last_combat_checkpoint_path is None -- vanishingly rare, see that same
    docstring). Either way, the reconstructed session has no memory of anything before the
    checkpoint/replay point, so `prefix` is spliced back onto session.action_log afterward --
    needed purely so CompactRecorder's start_step=len(prefix) indexing (set below) lines up with
    the action_log this chain's own recorder.save() is eventually handed. This is the chain's ONLY
    reconstruction -- unlike the branched predecessor design, nothing re-checkpoints or
    reconstructs this session again after this point; it just keeps stepping the same session
    forward, turn by turn and decision by decision, to natural death.

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
                    # facility_common._play_turn_only, looped until the combat itself resolves --
                    # no branch-worker dispatch, no session resync. bypass_reward_on_win=False
                    # keeps the terminal card_reward real (not bypassed) so it can be resolved for
                    # real via choose_action below, same as facility_common's own docstring on
                    # that flag describes for a meta-branch's horizon walk.
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
                outcome = "won" if decision.victory else "lost"  # reached game_over directly
                # (e.g. beat the final boss) rather than via a combat "lost" -- rare but not
                # impossible, and outcome=="won" here still correctly skips set_value below since
                # total_value has already accumulated every combat this chain actually won.
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
    at a time, then commits for real via ChoicesParadoxNet's own prediction
    (_choose_meta_option_claws) -- not necessarily whichever option scored best. Falls back to the
    plain heuristic pick (what choose_action would have done anyway) if fewer than 2 options exist
    to compare, or if every chain errored -- forward progress on the main line matters more than a
    missed training signal for this one decision.

    SEQUENTIAL, not concurrent -- an earlier version of this function ran every option's chain on
    its own thread. That was safe specifically because each chain's Sts2Session was its own OS
    subprocess with its own private results_queue -- provably independent. Since
    _evaluate_option_chain now runs each chain on an in-process Sts2BufferSession instead (see
    this module's docstring on why), concurrent chains would mean multiple BufferSessions alive at
    once sharing the one CoreCLR this process loaded -- exactly the shape of race
    debug_repro_branch_divergence.py already caught (RunSimulator's save/profile state diverging
    across merely-concurrent processes sharing an unlocked on-disk path; threads sharing the same
    in-process CLR are at least as exposed). job_queue/results_queue/main_worker_index/stop_event/
    queue_manager are no longer used by this function's own body -- kept only because
    main_worker_loop's meta_resolver contract calls every meta_resolver with this same positional
    signature (see its docstring), and facility_claws.py's combat_resolver still uses job_queue/
    results_queue/stop_event for the branch-worker pool the MAIN LINE's own combat still uses.

    last_combat_checkpoint_path/last_combat_entry_index (from main_worker_loop, see its docstring):
    the native checkpoint for the combat that produced this card_reward, and that combat's entry
    action's position in `prefix` below. Passed straight through to every option-chain, which
    reconstructs its own starting state by replaying just that one combat's tail of actions on top
    of it (_evaluate_option_chain's own load_and_resume_combat call) -- deliberately NOT a fresh
    checkpoint taken here at the pending card_reward decision itself: confirmed by hand that
    doesn't work (RunSimulator.SaveCheckpoint rolls back to a pre-room map state whenever the
    current room isn't a MapRoom -- see TryRollbackSerializedSaveToPreRoom in sts2-cli's
    RunSimulator.SaveLoad.cs -- so a checkpoint taken mid-card_reward reloads past it, at
    map_select, with no pending-reward signal left to recover; there's nothing to poll because the
    save format has no representation of "mid-room"/"room just finished" state outside MapRoom at
    all). Reusing the pre-combat checkpoint main_worker_loop already takes is the same trick
    facility_claws.py's own combat-internal fast path relies on for the exact same reason."""
    options = legal_options(decision, decision_kind)
    if len(options) < 2:
        if not options:
            return decision  # defensive -- card_reward always offers skip_card_reward at least
        name, args = options[0]
        return session.step(name, **args)

    prefix = list(session.action_log)
    # One shared walk_seed for every sibling option of THIS divergence -- see
    # _evaluate_option_chain's docstring on why (common random numbers: keeps sibling chains'
    # post-commit policy draws identical wherever their state hasn't actually diverged yet, so the
    # measured value difference reflects the option itself, not arbitrary map/combat-sampling
    # noise). Drawn fresh per divergence, not per trial -- a fixed cross-divergence seed would
    # make every divergence's chains replay the identical "random" choices too, which isn't
    # wanted; only siblings of the SAME divergence should share one.
    walk_seed = random.getrandbits(64)
    # One chain at a time -- see this function's docstring on why concurrent BufferSession-backed
    # chains aren't safe (unlike the Sts2Session-per-chain design this replaced).
    values: list[float | None] = [
        _evaluate_option_chain(
            cfg, character, seed, cfg.ascension, main_id, decision_kind, divergence_index,
            option_index, prefix, last_combat_checkpoint_path, last_combat_entry_index, name, args,
            walk_seed,
        )
        for option_index, (name, args) in enumerate(options)
    ]
    # Every chain above deliberately reseeded random/torch to the SAME walk_seed (see
    # _evaluate_option_chain's docstring) -- restore unpredictable global state now, before
    # _choose_meta_option_claws's own Dirichlet sampling below (the REAL adoption draw) and
    # before control returns to the main line's own subsequent play, both of which use these
    # same global random/torch streams and should NOT stay correlated with walk_seed.
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

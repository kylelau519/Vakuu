# Golden Compass model

Design doc for training Golden Compass the way `claws/model` and `choices_paradox/model` train
their own subsystems -- self-play, not supervised learning -- adapted for map_select's own
decision shape. This is the **second** design (see git history for the first, whole-trajectory
terminal-reward one) -- replaced once it became clear the terminal signal was degenerate at the
self-play facility's actual win rate. See "Why the first design was replaced" below before
assuming anything about a `+1`/`-1` terminal tag or a `combination`-pooled action space; neither
exists anymore.

## Why the first design was replaced

The original design tagged every `select_map_node` step on a trial's own main line with that
trial's own terminal outcome (`+1` win / `-1` death), plus a local Elite/Boss-win shaping bonus.
That's a fine idea in principle, but it turned out to be nearly useless in practice: at the self-
play facility's actual measured win rate (0% across every round sampled in production so far --
Claws isn't good enough at combat yet to win a run), a trial dies whether it took the safe route or
the risky one, so the terminal tag carries almost no information about path *quality* -- it's
±1 regardless of the choice. Loaded the live checkpoint trained under that design and replayed a
real trial through it: the policy had collapsed to near-one-hot (entropy ~0.000-0.1 nats out of a
max of ~1.4) at essentially every decision, and the value head had converged to predicting roughly
the dataset's mean outcome (~-0.8) everywhere, independent of state -- both symptoms of a REINFORCE
signal with real variance concentrated in a handful of rare outlier trajectories (an Elite win here
and there) dominating training, rather than a stable win/loss distinction tied to the actual
choice made.

Separately (found while diagnosing the above): the **live main line's map_select decisions were
never actually driven by any of this** -- `whispering_earring/policy.py`'s buffer-side dispatch
(`_choose_map_action_from_buffer`) called `random.choice` unconditionally, never consulting
`golden_compass/policy.py`'s DP heuristic or a trained net, because both `facility_claws.py`'s and
`facility_choices_paradox.py`'s main lines run on `Sts2BufferSession`, not the plain `Sts2Session`
the pydantic-side `_choose_map_action` was written against. Fixed as part of this redesign --
`_choose_map_action_from_buffer` now delegates to `golden_compass.choose_next_node_from_buffer`.

## Design: per-choice branching, not whole-trajectory tagging

Map decisions are now scored the same way Choices Paradox scores its own card_reward/rest_site/
event_choice decisions: branch every sibling option at the decision point (`MapOptionJob`,
`checkpoint.py`), play each one forward independently on the shared branch-worker pool
(`facility_common.py`'s `_resolve_map_decision`/`_run_map_option_job`), and score the walk
directly -- no whole-run outcome required. This is available and meaningful regardless of the
underlying win rate, since it's a *local* comparison between siblings from an identical starting
state, not "did the whole run eventually win."

**Reward**: each `MapOptionJob` plays forward `cfg.map_room_horizon` further `select_map_node`
decisions past its own commit (default 1 -- resolve the committed sibling's own room, whatever it
is, then stop the instant the walk is back at the map for a further junction) and scores:

- **`-1.0`** if the walk died within that horizon.
- **Otherwise, `_combat_value`'s hp-fraction-shaped read of the walk's HP at the cutoff** --
  `facility_common.py`'s existing >=50%-HP-vs-below-50%-HP-with-a-3x-penalty-slope shaping,
  already used elsewhere for per-turn/per-combat scoring, reused here as-is.

This is deliberately **not** a flat per-combat-win bonus (`_run_meta_option_job`'s own
`_WIN_BONUS` scheme for card_reward chains) despite being otherwise structurally identical to
that job. A card_reward pick can't affect whether a fight happens at all, so rewarding a chain for
every combat it happens to win along the way is a fine, non-gameable signal there. A map choice's
entire *job* is deciding whether to walk into a fight -- a flat win bonus would reward packing in
extra optional risk (chase an additional winnable fight, net a higher flat bonus) with no matching
downside, exactly the "chase risky Elites" failure mode the first design's flat terminal term was
built to guard against, just relocated to a local horizon instead of the whole trajectory. Scoring
by terminal HP fraction instead means an extra fight only pays off if the walk is still healthier
for it, and a route through a Rest Site (or around an optional Monster) scores on its own HP
merits -- no special-cased "avoid combat" term needed.

**Baseline**: a decision's siblings are dispatched with a shared `walk_seed` (common random
numbers, same variance-reduction `_run_meta_option_job`'s chains already use), and
`golden_compass/model/materialize.py` baselines each sibling's value against its own group's mean
-- a real causal comparison from an identical starting state, not the whole-trajectory design's
un-baselined raw terminal tag. There's no natural "skip" reference option the way card_reward has
`skip_card_reward` (every physical map child is a real move), so this uses the group mean rather
than choices_paradox's own skip-relative baseline.

## Architecture: scoring physical nodes directly

`GoldenCompassNet` (`model.py`) scores `map_select`'s own one-hop `choices` **directly** -- one
token, one logit, per real child coordinate (`legal_options`/`legal_options_from_buffer`). The
first design pooled every path reachable within a 5-floor lookahead into per-room-type-count
histograms and scored the histograms instead, specifically to give the net some visibility past
the immediate node without branching. That's no longer needed now that map decisions are branched:
the branch walk itself *is* the lookahead, played out for real against the live combat/meta policy
stack, so the histogram encoding just threw away signal for no remaining benefit -- worse, it
actively discarded exactly the HP-vs-risk tradeoff that matters most for judging a path (an Elite
at 90% HP and the same Elite at 20% HP produced an identical token under the old scheme, though the
old encoding wasn't why the trained policy collapsed -- see "Why the first design was replaced"
above; that was the reward, not the representation. Still worth fixing regardless).

Both a pydantic (`tokenize`/`legal_options`, off `MapSelect`) and a buffer-protocol
(`tokenize_from_buffer`/`legal_options_from_buffer`, off `BufferDecision`/`buffer_layout`) path
exist -- `forward()` dispatches on `isinstance(decision, BufferDecision)`, same convention
`ChoicesParadoxNet.forward` uses. This closes the "Phase 1" gap the first design's own Phasing
section flagged (no buffer-side tokenizer existed at all, so `_choose_map_action_from_buffer`
"currently pure-random" had nowhere to delegate to even once fixed) as part of this same redesign.

Value head has **no `tanh`** (matching `ChoicesParadoxNet`'s own convention, unlike the first
design's `tanh`-bounded head): the training target is a `MapOptionJob` walk's own terminal score
(roughly `[-1, 1]`-ish but not hard-bounded there), not a fixed `+1`/`-1`. `tanh`'s asymptotic
squash bought nothing here beyond saturated gradients whenever a target strayed outside its range,
which the first design's shaped terminal reward did routinely (confirmed: `value_loss` stuck
around 2.7-4.35 for dozens of epochs in production logs).

## Data

`golden_compass/model/materialize.py` reads `MapOptionJob` branch files directly (grouped by
`(main_id, divergence_index)`, both recorded explicitly in each branch file's own metadata --
no prefix-action matching needed the way `choices_paradox/model/materialize.py`'s own grouping
does, since its meta-branch files predate carrying `divergence_index` explicitly). No
main-trial-fusion optimization yet (`choices_paradox/model/materialize.py`'s
`materialize_main_and_groups` counterpart) -- every group replays its own recorded prefix once
from scratch, a deliberate v1 tradeoff; worth revisiting if materialize throughput becomes a
bottleneck.

Map branching is **off by default** (`--map-branches-per-trial 0` in both `facility_claws.py` and
`facility_choices_paradox.py`) -- pass `--map-branches-per-trial -1` (unlimited) or a positive cap,
plus `--map-room-horizon` to adjust the walk depth, to actually generate `MapOptionJob` training
data. The main line still commits via `golden_compass.choose_next_node_from_buffer`'s DP heuristic
either way (see below) -- branching only adds the training signal on top, it doesn't change what
the main line itself does when no trained net is loaded.

## What drives the live main line today

No trained net is wired into the live buffer path yet -- `_resolve_map_decision` adopts via
`golden_compass/policy.py`'s DP heuristic (`choose_next_node`/`choose_next_node_from_buffer`),
same "reasonable, not trained" bootstrap every other subsystem here starts from. Once
`MapOptionJob` data trains a checkpoint under this package's simplified one-hop action space,
`_resolve_map_decision` (and `whispering_earring/policy.py`'s `_choose_map_action`/
`_choose_map_action_net` for the pydantic path) are the places that need to change to sample from
it instead -- `GOLDEN_COMPASS_CHECKPOINT` is already threaded through
`training_alternating.sh`/`policy.py`'s `_load_golden_compass_net`, but nothing currently points
the live buffer-side decision at it.

**Do not warm-start a fresh training run from an old `golden_compass.pt`/its backups** -- those
were trained under the previous combination-pooled action space and whole-trajectory reward; the
state_dict shapes happen to still match (same token types, same `FEATURE_DIM`), so `train.py --init`
would silently *load* an old checkpoint, but the OPTION token's meaning changed (one physical node,
not a pooled histogram) and the reward scale changed too -- start from `--init ""` instead.

## Open questions

- No main-trial-fusion replay path yet for `materialize.py` (see Data above) -- purely a throughput
  concern, not a correctness one.
- `map_room_horizon`'s default (1) is an untested first guess, not measured against training
  outcomes yet -- worth revisiting once there's enough branch data to tell whether a deeper walk
  (2+ rooms) meaningfully changes what the net learns to prefer.
- Should Golden Compass eventually consume Claws' own value head as an input feature (a
  win-probability estimate) rather than only relic/deck/HP counts? Deferred -- see
  `agents/README.md`'s note on the three subsystems potentially converging.
- Once Claws' win rate is meaningfully above 0%, is it worth blending a small whole-trajectory
  terminal term back in on top of the branch-derived signal, so Golden Compass also learns to
  value "paths that historically led to actual wins," not just locally healthy ones? The
  `pending_map_steps`/`_apply_golden_compass_rewards` machinery from the first design is still
  intact in `facility_common.py` (it just naturally stops firing once `map_resolver` handles a
  trial's map decisions instead of the generic path) if this turns out to be worth reviving.

## Phasing

- **Phase 0 (superseded)**: whole-trajectory terminal-reward tagging, combination-pooled action
  space, no branching. See git history.
- **Phase 1 (this)**: per-choice `MapOptionJob` branching, causal group-mean baseline, one-hop
  action space (pydantic + buffer), buffer-side main-line fix (DP heuristic instead of
  `random.choice`). No trained net wired into the live path yet.
- **Phase 2**: wire a trained checkpoint into `_resolve_map_decision`'s live adoption (sampling
  from the net instead of always the DP heuristic), same "upgrade once a checkpoint exists" pattern
  `_choose_card_reward_decision`/`_load_choices_paradox_net` already follow.

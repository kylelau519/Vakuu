# Golden Compass model

Design doc for training Golden Compass the way `claws/model` and `choices_paradox/model` train
their own subsystems -- self-play, not supervised learning -- adapted for map_select's own
terminal, whole-run reward (now with a local Elite/Boss-win shaping bonus layered on top, see
below) rather than a purely local/shaped one.

## Why self-play, and why this reward

`agents/golden_compass/policy.py`'s DP heuristic already routes reasonably around HP and node
type, but it's static: no relic/deck awareness, a hard elite-HP cliff, and no way to improve from
experience. The reward that actually matters for a map choice is whether the *run* is eventually
won -- so training targets the run's own terminal outcome directly: **+1 for winning the run, -1
for dying, and no example at all for a step whose trial ended in `"error"`** (no signal either
way, not a 0 -- see `materialize.py`).

That flat terminal tag alone is a weak signal, though: a decision at floor 3 gets the exact same
±1 as one at floor 49, so credit assignment is smeared across the whole run instead of pointing at
what a choice actually led to. `facility_common.py`'s `main_worker_loop` layers a local shaping
bonus on top -- **+2 for a won Elite fight, +3 for winning the boss** -- backward-discounted
across `select_map_node` steps (`_GOLDEN_COMPASS_GAMMA`, ~0.9, a ~6.6-decision half-life) so a
choice a few floors before a win gets a real nudge instead of a flat fraction of credit. The flat
±1 term is kept, not replaced: without it, a trial that beats a couple of Elites (+2 each) and
then still dies (-1) would net positive, which would train the policy to chase risky Elites over
safe clears. This is standard potential-based reward shaping -- densifying the signal, not
overriding the underlying win/loss objective.

## Design: whole-trajectory tagging, no branching

Choices Paradox needs `MetaOptionJob`-style branching (replay the same prefix under each sibling
option) because a card pick's value depends on downstream deckbuilding that can't be read off the
pick itself. A map choice's *true* value is mostly "did this run end up won" -- already known,
exactly once, at the end of the very same trial that made the choice -- plus the Elite/Boss
shaping bonus described above, which is also fully known by trial end (every combat this trial
ever fought has already resolved). So Golden Compass needs no branching at all: every
`select_map_node` decision on a trial's own main line gets tagged with its value once it's known,
via `recorder.mark_step()` at the decision and `recorder.set_value()` at trial end (see
`facility_common.py`'s `pending_map_steps`). This also means training data comes for free from
*any* facility run (`facility_claws.py`, `facility_choices_paradox.py`, ...) -- no dedicated
Golden Compass facility script needed, and even round-1 heuristic-driven trials produce usable
examples (see `materialize.py`'s module docstring).

The tradeoff: no counterfactual baseline to subtract (no sibling branches to compare against), so
`advantage` is just the realized `+1`/`-1` terminal outcome plus the discounted shaping bonus,
normalized (zero-mean/unit-std) across the dataset by `train.py` -- plain REINFORCE on a shaped
trajectory return, not advantage-over-baseline. Noisier than Choices Paradox's causal-baseline
signal, but requires no new async job infrastructure and is a reasonable place to start; the
shaping bonus is itself a partial answer to that noise (see the reward section above).

## Architecture: scoring combinations, not nodes

`GoldenCompassNet` (`model.py`) does **not** score `map_select`'s physical one-hop `choices`
directly. For every path reachable within `_LOOKAHEAD_HORIZON` (5) floors -- pooled across ALL of
`choices`' own branches, not kept separate per branch -- it computes that path's own per-room-type
count histogram (`_path_combination`, walking the same full `map` graph
`golden_compass/policy.py`'s DP heuristic reads via `session.get_map()`), then groups every path
that produces the identical histogram into one candidate (`combination_options`). Two paths
through completely different physical nodes that happen to pass the same rooms (2 Elites + a
Shop, say) are literally the same candidate to this net; a network with only 2 physical branches
can end up choosing among 3 candidates if one branch forks internally into two different
histograms, or among 1 if both branches happen to lead to the same combination. Order/exact
distance within a path is thrown away on purpose -- a plain per-type count, not per-node
distance -- trading that signal for a representation simple enough for this net's still-small
dataset to actually learn from.

Once a combination is sampled from the softmax, it's resolved back to a real engine action by
picking any ONE of the (possibly several) paths that produced it (`resolve_combination_action`)
and walking only that path's first hop -- the rest of the path is discarded. This is re-planned
from scratch at the very next `map_select`, not committed to upfront: `facility_common.py`'s
`main_worker_loop` already calls into policy selection fresh at every `map_select` on its own, so
"walk one hop, replan" falls out of the existing control flow for free. Because more than one
combination can share a first hop, the resolved `(col, row)` alone doesn't tell you which
combination was actually picked -- `whispering_earring/policy.py`'s `_choose_map_action_net`
stashes the sampled index as `policy["chosen"]` (alongside the full distribution) specifically so
`materialize.py` can recover the real label when replaying a saved trial, rather than re-deriving
it ambiguously from the taken coordinate.

Value head is `tanh`-bounded (matching `ClawsNet`'s convention), unlike `ChoicesParadoxNet`'s
unbounded one -- the training target here is a `+1`/`-1` terminal outcome (now with a shaping
bonus added on top, so its magnitude can exceed 1 -- see the reward section above), not a growing
sum. A target outside `[-1, 1]` just pushes the head further into saturation than a pure `+1`/`-1`
target would; still fine for a policy-gradient sign/magnitude signal (see Open questions below).

## Data

Pydantic-only: `combination_options()`/`tokenize()` work off `MapSelect`/`MapChoice` and
`session.get_map()` directly. There is no buffer-side counterpart yet, so `materialize.py` replays
trial JSON through the plain pydantic `Sts2Session`, not the faster `Sts2BufferSession`
claws'/choices_paradox's own materialize scripts use. Slower per trial; a deliberate v1 tradeoff
(see Phasing). Replaying a saved trial recomputes `combination_options()` fresh against the same
decision/map -- its key order is deterministic, so an index recorded live (`policy["chosen"]`)
still lines up correctly against the recomputed list (see `materialize.py`'s `_action_index`).

## Open questions

- No counterfactual baseline (see Design above) means higher-variance policy gradients than
  Choices Paradox's causal-baseline signal. Worth watching whether training is noisy enough in
  practice to warrant adding `MetaOptionJob`-style per-choice branching later.
- `tanh`'s value target is exactly `+1`/`-1` at the edges of its range, which the MLP can only
  approach asymptotically -- fine for a policy-gradient sign/magnitude signal, but means the raw
  value head output should be read as "how confident," not literally saturating.
- Should Golden Compass eventually consume Claws' own value head as an input feature (a
  win-probability estimate) rather than only relic/deck/HP counts? Deferred -- see
  `agents/README.md`'s note on the three subsystems potentially converging.
- `_GOLDEN_COMPASS_GAMMA`/`_GOLDEN_COMPASS_ELITE_BONUS`/`_GOLDEN_COMPASS_BOSS_BONUS`
  (`facility_common.py`) are untuned first guesses, not measured against training outcomes yet --
  worth revisiting once there's enough round-over-round data to tell whether the shaping bonus is
  actually reducing policy-gradient variance in practice.
- Round-1 (heuristic-driven) steps have no recorded `policy["chosen"]`, so `materialize.py`'s
  `_action_index` falls back to the first combination (in `combination_options()`'s own
  deterministic key order) whose path set starts at the taken `(col, row)` -- an arbitrary but
  consistent tie-break when more than one combination shares that first hop, since the DP
  heuristic never reasoned in terms of combinations to begin with. Only matters for round-1
  bootstrap data; once a net is training itself, every step it drives carries the real index.

## Phasing

- **Phase 0 (this)**: whole-trajectory terminal-reward tagging on the main line only, pydantic-only
  tokenizer, no branching, no buffer-side replay path, no warm-start.
- **Phase 1**: buffer-side `tokenize_from_buffer`/`legal_options_from_buffer`, so a `MetaOptionJob`
  chain's own horizon walk (`_choose_map_action_from_buffer`, currently pure-random) can delegate
  to a trained net too, instead of adding pure noise to every chain that happens to cross a
  map_select.
  Also gives `materialize.py` a fast replay path.
- **Phase 2**: if Phase 0's no-baseline variance turns out to matter, add `MetaOptionJob`-style
  per-choice branching (map_select added to `META_ELIGIBLE_DECISIONS`) for a causal baseline, same
  shape as Choices Paradox's own.

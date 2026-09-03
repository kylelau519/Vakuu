# Model

Design doc for training Choices Paradox the way [`claws/model`](../../claws/model/) trains Claws --
self-play branch-and-score, not supervised learning on scraped human run data. **Phase 0 and Phase
1 are implemented** (`policy.py`'s purpose-threading refactor, and card_reward meta-branching via
[`whispering_earring/facility_choices_paradox.py`](../../whispering_earring/facility_choices_paradox.py),
validated end to end). Phase 2 (`card_select`) and Phase 3 (`shop`/`rest_site`/`event_choice`) are
not -- see Phasing below. Update this file as design decisions change.

## Why this exists

The working hypothesis for the Act 1 boss struggle: it may not be a combat-skill problem at all.
`choices_paradox/policy.py`'s current card-reward/removal/upgrade picker is a static heuristic --
greedy tier-list, zero synergy or curve awareness, zero connection to whether picks actually
correlate with winning. A combat-skilled Claws still can't win with a deck that heuristic can't
build well.

## Why self-play instead of supervised learning on human data

STS2 is too young and complex a game for scraped human runs to be a sufficient dataset, and even
with enough data, training on human outcomes optimizes for what wins *in human hands*, not
Claws's. Self-play sidesteps both: unlimited data, and the label answers the question that
actually matters -- does this pick win under the policy actually being deployed.

## Design: branch-and-score at meta decisions

Mirrors `whispering_earring`'s combat turn-branch-and-adopt one level up: at a branchable
decision (currently only `card_reward`), branch across every legal option (e.g. 3 offered cards +
skip). One chain per option, no replicate averaging (M=1 -- the noise that used to justify
replicating a raw single-turn rollout is largely absorbed now that every combat inside a chain
gets full turn-branch-and-adopt quality, not a single noisy rollout).

Each option-branch commits its card, then plays forward with the ordinary self-play policy
(Golden Compass for map_select, Claws for combat, the existing tier-list heuristic for anything
else) until it actually loses -- no fixed horizon, no TD-bootstrap. Value = `_WIN_BONUS` (0.5,
tuned down from 1.0 after a live smoke test showed raw survival-count swamping the actual
pick-quality signal) plus that combat's own `_combat_value`, summed over every combat won along
the chain. No terminal penalty for the eventual loss -- every chain loses eventually by
construction, so a flat penalty would just be a constant the group-baseline subtraction already
cancels out.

Options run concurrently, each on its own thread and its own forked `Sts2Session` (own OS
subprocess, own private results queue) -- safe because a chain never touches the live main-line
session or another chain's state. Each chain reuses `facility_claws`'s shared branch-worker pool
for its own combats (same `_resolve_combat`) rather than a single unbranched rollout or a
dedicated worker pool. A native checkpoint is taken before every combat entry along a chain
(confirmed safe to chain repeatedly, not just faithful for one hop). A chain reconstructs its
starting point from the main line's own pre-combat checkpoint -- not a fresh checkpoint at the
pending card_reward decision itself, since STS2's save format has no "mid-room" representation to
reload into -- by replaying just that one combat's tail of actions.

No recursive meta-branching inside a chain: any card_reward a chain hits past its own initial
commit resolves via a single self-play pick, same as the main line whenever meta-branching isn't
invoked. Combinatorial cost (`options^depth`) is why.

The real trial does **not** adopt whichever option scored best. It rewinds to the pre-divergence
state (never actually left, since every chain runs on its own forked session) and commits via
`ChoicesParadoxNet`'s own softmax-plus-Dirichlet-noise prediction instead -- keeps the trial
on-policy rather than coupled to whichever option an under-trained net happened to explore best
this round. Falls back to the tier-list heuristic until a checkpoint (`CHOICES_PARADOX_CHECKPOINT`)
exists.

Out of scope for now: `shop`/`rest_site`/`event_choice` (their buffer-protocol representations are
missing semantic fields a branch worker would need), and boss-fight combats are weighted equally
with normal ones inside a chain (accepted v1 simplification).

## Architecture

Same shape as [`claws/model/model.py`](../../claws/model/model.py)'s `ClawsNet`: typed-token
transformer trunk -> per-option query readout -> shared `policy_mlp`/`value_mlp` heads, matching
`dim`/`n_layers`/`n_heads` so `ChoicesParadoxNet.warm_started_from` can copy `ClawsNet.id_emb`
rows without passing matching kwargs by hand.

- **Token types**: `PLAYER`, `MODE`, `DECK_CARD`, `OFFER_CARD`, `RELIC`. `MODE` is a dedicated
  always-present pseudo-token keyed by `decision_kind`, sized at 3 rows from the start
  (`card_reward`, `card_select_remove`, `card_select_upgrade`) even though only `card_reward`
  ships first, so Phase 2 doesn't need a checkpoint-breaking resize.
- **`DECK_CARD` vs `OFFER_CARD`**: every card in `player.deck` gets a token (full-deck
  synergy/curve context the tier-list heuristic can't see); offered/candidate cards get
  `OFFER_CARD` tokens, both sharing the same card-id embedding space.
- **`legal_options(decision, decision_kind)`** / **`legal_options_from_buffer(...)`**: fully
  pre-expanded option list, following `claws/model/model.py`'s template -- `card_reward` yields
  one `select_card_reward` per offered card plus `skip_card_reward` if `can_skip`; `card_select_*`
  (Phase 2) yields one `select_cards` per offered card plus `skip_select`.
- **`score(batch, options)`**: each card option's query is that card's own contextualized
  `OFFER_CARD` hidden state; `skip_*`'s query is a dedicated learned parameter (`skip_query`),
  mirroring `ClawsNet`'s `end_turn_query`.
- **`forward(decision, decision_kind)`** dispatches pydantic (`CardReward`/`CardSelect`) vs.
  `BufferDecision` the same way `ClawsNet.forward` does.

Separate network from `ClawsNet` (not shared): deckbuilding and in-combat attention are different
enough questions that a shared trunk risks negative transfer, and training needs Claws pinned to a
frozen checkpoint for a round, which only works cleanly if the nets don't share live weights. The
card-identity embedding table is warm-started from `ClawsNet.id_emb` once per round.

## Data

Trial JSON reuses `CompactRecorder` unchanged: `role="branch"`, `branch_kind="meta"`, a `meta`
dict (`decision_kind`, `divergence_index`, `option_index`, `prefix_len`) at construction, and one
value attached to the commit action via `mark_step()`/`set_value()` (the value is only knowable
once the chain finishes, long after the commit action is logged). One file per option -- there is
no meta-specific job dataclass: an option-chain runs inline in the main worker process, dispatching
to the same branch-worker pool combat turn-branching already uses via
[`whispering_earring/checkpoint.py`](../../whispering_earring/checkpoint.py)'s `BranchJob`.

`materialize.py` (near-structural copy of
[`claws/model/materialize.py`](../../claws/model/materialize.py)): filters trial files on
`branch_kind == "meta"`, groups by `(main_id, decision_kind, prefix)`, replays the prefix,
tokenizes via `legal_options_from_buffer`, and builds the same example shape `train.py` expects.
`_group_baseline` = mean of the group's values, `advantage = value - baseline` -- one file per
option (M=1) is exactly the degenerate case of the grand-mean-equals-mean-of-equal-size-groups math
an earlier M-replicate version of this scheme relied on, so this file needed no changes when the
scheme was redesigned.

`train.py` is a near-verbatim copy of [`claws/model/train.py`](../../claws/model/train.py) -- same
loss formula, pointed at `ChoicesParadoxNet`.

**`FacilityConfig` knob** (in `whispering_earring/facility_common.py`, shared with
`facility_claws.py`, which never touches it): `meta_branches_per_trial: int | None = 0` -- `0`
disables meta-branching (`facility_claws.py`'s default); `None` branches every eligible
`card_reward` decision (`facility_choices_paradox.py`'s own CLI default, via
`--meta-branches-per-trial`); a positive int caps how many per trial (a decrementing per-trial
budget -- early-run rewards get branched more often than late-run ones, a known bias not blocking
v1).

## Open questions / accepted approximations

- `_WIN_BONUS` (0.5) was tuned once off a single live smoke test where three genuinely different
  picks landed within ~0.2 of each other while skipping landed ~1.1 lower -- retune again if
  trained output still looks survival-count-dominated rather than pick-quality-dominated.
- Chained-checkpoint fidelity confirmed for 13 hops across 4 seeds, but act-boundary crossing not
  yet directly exercised.
- No-recursive-branching's bias-cancellation argument assumes a chain's unbranched downstream
  picks don't systematically favor or hurt one option over another -- not separately validated.
- Boss/normal combats weighted equally inside a chain -- an accepted v1 simplification, not a
  considered decision to keep them equal forever.
- `shop`/`rest_site`/`event_choice` still blocked on buffer-protocol semantic gaps (no
  option_id-equivalent for rest_site, no title/description/vars for event_choice).

## Phasing

1. **Phase 0 -- DONE**: `policy.py`'s purpose-threading refactor (removes the `card_select`
   remove/upgrade global, threads it explicitly instead).
2. **Phase 1 -- DONE**: `card_reward` meta-branching, described above. Originally an
   M-replicate/fixed-horizon-TD-bootstrap scheme; redesigned to one chain per option played to
   natural death after the M-replicate version validated end to end (see Design above for why).
3. **Phase 2 -- not started**: extend to `card_select`'s `remove`/`upgrade` purposes -- additive
   to `model.py` (new option shape, reusing already-provisioned `mode_emb` rows), plus widening
   `META_ELIGIBLE_DECISIONS` and `main_worker_loop`'s branch-eligibility gate.
4. **Phase 3 (out of scope)**: `shop`/`rest_site`/`event_choice` branching -- blocked on the
   buffer-protocol semantic gaps noted above.

# Model

Design doc for training Choices Paradox the way [`claws/model`](../../claws/model/) trains Claws --
self-play branch-and-score, not supervised learning on scraped human run data. **Not implemented
yet** -- this is the spec being built against; see Phasing below for what ships first. Update this
file as design decisions change, and strip the "not implemented" framing from each section as it
lands.

## Why this exists

The working hypothesis for the Act 1 boss struggle: it may not be a combat-skill problem at all.
`choices_paradox/policy.py`'s current card-reward/removal/upgrade picker is a static heuristic --
greedy "take the highest Baalorlord-tier-ranked option among what's offered" -- with zero synergy
or curve awareness and zero connection to whether picks actually correlate with winning. A
combat-skilled Claws still can't win with a deck that heuristic can't build well. If that's true, no
amount of combat-side training fixes it.

## Why self-play instead of supervised learning on human data

STS2 is too young and too complex a game for scraped human runs to be a sufficient dataset the way
they might be for a decade-old game with a huge public corpus, and even with enough data, training
on human outcomes optimizes for what wins *in human hands* -- confounded by human combat skill, not
Claws's. Self-play sidesteps both problems: unlimited data, and the label answers the question that
actually matters -- does this pick win under the policy actually being deployed.

## Design: branch-and-score at meta decisions

The mechanism mirrors [`whispering_earring`](../../whispering_earring/)'s existing turn-level
branch-and-adopt for combat almost exactly, one level up: instead of branching at every combat
turn, branch at a card-reward or card-select (remove/upgrade) decision.

**Topology.** One seed, walk identically down to a branchable decision. At that point, branch
across every legal option there (e.g. 3 offered cards + skip). Not just one option each -- spawn
`M` independent replicate rollouts *per* option, because a single playout's value is dominated by
noise (did the card even get drawn? did Claws's Dirichlet-sampled combat policy happen to sequence
it well this time?). Average the M replicates within each option before comparing across options --
this is a genuine variance-reduction lever, not just more data for its own sake. After the
divergence, every replicate plays forward identically to how the main line would (same downstream
meta-decision policy, same Dirichlet-noised Claws combat policy) until a value can be assigned --
nothing else is allowed to diverge, so any difference in outcome is attributable to the one choice
being compared.

**Generic across decision types, not card-reward-specific.** The mechanism is built once, against
any decision with an enumerable legal-options list -- concretely, in scope for v1: `card_reward`'s
card pick, and `card_select`'s two disambiguated purposes, `remove` and `upgrade`. Out of scope:
`shop`/`rest_site`/`event_choice` -- their buffer-protocol representations are missing semantic
fields a branch worker would need (`rest_site`'s `option_id`/`name`, `event_choice`'s
title/description/vars aren't exported to the buffer at all, only card identity is), so branching
those blind would let the network see which option index it picked but not what the option was.

**Value formula: fixed-horizon TD-bootstrap, not play-to-boss.** Playing every branch all the way to
the act boss is untenable against the branch-worker pool's real budget (already oversubscribed for
combat-only turn-branching at `training.sh`'s own settings -- 15 workers shared across 3 main
workers x up to 30 branches/turn). Instead: play forward until `meta_horizon_combats` combats have
been *won*, or a loss occurs -- bounding the walk at combat boundaries (never mid-combat) so the
cutoff always lands on a clean decision. A loss before N wins is a real terminal value, not an
error or a discarded branch -- floor it the same way combat already does (`_combat_value`'s
existing `-1.0` convention). Reaching N wins alive bootstraps the remainder off Claws's own value
head (`get_claws_net()`, the same mechanism `_play_turn_only`'s non-terminal branch already uses for
its own intra-fight TD term) -- accepting some bias from Claws's imperfect value estimates in
exchange for not needing to simulate an entire act. Default `meta_horizon_combats = 1`: cheapest
starting point, but flagged explicitly as a guess -- a scaling card's payoff can take several
combats to show up, and 1 may prove too short once real training signal comes in.

**No nested branching inside a meta-branch's own continuation.** A meta-branch's post-divergence
combats are driven by a single noisy Dirichlet-sampled rollout (reusing `_play_turn_only` turn by
turn, unbranched), not their own inner turn-branch-and-adopt tree -- `num_options x M x
branches_per_turn x turns-per-combat x horizon_combats` blows past the worker pool's budget by
orders of magnitude otherwise. This does mean each replicate's combat play is a single sample rather
than the main line's own best-of-N adopted quality -- but since every sibling across every option
gets identical treatment, and the group baseline is built from those same branches (not from
main-line-quality play), `advantage = value - baseline` subtracts out any bias that's a uniform
additive offset shared equally across options. The one caveat: if a deck-defining, sequencing-
sensitive card's realized value depends much more on good tactical sequencing than a marginal
card's does, the bias isn't actually uniform across options, and sequencing-sensitive picks could be
systematically under-credited. Accepted as a v1 approximation, revisited only if trained output
looks suspiciously biased against tempo/combo cards.

**Shared worker pool, not a dedicated one.** Meta-branches dispatch through the same `job_queue`
combat branching already uses (`branch_worker_loop` dispatches on job type). A dedicated pool would
need its own sizing for a completely different workload shape (few, much longer-lived jobs) and
would sit idle whenever no meta-decision is in flight. The real cost lever is
`meta_branches_per_trial` (see Facility-side changes below), not pool partitioning -- though a
meta-branch job can legitimately occupy a worker far longer than a turn-branch job, and while it
does, it's unavailable to *any* concurrently running main line's ordinary combat branching. Worth
monitoring once this ships, not something the timeout alone resolves.

**Separate network, not shared with ClawsNet.** Combat's attention question ("given energy, enemy
intents, current powers, what's the best play this turn") and deckbuilding's ("given this deck's
composition and how much act is left, is this card worth the slot") are different enough that a
shared trunk risks negative transfer. More importantly: training Choices Paradox needs Claws pinned
to a frozen/lagged checkpoint for the duration of a round, so the ground truth it's learning against
doesn't shift out from under it mid-training -- only mechanically enforceable if the two nets don't
share live trainable weights. What *is* reused: the card-identity embedding table is warm-started by
copying `ClawsNet.id_emb`'s card rows into the new net's own `id_emb` once per round (not sharing
the live tensor -- both nets load the same committed `EntityVocab`, so the offsets line up), and the
architectural pattern below is a direct copy of `ClawsNet.score()`'s shape.

## Architecture

Same shape as [`claws/model/model.py`](../../claws/model/model.py)'s `ClawsNet`: typed-token
transformer trunk -> per-option query readout -> shared `policy_mlp`/`value_mlp` heads. Smaller
trunk than combat's (fewer layers/dim -- a card-reward/select decision's state is much smaller than
a full combat state).

- **Token types**: `PLAYER`, `MODE`, `DECK_CARD`, `OFFER_CARD`, `RELIC`. `MODE` is a dedicated
  always-present pseudo-token, not a feature bit tacked onto `PLAYER` -- its embedding comes from a
  small `mode_emb: nn.Embedding(3, dim)` keyed by `decision_kind in {card_reward,
  card_select_remove, card_select_upgrade}`. Sized at 3 rows from the start even though only
  `card_reward` ships first (Phase 1), so `card_select` (Phase 2) doesn't need an
  embedding-table-resize that breaks checkpoint compatibility.
- **`DECK_CARD` vs `OFFER_CARD`**: every card in `player.deck` gets a `DECK_CARD` token (full deck,
  not just what's offered -- this is exactly the synergy/curve context the tier-list placeholder
  can't see), and the offered/candidate cards get `OFFER_CARD` tokens, both drawing from the *same*
  card-id embedding space so the trunk can tell "already mine" apart from "being offered" even for
  the same card id.
- **`legal_options(decision, decision_kind) -> list[tuple[str, dict]]`** /
  **`legal_options_from_buffer(buf, decision_kind) -> list[tuple[str, dict]]`**, following
  `claws/model/model.py`'s exact template (fully pre-expanded options, no separate target lists):
  - `card_reward`: one `("select_card_reward", {"card_index": c.index})` per offered card, plus
    `("skip_card_reward", {})` if `can_skip`.
  - `card_select_remove` / `card_select_upgrade`: one `("select_cards", {"indices": [c.index]})` per
    offered card (both purposes are `min_select=max_select=1`, matching
    `choices_paradox.choose_removal_target`/`choose_upgrade_target`'s existing convention), plus
    `("skip_select", {})` if `min_select == 0`.
- **`score(batch, options)`**: each `select_card_reward`/`select_cards` option's query is that
  offered card's own contextualized `OFFER_CARD` token hidden state; `skip_*`'s query is a dedicated
  learned parameter (`skip_query + h[player_pos]`) mirroring `ClawsNet`'s `end_turn_query` -- the
  same "no corresponding token" pattern for an option that isn't about any specific card.
- **`forward(decision, decision_kind)`** dispatches pydantic (`CardReward`/`CardSelect`) vs.
  `BufferDecision` the same way `ClawsNet.forward` does.

## Data

**`MetaBranchJob`** (new dataclass, sibling to `whispering_earring/checkpoint.py`'s `BranchJob`, not
an extension of it -- `BranchJob`'s combat-scoring fields like `is_last_of_act`/`boss_ids` are
meaningless here):

```python
@dataclass
class MetaBranchJob:
    main_id: str
    decision_kind: str          # "card_reward" | "card_select_remove" | "card_select_upgrade"
    divergence_index: int       # monotonic per-trial counter, mirrors combat_index
    option_index: int           # which legal_options(...) entry this branch commits to
    replicate_index: int        # which of the M replicates of that option this is
    character: str
    seed: str
    ascension: int
    actions: list[dict] = field(default_factory=list)   # full replay prefix
    horizon_combats: int = 1

    @property
    def branch_id(self) -> str:
        return f"{self.main_id}_m{self.divergence_index:03d}_o{self.option_index:02d}_r{self.replicate_index:02d}"
```

No `checkpoint_path`/native-checkpoint fast path in v1 -- every replicate reconstructs via full
`open_from_checkpoint` replay. `BranchJob` grew that fast path because full-replay latency climbs
with run depth (measured ~1.7s early in a run to ~7s+ by ~200 actions in); a meta-branch multiplies
that cost by `num_options x M`, so it could dominate for meta-decisions late in long runs. Flagged
for empirical measurement before v2 adds a fast path here, not assumed fine by analogy -- and a
per-combat fast path inside a multi-combat horizon walk would need to repeat `_resolve_combat`'s
own post-win rebuild-via-full-replay at every combat boundary within the horizon window to stay
RNG-safe, which could erode much of the fast path's own advantage. Costed out later, not a v1
blocker.

Trial JSON output reuses `CompactRecorder` entirely unchanged (`attach_policy`/`attach_value`/
`log_action`/`save` are already decision-type-agnostic -- see
[`whispering_earring/recorder.py`](../../whispering_earring/recorder.py)) with a richer `meta` dict
at construction: `"branch_kind": "meta"`, `"decision_kind"`, `"divergence_index"`, `"option_index"`,
`"replicate_index"`, `"prefix_len"` in place of combat's `act`/`floor`/`combat_index`/`turn_index`/
`branch_index`. `attach_value` is called once, immediately before logging the option-commit action
-- no `begin_turn`/`end_turn` bracket needed, since only that one action is being scored, not a
whole turn's worth.

`materialize.py` (near-structural copy of
[`claws/model/materialize.py`](../../claws/model/materialize.py)): filters trial files on
`branch_kind == "meta"` up front, groups by `(main_id, decision_kind, prefix)` -- `decision_kind` in
the group key as cheap insurance against cross-purpose merging even though divergence points
naturally differ. Replays the prefix, asserts `decision.decision in ("card_reward", "card_select")`,
tokenizes via the new `legal_options_from_buffer`, and builds the same example shape
(`type_ids/global_ids/features/.../options/value/action_index/advantage/policy_real`) `train.py`
already expects. Baseline and advantage are computed exactly like combat's: `_group_baseline` = mean
of each option's own averaged-across-M value, `advantage = option_value - baseline`.

`train.py` is a near-verbatim copy of [`claws/model/train.py`](../../claws/model/train.py) --
same `policy_loss = -advantage * log_prob(action_index).clamp(min=-6.0)`, same
`_normalize_advantages`, pointed at `ChoicesParadoxNet` instead. Genuinely reusable as-is at the
loss-formula level; only the tokenizer/network underneath differs.

## Facility-side changes (in `whispering_earring/`, not this package)

**Prerequisite refactor: `policy.py`'s `_pending_card_select_purpose` global has to go.** Today,
`card_select`'s remove-vs-upgrade purpose is disambiguated by a module-level global set by
`_choose_shop_action`/`_choose_rest_site_action` and read by `_choose_card_select_action` on the
*next* call -- both setters only ever run on the main line's own pydantic session, so a
`BufferDecision`-typed `card_select` (how branch workers see everything) always has `purpose=None`.
This is latent and harmless today because branch workers never currently reach `shop`/`rest_site`.
It stops being harmless the moment a meta-branch's own forward continuation spans multiple combats
and routinely wanders through a shop or rest site inside a branch worker. Fix: widen
`choose_action`/`_choose_generic_action`/`_choose_shop_action`/`_choose_rest_site_action`'s return
contracts so purpose flows as an explicit value threaded by the caller into its next call, not a
global. Small, mechanical, behavior-preserving for today's main-line path (verify via a fixed-seed
trial diff before/after) -- but a load-bearing prerequisite for *any* horizon-spanning branch
continuation, including a card-reward-only first phase.

**New `FacilityConfig` knobs**:
- `meta_branches_per_trial: int | None = None` -- caps how many card_reward/card_select decisions
  in a trial actually spawn branches; `None` branches every eligible decision (matching combat's own
  always-branch default). This is the direct answer to wanting simulation cost under control per
  iteration -- a decrementing per-trial counter, not a sampling scheme, since a fair sampling scheme
  would need to know the total number of eligible decisions in advance, which isn't knowable without
  already having played the run. Known bias (early-run rewards branched more than late-run ones),
  not blocking for v1.
- `meta_options_replicates: int = 3` -- `M`.
- `meta_horizon_combats: int = 1` -- see Design above.
- `meta_branch_timeout_s: float = 600.0` -- generous vs. combat's own 120s turn-branch timeout,
  since a meta-branch job is many combats long, not one turn.

**`_resolve_meta_decision`** (mirrors `_resolve_turn`'s shape) spawns `num_options x M` jobs, waits
via a new `_wait_for_meta_results` (averages each option's realized value across its usable
replicates -- non-error, non-timeout; a *died* replicate is usable and contributes its floored
`-1.0`, only a genuine engine error is dropped), and adopts the best-averaged option. The adoption
step is just `session.step(name, **args)` on the live, never-torn-down main-line session -- no
`load_and_resume_combat`/`open_from_checkpoint` reconstruction at all, unlike turn-branch adoption.
This is the single biggest structural simplification versus combat branching, and it's correct
*because* no replicate's own downstream play is ever grafted onto the main line -- every replicate's
only job is producing a value estimate for the option it committed to. It also sidesteps
`_resolve_combat`'s documented `load_save`-into-fresh-process RNG desync class entirely, by
construction -- worth re-deriving explicitly before any future edit introduces `load_save`-based
reconstruction into this adoption step.

**`_play_meta_forward`** (runs inside `branch_worker_loop` for a `MetaBranchJob`): reconstructs via
`Sts2BufferSession.open_from_checkpoint`, re-derives `legal_options_from_buffer` locally (must be a
pure function of buffer state so `option_index` means the same thing the main line's own
enumeration assigned it), applies the committed option, then drives forward unbranched -- reusing
`_play_turn_only` per combat, purpose threaded per the refactor above -- until the horizon or a loss
is reached, scores per the Design section's value formula, and saves via the same
`recorder.save(...)` path `branch_worker_loop` already uses.

## Open questions / accepted approximations

- `meta_horizon_combats = 1` is a starting guess, not a validated value -- likely needs raising once
  training signal exists to check against.
- No-nested-branching's bias-cancellation argument holds rigorously only for a uniform-across-
  options bias; a sequencing-sensitive card could still be under-credited relative to a
  replacement-level one. Watch for this specifically once trained output exists, not before.
- No native-checkpoint fast path in v1 -- cost impact on late-run meta-decisions needs measuring,
  not assuming.
- `CardSelect.source_category`/`source_id` (populated by the engine per `RunSimulator.cs`'s
  `HeadlessCardSelector.PendingSourceCategory`/`PendingSourceId`) could replace the purpose-
  threading refactor with something structurally cleaner (purpose becomes self-contained on the
  decision object, no cross-call ordering dependency) -- but the concrete strings it resolves to for
  shop-removal vs. Smith-upgrade vs. True Grit's exhaust-select haven't been verified empirically.
  Worth a small probe script later; not required to ship Phase 0/1/2.

## Phasing

1. **Phase 0** (prerequisite, ship and verify alone): the `policy.py` purpose-threading refactor.
   Zero behavior change expected on the main line -- verify with a fixed-seed trial diff before
   either meta-branching phase.
2. **Phase 1**: card_reward-only meta-branching -- `MetaBranchJob`, the new `FacilityConfig` knobs,
   `_resolve_meta_decision`/`_play_meta_forward`/`branch_worker_loop` dispatch, and this package's
   `model.py`/`materialize.py`/`train.py` scoped to `card_reward`'s option shape (with `mode_emb`
   already sized for 3 rows). `card_select` keeps resolving via the existing unbranched heuristic
   for all of Phase 1. Validate end to end (small trial run -> materialize -> train smoke test)
   before Phase 2.
3. **Phase 2**: extend to `card_select`'s `remove`/`upgrade` purposes -- additive to Phase 1's
   `model.py` (new option shape, reusing the `mode_emb` rows already provisioned), plus loosening
   `main_worker_loop`'s branch-eligibility gate.
4. **Phase 3 (explicitly out of scope, not attempted)**: `shop`/`rest_site`/`event_choice`
   branching -- blocked on the buffer-protocol gaps noted above.

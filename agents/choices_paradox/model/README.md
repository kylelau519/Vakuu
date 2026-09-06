# Model

Design doc for training Choices Paradox the way [`claws/model`](../../claws/model/) trains Claws --
self-play branch-and-score, not supervised learning on scraped human run data. **Phase 0, Phase 1,
and rest_site/event_choice (originally scoped as part of Phase 3) are implemented** (`policy.py`'s
purpose-threading refactor; card_reward, rest_site, and event_choice meta-branching, all via
[`whispering_earring/facility_choices_paradox.py`](../../whispering_earring/facility_choices_paradox.py),
validated end to end -- event_choice also covers the ancient/Neow boon for free, since it's not a
separate decision type, see Design below). Phase 2 (`card_select`) and the rest of Phase 3
(`shop`) are not -- see Phasing below. Update this file as design decisions change.

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

No recursive meta-branching inside a chain: any card_reward/rest_site/event_choice a chain hits
past its own initial commit resolves via a single self-play pick, same as the main line whenever
meta-branching isn't invoked. Combinatorial cost (`options^depth`) is why.

Each chain can optionally be cut off after a fixed number of combats past its own commit
(`FacilityConfig.meta_horizon_combats` / `--meta-horizon-combats`, default unbounded) instead of
always playing to natural death -- scores on whatever `total_value` it accumulated up to the
cutoff, no terminal penalty either way (an early cutoff isn't a loss, same reasoning as the
unbounded case's own no-penalty-for-the-eventual-loss argument).

rest_site and event_choice both reuse this exact design (chain-and-score to death or to the
horizon, baseline-via-skip-value, same as card_reward) with one addition: `leave_room` -- not
itself one of `RestSite.options`/`EventChoice.options`, but a separate action always legal in a
non-combat room regardless of which real options are enabled or locked -- is included as a
synthetic sibling in `legal_options`/`legal_options_from_buffer`, giving both the same "did
nothing" baseline anchor skip_card_reward already gives card_reward (`materialize.py`'s
`_group_baseline` uses whichever sibling's action is in `_SKIP_ACTIONS`, now including
`leave_room`). Without it, every sibling would be a real intervention and there'd be no clean
reference point to measure "was doing anything here even worth it" against. (An all-locked
event_choice never reaches this code at all -- sts2-cli auto-leaves internally before it becomes a
decision -- so event_choice's `leave_room` sibling only ever coexists with at least one real,
unlocked option.)

The ancient/Neow boon needed no special-casing to pick up: it's not a separate decision type, just
an ordinary `event_choice` sourced from the `"ancients"` loc table instead of `"events"` for its
title (confirmed live -- seed 0 of a fresh Ironclad run lands on `event_choice` at step 0, before
any other decision).

The real trial does **not** adopt whichever option scored best. It rewinds to the pre-divergence
state (never actually left, since every chain runs on its own forked session) and commits via
`ChoicesParadoxNet`'s own softmax-plus-Dirichlet-noise prediction instead -- keeps the trial
on-policy rather than coupled to whichever option an under-trained net happened to explore best
this round. Falls back to the tier-list heuristic until a checkpoint (`CHOICES_PARADOX_CHECKPOINT`)
exists.

Out of scope for now: `shop` (its buffer-protocol representation is missing semantic fields a
branch worker would need -- rest_site/event_choice had the same kind of gap for their per-option
identity and shipped anyway with identity-less tokens, see Open questions below), and boss-fight
combats are weighted equally with normal ones inside a chain (accepted v1 simplification).

## Architecture

Same shape as [`claws/model/model.py`](../../claws/model/model.py)'s `ClawsNet`: typed-token
transformer trunk -> per-option query readout -> shared `policy_mlp`/`value_mlp` heads, matching
`dim`/`n_layers`/`n_heads` so `ChoicesParadoxNet.warm_started_from` can copy `ClawsNet.id_emb`
rows without passing matching kwargs by hand.

- **Token types**: `PLAYER`, `MODE`, `DECK_CARD`, `OFFER_CARD`, `RELIC`, `REST_OPTION`,
  `EVENT_OPTION`. `card_select_remove`/`card_select_upgrade`'s `MODE` rows were sized in from the
  start (before Phase 2 itself); `rest_site`/`REST_OPTION` and `event_choice`/`EVENT_OPTION` were
  each appended later, after card_reward-only (then card_reward+rest_site) checkpoints already
  existed in production -- `load_state_dict_allowing_growth()` (in `model.py`, used by both
  `ChoicesParadoxNet.from_checkpoint` and `train.py`'s `--init`) is what lets `mode_emb`/`type_emb`
  grow new rows without discarding an existing checkpoint's trained ones, since a plain
  `load_state_dict` hard-fails on that shape mismatch. Verified against the live checkpoint jumping
  straight from 3 `DECISION_KINDS`/5 `TOKEN_TYPES` to 5/7 in one load.
- **`DECK_CARD` vs `OFFER_CARD`**: every card in `player.deck` gets a token (full-deck
  synergy/curve context the tier-list heuristic can't see); offered/candidate cards get
  `OFFER_CARD` tokens, both sharing the same card-id embedding space. `REST_OPTION`/`EVENT_OPTION`
  tokens (one per enabled `RestSite` option / unlocked `EventChoice` option) are the same idea but
  currently identity-less -- see Open questions.
- **`legal_options(decision, decision_kind)`** / **`legal_options_from_buffer(...)`**: fully
  pre-expanded option list, following `claws/model/model.py`'s template -- `card_reward` yields
  one `select_card_reward` per offered card plus `skip_card_reward` if `can_skip`; `rest_site`
  yields one `choose_option` per enabled option plus `leave_room`; `event_choice` yields one
  `choose_option` per unlocked option plus `leave_room` (both `leave_room`s always legal, not
  themselves one of `RestSite.options`/`EventChoice.options` -- see Design above); `card_select_*`
  (Phase 2) yields one `select_cards` per offered card plus `skip_select`.
- **`score(batch, options)`**: each card/rest/event option's query is that option's own
  contextualized `OFFER_CARD`/`REST_OPTION`/`EVENT_OPTION` hidden state (looked up via
  `offer_pos`); `skip_*`/`leave_room`'s query is a dedicated learned parameter (`skip_query`),
  mirroring `ClawsNet`'s `end_turn_query`.
- **`forward(decision, decision_kind)`** dispatches pydantic (`CardReward`/`CardSelect`/
  `RestSite`/`EventChoice`) vs. `BufferDecision` the same way `ClawsNet.forward` does.

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
`_group_baseline` is the group's own skip-sibling's realized value -- whichever member's action is
in `_SKIP_ACTIONS` (`skip_card_reward`/`skip_select`/`leave_room`) -- and `advantage = value -
baseline`; a group with no skip sibling in it, or with only one distinct action across all
members, has no baseline and is dropped (`_group_baseline` returns `None`).

`train.py` is a near-verbatim copy of [`claws/model/train.py`](../../claws/model/train.py) -- same
loss formula, pointed at `ChoicesParadoxNet`.

**`FacilityConfig` knobs** (in `whispering_earring/facility_common.py`, shared with
`facility_claws.py`, which never touches either): `meta_branches_per_trial: int | None = 0` -- `0`
disables meta-branching (`facility_claws.py`'s default); `None` branches every eligible
decision in `META_ELIGIBLE_DECISIONS` (`card_reward`, `rest_site`, `event_choice`)
(`facility_choices_paradox.py`'s own CLI default, via `--meta-branches-per-trial`); a positive int
caps how many per trial, shared across all eligible decision kinds, not one budget each (a
decrementing per-trial budget -- early-run divergences get branched more often than late-run ones,
a known bias not blocking v1). `meta_horizon_combats: int | None = None` -- `None` plays every
option-chain to natural death (default); a positive int (`--meta-horizon-combats`) caps each chain
at that many combats past its own commit instead, useful for a lower-variance/faster-turnaround
signal at the cost of not seeing a chain's longer-tail consequences.

## Open questions / accepted approximations

- `_WIN_BONUS` (0.5) was tuned once off a single live smoke test where three genuinely different
  picks landed within ~0.2 of each other while skipping landed ~1.1 lower -- retune again if
  trained output still looks survival-count-dominated rather than pick-quality-dominated.
- Chained-checkpoint fidelity confirmed for 13 hops across 4 seeds, but act-boundary crossing not
  yet directly exercised.
- No-recursive-branching's bias-cancellation argument assumes a chain's unbranched downstream
  picks don't systematically favor or hurt one option over another -- not separately validated,
  and rest_site is a sharper case of it than card_reward: Golden Compass's own map policy
  hard-branches on `hp_frac` (`golden_compass/policy.py`'s `_ELITE_HP_FLOOR`), so a HEAL sibling
  and a SMITH sibling can walk genuinely different downstream routes (different fights entirely),
  breaking the shared-`walk_seed` correlated-noise-cancellation that keeps card_reward's
  comparison low-variance (siblings there rarely diverge in route). Not yet mitigated -- one
  option discussed but not implemented: pin the downstream route for a bounded window across a
  divergence's siblings, letting Golden Compass's own reward shaping
  (`facility_common._apply_golden_compass_rewards`) separately capture the "safer route" value
  resting can unlock, rather than rest_site's own signal having to absorb both effects at once.
- Boss/normal combats weighted equally inside a chain -- an accepted v1 simplification, not a
  considered decision to keep them equal forever.
- `REST_OPTION`/`EVENT_OPTION` tokens carry no per-option identity (HEAL vs. SMITH vs. ...; which
  specific boon an event option is) -- the buffer protocol's `RestSite`/`EventChoice` sections
  export only `(present, is_enabled)` / `is_locked` per slot, no option-id-equivalent to decode
  from a branch worker's `Sts2BufferSession` (`RestOption.option_id` and `EventOption.title`/
  `description`/`text_key`/`vars` all exist on the JSON side, `RunSimulator.Writers.Json.cs`, but
  not in the buffer one, `RunSimulator.States.Buffer.cs`'s `WriteRestSite`/`WriteEventChoice`).
  Branching/scoring/materialization all work today regardless (validated end to end for both,
  including a real training step against the live checkpoint), but the model can only learn from
  state features (HP, deck, upcoming-path context if that's added -- see below) plus a slot's
  position, not "this option is specifically HEAL" or "this is the boon that grants a relic" --
  and slot order isn't guaranteed stable run to run. Fixing this needs an sts2-cli patch (a
  tokenized option-id field per slot, mirroring how `event_token` already does
  `_tokenizer.Encode`); for event_choice specifically, a narrower first cut discussed but not
  built: relic-granting options only, reusing the `RELIC` token type/vocab space that already
  exists for relics *held* (`player.relics`), needing just a per-option relic-id buffer field plus
  an `OFFER_RELIC`-style `offer_pos` entry rather than solving `vars` in general. Tracked as a
  follow-up, not solved here.
- No context in `ChoicesParadoxNet`'s tokens about the map path ahead -- rest-vs-upgrade plausibly
  depends on what's coming (heal because an elite/boss is next, upgrade if the road ahead is calm),
  which the current token set can't express at all. Discussed direction: a `PATH_NODE` token type
  per upcoming node type toward the boss, sourced from the same DP `golden_compass/policy.py`'s
  `_best_path_values` already computes (next-N-nodes-regardless-of-route, not Golden Compass's own
  literal chosen route, to avoid coupling this net's inputs to Golden Compass's current training
  state). Not implemented.
- `shop` still blocked on its own buffer-protocol semantic gaps (not investigated in the same
  detail as rest_site/event_choice's).

## Phasing

1. **Phase 0 -- DONE**: `policy.py`'s purpose-threading refactor (removes the `card_select`
   remove/upgrade global, threads it explicitly instead).
2. **Phase 1 -- DONE**: `card_reward` meta-branching, described above. Originally an
   M-replicate/fixed-horizon-TD-bootstrap scheme; redesigned to one chain per option played to
   natural death after the M-replicate version validated end to end (see Design above for why).
3. **Phase 2 -- not started**: extend to `card_select`'s `remove`/`upgrade` purposes -- additive
   to `model.py` (new option shape, reusing already-provisioned `mode_emb` rows), plus widening
   `META_ELIGIBLE_DECISIONS` and `main_worker_loop`'s branch-eligibility gate.
4. **Phase 3, rest_site -- DONE**: `rest_site` meta-branching with a `leave_room` baseline sibling
   and an optional bounded horizon (`meta_horizon_combats`), validated end to end (branching,
   materialize, and a resumed training step against the live checkpoint). `REST_OPTION` tokens are
   identity-less and there's no map-path context yet -- both open, see Open questions above; the
   Golden-Compass-route-divergence signal-quality concern is also open, not yet mitigated.
5. **Phase 3, event_choice -- DONE**: same design as rest_site (including its own `leave_room`
   baseline sibling), validated end to end the same way -- also picks up the ancient/Neow boon for
   free, since it's not a separate decision type (confirmed live). `EVENT_OPTION` tokens are
   identity-less for the same reason `REST_OPTION`'s are; a narrower relic-only identity cut was
   scoped (see Open questions) but not built.
6. **Phase 3, shop -- not started**: blocked on its own buffer-protocol semantic gaps, not yet
   investigated to the same depth as rest_site/event_choice's.

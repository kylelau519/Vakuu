# whispering_earring

Self-play facility: plays full runs end-to-end and records every decision, to generate training
data for [`claws`](../claws/), the in-combat ML suggestion model, and (via a second kind of
branching layered on the same worker pool) for [`choices_paradox`](../choices_paradox/), the
card-reward/removal/upgrade picker. This package doesn't decide combat or deckbuilding well
itself -- it drives the game forward so that trajectories exist to train on.

Two concrete facilities share one core (`facility_common.py`):

- **`facility_claws.py`** -- combat turn-branching, generating Claws's training data. This is the
  original scheme, and the one to run if you only care about combat.
- **`facility_choices_paradox.py`** -- adds card_reward meta-decision branching on top, sharing
  the same branch-worker pool (its main line still turn-branches combat exactly like
  `facility_claws.py`'s does, reusing that module's `_resolve_combat` unchanged). Run this one to
  also generate `choices_paradox`'s training data. See
  [`choices_paradox/model/README.md`](../choices_paradox/model/README.md) for the design.

See `facility_claws.py`'s module docstring for the combat branching scheme (N concurrent
main-line workers, each forking off a checkpoint at every *turn* to explore alternate lines via a
shared pool of branch workers, discarded after that one turn resolves -- a main line adopts
whichever of its branches scores best, per `_turn_score_key`, once all have reported or timed
out) and `policy.py`
for the current action-selection policy: combat is uniform-random exploration by default
(non-combat decisions delegate to the Choices Paradox and Golden Compass placeholder policies,
since a random meta/path policy would produce decks and runs too weak to make combat outcomes
meaningful) -- or, with `CLAWS_CHECKPOINT` set, samples from a loaded `claws/model/ClawsNet`
checkpoint's own softmax policy instead (still no search, just the raw network). This
softmax-sample/uniform-random diversity across a turn's parallel branches is what gives
branch-and-adopt something to select between.

Each adopted turn is scored with a TD-bootstrapped value, not a flat per-combat constant (see
`facility_common.py`'s `_play_turn_only` and `_combat_value`): a turn that doesn't end the combat gets
`hp_weight * (hp_term + enemy_term) + V_net(state_after_turn)`, where `hp_term`/`enemy_term` are
each `±(hp_lost_or_dealt_this_turn)/max_hp` and `V_net` is the current `ClawsNet`'s own value-head
estimate (one forward pass, no further simulation); a turn that does end the combat gets the
existing terminal HP-loss formula (`_combat_value`) instead of a bootstrap. `hp_term`/`enemy_term`
are both potential-differences that telescope to a fixed total over the whole combat regardless of
which turn the damage happened on, so weighting them equally densifies credit assignment without
biasing the policy toward trading HP for damage at some fixed rate -- the actual tradeoff is still
governed by the terminal win/loss reward. This per-turn shaping is also what lets a turn that took
damage to set up a good follow-up score differently from a turn that didn't, which a single scalar
backfilled onto the whole combat couldn't do.

Branch workers drive their own turn via sts2-cli's buffer protocol (`buffer_env.Sts2BufferSession`,
in-process, no subprocess/JSON/pydantic decode) instead of the JSON `Sts2Session` -- see sts2-cli's
`src/Sts2Headless/README.md` for the wire format and `policy.py`'s combat/card-select choosers for how
they branch on the decision type (`isinstance(decision, BufferDecision)`) to read fields via
`buffer_layout.py` instead of a pydantic attribute tree. The main self-play line still uses
`Sts2Session`/pydantic `Decision` throughout, since it never plays `combat_play` directly (see
`_resolve_turn`) -- only branch workers do.

## Layout

| File | Role |
|---|---|
| `facility_common.py` | Shared core: worker-process orchestration (`Facility`), `FacilityConfig`, stderr relay/logging, combat scoring (`_combat_value`, `_play_turn_only`), the generic `main_worker_loop`/`branch_worker_loop` both concrete facilities build on |
| `facility_claws.py` | Combat turn-branching (Claws's training data): `_resolve_turn`/`_resolve_combat`, the `BranchJob` handler, CLI entry point |
| `facility_choices_paradox.py` | Adds card_reward meta-decision branching (choices_paradox's training data): `_resolve_meta_decision`/`_evaluate_option_chain`, CLI entry point. Only the main line's own combat uses the shared branch-worker pool (via `facility_claws._resolve_combat`, reused as-is); each option's own evaluation chain plays unbranched on its own in-process `Sts2BufferSession` instead, one chain at a time (see the module's docstring for why -- an earlier turn-branched, concurrent-threads design was the actual bottleneck, and concurrent `Sts2BufferSession`s aren't safe to begin with) |
| `policy.py` | `choose_action(decision, session, purpose=None)` -- combat is random, or ClawsNet-sampled if `CLAWS_CHECKPOINT` is set; meta decisions delegate out. `card_reward` is random/tier-list by default, or `ChoicesParadoxNet`-sampled if `CHOICES_PARADOX_CHECKPOINT` is set (`_choose_card_reward_decision`) -- same Dirichlet-noised-softmax sampling as the combat net. `purpose`/the returned next-purpose thread `card_select`'s remove/upgrade disambiguation across calls -- see the function's docstring. Also has the buffer-protocol (`BufferDecision`) counterparts for every non-combat decision kind, needed once `facility_choices_paradox.py`'s horizon walk started driving branch workers through card_reward/map_select/shop/rest_site/event_choice/bundle_select -- something no branch worker did before it existed |
| `recorder.py` | `CompactRecorder` -- per-trial JSON capture (action log, policy, per-turn value; `mark_step`/`set_value` for a value only knowable after playing forward, e.g. a meta-branch's horizon walk) |
| `checkpoint.py` | `BranchJob` -- the replay-prefix format passed to branch workers via an in-memory queue (combat turn-branching only; `facility_choices_paradox.py`'s meta-branch option-chains no longer dispatch to the branch-worker pool at all) |
| `sts2_path.py` | Adds sts2-cli's `python/` to `sys.path`; import before anything from sts2-cli |
| `data/raw/` | Recorded trial JSON per trial (gitignored) |

## Running

```sh
conda activate sts2
python facility_claws.py --trials 10 --branch-workers 4 --branches-per-turn 3

# multiple concurrent main lines sharing that same branch-worker pool, so branch workers stay
# fed even while any given main line is off doing map/shop/reward decisions between combats
python facility_claws.py --trials 10 --branch-workers 12 --branches-per-turn 3 --main-workers 4

# card_reward meta-branching for choices_paradox, on top of the same combat turn-branching --
# see choices_paradox/model/README.md's Design section
python facility_choices_paradox.py --trials 10 --branch-workers 12 --branches-per-turn 3 \
  --main-workers 4 --meta-options-replicates 3 --meta-horizon-combats 1
```

To drive combat from a trained (or just initialized -- see `claws/model/train.py --epochs 0`)
checkpoint instead of uniform-random -- this also feeds the TD-bootstrap value term's `V_net`
estimate, not just action selection (works with either facility):

```sh
CLAWS_CHECKPOINT=../claws/model/claws.pt python facility_claws.py --trials 10 ...
```

Requires `sts2-cli` checked out next to this repo (or `STS2_CLI_ROOT` set) and set up per its own
`./setup.sh`.

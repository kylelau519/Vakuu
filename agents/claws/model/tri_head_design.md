# Claws tri-head value/policy design

Status: implemented (`model.py`, `materialize.py`, `train.py`). Written up from a design
discussion; see git blame / chat history on this file's introduction for the original reasoning,
and the Aggregator/Training sections below for a since-revised piece: the gate started as a
proposed differentiable probability-space blend, then was changed to hard/stochastic gating once
self-play data showed the blend couldn't recover a head it had driven to ~0 weight. This replaced
`ClawsNet`'s single `policy_mlp`/`value_mlp` pair with three actor-critic pairs (offense, defense,
scale) plus one auxiliary outcome estimator and a gate.

## Problem this is solving

The current single value head conflates offense, defense, and win-margin into one scalar. This
produces a specific, observed failure: against a low-max-HP enemy, `enemy_term` (fractional enemy
HP lost) dominates the shaped reward, since each point of damage is a large fraction of a small
`enemy_max_hp`. The policy over-plays aggressively and takes unnecessary damage securing kills
that were already assured, because the single scalar has no way to say "the fight is already won,
stop spending HP." A single value head *could* in principle represent this correctly given enough
training, but a single scalar target also entangles three concerns whose evidence looks very
different (immediate, low-variance HP/enemy swings vs. sparse, high-variance win/lose outcomes),
which empirically fights itself during training.

Also considered and rejected as the fix:
- Just reducing `_COMBAT_DIRICHLET_EPS` scaling or hand-tuning `hp_weight` — treats the symptom,
  not the conflated-objective root cause.
- A single win-probability head used only as a 2-way (aggro/defense) gate — mathematically
  incapable of ever proposing a pure setup/scaling action, since a mixture of two experts is
  confined to the convex hull of what those two experts already propose. If neither expert ever
  assigns real probability to "play Strength, do nothing else," no gate weighting can produce it.
  This is why scale needs its own *policy*, not just its own *value*.

## Architecture

One shared trunk (`ClawsNet.encode()`, unchanged) producing per-token hidden states `h`. Three
actor-critic pairs and one auxiliary head, all reading off the same shared `h` -- no second
transformer, no per-head re-encoding:

- `policy_aggro` + `value_aggro` -- reward stream: `enemy_term` (existing per-turn fractional
  enemy-HP-loss potential difference, see `facility_common.py`'s `_play_turn_only`)
- `policy_defense` + `value_defense` -- reward stream: `hp_term` (existing per-turn fractional
  own-HP-loss potential difference, same location)
- `policy_scale` + `value_scale` -- reward stream: `scale_term` (new, defined below)
- `win_value` -- NOT a fourth peer head. A single auxiliary scalar estimating graded combat
  outcome (what `v_net` approximates today: `_combat_value(...)` on a win, `-1` on a loss,
  TD-bootstrapped on continuing turns). Two jobs only: (1) supplies the per-turn `scale_term`
  reward via a residual (below), (2) feeds the gate.

Each `value_i` is "expected future discounted accumulation of reward stream i from this state" --
same relationship `hp_term`/`enemy_term` have to today's single `v_net`, just split three ways
instead of summed into one.

## Reward stream definitions

`enemy_term` and `hp_term`: unchanged from today's `_play_turn_only` (facility_common.py:301-309).

`scale_term` (per turn):

    scale_term(turn) = [win_value(s') - win_value(s)] - hp_term(turn) - enemy_term(turn)

Read as: "how much did my expected outcome change this turn, for reasons other than the direct
HP/enemy swings already credited to the other two heads." This is deliberately NOT a hand-weighted
sum of power-stack amounts (rejected -- some powers are worth far more than others, and several
real scaling effects, e.g. extra-energy relics, aren't `Power` objects at all and have no field to
sum). Instead the "how good is this power/relic/setup" question is answered implicitly: states
with more Strength (or any other scaling effect) correlate with higher `win_value` across
self-play, the same way AlphaZero's value net learns relative piece values with no hand-coded
weights -- because `win_value` is itself regressed against real win/lose outcomes.

The subtraction of `hp_term`/`enemy_term` is not primarily a variance-reduction trick -- it exists
so `scale_term` is *orthogonal* to what the other two heads already claim credit for. If
`enemy_term`/`hp_term` are properly n-step (see below), they already include the future
damage/block that scaling enables; without subtracting them off, `scale_term` would just be
redundant with the other two heads, not a distinct third axis.

Known gap: enemy-applied debuffs (Vulnerable/Weak/Poison on the enemy) are readable today for
reward purposes (`Enemy.powers` in `models.py`, `bl.enemy_powers()` in `buffer_layout.py`) but are
NOT tokenized as network input (`ClawsNet`'s `ENEMY` token has no debuff-amount feature, only
`[hp_frac, hp, block, intent_damage, intends_attack]`). `scale_term` can reward applying a debuff,
but `policy_scale` can't condition on how much debuff is already stacked until `ENEMY_POWER`
tokens (or equivalent features) are added. Not a blocker to a first experimental pass, but a real
ceiling on what `policy_scale` can learn about enemy-side scaling until it's fixed.

## Propagation: n-step / TD(lambda) instead of 1-step bootstrap

Today's `turn_value = shaped_this_turn + v_net(s')` is a 1-step bootstrap -- a scaling move whose
payoff lands several turns later has to travel back through that many noisy single-step hops,
each only as good as the net's current (possibly immature) guess. Since full self-play
trajectories are already saved to disk before `materialize.py` runs (not a true online/streaming
setting), targets for all three heads should be computed as n-step or TD(lambda)-blended returns
over the actual recorded continuation instead of a single bootstrap hop:

    target_n = r_1 + gamma*r_2 + ... + gamma^(n-1)*r_n + gamma^n * V(s_{n+1})

with `r_i` the real recorded per-turn reward (whichever stream) and `V` the corresponding head's
value estimate. `n=1` is today's behavior. TD(lambda)/GAE blends all `n` with weights `(1-lambda)*
lambda^(n-1)` instead of committing to one fixed `n` -- standard bias/variance interpolation
(`lambda=0` = today's 1-step bootstrap, low variance/high bias; `lambda=1` = full Monte Carlo,
high variance/low bias). All three heads' targets, and `win_value`'s own target, should use the
same propagation scheme and horizon so `scale_term`'s subtraction stays well-defined (it's
subtracting `enemy_term`/`hp_term` computed over the same window `win_value`'s delta spans).

Caveat: `scale_term` depends on `win_value(s') - win_value(s)`, a difference of two outputs from a
still-training critic. Early in training, before `win_value` is well-calibrated, this residual
will be noisy/biased -- not a flaw unique to this design, the same bootstrapping-off-an-imperfect-
critic issue every actor-critic method has, including today's single `v_net`. Expect `scale_term`
to be the least reliable of the three signals early on, improving as `win_value` converges.

## Aggregator

`gate_mlp(h[player_pos], value_aggro, value_defense, value_scale, win_value) -> (w1, w2, w3)`,
softmax-normalized -- `values["gate"]` in `model.py`'s `score()`.

**Originally proposed, and initially implemented, as a differentiable probability-space blend**:
`P_final(a) = w1*P_aggro(a) + w2*P_defense(a) + w3*P_scale(a)`, sampling the action straight from
that one mixed distribution. Rejected in favor of hard/stochastic gating below after a real
training run showed the specific failure this section used to dismiss as unlikely: the gate drove
one head's weight to ~0.00 within a single epoch and never moved it again across 8 further epochs,
while that head's OWN value head kept fitting its regression target just fine -- proof the
representation was learnable, the gate just never routed it any policy gradient to learn from
(both REINFORCE terms below are scaled by that head's own weight, so `w_i≈0` means ~0 gradient for
head `i` regardless of whether its suggestions were actually good). `model.py`'s `score()` still
computes this blend and returns it as `probs` for diagnostics; it is no longer what self-play
samples from.

**Implemented: hard/stochastic gating** (the "options" framework -- an option/sub-policy pair per
head, not a soft mixture of experts). `policy.py`'s `_choose_combat_action_claws` samples a
strategy `z ~ Categorical(w1, w2, w3)` first, then samples the action from ONLY that head's own
distribution, `P_z(a)`. The self-play trial JSON records which head was actually used
(`entry["policy"]["head"]`, read back by `materialize.py`'s `_head_of`) alongside the action, so
training can route credit correctly.

Why this fixes the starvation failure: `z` is a genuine sample from `Categorical(w1,w2,w3)`, so as
long as a head's gate weight is nonzero at all, self-play will occasionally still route a real
decision through it and record a real (state, action, advantage) triple for it -- self-correcting
exploration over STRATEGY, not just over which action a fixed blend prefers. A head can only go
fully silent if its gate weight is driven to exact 0, which softmax approaches asymptotically but
never truly reaches -- but "never truly reaches" isn't the same as "recovers in practice": pure
REINFORCE still has no restoring force pulling weight back once it's near 0, and one specific
mechanism pushes on it consistently: `hp_term` (defense's own reward stream) is turn-level, not
action-level (recorder.py's `end_turn()` backfills one identical reward dict onto every action
recorded that turn), so `end_turn` inherits whatever `hp_term` the turn produced. A quiet turn
(nothing happened, no damage taken) hands defense a free non-negative advantage that aggro/scale
structurally can't get the same way (their streams are 0 unless the agent actually acted) --
observed in a real run as the gate collapsing to `defense=1.00, aggro=scale=0.00` within one epoch
and staying there across 8 more, aggro/defense/scale usage in the underlying data notwithstanding.
`train.py`'s `--gate-entropy-coef` (default 0.01) adds an entropy bonus on the gate's own output
per example -- `loss -= gate_entropy_coef * entropy(gate_weights)` -- so pushing a head's weight
all the way to 0 costs loss even when REINFORCE alone would keep pushing it there, giving the
gate an explicit reason to keep every head's routing probability off the floor.

Why this also sharpens credit assignment: under the old blend, every example nudged all 3 heads'
logits in the same direction (toward the same win-advantage-driven target), differing only in
magnitude by `w_i` -- so absent some other force, the three heads had no structural reason to stay
distinct beyond whichever states the gate happened to route them to. Under hard gating, a head
only ever gets a policy-gradient update from examples where IT was the one self-play sampled, so
its specialization comes from genuinely separate training data, not a differently-scaled copy of
the same signal every other head also received.

A small MLP is deliberately recommended over a second transformer for `gate_mlp`: the shared trunk
already computes a fully contextual `h` via self-attention over the whole board; a second
transformer would mostly re-derive context that already exists. Attention's advantage (permutation
invariance over a variable/large set) doesn't apply to gating exactly 3 fixed experts -- a concat +
MLP does the job with far fewer parameters and much easier training dynamics.

Design choice not yet settled: state-level gate (one `(w1,w2,w3)` per decision, applied uniformly
across every candidate option this turn -- `gate_mlp(h[player_pos], ...)`) vs. option-level gate
(a different weighting per candidate action -- `gate_mlp(h[option_pos], ...)`). State-level is
simpler to validate and already sufficient to express the original bug fix ("mostly defense once
winning is assured"); option-level is strictly more expressive but harder to train/diagnose.
Recommendation: start state-level, only move to option-level if a state-level gate demonstrably
can't express the desired behavior.

Training: no separate reward or separate training loop for the gate -- two REINFORCE terms off the
same win-derived advantage signal as every other update in this codebase, not a new objective:

    gate_loss = -advantage * log(w_z)              # was picking head z here a good idea?
    head_loss = -advantage * log(P_z(action_taken)) # was picking this action, given z, a good idea?

(`train.py`'s `_batch_step`; `z` is whichever head `ex["head"]` says self-play actually used for
that recorded example.)

Explicitly rejected alternative: supervising the gate directly on "which head's suggested action
was adopted, and did that turn win" (a literal classification target). This reintroduces exactly
the survivorship bias `facility_claws.py`'s `_wait_for_turn_result` was deliberately built to
avoid -- it adopts branches by a fixed index, never by outcome, specifically because "the branch
that won" is a biased signal (see that function's docstring). The advantage-weighted approach
above doesn't have this problem since it's already baseline-subtracted, same as the rest of the
training pipeline.

## Summary of what's new vs. the original single-head net

| Before | Now |
|---|---|
| 1 `policy_mlp`, 1 `value_mlp` | 3 policy heads, 3 value heads, 1 auxiliary `win_value`, 1 gate |
| `turn_value = hp_weight*(hp_term+enemy_term) + v_net`, one combined scalar | `enemy_term`, `hp_term`, `scale_term` kept as three separate reward streams, never summed |
| 1-step bootstrap (`v_net(s')`) | n-step / TD(lambda) targets computed offline over recorded trajectories, same scheme for all three heads and `win_value` |
| Single scalar picks the action | Hard/stochastic gating: sample which head from the gate, then sample the action from only that head -- two REINFORCE terms off the same advantage signal, not a differentiable blend (see Aggregator's revision history) |

## Known issue, not yet fixed

`_combat_value` (facility_common.py) grades a win purely on ending `hp_frac`. Since `hp_term`
already sums to ~that same fractional HP change over the fight, `scale_term`'s terminal-turn
residual (`win_value_after - win_value_before - hp_term - enemy_term`) mostly cancels against the
fight's own accumulated `hp_term` -- there is currently no reward signal at all for ending a fight
with a better RELIC/POTION/GOLD/DECK position, only for ending it with more HP. Candidate fix: fold
a non-HP "status" term (buffer_layout.py already exposes `player_relics`/`player_potions`/
`player_gold`/`player_deck_size` at combat end) into `win_value_after`, orthogonal to `hp_frac`, so
`scale_term` has real independent signal instead of an echo of `hp_term`.

## Open follow-ups, not blocking a first experimental pass

- Tokenize enemy-applied debuffs (`ENEMY_POWER` or equivalent) so `policy_scale` can condition on
  existing enemy debuff stacks, not just be rewarded for applying them.
- Decide state-level vs. option-level gate empirically once a state-level version is running.
- Pick a concrete `lambda`/horizon for the n-step propagation; likely needs a small sweep rather
  than a single guessed value.
- Tune `--gate-entropy-coef` (added to counteract observed gate collapse, see Aggregator section)
  against real self-play rounds -- 0.01 is an untested starting guess, not a validated value.
- The gate-entropy bonus treats the symptom (softmax driving a head to 0) but not the root cause
  identified alongside it: `hp_term` is turn-level, so `end_turn` free-rides on whatever the turn's
  other actions already earned. Worth a real fix if entropy regularization alone doesn't hold
  `head_used_*` balanced -- e.g. a baseline that doesn't credit `end_turn` with a positive
  advantage on turns where no other action was taken.

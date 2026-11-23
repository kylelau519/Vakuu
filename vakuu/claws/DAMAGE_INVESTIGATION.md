# Why Claws takes so much damage -- investigation notes (2026-11-22)

Question: is the damage Claws takes (act 1: about 12.5 HP per combat, act 2: about 18) caused by an over-aggressive policy, by the training
method, or by the decks the macro player (Golden Compass / tier list) builds? These notes are what was measured, with the sample sizes, and what
is still open. Everything below is one checkpoint (`runs/spire/claws/claws.pt` as of 22 Nov 13:08) and one compass (`golden_compass.pt`, 09:32).
"Greedy" = `CLAWS_GREEDY=1` (Claws takes its top option); "greedy compass" = `COMPASS_EPS=0 COMPASS_TAU=0` (argmax over afterstate scores).

## The expectation (from the project owner, act 1)

Normal elite 15-20 HP lost (8-15 if entered later), boss about 30, normal fights 5-7. A good player would take far less than the 12.5 / 18 per
combat that the evals show.

## Indicative results at a glance

These are the results that changed what we think; sample sizes are in the sections below. Act 1 unless noted; greedy Claws.

| Test | Result | Verdict |
|---|---|---|
| Compass sampled (tau 0.35) vs greedy, same 100 saves | 12.5 vs 10.0 HP/combat (paired -2.45 +/- 0.70); boss beaten 29 vs 62 | **Sampling the compass is very harmful; the biggest single effect found** |
| Compass eps 0.10 (tau 0) vs greedy | 10.6 vs 10.0 (+0.43 +/- 0.34); boss beaten 63 vs 62 | Epsilon-greedy keeps the benefit and some card diversity (now the data-generation default) |
| Cards added at combat start only (hidden from the macro player), sampled compass | block cards 11.1, strong attacks 11.1, 4 Strikes 13.4 vs 12.5 base; elite 30 -> 22-23 | Card quality matters; block is not special |
| Replay regret, first play only (act 1: 1300 decisions; act 2: 412) | 0.93 / 1.26 HP per decision; block-instead-of-attack clearly better in 6 of 94 big-incoming attacks | The policy's plays are not systematically too aggressive |
| Replay regret by encounter, act 1 | Gardeners 2.13, Skulking Colony 1.88, Bygone Effigy 1.69 vs Terror Eel -0.24, The Kin 0.01 | Supports "Claws misplays specific enemy mechanics" (about 1.5-2 SE) |
| Value head vs real loss (616 combats, act 2) | corr 0.73, RMSE 13.9 vs 19.3 for the mean, bias +4 | Moderate predictor; cannot rank plays inside a turn |
| Fork training signal by incoming damage | Q tracks the true return; block over attack worth only 0.6 HP/energy at 16-25 incoming | Aimed correctly, weak |
| Compass card scores (seed 4242, greedy, 7 rewards) | options within 0.04-0.1 floors in 5 of 7 | Weak separation: its picks are close to coin flips |
| Alternating loop evals, act 2, 14 rounds after the relaunch | end floor 31.6-33.1, combat win 0.842-0.863; HP/combat 17.4-18.4 (only the last 4 rounds logged it) | No measurable improvement from more Claws training under the old data |

Act-1 damage by kind with the greedy compass: weak 0.8, normal 4.5, elite 20.0, boss 43.0 (project owner's expectation: elite 15-20, boss about 30).

## What was measured

### 1. Eval metrics were missing; they are in `evaluate.py` now
`evaluate.py` rows carry `hp_loss_per_combat`, `hp_loss_by_kind` (weak / normal / elite / boss) and the value head's first-decision forecast error
(`v_bias`, `v_rmse`; each combat also stores `v0`). Per-combat play statistics are stored under `pl` in the detail log: block-capable cards in
hand and played, energy by attack / block / other, incoming damage, block held at end of turn, unblocked damage, energy left.
Act-2 baseline (100 greedy runs): 18.5 HP per combat, V bias +3.7, V RMSE 13.9.

### 2. The value head is a moderate predictor, not a sharp one
Per combat (616 combats): corr 0.73, RMSE 13.9 HP against 19.3 for guessing the mean; biased high by about 4 HP. Worst by encounter: Ovicopter,
Entomancer, Infested Prisms, Decimillipede (RMSE 18-20). Its error is the same size as the differences between plays in a turn, so it cannot rank
individual plays.

### 3. The training signal is not biased toward attacking
Within groups of forks from the same state, Q = loss + V(next turn start) tracks the true combat return (corr 0.53; slope on V(next) 0.8) and
values energy sensibly by incoming damage (below 5 incoming: attack worth 4x skill; 16-25: skill worth 0.6 HP/energy more; above 25: equal).
The signal is correctly aimed but weak: Q explains about 28% of the variance in the return within a group, and the preference for block over
attack at 16-25 incoming is only about 0.6 HP per energy against noise of about 5 HP.
Note: forks are clones of the exact export (`World.from_export(..., sample=False)`), so all forks share one hidden draw order. `sample=True` does
not resample anything (it only changes what `hidden_key` means).

### 4. Replay in eval (`vakuu/claws/replay.py`, `--replay-per-run`) -- the policy's own choices cost little
At a turn start the combat is re-imported under N independent hidden-state samples (`resample_export`: draw pile order and every RNG state redrawn,
visible state untouched), every first play (no potions) is tried and the combat is finished greedily. Regret = cost of the greedy play minus the
cost of the best alternative picked on half the samples and measured on the other half (cross-fitted).
Act 2, 40 runs, 412 decisions, 12 hidden samples: mean regret 1.26 HP (SE 0.21), median 0, above 3 HP in 15%, above 6 HP in 8%. Attacking with 16+
incoming: a block was clearly better in 6 of 94 decisions. Switches to the better play: attack->block 35, block->attack 47 -- not skewed toward
aggression. Mostly which attack was chosen, not the target. Lower bound only (first play, policy continuation).

### 5. Claws spends its energy and plays block cards when it has them
Act 1 (100 runs): fewer than 2% of turns end with unspent playable energy. Elite turns: 1.64 block-capable cards in hand, 0.77 played (47%);
about 40% of incoming damage goes through. Giving it more block cards raises both its block played and its block held at end of turn. Energy split in
training mainlines is about 63% attack / 26% block / 11% other, and is the same in high- and low-damage fights (corr with damage about 0).
Act-1 elite fights last 6.2 turns on average, act-2 elites 4.4.

### 6. The deck matters (injection test)
Cards added to the starting deck of 100 fixed act-1 saves (same seeds), hidden from the macro player (`EVAL_HIDE_ADDED`) so only combat sees them
(sampled compass, `EVAL_STOP_ACT=0`):

| Added cards | HP / combat | Elite | Act-1 boss beaten |
|---|---|---|---|
| none | 12.5 | 30.1 | 29 |
| +2 Shrug It Off, +2 Iron Wave | 11.1 | 23.5 | 40 |
| +2 Twin Strike, +2 Pommel Strike | 11.1 | 22.1 | 43 |
| +4 Strike | 13.4 | 28.6 | 20 |

Better cards help whether they attack or block (paired: block -1.59 +/- 0.43, strong attacks -1.82 +/- 0.37, Strikes +0.82 +/- 0.32 HP per combat).
Act 2 (4 cards added to a built deck): no reliable effect. Effects of a single card are about 0.5 HP per combat and need roughly 300+ runs to read.

### 7. The compass: sampling hurts, and its card scores are almost flat
Same 100 act-1 saves, same seeds, same Claws:

| Compass | HP / combat | Normal | Elite | Boss | Boss beaten |
|---|---|---|---|---|---|
| sampled (tau 0.35, 3% uniform) | 12.5 | 8.1 | 30.1 | 47.7 | 29 |
| greedy | **10.0** | 4.5 | 20.0 | 43.0 | **62** |

Paired -2.45 +/- 0.70 HP per combat. Card rewards use the hazard net's expected-floors score of each deck-with-card (the tier list is only a fallback
and for shop / remove / upgrade); at most reward screens the four options (3 cards + skip) are within 0.04-0.1 floors, so the sampled compass is
close to a random pick and skip is as likely as the best card. Upgrade prompts (and event card choices) have no compass score and fall back to a
random pick. Seed 4242 (greedy compass, greedy Claws, reproducible): 77 identical actions across two runs, dies to the Vantom on floor 17.
Where the compass samples (data generation, compass runs, the Claws-only loop's eval) Claws sees weaker decks than a greedy compass builds.
`scripts/train_alternating.sh` already evaluates with `COMPASS_EPS=0 COMPASS_TAU=0`; `scripts/train_claws.sh` does not.

### 8. Act 1 by encounter, greedy compass + greedy Claws (100 runs)
Elites: Phrog Parasite 16.2, Terror Eel 14.7, Byrdonis 18.1 (in range); Skulking Colony 22.1, Phantasmal Gardeners 23.9 (median 16), Bygone Effigy 27.2.
Bosses (target about 30): Soul Fysh 33.7, Lagavulin Matriarch 31.2 (n=4), The Kin 37.4, Ceremonial Beast 40.2, Vantom 46.8 (win 0.43), Waterfall
Giant 54.0. 32% of boss fights end in death and a death counts as the HP it entered with; in won boss fights the loss is 37.9. Boss entry HP is
54-72 of 80.

### 9. Replay regret by encounter, act 1 (greedy compass + greedy Claws, 100 runs, 1300 decisions, 12 hidden samples; elite and boss turns always replayed)
Overall 0.93 HP per decision (SE 0.10); elite 1.36, boss 0.69, normal 0.81, weak 0.40. By encounter (n >= 60): Phantasmal Gardeners 2.13 (SE 0.56),
Skulking Colony 1.88 (0.51), Bygone Effigy 1.69 (0.53), Phrog Parasite 1.20 (0.30), Soul Fysh 1.18 (0.50), Waterfall Giant 0.86 (0.33), Vantom 0.71
(0.30), Ceremonial Beast 0.26 (0.47), The Kin 0.01 (0.38), Terror Eel -0.24 (0.26). Skulking Colony and Bygone Effigy are about twice the average
(about 1.5-2 SE above it), and for Skulking Colony the better play is usually a block instead of the attack (9 of 16 large-regret cases), which
fits the damage cap. Waterfall Giant is not above average on first-play regret: a "rush it, then block after the kill" plan needs several turns
and a first-play replay cannot see it. Regret is a lower bound and the per-encounter samples are small.

### 10. Seed 4242 (greedy compass, greedy Claws): the owner's card judgement against the compass
Compass picks were Spite (f2), Pommel Strike (f3), Blood Wall (f5), Thunderclap (f8, f13, f15), Second Wind (f14), with shop buys Fight Me,
Block Potion (f4) and Shrug It Off, Swift Potion (f11); it ended with three Thunderclaps and died to the Vantom on floor 17. The owner would
have taken Spite / Pommel Strike / Blood Wall or Tremble, Armaments (f8), Bloodletting (f13), Second Wind or Demon Form (f14), Skip or Dismantle
(f15). Every disagreement (f8, f13, f15) is within 0.1 expected floors of the compass's own top score (their picks ranked 4th, 4th and 2nd-3rd),
so the compass could not tell them apart. Upgrade and event card prompts fall back to a random pick. The run reproduces exactly under the same
seed with `COMPASS_TAU=0`.

## Conclusions so far
- No evidence of a systematic over-aggression in the policy's plays; the choice it makes at a turn start is rarely more than a few HP from the best
  alternative, and the training signal points the right way (but weakly).
- A large part of the damage comes from the decks, and the compass's card choices are weak (flat scores) and, when sampled, near random.
- Partly supported (project owner's observation; see section 9): Claws seems to lack per-turn / per-enemy mechanics knowledge -- Bygone Effigy (Slow: each card
  played raises damage taken), Skulking Colony (damage capped at 20 per turn, so extra damage is wasted), Waterfall Giant (rush it, then big block
  after the kill). The state carries enemy powers as tokens (`SLOW_POWER`, `HARDENED_SHELL_POWER`, ...), but these fights are rare in the data and a
  capped or wasted damage has no effect on the HP-loss label.

## Reproducing
- `CLAWS_GREEDY=1 python -m vakuu.claws.evaluate --runs N --workers W --resume-act 2 --replay-per-run 12 --replay-hidden 12 ...` (replay needs
  `CLAWS_GREEDY=1`; it runs the host in the SL env so the combat can be exported).
- Act-1 tests: `EVAL_STOP_ACT=0` ends a run after act 1; act-1 start saves were captured from `SpireHost` (env SL, `host.checkpoint`, key `"Save"`)
  and had cards appended to `players[0].deck`; `EVAL_HIDE_ADDED='{"SHRUG_IT_OFF": 2}'` hides them from the macro player.
- Greedy compass: `COMPASS_EPS=0 COMPASS_TAU=0` together with `GOLDEN_COMPASS_CHECKPOINT=...`.
- The scratch scripts used for the saves, the per-seed card log and the aggregations were not added to the repo.

## Script changes made with this investigation
`scripts/train_alternating.sh` and `scripts/train_claws.sh` run the compass epsilon-greedy (`COMPASS_EPS=0.1 COMPASS_TAU=0`; overridable with
`CLAWS_COMPASS_EPS` / `CLAWS_COMPASS_TAU`) where Claws' data is generated and in its held-out greedy lines; both evals are fully greedy
(`COMPASS_EPS=0 COMPASS_TAU=0`). Why 0.1: act-1 test, same 100 saves and seeds, HP lost per combat: sampled (tau 0.35, eps 0.03) 12.5,
eps 0.25 11.1, eps 0.10 10.6, greedy 10.0 (eps 0.10 minus greedy +0.43 +/- 0.34; boss clears 63 vs 62 vs 29 sampled). The remaining
exploration is meant to keep card diversity so neither Claws nor the compass gets stuck with a deck pattern the other never sees.
`train_compass.sh` still samples its compass for exploration.

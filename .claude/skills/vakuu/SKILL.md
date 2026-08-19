---
name: vakuu
description: Orients work on Vakuu, the Slay the Spire 2 autoplay project (Choices Paradox meta-option picker, Golden Compass path planner, Claws in-combat ML model, Whispering Earring self-play facility). Use when working on STS2 state extraction, the C# mod, headless/AutoSlay trial generation, in-combat state capture or restore, RNG determinism, the shared state schema, or the training data pipeline for any of the four subsystems. Also use when reading the decompiled STS2 source to answer "how does the game actually do X".
---

# Vakuu — Slay the Spire 2 autoplay

## The project

**Vakuu** is an AI that plays STS2. Four subsystems, deliberately not yet merged — whether they stay
separate is an open question, not a settled architecture:

| Subsystem | Decides | Engine action surface |
|---|---|---|
| **Choices Paradox** | Meta options: take/skip a card reward, shop priorities, relic vs. removal vs. card | `PickRelicAction`, card-reward selection, merchant screens |
| **Golden Compass** | Which path to take, given current state, foreseeable elite pool, and boss | `MoveToMapCoordAction`, `VoteForMapCoordAction`, `Core.Map` |
| **Claws** | Actually playing combat — the trained ML suggestion model. Hardest of the four. | `PlayCardAction`, `EndPlayerTurnAction`, `UsePotionAction` |
| **Whispering Earring** | Nothing itself — self-play facility that plays full runs (random/heuristic policy, delegating to Choices Paradox/Golden Compass for meta decisions) to generate Claws's training data | same surface as Claws, driven by whichever policy is wired in |

Conda environment: `conda activate sts2`

## Repo layout

```
mod/       C# Godot mod — state extraction + headless trial driver
schema/    state/action schema; the contract mod/ and agents/ both validate against
agents/    Python: choices_paradox/, golden_compass/, claws/, whispering_earring/
data/      generated trials — gitignored, reproduced from manifests, never committed
docs/
```

`mod/`'s "headless trial driver" job is now largely covered by `sts2-cli` (see below) — re-scope it to
extraction work that goes past what `sts2-cli`'s JSON protocol already exports, rather than building a
driver from scratch. Whether `mod/` is still needed at all, versus `agents/` just talking to
`sts2-cli` directly, is an open question — see `docs/README.md`.

**Two paths outside this repo:**

- `/Users/kylelau519/Programming/ChoicesParadox` — the **prototype**. A first pass at Choices Paradox:
  per-character XGBoost hurdle models trained on scraped run histories. Kyle has said it "has
  problems" and expects to redesign it; parts may be reused, but it is "just an idea." Treat it as
  prior art, not architecture. Don't carry over its structure, config layout, or feature engineering
  by default, and don't propose work whose value depends on keeping it.
- `/Users/kylelau519/Programming/sts2-decompiled` — read-only decompiled game source. **Never copy it
  into this repo and never commit any of it**; it's derived from a copyrighted game and Vakuu has a
  public remote. Reference it in place.
- `/Users/kylelau519/Programming/sts2-cli` — a **third-party headless driver**, not a Vakuu subsystem.
  It runs `sts2.dll` directly on bare .NET with **no Godot process at all** — `src/GodotStubs/`
  replaces `GodotSharp.dll` with no-op types, and `setup.sh` IL-patches `sts2.dll` (Mono.Cecil) so
  async waits, tweens, and `Cmd.Wait` resolve instantly instead of blocking on a renderer that isn't
  there. Driven over a JSON stdin/stdout protocol (`start_run`, `action`, `load_save`,
  `write_continue_save`, …; see its README). This has quietly solved most of what "Current phase"
  below used to treat as open engineering problems — read that section before assuming any of them
  still need solving. It needs its own `./setup.sh` run against a local Steam install (copies DLLs +
  patches them); those `lib/*.dll` are copied game binaries, same copyright status as
  `sts2-decompiled` — reference the repo, never vendor its DLLs into this one. It's under active
  development (see its own `CLAUDE.md`), so re-check its current state rather than trusting a stale
  read of it.

## Current phase: generating pseudo-play data

Real runs can't supply enough data, and STS2 ships no combat simulator you can drive. Three things
are needed; `sts2-cli` has changed the cost of each — read this before treating any of them as open:

1. **Extract everything in combat** — full observable state at each decision point. **Largely solved
   as a byproduct.** `sts2-cli`'s `combat_play` decision (`RunSimulator.cs: CombatPlayState`) already
   exports hand cards with *resolved* stats (replays `DynamicVars` preview so damage/block reflect
   Strength/Weak/Frail/current Block — the hard part, not the base card values), per-target damage for
   multi-target attacks, enemy HP/block/intent-damage/powers, player powers, and character mechanics
   (Defect orbs, Regent stars, Necrobinder's Osty). Gaps if training data needs more: draw/discard/
   exhaust piles are exported as counts only, not card lists; no floating-card tracking; no RNG stream
   cursor export. `ICombatState` is still not sufficient on its own if extending past what `sts2-cli`
   gives you — the card piles live on `PlayerCombatState`, not on the combat state.
2. **A restorable checkpoint format for in-combat state.** **Solved — via replay, not snapshot.**
   `sts2-cli`'s own checkpoint format *is* `{character, seed, actions: [...]}` (see `saves/*.json`
   there): every decision is index-based, so seed + the full action-index log reproduces a run
   deterministically. `agent/sts2_bridge.py replay <logfile> --until STEP` is a working reference for
   "replay to a branch point, then diverge" — exactly route A below, already implemented outside this
   repo. Don't build a custom `CombatState` snapshot serializer (route B) unless replay throughput
   measured through `sts2-cli` turns out insufficient — the native save (`write_continue_save`/
   `load_save`, `SerializableRun`) stays room-boundary only either way, so a snapshot serializer would
   still have to be hand-built from scratch; it isn't a smaller project than it looks.
3. **Headless mode that clones itself for new trials and records the results.** **Solved, differently
   than this repo assumed.** `sts2-cli` doesn't launch Godot at all, so the entire
   Steam-init/`--clientId`/`PreloadManager` chase in `agents/headless_runner/probe.py` (SIGABRT on
   asset preload) is moot for this route — no display driver, no asset preload, nothing to crash.
   Concurrency is just N `subprocess.Popen` of the `dotnet run` host; no save-dir collision handling
   needed since `SaveManager.InitProfileId(0)` is a dummy in-process profile, not a Steam-scoped tree.

Search is bounded by a policy-value approach, so breadth of generated data is not a constraint —
fidelity and determinism are. A determinism test (same seed + same action log → identical outcome) is
still worth running once against `sts2-cli` directly to confirm the replay route holds for this
project's needs before leaning on it for real trial generation.

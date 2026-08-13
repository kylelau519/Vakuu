---
name: vakuu
description: Orients work on Vakuu, the Slay the Spire 2 autoplay project (Choices Paradox meta-option picker, Golden Compass path planner, Whispering Earring in-combat agent). Use when working on STS2 state extraction, the C# mod, headless/AutoSlay trial generation, in-combat state capture or restore, RNG determinism, the shared state schema, or the training data pipeline for any of the three subsystems. Also use when reading the decompiled STS2 source to answer "how does the game actually do X".
---

# Vakuu — Slay the Spire 2 autoplay

## The project

**Vakuu** is an AI that plays STS2. Three subsystems, deliberately not yet merged — whether they stay
separate is an open question, not a settled architecture:

| Subsystem | Decides | Engine action surface |
|---|---|---|
| **Choices Paradox** | Meta options: take/skip a card reward, shop priorities, relic vs. removal vs. card | `PickRelicAction`, card-reward selection, merchant screens |
| **Golden Compass** | Which path to take, given current state, foreseeable elite pool, and boss | `MoveToMapCoordAction`, `VoteForMapCoordAction`, `Core.Map` |
| **Whispering Earring** | Actually playing combat. Hardest of the three. | `PlayCardAction`, `EndPlayerTurnAction`, `UsePotionAction` |

Conda environment: `conda activate sts2`

## Repo layout

```
mod/       C# Godot mod — state extraction + headless trial driver
schema/    state/action schema; the contract mod/ and agents/ both validate against
agents/    Python: choices_paradox/, golden_compass/, whispering_earring/
data/      generated trials — gitignored, reproduced from manifests, never committed
docs/
```

`mod/`'s "headless trial driver" job is now largely covered by `sts2-cli` (see below) — re-scope it to
extraction work that goes past what `sts2-cli`'s JSON protocol already exports, rather than building a
driver from scratch. Whether `mod/` is still needed at all, versus `agents/` just talking to
`sts2-cli` directly, is an open question — see `docs/README.md`.

`schema/` is the real interface between the C# and Python halves. A change to extracted state touches
`mod/` and `agents/` together — keep those commits atomic, and version the schema explicitly so a
dataset can name the schema version that produced it.

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

## Engine constraints that shape every design here

Verified against the decompiled source. Re-verify before relying on them for a new game build; the
game is in active development.

**There is no combat snapshot anywhere in the engine — saves stop at room boundaries, and even
multiplayer resyncs by replaying actions plus RNG, not by transferring state.** This is *why* route A
(replay, via `sts2-cli`) is the default per "Current phase" above, and why route B (a hand-built
`CombatState` snapshot serializer) stays the fallback rather than a shortcut — nothing in the engine
or in `sts2-cli` gives that serializer away for free. Full detail: `references/engine-map.md`
("Saves — and why they don't solve state restore").

**Why B can't be shortcut with the existing reflection machinery.** `SavedProperties.FromInternal`
walks `[SavedProperty]`-annotated properties on any `AbstractModel`, which looks like a free generic
snapshot serializer. It isn't: `grep -rl SavedProperty Models.Powers` returns **0 of 267** power
classes (vs. 40 of the relics), because powers never persist across room boundaries so nobody
annotated them. A generic reflection snapshot would get cards and relics right and silently drop
per-power state — exactly the failure mode that stays invisible until the training data is already
poisoned. `PowerModel` exposes `Amount`, `AmountOnTurnStart`, `SkipNextDurationTick`,
`Owner`/`Applier`/`Target`, `DynamicVars`; anything a specific power keeps in private fields is on
you to find.

**RNG is seedable, and full combat RNG state is ~12×5 integers.** A `Rng` is exactly
`{counter, s0, s1, s2, s3}` (`SerializableRng`). The combat-relevant streams are on **`RunRngSet`**
(12 streams, `RunRngType`): `Shuffle`, `CombatCardGeneration`, `CombatPotionGeneration`,
`CombatCardSelection`, `CombatEnergyCosts`, `CombatTargets`, `MonsterAi`, `CombatOrbs`, `Niche`, plus
three meta streams. `PlayerRngSet`'s `Rewards`/`Shops`/`Transformations` are meta-progression, *not*
combat — don't confuse the two sets. Monsters also carry a per-creature `Rng` derived at
`CombatState.cs:245` from seed + map coord + act + combat id: reconstructable, but its counter
advances, so capture it. `Core.TestSupport/TestRngInjector` overrides initial shuffle, combat card
generation, and relic rolls — the control surface for holding everything fixed except the decision
under test.

**`MegaCrit.Sts2.Core.AutoSlay/*` is dead code for this project — don't read into it.** The CLI path to
it is unreachable in the shipped build (`NGame.IsReleaseGame()` hard-returns `true` at `NGame.cs:751`,
gating `NGame.cs:684`), and `sts2-cli` doesn't route through `AutoSlayer` either — it drives
`RunSimulator` directly against `sts2.dll`. Nothing here is a live design option anymore; skip it
rather than spending tokens on it. One fact worth keeping without reading further: `AutoSlayConfig`
ramps +200 Strength/turn from turn 3, so if anything ever does touch `AutoSlayer`, data recorded
through it unmodified is not usable for training.

**Parallel headless instances — solved differently by `sts2-cli` than by launching the real game.**
`sts2-cli` never starts Godot, so none of this applies to it: no Steam popup-and-quit on failed init,
no `--clientId`/`UserDataPathProvider` save-tree collision to manage, nothing gated on
`NonInteractiveMode`/`TestMode.IsOn`. Concurrency is just N `subprocess.Popen` of the `dotnet run`
host. The bullets below are the **mod-based fallback** (launching the actual Godot game headless),
kept for if that route is ever needed instead of `sts2-cli`:

- `--force-steam=off` skips platform init and returns success (`NGame.cs:1376-1380`); without it, a
  failed init shows a popup and calls `GetTree().Quit()` (`.cs:1383-1400`).
- Save-dir isolation is a shipped flag, not a patch: `--clientId=<N>` sets the userId
  `NullPlatformUtilStrategy` reads (`.cs:29-32`), which is a path segment in
  `UserDataPathProvider.cs:11-17`.
- `NonInteractiveMode.IsActive` (gates animation/audio waits) is true when `TestMode.IsOn`, which has
  a public setter (`TestMode.cs:12`) — but setting it disables mod loading (`ModManager.cs:89`), so a
  mod must set it *after* mods finish loading.
- **Measured on v0.110.1, 2026-08-12** (`agents/headless_runner/probe.py`, superseded by `sts2-cli` for
  the headless-driver problem but kept as a record of this dead end): six instances booted to the main
  menu in 12.2s each, fully isolated save trees — concurrency wasn't the problem. Every instance then
  SIGABRT'd right after asset preload, inside Godot's dummy rendering driver. Untried fix:
  `PreloadManager.Enabled = false` (`Core.Assets/PreloadManager.cs:32`) set from a mod initializer,
  before the crashing preload at `NGame.cs:1031`.

**Mod hooks are official.** `ModHelper.SubscribeForCombatStateHooks` and `SubscribeForRunStateHooks`;
`ModManager` respects `--nomods`. Mods are ordinary .NET 9 assemblies loaded via
`AssemblyLoadContext` with HarmonyLib available, so patching anything is on the table.

**`ActionExecutor` is the recording seam.** It exposes `BeforeActionExecuted`,
`JustBeforeActionFinishedExecuting`, and `AfterActionExecuted`. Subscribing there yields (state,
action, next-state) transitions without patching individual actions.

See `references/engine-map.md` for where each of these lives.

## Working rules

- **Answer engine questions from the decompiled source, not from STS1 knowledge or memory.** STS2
  differs substantially. Grep it and cite `namespace-dir/File.cs:line`.
- Decompiled output is ILSpy-style: expect compiler-generated names (`--y__InlineArray*`,
  `<>c__DisplayClass`), inlined iterators, and lost local names. Structure and constants are
  reliable; control flow around `async`/`yield` is reconstructed, so read it skeptically.
- **Don't version generated data — version the recipe.** A dataset is fully described by
  `(game build, mod commit SHA, schema version, seed, policy checkpoint, config)`. Write that
  manifest next to each dataset; determinism makes the data itself reproducible.
- Game builds move fast. Model IDs, save schema, and `Core.Saves.Migrations.SerializableRuns/` are
  the things that break parsed data between versions — check them after an update.
- Redesign proposals are welcome and expected. When suggesting one, say plainly what it discards.

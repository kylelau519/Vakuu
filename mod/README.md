# mod

The C# Godot mod. Was scoped as two jobs — **extract game state** and **drive headless trials** —
but `/home/kylelau519/Programming/sts2-cli` (outside this repo) now does the headless-driving job
without a mod at all, and covers a good chunk of state extraction as a byproduct of its JSON
protocol. See `.claude/skills/vakuu/SKILL.md`'s "Current phase" section for specifics on what it
covers and what it doesn't, and `docs/README.md` for the open question of whether `mod/` is still
needed. Don't start scaffolding this without checking that section first — it may turn out `agents/`
can talk to `sts2-cli` directly instead.

If it does turn out to be needed (for gaps `sts2-cli` doesn't cover), generate it from Alchyr's STS2
mod template:

```bash
dotnet new install Alchyr.Sts2.Templates        # or: dotnet new install ./path/to/ModTemplate-StS2
dotnet new alchyrsts2mod --ModAuthor kylelau519 --PublicizeSts true
```

`PublicizeSts` matters here. It exposes non-virtual private and protected symbols in the STS2 DLL,
and much of what needs extracting — `CombatState._allCards`, `_nextCreatureId`, private pile state —
is not on the public `ICombatState` surface. Without it, extraction needs reflection.

## Design starting points

All paths are in the decompiled source (see `.claude/skills/vakuu/references/engine-map.md`).

- **Recording seam:** `ActionExecutor` exposes `BeforeActionExecuted`,
  `JustBeforeActionFinishedExecuting`, `AfterActionExecuted`. Subscribing there yields
  (state, action, next-state) transitions without patching individual actions.
- **Mod hooks:** `ModHelper.SubscribeForCombatStateHooks` / `SubscribeForRunStateHooks` are the
  official entry points.
- **Headless driver:** the game ships `--autoslay --seed=<s> --log-file=<f>`, gated on
  `!IsReleaseGame()`. Fork `AutoSlayer` rather than driving the UI from outside.
  **Disable the `AutoSlayConfig` Strength ramp first** — stock AutoSlay adds +200 Strength per turn
  from turn 3 to force fights to end, which makes any recorded data useless for training.
- **Determinism:** `Rng`, `PlayerRngSet` (serializable, named streams), and
  `Core.TestSupport/TestRngInjector` (overrides initial shuffle, combat card generation, relic rolls).
- **If a custom state serializer is written:** build on `PacketWriter`/`PacketReader` and
  `ModelIdSerializationCache` from `Core.Multiplayer.Serialization` rather than inventing a format.

Everything written here must match the version in `../schema/`.

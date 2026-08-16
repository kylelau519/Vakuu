# STS2 decompiled source map

Root: `/Users/kylelau519/Programming/sts2-decompiled` (read-only). Directories are namespace names,
so `MegaCrit.Sts2.Core.Combat/CombatState.cs` is the whole path. Verified against the build present
as of 2026-08-12 — re-check after game updates.

Always check `/Users/kylelau519/Programming/sts2-cli` before digging into the decompiled codes to save tokens.

## Combat

| File | What's there |
|---|---|
| `MegaCrit.Sts2.Core.Combat/CombatState.cs` | The live combat state object. Allies/enemies/creatures, `_allCards` (including floating cards not in any pile), `_nextCreatureId`, escaped creatures, encounter. Implements `ICardScope`. |
| `MegaCrit.Sts2.Core.Combat/ICombatState.cs` | The read surface: `Allies`, `Enemies`, `Creatures`, `PlayerCreatures`, `Players`, `Modifiers`, `RoundNumber`, `CurrentSide`, `Encounter`, `EscapedCreatures`, `CreaturesOnCurrentSide`, `HittableEnemies`, `RunState`. Start here — but it is **not sufficient**; see the two rows below. |
| `MegaCrit.Sts2.Core.Entities.Players/PlayerCombatState.cs` | **Where the piles actually are.** `Hand`, `DrawPile`, `DiscardPile`, `ExhaustPile`, `PlayPile` (`.cs:60-70`), `AllPiles`, `AllCards`, `Energy`, `MaxEnergy`, `Stars`, `TurnNumber` (`.cs:37` — prefer this over `ICombatState.RoundNumber`), `OrbQueue`, `Pets`. `CombatState._allCards` additionally holds floating cards in no pile. |
| `MegaCrit.Sts2.Core.Entities.Creatures/Creature.cs` | Per-creature state: `Block`, `CurrentHp`, `MaxHp`, `CombatId`, `Side`, `SlotName`, `Powers`, `IsHittable`, `IsStunned`, `PetOwner`. Enemy intent is `Monster.NextMove`; per-monster RNG is `Monster.Rng`. |
| `MegaCrit.Sts2.Core.Combat/CombatManager.cs` | Combat lifecycle. Also touches `NonInteractiveMode`. |
| `MegaCrit.Sts2.Core.Combat/CombatStateTracker.cs`, `CombatTurnState.cs`, `PlayerTurnPhase.cs`, `CombatSide.cs` | Turn/phase bookkeeping — needed to know *which* decision point a transition belongs to. |
| `MegaCrit.Sts2.Core.Combat.History/`, `MegaCrit.Sts2.Core.Combat.History.Entries/` | Per-combat event log the game itself keeps (~18 entry types). Possibly a cheaper extraction path than walking state. |

Card/creature/modifier models live under `MegaCrit.Sts2.Core.Entities.*` (`Cards`, `Creatures`,
`Players`, `Ancients`, `Ascension`, …).

## Actions — the decision + recording surface

`MegaCrit.Sts2.Core.GameActions/`

- Player-originated: `PlayCardAction`, `EndPlayerTurnAction`, `UndoEndPlayerTurnAction`,
  `UsePotionAction`, `DiscardPotionGameAction`, `PickRelicAction`, `MoveToMapCoordAction`,
  `VoteForMapCoordAction`, `VoteToMoveToNextActAction`, `ReadyToBeginEnemyTurnAction`.
- Each has a `Net*` twin (`NetPlayCardAction`, …) — that pairing is the evidence that actions, not
  states, are what the engine replicates.
- `ActionExecutor.cs` — `BeforeActionExecuted`, `JustBeforeActionFinishedExecuting`,
  `AfterActionExecuted`, `Pause()`, `Unpause()`, `Cancel()`, `FinishedExecutingActions()`,
  `CurrentlyRunningAction`, `IsPaused`. The hook point for recording transitions and for stepping a
  headless trial deterministically.
- `PlayerChoiceResult.cs` — how mid-action player choices resolve (discard-N, choose-a-card). These
  are extra decision points inside a single action; don't miss them when enumerating the action space.

`MegaCrit.Sts2.Core.Commands/` (`Cmd.cs` and ~20 builders) is the lower-level effect layer beneath
actions.

## Determinism and RNG

- `MegaCrit.Sts2.Core.Random/Rng.cs` — `Rng(ulong seed)`, `Rng(SerializableRng)`,
  `Rng(Player, ModelId, ulong mixin)`, `Rng(ulong seed, string name)`, `LoadFromSerializable`. Full
  `Next*` family plus `NextItem`/`WeightedNextItem`. Every `Next*` bumps `_counter`.
- `MegaCrit.Sts2.Core.Saves/SerializableRng.cs` — the whole of an RNG's state is
  `{counter, s0, s1, s2, s3}`. Both JSON and `IPacketSerializable`.
- **`MegaCrit.Sts2.Core.Runs/RunRngSet.cs` — the combat-relevant set.** 12 streams keyed by
  `Entities.Rngs/RunRngType`: `UpFront`, `Shuffle`, `UnknownMapPoint`, `CombatCardGeneration`,
  `CombatPotionGeneration`, `CombatCardSelection`, `CombatEnergyCosts`, `CombatTargets`, `MonsterAi`,
  `Niche`, `CombatOrbs`, `TreasureRoomRelics`. Reached as `RunState.Rng.<stream>`. Has `StringSeed`
  (display) and `Seed` (hashed, actual).
- `MegaCrit.Sts2.Core.Random/PlayerRngSet.cs` — a **different, meta-progression** set: `Rewards`,
  `Shops`, `Transformations` via `GetRng(PlayerRngType)`; `ToSerializable`/`FromSerializable`/
  `LoadFromSerializable`, `Seed`. Nothing in combat draws from here.
- Per-monster RNG: `CombatState.cs:236-245` — `monster.RunRng = RunState.Rng`, and
  `monster.Rng = new Rng(RunState.Rng.Seed + mapCoord.col + mapCoord.row + actIndex + creature.CombatId)`.
  Deterministically derivable, but the counter advances during combat, so snapshot it.
- Shuffle sites: `Entities.Cards/CardPile.cs:71-73` (initial, `UnstableShuffle` + `TestRngInjector`
  override + `Hook.ModifyShuffleOrder`) and `Commands/CardPileCmd.cs:1051-1072` (reshuffle,
  `StableShuffle(player.RunState.Rng.Shuffle)`, plus a `DebugForcedTopCardOnNextShuffle` escape hatch).
- `MegaCrit.Sts2.Core.Saves.Runs/SerializableRunRngSet.cs`, `SerializablePlayerOddsSet.cs`,
  `SerializableRunOddsSet.cs` — the persisted form.
- `MegaCrit.Sts2.Core.TestSupport/TestRngInjector.cs` — `SetRelicOverride<T>`,
  `SetRelicRarityOverride`, `SetCombatCardGenerationOverride(List<CardModel>)`,
  `SetInitialShuffleOverride(Action<List<CardModel>>)`, `Cleanup()`. Consume-once semantics on the
  `Consume*` getters.
- `MegaCrit.Sts2.Core.TestSupport/` also has `TestMode.cs`, `TestFlags.cs`, `ICardSelector.cs`,
  `TestCardSelector.cs`, `CardRewardSelection.cs` — a programmatic card-choice injection point,
  directly relevant to Choices Paradox trials.
- `MegaCrit.Sts2.Core.Multiplayer.Messages.Game/SyncRngMessage.cs` — how the game re-syncs RNG
  between peers; read it before designing branch points.

## Headless / AutoSlay

**Superseded by `/Users/kylelau519/Programming/sts2-cli`** (outside this repo) for the headless-driver
problem — it doesn't launch `NGame`/Godot at all, so none of this is on the path for that route. See
`../SKILL.md`'s "Current phase" section. Kept here as a compressed pointer, not a live plan, in case a
mod-based driver is ever revisited:

## Saves 

- `sts2-cli` offers up to in-combat restore through recording actions driven by option index.
It solved the in combat restore problem the native sts2 has. Running with same seed and same actions are checked to be faithful and identical to the in game.


## Multiplayer 

- Only single player are in the focus here, ignore everything about multiplayer.

## Modding

- Potentially we will mod through existing sts2 base-mod packages for simplicity.


`MegaCrit.Sts2.Core.Modding/` — `ModManager.cs` (honours `--nomods`), `Mod.cs`, `ModManifest.cs`,
`ModSettings.cs`, `ModInitializerAttribute.cs`, `ModLoadState.cs`, `ModSource.cs`,
`CombatHookSubscriptionDelegate.cs`, `RunHookSubscriptionDelegate.cs`, and `ModHelper.cs`:

- `SubscribeForRunStateHooks(string id, RunHookSubscriptionDelegate del)`
- `SubscribeForCombatStateHooks(string id, CombatHookSubscriptionDelegate del)`
- `IterateAllRunStateSubscribers(RunState)`, `IterateAllCombatStateSubscribers(CombatState)`
- `AddModelToPool<TPoolType, TModelType>()` — inject custom models into game pools.

Building one:

- The game assembly is `sts2.dll`, `net9.0`, `LangVersion 14.0`, `PlatformTarget Arm64`. Its own
  references (`sts2.csproj`) are `GodotSharp`, `Sentry`/`Sentry.Godot`, `Steamworks.NET`,
  `System.IO.Hashing`, `0Harmony`, `SmartFormat` — all sitting in the game install, so a mod project
  references them from there.
- `ModManifest.cs` — manifest JSON fields: `id`, `name`, `author`, `description`, `version`,
  `has_pck`, `has_dll`, `dependencies` (`{id, min_version}`), `affects_gameplay`, `min_game_version`.
- `ModInitializerAttribute.cs` — `[ModInitializer("MethodName")]` on a class names the entry point.
  Without it, `ModManager.cs:860` creates a Harmony instance and calls `PatchAll` for you.
- `ModManager.cs` loads DLLs through `AssemblyLoadContext` (`.cs:803`) and reads from both a mods
  directory and Steam Workshop (`ModSource`). Gates: `--nomods` (`.cs:84`) and
  `PlayerAgreedToModLoading` (`.cs:676`), the latter a one-time settings-save flag.
- `HasHarmonyPatches()` (`.cs:1085`) is what the game uses to decide a session is "modded".

## Dev console — manual state manipulation

`MegaCrit.Sts2.Core.DevConsole.ConsoleCommands/` (~40 commands): `FightConsoleCmd`, `RoomConsoleCmd`,
`TravelConsoleCmd`, `CardConsoleCmd`, `DrawConsoleCmd`, `ApplyPowerConsoleCmd`, `DamageConsoleCmd`,
`BlockConsoleCmd`, `EnergyConsoleCmd`, `HealConsoleCmd`, `KillConsoleCmd`, `RelicConsoleCmd`,
`PotionConsoleCmd`, `GoldConsoleCmd`, `ActConsoleCmd`, `EnchantConsoleCmd`, `UpgradeCardConsoleCmd`,
`RemoveCardConsoleCmd`, `DumpConsoleCmd`, `InstantConsoleCmd`, `GodModeConsoleCmd`, `WinConsoleCmd`,
`GetLogsConsoleCmd`. `DumpConsoleCmd` and `FightConsoleCmd` are the ones worth reading first — they
show how to construct an arbitrary encounter and dump state without a full run.

`FightConsoleCmd.cs:44` is the direct-to-combat primitive:
`RunManager.Instance.EnterRoomDebug(RoomType.Monster, MapPointType.Unassigned, encounterModel)` after
`ModelDb.GetById<EncounterModel>(id).ToMutable()`. This makes branch points much cheaper than
replaying a run from floor 1. **Skip the `encounterModel.DebugRandomizeRng()` call on the line above
it** — it destroys determinism, which is the one property the whole pipeline rests on.

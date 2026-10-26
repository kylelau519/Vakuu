# Vakuu

# Huge refactoring going on in refractor branch. The signal is too weak to be trained. RL Ideas cannot be tested because of unexpected problems. 

An AI that plays Slay the Spire 2.

Three subsystems, currently kept separate:

| Subsystem | Decides |
|---|---|
| **Choices Paradox** | Meta options — take/skip a card reward, shop priorities, relic vs. removal vs. card |
| **Golden Compass** | Which path to take, given current state, foreseeable elite pool, and boss |
| **Whispering Earring** | Playing combat itself |

## Layout

| Path | Contents |
|---|---|
| `mod/` | C# Godot mod — extracts game state and drives headless trials |
| `schema/` | State/action schema; the contract `mod/` and `agents/` both validate against |
| `agents/` | Python — models, training, and inference for the three subsystems |
| `data/` | Generated trial data. Gitignored; reproduced from manifests, never committed |
| `docs/` | Design notes |

## Status

Early. The current focus is generating pseudo-play data, because real runs can't supply enough of it
and STS2 ships no combat simulator you can drive:

1. Extract everything in combat.
2. Store in-combat state in a restorable format.
3. Headless mode that clones itself for new trials and records the results.

A prior prototype of Choices Paradox — per-character XGBoost hurdle models over scraped run
histories — lives in a separate `ChoicesParadox` repo. It's being redesigned, not carried over
wholesale.

## Requirements

- .NET 9 and Godot (mod)
- Python (agents)
- A local copy of Slay the Spire 2

Decompiled game source is used as reference during development but is never checked into this
repository.

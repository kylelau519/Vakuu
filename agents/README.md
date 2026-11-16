# agents

The Python half: models, training, and inference for the subsystems.

| Directory | Decides |
|---|---|
| `golden_compass/` | The non-combat run — map path and card reward (shop, rest, events to follow) — as a per-floor death-hazard net over run-state afterstates. Replaces the old Golden Compass and Choices Paradox |
| `claws/` | In-combat play — the ML suggestion model itself |
| `whispering_earring/` | Self-play facility — plays full runs to generate `claws`'s training data. After everything is trained this will be the module to pass live data to all the agents and return suggestions through modding. |

Whether these stay separate models or converge on a shared policy-value network is an open
question. The split is how the problem is currently carved up, not a committed architecture — a
shared state representation may well force the decision.

## Prior art

A prototype of Choices Paradox — per-character XGBoost hurdle models over scraped run histories —
lives in the separate `ChoicesParadox` repo; spire-zero's run value net inspired the current
golden_compass. Borrow ideas from them; don't inherit their structure by default.

## Reading data

Datasets are described by a manifest, not stored in git. Read the schema version out of the manifest
and validate against `../schema/` rather than assuming the current schema.

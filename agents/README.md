# agents

The Python half: models, training, and inference for the three subsystems.

| Directory | Decides |
|---|---|
| `choices_paradox/` | Meta options — card reward take/skip, shop priorities, relic vs. removal vs. card |
| `golden_compass/` | Path selection, given current state, foreseeable elite pool, and boss |
| `whispering_earring/` | In-combat play |

Whether these stay three separate models or converge on a shared policy-value network is an open
question. The split is how the problem is currently carved up, not a committed architecture — a
shared state representation may well force the decision.

## Prior art

A prototype of Choices Paradox — per-character XGBoost hurdle models over scraped run histories —
lives in the separate `ChoicesParadox` repo. It has known problems and is expected to be redesigned.
Borrow ideas from it; don't inherit its structure by default.

## Reading data

Datasets are described by a manifest, not stored in git. Read the schema version out of the manifest
and validate against `../schema/` rather than assuming the current schema.

# schema

Typed Python models for `sts2-cli`'s JSON protocol. `agents/` imports these directly rather than
re-deriving the wire format by reading `RunSimulator.cs` each time.

## Why this exists, and why it's Pydantic, not JSON Schema

`sts2-cli` (`github.com/wuhao21/sts2-cli`, outside this repo) drives the real STS2 engine headless
and emits decision-point JSON over stdin/stdout. Its README only documents the happy path; the real
shape — which fields are conditionally absent, what `stats` means in one decision vs. another, that
`select_cards` wants a comma-string not a JSON array, that enemy indices shift when something dies —
only exists by reading the C#. This directory is that knowledge, written down once as code instead
of prose, so it doesn't have to be re-derived.

It started as raw JSON Schema (draft 2020-12, `$ref`s across files) on the theory that the format
needed to be language-neutral — a contract between a C# writer and a Python reader, mirroring how
`mod/`'s README frames things. That's not the actual situation: nothing on the `sts2-cli` side reads
this schema, Python is the only consumer, and Kyle maintains `sts2-cli` directly rather than it being
an unpredictable third-party dependency. Given that, Pydantic models are strictly more useful than
inert JSON Schema files sitting behind a validator harness — they're directly importable, they
validate on construction, and an editor can autocomplete against them. If a JSON Schema export is
ever needed again, Pydantic can generate one from these models (`Model.model_json_schema()`).

## Usage

```python
import json
from schema.models import parse_decision, CombatPlay
from schema.actions import action

decision = parse_decision(json.loads(line))
if isinstance(decision, CombatPlay):
    for card in decision.hand:
        if card.can_play:
            ...

cmd = action("play_card", card_index=0, target_index=1)
proc.stdin.write(json.dumps(cmd) + "\n")
```

- `models.py` — one class per decision type (`CombatPlay`, `MapSelect`, `CardReward`,
  `CardSelect`, `BundleSelect`, `EventChoice`, `RestSite`, `Shop`, `GameOver`) plus shared pieces
  (`Card`, `PlayerSummary`, `Relic`, `Potion`, `Power`, `RunContext`). `parse_decision(raw_dict)`
  dispatches on the `"decision"` field and returns the right type; it raises
  `pydantic.ValidationError` on a shape mismatch.
- `actions.py` — `action(name, **kwargs)` builds and validates the outgoing
  `{"cmd": "action", ...}` command for all 17 action names, including the conditional-arg rules
  (`target_index` required only for `AnyEnemy` cards with ≥2 enemies alive, etc.) and normalizing
  ergonomic inputs where the wire format is awkward (`select_cards` accepts `indices=[0, 2]` and
  converts it to `"0,2"` — passing a raw list to `sts2-cli` itself silently fails).
- `VERSION.json` — the `sts2-cli` commit these were read from.

Treasure rooms, an already-resolved `rest_site`, and an all-locked `event_choice` never reach the
client as their own decision — `sts2-cli` fast-forwards through them internally — so there's no
model for a "treasure" decision; it doesn't exist on the wire.

## Why almost every field is Optional

`sts2-cli` serializes with `JsonIgnoreCondition.WhenWritingNull` — an absent key and a null value
are indistinguishable on the wire, and nearly every field *is* conditionally absent (`stats`,
`keywords`, `damage_by_target`, character-specific fields like `orbs`/`stars`/`osty`, …). Every
model here also allows extra fields (`extra="allow"`) rather than rejecting anything this file
hasn't caught up to yet.

## Keeping this in sync with `sts2-cli`

Since Kyle maintains both repos, the practical rule is: **when a `RunSimulator.cs` change touches a
decision's JSON output, update the matching class in `models.py` (or `actions.py` for the command
side) in the same sitting.** `ValidationError` at parse time is the fallback signal if that slips —
run a real captured decision through `parse_decision` after any `sts2-cli` update and let it tell you
what changed.

## Needs verification

`Potion` fields beyond `index` and `name` weren't independently confirmed against real captured
output while drafting this — extend `models.py`'s `Potion` class once real `combat_play`/`shop` JSON
has been captured and diffed against it.

## Dependency

`pydantic>=2` (`requirements.txt`). `requirements-dev.txt` adds `pytest` for the tests below. No
repo-wide packaging setup exists yet — fold these into whatever `agents/` ends up using when that
gets set up.

## Tests

```bash
pip install -r schema/requirements-dev.txt
pytest schema/
```

`tests/test_models.py` and `tests/test_actions.py`. The models are checked against
`tests/fixtures/ironclad_run_01.jsonl` — **real captured `sts2-cli` output**, not hand-written data
(a full Ironclad run driven start to finish by a scripted policy; see `tests/fixtures/README.md` for
exact provenance). That fixture is what caught a real gap while writing this: `Card.rarity` was
modeled as always-required, but `PlayerSummary.deck` cards omit it entirely (deck cards do carry
`upgraded`, which the original source-reading pass hadn't placed on the shared `Card` base at all).
Reading `RunSimulator.cs` gets the shape mostly right; it doesn't substitute for validating against
what the engine actually emits.

Four decision types (`card_select`, `bundle_select`, `rest_site`, `shop`) weren't reached by that run
and are covered by hand-built samples instead — same "unverified" status as everything else here that
hasn't been checked against live output. Capturing a run that reaches them and swapping in real
fixtures is a good task the next time this needs touching.

There's no test that drives `sts2-cli` itself yet (that requires a local Steam install +
`STS2_GAME_DIR` + a built `Sts2Headless` project, so it can't run unconditionally in every
environment) — the fixture file is the durable substitute. Re-capturing a fresh fixture after a
`sts2-cli` update, the way `tests/fixtures/README.md` describes, is the manual equivalent.

# fixtures

`ironclad_run_01.jsonl` is **real captured output**, not hand-written — every line is a raw decision
JSON object exactly as `sts2-cli` returned it while a dumb scripted policy (first playable card, else
end turn; first map choice; skip rewards when possible) played a full Ironclad run start to finish.

- Source: `sts2-cli` commit `f4d7538` (2026-08-13), character `Ironclad`, seed `vakuu-fixture-01`,
  ascension 0.
- 79 decisions: 70 `combat_play`, 4 `map_select`, 3 `card_reward`, 1 `event_choice` (Neow), 1
  `game_over`.
- Captured by running the real headless process end to end (`STS2_GAME_DIR` pointed at a local Steam
  install) and dumping every reply verbatim — this is what caught the `Card.rarity` bug (`rarity` is
  absent on `PlayerSummary.deck` cards, present on hand cards) that reading `RunSimulator.cs` alone
  missed. See `../../models.py`'s `Card`/`HandCard` split.

This run never reached `card_select`, `bundle_select`, `rest_site`, or `shop` — those four decision
types have no real fixture yet. The samples for them in `test_models.py` are hand-built from reading
`RunSimulator.cs`, same as this file's models originally were, and carry the same "unverified until a
real capture confirms it" caveat the `Potion` model has. Capture a run that reaches them (a longer
scripted policy, or `--character` variety, will eventually hit a shop or rest site) and replace those
hand-built samples with real ones when convenient.

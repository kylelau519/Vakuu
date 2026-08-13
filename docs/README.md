# docs

Design notes and decisions.

Durable engine facts about STS2 — what the save format does and doesn't contain, how the game
achieves determinism, where AutoSlay can and can't be reused — are recorded in
`.claude/skills/vakuu/` (`SKILL.md` and `references/engine-map.md`) so they load automatically when
working on this project. Put longer-form design rationale here.

Open questions worth writing up as they resolve:

- Do the three subsystems stay separate, or converge on a shared policy-value model?
- Replay-based trial cloning vs. a custom `CombatState` snapshot — and whether the snapshot ever
  earns its divergence risk. (Leaning replay: `sts2-cli` already implements it end-to-end — see
  `.claude/skills/vakuu/SKILL.md`.)
- What the reward signal is for each subsystem, given that only run outcome is directly observed.
- **Does `mod/` still need to exist? Decided (for now): no, not as a starting point.** `schema/` was
  drafted directly from `sts2-cli`'s decision JSON (`RunSimulator.cs`, commit `f4d7538`,
  2026-08-13 — see `../schema/VERSION.json`), not from `ICombatState`, and `agents/` will talk to
  `sts2-cli` over its stdin/stdout protocol rather than reading a mod-written export. `mod/` stays
  unscaffolded until a specific gap shows up that `sts2-cli` doesn't cover — full pile contents (only
  counts are exported), floating-card tracking, or RNG stream cursors are the known candidates. This
  is a working call from the current schema/trial-runner push, not a closed question forever: if
  replay throughput through `sts2-cli` turns out insufficient, or a subsystem needs one of those gaps,
  revisit.

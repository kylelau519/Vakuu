# data

Generated trial data. **Everything in here except this file is gitignored.**

Don't version datasets — version the recipe. With deterministic replay, a dataset is fully described
by the tuple that produced it, so each dataset directory carries a `manifest.json` recording at
minimum:

```json
{
  "game_build":      "0.10x",
  "mod_commit":      "<sha>",
  "schema_version":  "<n>",
  "seed":            "<run seed>",
  "policy_checkpoint": "<path or hash>",
  "config":          { "...": "generation settings" },
  "generated_at":    "<iso8601>"
}
```

That keeps the repo small, makes datasets reproducible rather than merely archived, and means a
dataset whose schema has since moved is still interpretable.

Game builds move fast. A dataset without `game_build` and `schema_version` recorded is not
trustworthy after an update.

# whispering_earring

Self-play data generation. It plays full runs end to end and records every decision, so that
trajectories exist to train on -- it doesn't decide combat or deckbuilding well itself. Today it
generates data for [`claws`](../claws/model/README.md), the in-combat model.

## How it works

A pool of worker processes, each owning one in-process `Sts2BufferSession`, runs **jobs** pulled from
two queues and reports `(job.uid(), JobResult)` on one result queue. Every job starts from a
`Checkpoint` and nothing else (`jobs.py`'s `Job.begin`).

- `MainlineJob` plays a whole run from the policy. Each combat is played from full HP (the run's HP is
  recorded, the player healed, and the combat's cost deducted afterwards). It saves natively before every
  room entry, so any turn of any combat can be reloaded cheaply, and returns a per-combat record (enemies,
  entry HP, loss, potions used, and per turn: start index, HP lost, HP lost to the combat's end).
- `ForkJob` reloads one mainline turn's checkpoint and plays the combat out. `forks_per_turn` of them
  are queued for every turn of every combat, and each writes its own file. The mainline never adopts a
  fork, so nothing waits on anything.
- `TrialJob` plays a whole run with the generic policy (random combat, heuristic cards and map), no HP
  tricks -- a plain job for checking the pipeline.

`facility_claws.py` keeps `--main-workers` runs open at once (a mainline plus its forks) until
`--trials` have started. Output goes to `--raw-dir`: `{main_id}.json` for a mainline and
`{main_id}_c###t##_f###.json` for each fork.

```
python facility_claws.py --trials 10 --branch-workers 12 --main-workers 2 --forks-per-turn 8 --raw-dir data/raw
CLAWS_CHECKPOINT=../claws/model/claws.pt python facility_claws.py --trials 10 ...   # claws net plays combat
```

## Layout

| File | Role |
|---|---|
| `jobs.py` | `Job`, `JobResult`, `MainlineJob`, `ForkJob`, `TrialJob` |
| `checkpoint.py` | `Checkpoint`: a run's full action history plus an optional native save, so any job resumes by `load_save` + replaying only what followed it |
| `workers.py` | `BufferWorker`: one session, pulls jobs, reports results |
| `facility_common.py` | `FacilityConfig`, `NewFacility` (worker pool, queues, stop), stderr relay / live status board, shared CLI flags |
| `facility_claws.py` | `FacilityClaws`: mainline + fork scheduling, CLI |
| `policy.py` | `choose_action` -- combat is uniform-random, or sampled from `ClawsNet` when `CLAWS_CHECKPOINT` is set; other decisions delegate to the choices_paradox / golden_compass policies |
| `recorder.py` | `CompactRecorder`: per-run action log with the policy recorded at each step |
| `utils.py` | seeds, log regexes, shared constants |
| `debug/` | repro and verification scripts (`debug_resume_job.py`: reload equals full replay; `debug_set_hp.py`: the `set_hp` log entry) |

`Checkpoint.reload` rebuilds a state from the last native save (always a map-level save: the engine rolls
an in-room save back to just before the room was entered) plus the actions since, which is bounded by one
room rather than the run.

Run the scripts in the `sts2` conda env, with `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1`.

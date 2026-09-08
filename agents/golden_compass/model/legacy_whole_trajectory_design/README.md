# Legacy checkpoint/dataset (whole-trajectory design)

`golden_compass.pt`, `dataset.pt`, and `backups/` here were produced under the *previous*
Golden Compass design (whole-trajectory terminal-reward tagging, combination-pooled action
space) -- see `../README.md`'s "Why the first design was replaced". Moved aside rather than
deleted: the state_dict shapes happen to still match the current architecture (same token types,
same `FEATURE_DIM`), so `train.py --init` would silently *load* one of these without erroring,
but the OPTION token's meaning changed (one physical node, not a pooled histogram) and the reward
scale changed too. Do not point `--init` at anything in here.

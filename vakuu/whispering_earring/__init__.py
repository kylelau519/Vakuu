"""Whispering Earring: the self-play facility. It decides nothing itself; it plays full runs on spire-zero in a pool of worker
processes and records every decision as training data for Claws and the Golden Compass.

    facility_claws.py    mainline runs whose every combat turn is forked in the Rust engine  (claws/job.py)   -> raw/*.pkl
    facility_compass.py  whole real-HP runs played by Claws + the compass                    (golden_compass/job.py) -> raw_runs/*.json
    facility_branch.py   paired branch evaluation of card-reward and route choices           (branch/job.py) -> raw_branch/*.json
    workers.py           the worker-process lifecycle they share, and act-save selection

The jobs themselves live with the subsystem whose data they produce.
"""

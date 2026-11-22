"""Where things live. Source is vakuu/; everything generated is gitignored: game data under data/, trained nets under runs/.

    data/act_checkpoints/   act-boundary saves a run can resume from (--resume-act)
    data/spire/             raw runs, training history and logs of the active run (SPIRE_DATA overrides)
    runs/spire/             the active run's nets: claws/claws.pt, compass/golden_compass.pt (SPIRE_RUN overrides)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("SPIRE_DATA", REPO / "data" / "spire"))
ACT_SAVES = REPO / "data" / "act_checkpoints"
RUN = Path(os.environ.get("SPIRE_RUN", REPO / "runs" / "spire"))
CLAWS_PT = RUN / "claws" / "claws.pt"
COMPASS_PT = RUN / "compass" / "golden_compass.pt"

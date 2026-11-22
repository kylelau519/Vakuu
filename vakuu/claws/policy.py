"""How Claws plays: which checkpoint, and how its action probabilities are turned into a move. Configured by the environment and
read once per process, so every worker of a facility plays by the same settings:

    CLAWS_CHECKPOINT   the net's .pt file (job.py raises when it is unset)
    CLAWS_GREEDY       any value: always the top-probability option (eval, and a fork's turns after the compared one)
    CLAWS_NOISE_EPS    fraction of probability mass replaced by Dirichlet noise (default 0.25, AlphaZero's own); the compass stage
                       sets it near 0 to play the deployed agent, so its labels are not mostly combat luck
"""
from __future__ import annotations

import os

CHECKPOINT = os.environ.get("CLAWS_CHECKPOINT")
GREEDY = bool(os.environ.get("CLAWS_GREEDY"))
NOISE_EPS = float(os.environ.get("CLAWS_NOISE_EPS", "0.25"))
# At or below this eps the noise is a rare deviation from the top move: (1 - eps) argmax + eps noise, so with 0.05 Claws plays its top
# move ~95% of the time. Above it the noise is exploration blended into Claws's own softmax, which is then sampled (the mainline and
# its forks, which want varied play).
ARGMAX_MIX_MAX_EPS = 0.1
DIRICHLET_ALPHA_SCALE = 10.0  # concentration = this / n_legal_options

_net = None  # lazy-loaded once per worker process


def get_net():
    """The ClawsNet of CLAWS_CHECKPOINT (None when it is unset)."""
    global _net
    if _net is None and CHECKPOINT:
        from vakuu.claws.model import ClawsNet
        _net = ClawsNet.from_checkpoint(CHECKPOINT)
    return _net

"""Per-trial JSONL recorder: one line per state received and per action sent, plus a
leading meta line -- the raw capture format offline preprocessing (segmentation +
encoding) reads later. Written for every trial regardless of outcome (win, loss, or a
mid-combat crash), since the point of a trial is the trajectory, not just its ending.
"""
from __future__ import annotations

import json
import os


class TrialRecorder:
    def __init__(self, trial_id: str, out_dir: str, meta: dict):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"{trial_id}.jsonl")
        self._file = open(self.path, "w")
        self._file.write(json.dumps({"type": "meta", "data": meta}) + "\n")

    def log_state(self, decision) -> None:
        self._file.write(json.dumps({"type": "state", "data": decision.model_dump(mode="json")}) + "\n")
        self._file.flush()

    def log_action(self, cmd: dict) -> None:
        self._file.write(json.dumps({"type": "action", "data": cmd}) + "\n")
        self._file.flush()

    def log_error(self, message: str) -> None:
        """Marks the trial as ended by an exception rather than game_over -- e.g. an
        Sts2ActionError from sts2-cli, or anything else that aborted the trial early.
        Lets offline preprocessing tell "the game ended" apart from "the trial crashed"."""
        self._file.write(json.dumps({"type": "error", "data": {"message": message}}) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

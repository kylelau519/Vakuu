"""Checkpoint / branch-job format for combat-level self-play amplification.

A checkpoint is the replay prefix (character, seed, ascension, actions-so-far) needed to
reconstruct a state in a fresh sts2-cli subprocess -- same shape as sts2-cli's own save
format ({character, seed, actions}), plus the bookkeeping (main_id, floor, combat_index,
branch_index) that gives every branch a collision-free id even across many floors and
many concurrently-running main lines.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class BranchJob:
    main_id: str
    floor: int
    combat_index: int
    branch_index: int
    character: str
    seed: str
    ascension: int
    actions: list[dict] = field(default_factory=list)
    policy: str = "explore"

    @property
    def branch_id(self) -> str:
        return f"{self.main_id}_f{self.floor:02d}c{self.combat_index:02d}_b{self.branch_index:02d}"


def write_branch_job(job: BranchJob, queue_dir: str) -> str:
    os.makedirs(queue_dir, exist_ok=True)
    path = os.path.join(queue_dir, f"{job.branch_id}.json")
    with open(path, "w") as f:
        json.dump(asdict(job), f)
    return path


def read_branch_job(path: str) -> BranchJob:
    with open(path) as f:
        data = json.load(f)
    return BranchJob(**data)

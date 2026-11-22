"""workers.act_saves: which files of a checkpoint directory a --resume-act run may start from."""
from __future__ import annotations

import pytest

from vakuu.whispering_earring.workers import act_saves


def test_act_saves_picks_only_that_acts_saves_not_sidecars(tmp_path):
    for name in ("Ironclad_a-1_act2.json", "Ironclad_a-1_act2.ckpt.json", "Ironclad_b-2_act2.json", "Ironclad_a-1_act3.json", "notes.txt"):
        (tmp_path / name).write_text("{}")
    got = [p.rsplit("/", 1)[-1] for p in act_saves(str(tmp_path), 2)]
    assert got == ["Ironclad_a-1_act2.json", "Ironclad_b-2_act2.json"]  # sorted, no act 3, no .ckpt.json sidecar


def test_act_saves_without_any_is_an_error_naming_the_directory(tmp_path):
    with pytest.raises(SystemExit, match="no act-2 saves"):
        act_saves(str(tmp_path), 2)

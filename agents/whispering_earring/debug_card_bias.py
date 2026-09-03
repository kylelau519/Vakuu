"""Checks the trained combat policy's play_card probability distribution for card-identity bias
-- whether some cards get assigned disproportionately high/low probability regardless of game
context, versus real context-dependent evaluation.

Reuses materialize.py's shared-prefix grouping (_scan_trials) so each turn-branch group's
decision point is replayed exactly once no matter how many sibling branches share it, and reads
each sibling's own recorded `policy` field (the network's logged softmax at generation time)
rather than resampling or re-scoring anything -- this is a read of what the net actually did
during the overnight run, not a fresh evaluation.

Usage:
    conda activate sts2
    python debug_card_bias.py data/raw/*.json
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "claws" / "model"))
import materialize  # noqa: E402  (_scan_trials)

import sts2_path  # noqa: E402,F401  (sets sys.path to sts2-cli/python before the imports below)
import buffer_layout as bl  # noqa: E402
from buffer_env import Sts2BufferSession, decode_token, Sts2ActionError  # noqa: E402


def _option_name(action: str, args: dict, hand_names: dict[int, str]) -> str:
    if action == "play_card":
        return hand_names.get(args["card_index"], "?")
    if action == "use_potion":
        return "<potion>"
    return "<end_turn>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    _, groups = materialize._scan_trials(args.paths)
    print(f"{len(groups)} turn-branch groups")

    prob_mass: dict[str, float] = defaultdict(float)   # card -> summed prob across all sightings
    sightings: dict[str, int] = defaultdict(int)        # card -> number of decisions offered in
    picks: dict[str, int] = defaultdict(int)            # card -> number of siblings that chose it

    session = Sts2BufferSession()
    skipped = 0
    for group in groups.values():
        try:
            decision = session.start(group.character, group.seed, group.ascension)
            for entry in group.prefix_actions:
                decision = session.step(entry["action"], **entry.get("args", {}))
        except Sts2ActionError:
            skipped += 1
            continue
        if decision.decision != "combat_play":
            skipped += 1
            continue

        arr = decision.arr
        hand_names = {idx: decode_token(bl.card_token(slot, 0)) for idx, slot in bl.hand_cards(arr)}

        for member in group.members:
            policy = member.get("policy")
            if not policy:
                continue
            for opt in policy:
                name = _option_name(opt["action"], opt["args"], hand_names)
                prob_mass[name] += opt["prob"]
                sightings[name] += 1
            picks[_option_name(member["action"], member.get("args", {}), hand_names)] += 1

    print(f"skipped {skipped}/{len(groups)} groups (replay failed or landed off combat_play)")

    rows = []
    for name, n in sightings.items():
        rows.append((name, n, picks.get(name, 0), prob_mass[name] / n))
    rows.sort(key=lambda r: -r[3])

    print(f"\n{'card':<24} {'n_offered':>10} {'n_picked':>10} {'avg_prob':>10}")
    for name, n, p, avg in rows:
        print(f"{name:<24} {n:>10} {p:>10} {avg:>10.4f}")


if __name__ == "__main__":
    main()

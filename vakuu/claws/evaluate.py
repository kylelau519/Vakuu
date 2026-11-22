"""Greedy act-2 evaluation of Claws on spire-zero: real HP, no healing, no forks, from random act-boundary saves; the macro
decisions are macro/heuristics.py's (stage 1). Prints one summary row and appends it to --log as a JSON line.

    CLAWS_CHECKPOINT=... CLAWS_GREEDY=1 python -m vakuu.claws.evaluate --runs 60 --workers 17 --ckpt-dir data/act_checkpoints \
        --log data/run2/eval_claw_spire.jsonl --label 12
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

from vakuu.workers import act_saves, end_pool, exit_now, make_pool


def _cid(c) -> str:
    if not isinstance(c, dict):
        return str(c)
    up = c.get("upgrade") or c.get("upgrades") or c.get("upgraded") or c.get("up")
    return f"{c.get('id')}{'+' if up else ''}"


def _hazard_probe(d, asc: int):
    """The compass's death hazard for the next floor, from a map decision (None without a compass checkpoint or off a map screen)."""
    if d.surface != "map" or not os.environ.get("GOLDEN_COMPASS_CHECKPOINT"):
        return None
    try:
        import torch
        from vakuu.buffer.run_encode import encode_run
        from vakuu.golden_compass import agent as compass, hazard_net as hn, run_state as rs
        net = compass.get_net()
        dec = encode_run(d)
        if net is None or dec is None:
            return None
        st = rs.encode_from_buffer(dec.arr, d.obs["run"]["character"], asc, getattr(d, "remaining", None))
        with torch.no_grad():
            lg, _aux = net(compass._encoder.collate([st]))
        h = torch.sigmoid(lg)[0]
        i = hn.head_floor(st.act, st.floor)
        return {"floor": int(st.floor), "act": int(st.act), "hazard": round(float(h[i]), 4), "next": [round(float(x), 4) for x in h[i:i + 6]],
                "h": [round(float(x), 3) for x in h]}  # the whole head: index = hn.head_floor(act, floor) of the combat it precedes
    except Exception:  # noqa: BLE001 -- a probe must never end a run
        return None


def _play(task: tuple) -> dict:
    save, main_id, asc, character, seed = task
    import torch
    torch.set_num_threads(1)
    from vakuu.claws import job as J
    from vakuu.engine.actions import random_pick, to_action
    from vakuu.engine.host import SpireHost
    J._load()
    from vakuu.macro.player import MacroPlayer
    player = MacroPlayer(random.Random(J.zlib_hash(main_id)), asc)
    J.seed_everything(J.zlib_hash(main_id))
    rng = random.Random(J.zlib_hash(main_id))
    out = {"main_id": main_id, "floor": 0, "act": 0, "won": False, "combats": [], "error": None}
    host = None
    try:
        host = SpireHost.start(save=save, character=character, seed=seed, asc=asc, env="NoSL")  # no exports: nothing forks here
        d, n, same, last, probe = host.decision, 0, 0, None, None
        while d is not None and n < 4000:
            n += 1
            run = d.obs.get("run", {})
            out["floor"], out["act"] = run.get("floor", out["floor"]), run.get("act", out["act"])
            if d.surface == "map":
                probe = _hazard_probe(d, asc) or probe
            if d.surface == "combat":
                obs = d.obs["screen"]["combat"]
                if not out["combats"] or out["combats"][-1].get("_enc") != (obs["encounter"], run.get("floor")):
                    out["combats"].append({"encounter": obs["encounter"], "act": run.get("act"), "floor": run.get("floor"), "hp_in": obs["player"]["hp"],
                                           "max_hp": run.get("max_hp"), "deck_n": len(run.get("deck") or []), "relics_n": len(run.get("relics") or []),
                                           "pred": probe, "turns": 0, "potions_n": len(run.get("potions") or []), "gold": run.get("gold"),
                                           **({"deck": sorted(_cid(c) for c in run.get("deck") or []), "relics": [r.get("id") for r in run.get("relics") or []]}
                                              if str(obs["encounter"]).endswith(("_BOSS", "_ELITE")) else {}), "_enc": (obs["encounter"], run.get("floor")), "won": None})
                out["combats"][-1]["turns"] = max(out["combats"][-1]["turns"], int(obs.get("round") or 0))
                dec, options, idx, _p, _v = J.decide(obs, d.legal, J.run_context(run))
                out["combats"][-1].setdefault("v0", round(float(_v), 2))  # the value head's HP-loss forecast at the combat's first decision
                d = host.act(to_action(*options[idx], obs, d.legal))
                continue
            if d.surface == "card_choice" and "combat" in d.obs["screen"]:
                ch = d.obs["screen"]["choice"]
                d = host.act(random_pick(ch["min"], ch["max"], len(ch["options"]), rng))
                continue
            if out["combats"] and out["combats"][-1]["won"] is None:
                out["combats"][-1]["won"] = True
                out["combats"][-1]["hp_out"] = run.get("hp")
            sig = (d.surface, json.dumps(d.legal), run.get("hp"))
            same = same + 1 if sig == last else 0
            last = sig
            if same >= 25:
                out["error"] = "stalled"
                break
            d = host.act(player.act(d, same))
        if d is None:
            if out["combats"] and out["combats"][-1]["won"] is None:
                out["combats"][-1]["won"] = False
                out["combats"][-1]["hp_out"] = 0
            out["won"] = bool(host.end and host.end.get("won"))
            if host.end and "floor" in host.end:
                out["floor"] = host.end["floor"]
    except BaseException as e:  # noqa: BLE001
        import traceback
        out["error"] = f"{type(e).__name__}: {e} [" + " < ".join(f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in reversed(traceback.extract_tb(e.__traceback__)[-4:])) + "]"
    finally:
        if host is not None:
            host.close()
    for c in out["combats"]:
        c.pop("_enc", None)
    return out


def summarize(rows: list[dict], label: str) -> dict:
    ok = [r for r in rows if not r["error"]]
    floors = [r["floor"] for r in ok]
    mean = sum(floors) / max(1, len(floors))
    se = math.sqrt(sum((f - mean) ** 2 for f in floors) / max(1, len(floors) - 1) / max(1, len(floors))) if len(floors) > 1 else 0.0
    combats = [c for r in ok for c in r["combats"] if c["won"] is not None]
    def boss_of(act: int):
        return [c for c in combats if str(c["encounter"]).endswith("_BOSS") and c.get("act") == act]
    b1, b2 = boss_of(0), boss_of(1)
    # HP lost per combat (hp_in - hp_out; a death counts as the HP it entered with) and how well the value head's first forecast matched it
    def loss_of(c):
        return c["hp_in"] - (c.get("hp_out") or 0) if c.get("hp_in") is not None else None
    def kind_of(c):
        e = str(c["encounter"])
        return "boss" if e.endswith("_BOSS") else "elite" if e.endswith("_ELITE") else "weak" if e.endswith("_WEAK") else "normal"
    losses = [(c, loss_of(c)) for c in combats if loss_of(c) is not None]
    by_kind = {k: [l for c, l in losses if kind_of(c) == k] for k in ("weak", "normal", "elite", "boss")}
    vp = [(c["v0"], l) for c, l in losses if c.get("v0") is not None]
    vstat = ({"v_bias": round(sum(v - l for v, l in vp) / len(vp), 2), "v_rmse": round(math.sqrt(sum((v - l) ** 2 for v, l in vp) / len(vp)), 2)}
             if vp else {})
    acts = [r["act"] + 1 for r in ok]
    return {
        "label": label, "runs": len(ok), "errors": len(rows) - len(ok), "mean_end_floor": round(mean, 2), "floor_se": round(se, 2),
        "best_floor": max(floors, default=0), "wins": sum(r["won"] for r in ok),
        "combats": len(combats), "combat_win_rate": round(sum(c["won"] for c in combats) / max(1, len(combats)), 3),
        "act1_reached_boss": len(b1), "act1_beat_boss": sum(c["won"] for c in b1),
        "act2_reached": sum(a >= 2 for a in acts), "act2_reached_boss": len(b2), "act2_beat_boss": sum(c["won"] for c in b2),
        "hp_loss_per_combat": round(sum(l for _, l in losses) / max(1, len(losses)), 2),
        "hp_loss_by_kind": {k: round(sum(v) / len(v), 1) for k, v in by_kind.items() if v}, **vstat,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ascension", type=int, default=1)
    ap.add_argument("--ckpt-dir", default="data/act_checkpoints")
    ap.add_argument("--resume-act", type=int, default=0, help="0: fresh runs from the start (default); N: from random act-N saves")
    ap.add_argument("--characters", nargs="*", default=["IRONCLAD"])
    ap.add_argument("--log", default=None)
    ap.add_argument("--detail-log", default=None, help="one JSON line per run with every combat (encounter, hp in/out, turns, the compass's hazard), for the analysis panel")
    ap.add_argument("--label", default=time.strftime("%d.%H%M"))
    ap.add_argument("--seed", type=int, default=None, help="fix the games (game seeds and run ids), so two checkpoints play the same ones")
    args = ap.parse_args()
    rng = random.Random(args.seed if args.seed is not None else os.urandom(4))
    run_id = (lambda: f"eval{rng.randrange(16**6):06x}") if args.seed is not None else (lambda: f"eval{os.urandom(3).hex()}")
    tasks = []
    if args.resume_act > 0:
        saves = act_saves(args.ckpt_dir, args.resume_act)
        tasks = [(rng.choice(saves), run_id(), args.ascension, None, 1) for _ in range(args.runs)]
    else:
        tasks = [(None, run_id(), args.ascension, rng.choice(args.characters).upper(), rng.randrange(1, 2**31 - 1))
                 for _ in range(args.runs)]
    from concurrent.futures import as_completed
    pool = make_pool(args.workers)
    rows = []
    try:
        futs = [pool.submit(_play, t) for t in tasks]
        for f in as_completed(futs):
            rows.append(f.result())
            print(f"\r{len(rows)}/{args.runs}", end="", flush=True)
    finally:
        end_pool(pool)
    row = summarize(rows, args.label)
    print()
    print(f"runs {row['runs']}/{args.runs} (errors {row['errors']}) | floor {row['mean_end_floor']}±{row['floor_se']} best {row['best_floor']} | "
          f"combats {row['combats']} win {row['combat_win_rate']:.0%} | act-1 boss {row['act1_beat_boss']}/{row['act1_reached_boss']} | "
          f"reached act 2 {row['act2_reached']} (boss {row['act2_beat_boss']}/{row['act2_reached_boss']}) | wins {row['wins']}")
    print(f"HP lost per combat {row['hp_loss_per_combat']} {row['hp_loss_by_kind']} | V bias {row.get('v_bias')} rmse {row.get('v_rmse')}")
    for r in rows:
        if r["error"]:
            print("  error:", r["error"][:150])
            break
    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
        with open(args.log, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    if args.detail_log:
        with open(args.detail_log, "a") as fh:
            for r in rows:
                fh.write(json.dumps({"label": args.label, **{k: r[k] for k in ("main_id", "floor", "act", "won", "error", "combats")}}) + "\n")
    exit_now()


if __name__ == "__main__":
    main()

"""Text panel over data/spire/eval_combats.jsonl (written by evaluate.py --detail-log): deaths and damage per encounter, and the compass's
predicted death chance against what happened per floor.   python -m vakuu.claws.eval_panel [LABEL ...] [--bosses] [--elites]   (default: the newest label, the general panel;
the flags print the boss / elite panels; LABEL 'all' pools every label in the file)"""
import json
import statistics as st
import sys
from collections import defaultdict

PATH = "data/spire/eval_combats.jsonl"


def main() -> None:
    rows = [json.loads(l) for l in open(PATH)]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    labels = [a for a in sys.argv[1:] if not a.startswith("--")] or [rows[-1]["label"]]
    runs = [r for r in rows if (r["label"] in labels or labels == ["all"]) and not r["error"]]
    C = []
    for r in runs:
        for c in r["combats"]:
            if c.get("won") is None or c.get("hp_out") is None:
                continue
            c["share"] = max(0, c["hp_in"] - c["hp_out"]) / max(1, c["max_hp"])
            h = (c.get("pred") or {}).get("h")
            c["pred_h"] = h[c["floor"] - 1] if h and 0 <= c["floor"] - 1 < len(h) else None  # head index = the combat's floor - 1, both acts
            C.append(c)
    bar = lambda x, mx, w=20: "#" * round(w * x / mx) if mx else ""
    q90 = lambda v: sorted(v)[int(.9 * (len(v) - 1))]
    print(f"labels {labels}: {len(runs)} runs, {len(C)} combats, {sum(not c['won'] for c in C)} deaths\n")

    by = defaultdict(list)
    for c in C:
        by[c["encounter"]].append(c)
    tab = sorted(((sum(not c["won"] for c in cs), e, cs) for e, cs in by.items()), key=lambda t: (-t[0], -len(t[2])))
    print("BY ENCOUNTER   (dmg = HP lost as % of max HP; 'HP in' = HP going in)")
    print(f"{'encounter':28s} {'n':>4s} {'died':>4s} {'death%':>6s} {'mean':>5s} {'p90':>5s} {'HP in':>5s} {'turns':>5s}  mean damage")
    for d, e, cs in tab[:24]:
        sh = [c["share"] for c in cs]
        print(f"{e[:28]:28s} {len(cs):4d} {d:4d} {d/len(cs):6.0%} {st.mean(sh):5.0%} {q90(sh):5.0%} {st.mean(c['hp_in']/c['max_hp'] for c in cs):5.0%} {st.mean(c['turns'] for c in cs):5.1f}  {bar(st.mean(sh), 1)}")
    print(f"... {max(0, len(tab) - 24)} more\n")

    print("BY FLOOR   compass predicted death chance (before entering) vs actual")
    print(f"{'floor':>7s} {'n':>4s} {'pred':>6s} {'actual':>6s} {'mean dmg':>8s} {'p90':>5s}  pred | actual")
    bf = defaultdict(list)
    for c in C:
        bf[(c["act"], c["floor"])].append(c)
    for k in sorted(bf):
        cs = bf[k]
        ps = [c["pred_h"] for c in cs if c["pred_h"] is not None]
        if len(cs) < 8 or not ps:
            continue
        pr, ac, sh = st.mean(ps), sum(not c["won"] for c in cs) / len(cs), [c["share"] for c in cs]
        print(f"a{k[0]+1} f{k[1]:<3d} {len(cs):4d} {pr:6.1%} {ac:6.1%} {st.mean(sh):8.0%} {q90(sh):5.0%}  {bar(pr, .8, 16):<16s}|{bar(ac, .8, 16)}")

    pr = [(c["pred_h"], 0 if c["won"] else 1) for c in C if c["pred_h"] is not None and c["pred_h"] > 0.001]
    if len(pr) > 50:
        print("\nCALIBRATION (combats with predicted hazard > 0.1%)")
        for lo, hi in ((0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, .4), (.4, .6), (.6, 1.01)):
            s = [(p, y) for p, y in pr if lo <= p < hi]
            if len(s) >= 5:
                print(f"  predicted {lo:4.0%}-{min(hi,1):4.0%}  n={len(s):4d}  mean pred {st.mean(p for p, _ in s):5.1%}  actual {st.mean(y for _, y in s):5.1%}")


    def hist(vals, edges, w=8):
        n = [sum(lo <= v < hi for v in vals) for lo, hi in zip(edges, edges[1:])]
        m = max(n) or 1
        return " ".join(f"{'#' * round(w * x / m):<{w}s}" for x in n), n


    def group_panel(title: str, suffix: str) -> None:
        G = [c for c in C if str(c["encounter"]).endswith(suffix)]
        if not G:
            print(f"\n{title}: no combats")
            return
        won = lambda cs: [c for c in cs if c["won"]]
        lost = lambda cs: [c for c in cs if not c["won"]]
        mean = lambda v: st.mean(v) if v else float("nan")
        print(f"\n==================== {title}: {len(G)} fights, {len(lost(G))} lost ({len(lost(G))/len(G):.0%}) ====================")
        print("PER ENCOUNTER   win%  | HP lost, histogram of 0-20 20-40 40-60 60-80 80-99 100% (died) | HP in | turns win/loss | deck win/loss | compass pred vs actual death")
        edges = [0, .2, .4, .6, .8, .995, 9]
        by = defaultdict(list)
        for c in G:
            by[c["encounter"]].append(c)
        for e, cs in sorted(by.items(), key=lambda t: (-sum(not c["won"] for c in t[1]) / len(t[1]), -len(t[1]))):
            h, cnt = hist([c["share"] for c in cs], edges, 6)
            ps = [c["pred_h"] for c in cs if c["pred_h"] is not None]
            print(f"{e[:26]:26s} n={len(cs):3d} {1-len(lost(cs))/len(cs):4.0%} | {h} | {mean([c['hp_in']/c['max_hp'] for c in cs]):4.0%} | "
                  f"{mean([c['turns'] for c in won(cs)]):4.1f}/{mean([c['turns'] for c in lost(cs)]):4.1f} | "
                  f"{mean([c['deck_n'] for c in won(cs)]):4.1f}/{mean([c['deck_n'] for c in lost(cs)]):4.1f} | "
                  f"{mean(ps):4.0%} vs {len(lost(cs))/len(cs):4.0%}")

        def bucket(title2, key, edges2, fmt):
            print(f"\nLOSS RATE BY {title2}")
            for lo, hi in zip(edges2, edges2[1:]):
                cs = [c for c in G if lo <= key(c) < hi]
                if len(cs) >= 5:
                    print(f"  {fmt(lo, hi):>12s}  n={len(cs):4d}  lost {len(lost(cs))/len(cs):4.0%}  {bar(len(lost(cs))/len(cs), 1, 30)}")
        bucket("HP GOING IN (share of max)", lambda c: c["hp_in"] / c["max_hp"], [0, .5, .7, .85, .95, 1.01], lambda lo, hi: f"{lo:.0%}-{min(hi,1):.0%}")
        bucket("DECK SIZE", lambda c: c["deck_n"], [0, 15, 18, 21, 24, 99], lambda lo, hi: f"{lo}-{hi-1}")
        bucket("RELIC COUNT", lambda c: c["relics_n"], [0, 3, 4, 5, 6, 99], lambda lo, hi: f"{lo}-{hi-1}")
        bucket("FLOOR", lambda c: c["floor"], [0, 9, 13, 17, 25, 99], lambda lo, hi: f"{lo}-{hi-1}")
        with_deck = [c for c in G if c.get("deck")]
        print(f"\nCARDS IN THE DECK, winners vs losers ({len(with_deck)} fights with deck data; only cards in >= 12 decks; presence rate)")
        if len(with_deck) < 40:
            print("  (not enough yet: the loop's evals log decks for boss and elite fights, so this fills in as rounds finish)")
            return
        cnt = defaultdict(lambda: [0, 0, 0, 0])  # in winning decks, in losing decks
        nw, nl = len(won(with_deck)), len(lost(with_deck))
        for c in with_deck:
            for card in set(c["deck"]):
                cnt[card][0 if c["won"] else 1] += 1
        rows2 = [(card, a / nw, b / nl, a + b) for card, (a, b, _, _) in cnt.items() if a + b >= 12 and nw and nl]
        rows2.sort(key=lambda r: r[1] - r[2])
        print(f"  more common in LOSING decks (n lost {nl}):  " + ", ".join(f"{c[:18]} {w:.0%}/{l:.0%}" for c, w, l, _ in rows2[:8]))
        print(f"  more common in WINNING decks (n won {nw}): " + ", ".join(f"{c[:18]} {w:.0%}/{l:.0%}" for c, w, l, _ in rows2[::-1][:8]))
        print("  (win%/loss% = share of winning / losing fights whose deck had the card: associations only, not causes)")


    if "--bosses" in flags:
        group_panel("BOSS PANEL", "_BOSS")
    if "--elites" in flags:
        group_panel("ELITE PANEL", "_ELITE")


if __name__ == "__main__":
    main()

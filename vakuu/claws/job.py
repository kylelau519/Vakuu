"""One Claws data job on spire-zero: a MAINLINE run from an act-boundary save, every combat played from full HP (healed at
its start, the loss taken off the run's real HP after it), and at every
combat turn of the mainline `forks_per_turn` FORKS: the turn's live combat is imported into the Rust engine, cloned, and
each clone is played out by the same policy (sampled differently), no reload and no game involved.

The result is one raw file (raw_dir/{main_id}.pkl) of per-turn costs and ready tokenized examples; materialize.py turns
the files of many runs into the dataset the Claws trainer reads (advantages from the groups, no simulator needed)."""
from __future__ import annotations

import json
import os
import pickle
import random
import time
import zlib

import numpy as np
import torch

from vakuu.buffer.decision import BufferDecision
from vakuu.buffer.encode import encode_combat
from vakuu.buffer.state_key import combine, visible_key
from vakuu.claws import fast_tokens, policy as P
from vakuu.engine.actions import random_pick, to_action
from vakuu.engine.engine import World
from vakuu.engine.host import HostError, SpireHost
from vakuu.macro.player import MacroPlayer

MAX_COMBAT_TURNS = 20
# A fork explores (sampled, noisy) only in the turn being compared; every later turn it plays greedily, as eval does. The cost to the
# combat's end then carries less exploration noise, and the value head learns what a greedy continuation costs.
MAX_RUN_DECISIONS = 4000

_net = None
_gvocab = None
_tok = None   # model.tokenize_from_buffer, set by _load()
_opts = None  # model.legal_options_from_buffer, set by _load()


def _load():
    global _net, _gvocab, _tok, _opts
    if _net is None:
        _net = P.get_net()
        if _net is None:
            raise SystemExit("CLAWS_CHECKPOINT is not set")
        from vakuu.claws import model as claws_model
        from vakuu.common.tokenizer import EntityVocab
        fast_tokens.install()  # model._pile_tensors := the NumPy version (identical output, about 5x faster)
        _tok, _opts = claws_model.tokenize_from_buffer, claws_model.legal_options_from_buffer
        _gvocab = claws_model.GlobalVocab(EntityVocab.load())
    return _net


def zlib_hash(text: str) -> int:
    return zlib.crc32(text.encode())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def run_context(run: dict) -> dict:
    """What the combat observation alone does not know, from the host's run view (also used by the forks)."""
    return {"floor": run.get("floor", 0), "act_index": run.get("act", 0) + 1, "deck": run["deck"], "relics": run["relics"]}


class ForkRng:
    """One fork's own random draws (seeded by its uid), so forks that advance in lockstep still reproduce one by one."""

    def __init__(self, seed: int):
        self.np = np.random.default_rng(seed)
        self.torch = torch.Generator().manual_seed(seed)
        self.py = random.Random(seed)


def _sample(options, logits, obs, legal, rng: ForkRng | None = None, greedy: bool = False) -> tuple[int, np.ndarray]:
    """Claws' choice among the options the host lists: policy.py's sampling (softmax mixed with root Dirichlet noise, an argmax mix
    when CLAWS_NOISE_EPS is small, or argmax under CLAWS_GREEDY). Plain NumPy: a decision has a dozen options, and small torch ops cost
    more than the arithmetic."""
    legal_set = set(legal)
    ok = np.array([to_action(*o, obs, legal) in legal_set for o in options])
    lg = logits.detach().numpy().astype(np.float64)
    if not ok.all():  # an option the host does not list is not playable now (a potion that cannot be used, say)
        lg = np.where(ok, lg, -np.inf)
    e = np.exp(lg - lg.max())
    probs = e / e.sum()
    if greedy:  # the top option, no exploration (a fork's turns after the compared one)
        return int(np.argmax(probs)), probs
    if not P.GREEDY:
        n = len(options)
        alpha = P.DIRICHLET_ALPHA_SCALE / n
        dirichlet = rng.np.dirichlet([alpha] * n) if rng is not None else np.random.dirichlet([alpha] * n)
        if P.NOISE_EPS <= P.ARGMAX_MIX_MAX_EPS:
            probs = np.eye(n)[int(np.argmax(probs))]
        dirichlet = dirichlet * ok
        dirichlet = dirichlet / max(dirichlet.sum(), 1e-9)
        probs = (1 - P.NOISE_EPS) * probs + P.NOISE_EPS * dirichlet
    if P.GREEDY:
        return int(np.argmax(probs)), probs
    probs = probs / probs.sum()
    gen = rng.np if rng is not None else np.random
    return int(gen.choice(len(probs), p=probs)), probs


def decide_full(obs: dict, legal: list[str], ctx: dict, rng: ForkRng | None = None):
    """Claws at one combat decision -> (dec, options, idx, probs, value, batch): the state is tokenized once and the same tokens feed
    the net and, later, the training example."""
    net = _load()
    arr = encode_combat(obs, legal, ctx)
    dec = BufferDecision(arr)
    batch = _tok(dec, _gvocab)
    options = _opts(dec)
    with torch.no_grad():
        logits, value = net.score(batch, options)
    idx, probs = _sample(options, logits, obs, legal, rng)
    return dec, options, idx, probs, float(value.item()), batch


def decide(obs: dict, legal: list[str], ctx: dict):
    """Claws at one combat decision -> (dec, options, idx, probs, value)."""
    dec, options, idx, probs, value, _batch = decide_full(obs, legal, ctx)
    return dec, options, idx, probs, value


def make_example(dec: BufferDecision, options: list, idx: int | None, hp: float, batch=None) -> dict:
    """One combat decision as the trainer's example, minus value and advantage (known once the combat is over)."""
    batch = batch if batch is not None else _tok(dec, _gvocab)
    return {
        "type_ids": batch.type_ids, "global_ids": batch.global_ids, "group_ids": batch.group_ids, "features": batch.features,
        "player_pos": batch.player_pos, "hand_pos": batch.hand_pos, "enemy_pos": batch.enemy_pos,
        "potion_pos": batch.potion_pos, "options": options, **batch.piles,
        "action_index": idx, "hp": float(hp),
    }


def state_key_of(arr, world: World) -> str:
    return combine(visible_key(arr), str(world.hidden_key()))


class Dedup:
    """Per (combat, turn) group, the states the members' first turns ended in: a fork that reaches a known one stops."""

    def __init__(self):
        self.seen: dict[tuple[int, int], dict[str, str]] = {}
        self.stats: dict[str, dict] = {}

    def claim(self, combat: int, turn: int, key: str, owner: str) -> str:
        group = self.seen.setdefault((combat, turn), {})
        return group.setdefault(key, owner)

    def note(self, combat: int, turn: int, forks: int) -> dict:
        return self.stats.setdefault(f"c{combat:03d}t{turn:02d}", {"forks": forks, "discarded": 0, "multiplicity": {}})


class _Fork:
    """One clone being played out in lockstep with its siblings."""
    __slots__ = ("world", "rng", "uid", "dedup_cb", "turns", "examples", "last_round", "last_hp", "result", "pending")

    def __init__(self, world: World, uid: str, dedup_cb, hp0: float):
        self.world, self.uid, self.dedup_cb = world, uid, dedup_cb
        self.rng = ForkRng(zlib_hash(uid))
        self.turns, self.examples, self.last_round, self.last_hp = [], [], None, hp0
        self.result, self.pending = None, None


def _finish(f: _Fork, outcome: str | None = None, **extra) -> None:
    w = f.world
    if outcome is None:
        st = w.status()
        won = st == 1
        final = f.last_hp
        if won:
            try:
                final = w.observe()["player"]["hp"]
            except Exception:  # noqa: BLE001  (a finished combat may have no observation)
                final = f.last_hp
        else:
            final = 0
        outcome = "won" if won else "lost" if st == 2 else "stalled"
        extra["final_hp"] = final
    f.result = {"outcome": outcome, "turns": f.turns, "examples": f.examples, **extra}


def _advance(f: _Fork, ctx: dict, ready: list) -> None:
    """Plays fork `f` forward until it needs a Claws decision (appended to `ready`) or its combat is over (f.result is set)."""
    w = f.world
    while True:
        if w.status() != 0:
            _finish(f)
            return
        obs = w.observe()
        prompt = obs.get("prompt")
        if prompt:
            w.apply(random_pick(prompt["min"], prompt["max"], len(prompt.get("options") or []), f.rng.py))
            continue
        legal = w.legal()
        if not legal:
            _finish(f)
            return
        rnd, hp = obs["round"], obs["player"]["hp"]
        new_turn = rnd != f.last_round
        if new_turn and len(f.turns) >= MAX_COMBAT_TURNS:
            _finish(f, "stalled", final_hp=hp)
            return
        arr = encode_combat(obs, legal, ctx)
        dec = BufferDecision(arr)
        if new_turn:
            f.last_round = rnd
            f.turns.append({"hp": hp, "v": None})
            if len(f.turns) == 2 and f.dedup_cb is not None:  # the compared turn just ended: has this state been reached?
                owner = f.dedup_cb(state_key_of(arr, w))
                if owner is not None:
                    _finish(f, "duplicate", duplicate_of=owner, final_hp=hp)
                    f.examples = []
                    return
        f.pending = (obs, legal, dec, hp, new_turn)
        ready.append(f)
        return


def play_engines(worlds: list[World], ctx: dict, first_turn_hp: float, uids: list[str] | None = None, dedup_cbs: list | None = None) -> list[dict]:
    """Plays every imported combat to its end with Claws, all of them advancing one decision at a time TOGETHER: each step encodes and
    tokenizes the live forks' decisions and runs the net once over all of them. The worlds are consumed. Per world:
    {"outcome": won|lost|stalled|duplicate, "turns": [{"hp", "v"}], "final_hp", "examples", "duplicate_of"}.
    Examples: every decision of the first (compared) turn, then only each later turn's first decision."""
    net = _load()
    n = len(worlds)
    uids = uids or [f"f{i}" for i in range(n)]
    dedup_cbs = dedup_cbs or [None] * n
    forks = [_Fork(w, u, cb, first_turn_hp) for w, u, cb in zip(worlds, uids, dedup_cbs)]
    live = list(forks)
    while live:
        ready = []  # forks that now need a Claws decision
        for f in live:
            try:
                _advance(f, ctx, ready)
            except Exception as e:  # noqa: BLE001  (one fork failing must not take its siblings with it)
                f.result = {"outcome": "error", "error": f"{type(e).__name__}: {str(e)[:80]}", "turns": f.turns, "examples": []}
                f.pending = None
                if f in ready:
                    ready.remove(f)
        if ready:
            batches = [_tok(f.pending[2], _gvocab) for f in ready]
            options = [_opts(f.pending[2]) for f in ready]
            logits, values = fast_tokens.score_batch(net, batches, options)
            for k, f in enumerate(ready):
                obs, legal, dec, hp, new_turn = f.pending
                f.pending = None
                idx, _probs = _sample(options[k], logits[k], obs, legal, f.rng, greedy=len(f.turns) > 1)
                if new_turn:
                    f.turns[-1]["v"] = float(values[k].item())
                if len(f.turns) == 1:
                    f.examples.append(make_example(dec, options[k], idx, hp, batches[k]))
                elif new_turn:
                    f.examples.append(make_example(dec, options[k], None, hp, batches[k]))
                try:
                    f.world.apply(to_action(*options[k][idx], obs, legal))
                except Exception as e:  # noqa: BLE001
                    f.result = {"outcome": "error", "error": f"{type(e).__name__}: {str(e)[:80]}", "turns": f.turns, "examples": []}
                    continue
                f.last_hp = hp
        live = [f for f in live if f.result is None]
    return [f.result for f in forks]


def play_engine(world: World, ctx: dict, first_turn_hp: float, compared_examples: bool = True, dedup_cb=None) -> dict:
    """One combat played out (play_engines with a single world)."""
    return play_engines([world], ctx, first_turn_hp, dedup_cbs=[dedup_cb])[0]


def finish_turns(turns: list[dict], final_hp: float) -> None:
    """loss of each turn = HP at its start minus HP at the next turn's start (or at the end); g = HP lost from here to the end."""
    for i, t in enumerate(turns):
        nxt = turns[i + 1]["hp"] if i + 1 < len(turns) else final_hp
        t["loss"] = max(0.0, t["hp"] - nxt)
        t["g"] = max(0.0, t["hp"] - final_hp)


def patched_export(export: bytes, hp: int) -> bytes:
    """The combat's export with the player's current HP set (the mainline heals at the first decision; the export taken
    there still holds the HP the run really had)."""
    e = json.loads(export)
    e["arenas"]["creature"][0]["Creature"]["_current_hp"] = hp
    return json.dumps(e, separators=(",", ":")).encode()


class ClawsRun:
    def __init__(self, save: str | None, main_id: str, asc: int, raw_dir: str, forks_per_turn: int = 40, dedup: bool = True,
                 character: str | None = None, seed: int = 1):
        """`save` set: the run starts from that native save (an act boundary); None: a fresh run of `character` from seed."""
        self.save, self.main_id, self.asc, self.raw_dir = save, main_id, asc, raw_dir
        self.character, self.seed = character, seed
        self.forks_per_turn, self.use_dedup = forks_per_turn, dedup
        self.rng = random.Random(zlib_hash(main_id))
        self.dedup = Dedup()
        self.combats: list[dict] = []
        self.forks: list[dict] = []
        self.stats = {"forks": 0, "dup": 0, "err": 0, "stalled": 0}

    # --- the run -----------------------------------------------------------------------------------------------------

    def run(self) -> dict:
        _load()
        seed_everything(zlib_hash(self.main_id))
        t0 = time.time()
        outcome, error = "error", None
        self.player = MacroPlayer(self.rng, self.asc)
        host = SpireHost.start(save=self.save, character=self.character, seed=self.seed, asc=self.asc)
        try:
            d = host.decision
            same, last_sig, n = 0, None, 0
            while d is not None and n < MAX_RUN_DECISIONS:
                n += 1
                if d.surface == "combat":
                    d = self._combat(host, d)
                    continue
                sig = (d.surface, json.dumps(d.legal), d.obs.get("run", {}).get("hp"))
                same = same + 1 if sig == last_sig else 0
                last_sig = sig
                if same >= 25:
                    outcome = "stalled"
                    break
                d = host.act(self.player.act(d, same))
            else:
                if d is None:
                    outcome = "won" if host.end and host.end.get("won") else "lost"
                else:
                    outcome = "truncated"
            if d is None and outcome == "error":
                outcome = "won" if host.end and host.end.get("won") else "lost"
        except HostError as e:
            error = f"host: {e}"
        except Exception as e:  # noqa: BLE001
            import traceback
            where = " < ".join(f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in reversed(traceback.extract_tb(e.__traceback__)[-4:]))
            error = f"{type(e).__name__}: {e} [{where}]"
        finally:
            host.close()
        result = {
            "main_id": self.main_id, "save": self.save, "ascension": self.asc, "outcome": outcome, "error": error,
            "combats": self.combats, "forks": self.forks, "dedup": self.dedup.stats, "stats": self.stats,
            "seconds": round(time.time() - t0, 1),
        }
        os.makedirs(self.raw_dir, exist_ok=True)
        tmp = os.path.join(self.raw_dir, f".{self.main_id}.pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(result, f, protocol=5)
        os.replace(tmp, os.path.join(self.raw_dir, f"{self.main_id}.pkl"))
        return result

    def _combat(self, host: SpireHost, d):
        """Plays one combat of the mainline from full HP; returns the first decision after it (None: the run ended)."""
        run = d.obs["run"]
        entry_hp, max_hp = run["hp"], run["max_hp"]
        ctx = run_context(run)
        cidx = len(self.combats) + 1
        host.set_hp(max_hp)
        combat = {"combat_index": cidx, "floor": run.get("floor"), "encounter": d.obs["screen"]["combat"].get("encounter"),
                  "entry_hp": entry_hp, "max_hp": max_hp, "turns": []}
        turns: list[dict] = []
        examples: list[dict] = []  # (turn_number, example)
        last_round, last_hp, first = None, max_hp, True
        worlds: dict[int, World] = {}
        try:
            while d is not None and (d.surface == "combat" or (d.surface == "card_choice" and "combat" in d.obs["screen"])):
                if d.surface == "card_choice":
                    ch = d.obs["screen"]["choice"]
                    d = host.act(random_pick(ch["min"], ch["max"], len(ch["options"]), self.rng))
                    continue
                obs = d.obs["screen"]["combat"]
                if first:
                    obs["player"]["hp"] = max_hp
                rnd, hp = obs["round"], obs["player"]["hp"]
                new_turn = rnd != last_round
                dec, options, idx, _probs, value = decide(obs, d.legal, ctx)
                if new_turn:
                    if len(turns) >= MAX_COMBAT_TURNS:
                        host.abort("stalled combat")
                        combat.update(outcome="stalled", died=False, loss=0)
                        self.combats.append(combat)
                        return None
                    last_round = rnd
                    export = patched_export(d.export, hp) if first and d.export else d.export
                    turns.append({"hp": hp, "v": value, "turn_index": len(turns) + 1})
                    if export is not None:
                        w = World.from_export(export)
                        worlds[len(turns)] = w
                        if self.use_dedup and len(turns) >= 2:  # the previous turn's mainline end state, as its forks' first turn will end
                            self.dedup.claim(cidx, len(turns) - 1, state_key_of(dec.arr, w), "main")
                first = False
                examples.append((len(turns), make_example(dec, options, idx, hp)))
                last_hp = hp
                d = host.act(to_action(*options[idx], obs, d.legal))
            # the combat is over (or the run is)
            if d is None and not (host.end and host.end.get("reason") == "lost"):
                raise HostError(f"the run ended unexpectedly: {host.end}")
            died = d is None
            if died:
                final_hp = 0
            else:
                final_hp = d.obs["run"]["hp"]  # hp after the combat as the real game left it (relic heals included)
            fin_for_labels = 0 if died else last_hp  # labels: HP at the last decision, as forks measure it
            finish_turns(turns, fin_for_labels)
            combat["turns"] = [{k: t[k] for k in ("turn_index", "hp", "v", "loss", "g")} for t in turns]
            for tn, ex in examples:
                ex["value"] = max(0.0, ex["hp"] - fin_for_labels)
                ex["turn_index"] = tn
            combat["examples"] = [ex for _tn, ex in examples]
            combat["loss"] = sum(t["loss"] for t in turns)
            if died:
                combat.update(outcome="lost", died=True)
                self.combats.append(combat)
                self._fork_all(combat, worlds, ctx)
                return None
            loss_real = max_hp - final_hp
            new_hp = entry_hp - loss_real
            combat.update(outcome="won", died=new_hp <= 0, exit_hp=new_hp if new_hp > 0 else None)
            self.combats.append(combat)
            self._fork_all(combat, worlds, ctx)
            if new_hp <= 0:
                host.abort("run lost to attrition")
                return None
            host.set_hp(new_hp)
            d.obs["run"]["hp"] = new_hp
            return d
        finally:
            for w in worlds.values():
                w.close()

    # --- forks -------------------------------------------------------------------------------------------------------

    def _fork_all(self, combat: dict, worlds: dict[int, World], ctx: dict) -> None:
        """Every turn of the mainline combat is forked `forks_per_turn` times: the clones advance in lockstep (one batched net pass per
        step), each with its own random draws seeded by its uid."""
        cidx = combat["combat_index"]
        for t, w in worlds.items():
            if t > len(combat["turns"]):
                continue
            hp0 = combat["turns"][t - 1]["hp"]
            self.dedup.note(cidx, t, self.forks_per_turn)
            uids = [f"{self.main_id}_c{cidx:03d}t{t:02d}_f{f:03d}" for f in range(self.forks_per_turn)]
            cbs = [(lambda key, c=cidx, tt=t, u=u: (lambda o: None if o == u else o)(self.dedup.claim(c, tt, key, u))) if self.use_dedup else None
                   for u in uids]
            clones = [w.clone() for _ in uids]
            try:
                results = play_engines(clones, ctx, hp0, uids, cbs)
            finally:
                for c in clones:
                    c.close()
            for f, r in enumerate(results):
                if r["outcome"] == "error":
                    self.stats["err"] += 1
                    errs = self.stats.setdefault("errors", {})
                    errs[r["error"]] = errs.get(r["error"], 0) + 1
                    continue
                self.stats["forks"] += 1
                if r["outcome"] == "duplicate":
                    g = self.dedup.stats[f"c{cidx:03d}t{t:02d}"]
                    g["discarded"] += 1
                    g["multiplicity"][r["duplicate_of"]] = g["multiplicity"].get(r["duplicate_of"], 1) + 1
                    self.stats["dup"] += 1
                    continue
                if r["outcome"] == "stalled":
                    self.stats["stalled"] += 1
                    continue
                turns = r["turns"]
                finish_turns(turns, r["final_hp"])
                for ex in r["examples"]:
                    ex["value"] = max(0.0, ex["hp"] - r["final_hp"])
                self.forks.append({
                    "combat_index": cidx, "turn_index": t, "fork_index": f, "outcome": r["outcome"],
                    "turns": [{k: tt[k] for k in ("hp", "v", "loss", "g")} for tt in turns], "examples": r["examples"],
                })

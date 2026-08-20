"""PUCT tree search over combat_play decisions, per ../README.md's MCTS strategy section --
wraps a (policy, value) evaluator (normally ClawsNet.forward) the way README.md describes:
"expand it and call the network once -- (p, v) = f_theta(state) -- then immediately back up
that v, without continuing the simulation further."

A node's state is a replay prefix (an action_log), not a snapshot -- same approach as every
other part of this project (whispering_earring/checkpoint.py): there's no in-process fork/rewind
of a combat_play session, so reaching a new node means a fresh Sts2Session replaying the same
prefix (see Sts2Env.step). This is why the tree only ever calls the environment when it creates a
genuinely new child -- revisiting an already-expanded edge costs nothing but walking Python
object references, no replay.

The search mechanics (Node, select, expand, backup) are decoupled from both sts2-cli and
ClawsNet, via a plain `evaluate_fn(decision) -> (options, priors, value)` and an `Env` protocol
(`step`) -- see the __main__ self-test, which verifies PUCT/backup actually converge toward the
better-valued branch using a synthetic environment in milliseconds, before ever touching a live
game session (Sts2Env.step costs ~1-2s per call -- a real run's ~200 simulations/combat, per
README.md, is a multi-minute affair, not something to iterate against while debugging the
algorithm itself).

Chance-node trigger (confirmed, not a guess): a chance node is created whenever the chosen action
draws cards *during the resolving turn* -- i.e. a `play_card` whose card has a "cards" DynamicVar
(Pommel Strike, etc. -- see `draws_cards`), not `end_turn`. The once-per-turn draw at the start of
the *next* turn doesn't need one: different simulations reach many different next-turn hands
anyway just by taking different paths through the tree (each path consumes the seeded RNG stream
differently before it gets there). A same-turn draw effect doesn't get that for free -- revisited
from the exact same node, `play_card` on the exact same card always resolves against the exact
same RNG state (same action_log prefix), so without an explicit chance node every visit to a
popular draw-card edge would silently back up the same one lucky-or-unlucky draw over and over.

Unresolved from README.md -- flagged rather than silently decided, worth confirming before
trusting this for real self-play:
- k(N) = ceil(C * N**alpha)'s C/alpha aren't given numeric values; CHANCE_C/CHANCE_ALPHA below are
  placeholders (2.0/0.5), not tuned or verified against anything.
- Final action selection: README.md says "Live game play will use argmax[v]". Read literally as
  argmax over Q(s,a) = W/N (the closest thing to "v" in the node's own stats) below, not the more
  common argmax(N) visit-count convention other AlphaZero write-ups use. Flagged, not silently
  substituted -- if "v" actually meant something else, `best_action`'s the only place to change.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Protocol

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # see model.py's top -- two OpenMP
# runtimes (conda's llvm-openmp + torch's bundled copy) otherwise abort on import. Set here too,
# even though this module only imports torch lazily (net_evaluate), so it's set before torch
# loads no matter what order a caller imports mcts.py vs. model.py in.

C_PUCT = 1.5
N_SIMULATIONS = 200  # per combat, not per decision -- README.md's "## MCTS strategy"
MAX_DEPTH = 100  # README.md: terminate + evaluate if a combat runs past this many decisions
CHANCE_MAX_CHILDREN = 80
CHANCE_C = 2.0
CHANCE_ALPHA = 0.5

# Matches recorder.py's _OUTCOME_VALUE -- kept as a separate small constant here rather than
# imported, since whispering_earring isn't (and shouldn't need to be) an mcts.py dependency.
OUTCOME_VALUE = {"won": 1.0, "lost": -1.0}


def draws_cards(decision, action: tuple[str, dict]) -> bool:
    """True iff resolving `action` against `decision` draws cards as a direct effect of that one
    decision (e.g. Pommel Strike's "draw 1"), which is what needs a chance node -- see the module
    docstring. `card.stats` is sts2-cli's already-resolved DynamicVar preview (claws/README.md's
    Hand card table); "cards" is that var's exported name (RunSimulator.cs:
    `stats[dv.Name.ToLowerInvariant()]`, and CardsVar's default Name is "Cards" -- see e.g.
    PommelStrike.cs's `new CardsVar(1)`).

    Only checks play_card -- use_potion has no `stats` field in the schema to check (Potion only
    exposes `vars`), so a draw-on-potion-use effect (if one exists) isn't covered here yet."""
    name, args = action
    if name != "play_card":
        return False
    card = next((c for c in decision.hand if c.index == args["card_index"]), None)
    return bool(card and card.stats and card.stats.get("cards"))


# --- tree ----------------------------------------------------------------------------------------


@dataclass
class Node:
    action_log: list[dict]
    decision: object | None = None  # CombatPlay in real use; opaque to this module
    outcome: str | None = None  # "won"/"lost" -- set instead of decision iff terminal
    options: list[tuple[str, dict]] | None = None  # None until expand()ed
    priors: list[float] = field(default_factory=list)
    N: list[int] = field(default_factory=list)
    W: list[float] = field(default_factory=list)
    children: list[list["Node"]] = field(default_factory=list)  # per edge -- >1 only for chance
    value: float | None = None  # this node's own network value estimate, cached at expand()


EvaluateFn = Callable[[object], tuple[list[tuple[str, dict]], list[float], float]]


class Env(Protocol):
    def step(
        self, action_log: list[dict], action: tuple[str, dict], draw_seed: int | None = None
    ) -> Node:
        """Replays action_log, optionally reseeds the draw pile (chance-node sampling), takes
        `action`, and returns the resulting Node -- terminal (outcome set) or not (decision set)."""
        ...


def expand(node: Node, evaluate_fn: EvaluateFn) -> float:
    options, priors, value = evaluate_fn(node.decision)
    node.options = options
    node.priors = priors
    node.N = [0] * len(options)
    node.W = [0.0] * len(options)
    node.children = [[] for _ in options]
    node.value = value
    return value


def select_edge(node: Node, c_puct: float = C_PUCT) -> int:
    """Standard PUCT: argmax_a Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)).
    Unvisited edges (N=0) use Q=0 (no first-play-urgency heuristic -- keep it simple until
    there's a reason not to)."""
    total_n = sum(node.N)
    sqrt_total = math.sqrt(total_n)
    best_idx, best_score = 0, -math.inf
    for i in range(len(node.options)):
        q = node.W[i] / node.N[i] if node.N[i] > 0 else 0.0
        u = c_puct * node.priors[i] * sqrt_total / (1 + node.N[i])
        score = q + u
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx


def select_child(node: Node, edge_idx: int, env: Env) -> Node:
    action = node.options[edge_idx]
    children = node.children[edge_idx]

    if not draws_cards(node.decision, action):
        if not children:
            children.append(env.step(node.action_log, action))
        return children[0]

    # Chance edge: children are different sampled draw orders, not different actions -- selection
    # among them is uniform (an expectation, not a strategic choice), and the pool grows toward
    # k(N) as the edge gets visited more (README.md's dynamical formula).
    n_wanted = min(CHANCE_MAX_CHILDREN, max(1, math.ceil(CHANCE_C * (node.N[edge_idx] ** CHANCE_ALPHA))))
    if len(children) < n_wanted:
        seed = len(children)
        children.append(env.step(node.action_log, action, draw_seed=seed))
    return random.choice(children)


def simulate(root: Node, evaluate_fn: EvaluateFn, env: Env) -> float:
    path: list[tuple[Node, int]] = []
    current = root
    depth = 0

    while True:
        if current.outcome is not None:
            value = OUTCOME_VALUE[current.outcome]
            break
        if current.options is None:
            value = expand(current, evaluate_fn)  # network call -- stop here, don't descend further
            break
        if depth >= MAX_DEPTH:
            value = current.value if current.value is not None else expand(current, evaluate_fn)
            break

        edge_idx = select_edge(current)
        path.append((current, edge_idx))
        current = select_child(current, edge_idx, env)
        depth += 1

    for node, edge_idx in path:
        node.N[edge_idx] += 1
        node.W[edge_idx] += value
    return value


def run(
    root_decision: object,
    action_log_prefix: list[dict],
    evaluate_fn: EvaluateFn,
    env: Env,
    n_simulations: int = N_SIMULATIONS,
) -> Node:
    root = Node(action_log=list(action_log_prefix), decision=root_decision)
    expand(root, evaluate_fn)
    for _ in range(n_simulations):
        simulate(root, evaluate_fn, env)
    return root


def best_action(root: Node) -> tuple[str, dict]:
    """argmax(Q) -- see module docstring's note on README.md's "argmax[v]"."""
    def q(i: int) -> float:
        return root.W[i] / root.N[i] if root.N[i] > 0 else -math.inf

    best_idx = max(range(len(root.options)), key=q)
    return root.options[best_idx]


# --- real adapters ---------------------------------------------------------------------------------


def net_evaluate(net, decision) -> tuple[list[tuple[str, dict]], list[float], float]:
    """EvaluateFn adapter for model.py's ClawsNet."""
    import torch

    options, logits, value = net.forward(decision)
    priors = torch.softmax(logits, dim=-1).tolist()
    return options, priors, value.item()


class Sts2Env:
    """Env adapter over a real sts2-cli session. Every call spawns a fresh subprocess and
    replays action_log from scratch (open_from_checkpoint's approach) -- expensive (~1-2s), but
    only paid when the tree creates a genuinely new node (see module docstring)."""

    def __init__(self, character: str, seed: str, ascension: int = 0):
        self.character = character
        self.seed = seed
        self.ascension = ascension

    def step(
        self, action_log: list[dict], action: tuple[str, dict], draw_seed: int | None = None
    ) -> Node:
        from env import Sts2Session  # sts2_path must already be on sys.path -- see __main__

        name, args = action
        session = Sts2Session()
        try:
            decision = session.start(self.character, self.seed, self.ascension)
            for cmd in action_log:
                decision = session.step(cmd["action"], **cmd.get("args", {}))
            if draw_seed is not None:
                session.randomize_draw_pile(decision, seed=draw_seed)
            decision = session.step(name, **args)
            new_log = action_log + [{"cmd": "action", "action": name, "args": args}]
            if decision.decision != "combat_play":
                won = decision.decision == "game_over" and getattr(decision, "victory", False)
                outcome = "won" if (won or decision.decision != "game_over") else "lost"
                return Node(action_log=new_log, outcome=outcome)
            return Node(action_log=new_log, decision=decision)
        finally:
            session.close()


# --- self-test -------------------------------------------------------------------------------------

if __name__ == "__main__":
    import itertools

    # No sts2-cli/torch involved -- verifies expand/select/backup, and specifically the
    # chance-node fan-out, without the cost of a live game. Root has two options: "end_turn"
    # (non-chance -- always resolves to a guaranteed loss, one cached child) and "play_card" on a
    # card whose stats carry a "cards" DynamicVar (a chance action per draws_cards -- half its
    # sampled draw seeds win, half lose). A working search should (a) fan "play_card" out into
    # more than one child, and (b) still prefer it over end_turn, since ~50% win beats a sure loss.
    class _MockCard:
        def __init__(self, index: int, stats: dict):
            self.index, self.stats = index, stats

    class _MockDecision:
        def __init__(self, hand: list[_MockCard]):
            self.hand = hand

    _root_decision = _MockDecision(hand=[_MockCard(index=0, stats={"cards": 1})])

    def _mock_evaluate(decision):
        assert decision is _root_decision
        options = [("end_turn", {}), ("play_card", {"card_index": 0})]
        return options, [0.5, 0.5], 0.0

    class _MockEnv:
        def step(self, action_log, action, draw_seed=None) -> Node:
            name, _ = action
            new_log = action_log + [{"cmd": "action", "action": name}]
            if name == "end_turn":
                return Node(action_log=new_log, outcome="lost")
            outcome = "won" if (draw_seed or 0) % 2 == 0 else "lost"
            return Node(action_log=new_log, outcome=outcome)

    root = run(_root_decision, [], _mock_evaluate, _MockEnv(), n_simulations=200)
    print("root options:", root.options)
    print("N:", root.N, "W:", root.W)
    print("Q:", [w / n if n else None for w, n in zip(root.W, root.N)])
    print("children fanned out per edge:", [len(c) for c in root.children])
    print("best_action:", best_action(root))
    assert len(root.children[0]) == 1, "end_turn is not a chance action -- should have one cached child"
    assert len(root.children[1]) > 1, "play_card draws a card -- should have fanned out into multiple children"
    assert best_action(root) == ("play_card", {"card_index": 0}), \
        "PUCT should prefer the ~50%-win chance branch over end_turn's guaranteed loss"
    print("mock self-test passed")

    # Small live smoke test -- proves Sts2Env/net_evaluate actually wire up against a real
    # session + ClawsNet, kept tiny (12 sims, not the doc's 200) since each simulation here costs
    # a real subprocess spawn.
    try:
        import sys
        from pathlib import Path

        import sts2_path  # noqa: F401
        from env import Sts2Session
        from installation import ensure_setup

        from model import ClawsNet, GlobalVocab  # sets up the `common` sys.path entry as a side effect
        from common.tokenizer import EntityVocab

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "whispering_earring"))
        from policy import choose_action  # whispering_earring's meta-decision placeholder

        ensure_setup()
        character, seed = "Ironclad", "mcts-live-smoke-1"
        session = Sts2Session()
        decision = session.start(character, seed, 0)
        while decision.decision != "combat_play":  # walk past Neow/map to the first fight
            name, kwargs, _ = choose_action(decision, session)
            decision = session.step(name, **kwargs)
        root_prefix = list(session.action_log)
        root_decision = decision
        session.close()

        net = ClawsNet(GlobalVocab(EntityVocab.load()))
        live_root = run(
            root_decision, root_prefix,
            lambda d: net_evaluate(net, d), Sts2Env(character, seed, 0),
            n_simulations=12,
        )
        print(f"\nlive smoke test: {sum(live_root.N)} simulations run, "
              f"best_action={best_action(live_root)}")
        for (a, args), n, w in itertools.islice(zip(live_root.options, live_root.N, live_root.W), 6):
            print(f"  N={n:3d} Q={(w/n if n else 0):+.3f}  {a} {args}")
    except Exception as e:
        print(f"\n(skipped live smoke test: {type(e).__name__}: {e})")

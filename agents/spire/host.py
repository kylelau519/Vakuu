"""A run hosted by spire-zero's SimWorker (the real game v0.113.1, headless, deterministic single-threaded pump), driven from
Python over its wire protocol: the host asks (hello, then one `decision` per thing the player must answer), Python answers
(`ready`, then one `recommendation` per decision). One SpireHost is one process and one run at a time.

    host = SpireHost.start(save="data/act_checkpoints/X_act2.json", asc=1)
    d = host.decision                 # first Decision: surface, obs, legal, export (bytes, combat decisions only)
    d = host.act(rng.choice(d.legal)) # answer it, get the next; None when the run is over (host.end has the reason)
    host.set_hp(n)                    # not an action: sets the player's current HP (answered before the next act)
"""
from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
from dataclasses import dataclass

_SZ = os.environ.get("SPIRE_ZERO", os.path.expanduser("~/Programming/spire-zero"))
_WIRE_VERSION = 4


@dataclass
class Decision:
    id: int
    hash: str
    surface: str
    obs: dict           # surface-specific: combat -> the engine's CombatObservation plus run; others -> {"run":..., "screen":...}
    legal: list[str]
    export: bytes | None = None  # the live combat for the engine (combat surface, SL env)
    trace: list | None = None    # SL env: the steps since the latest node checkpoint (with host.checkpoint, a fork point)


class HostError(RuntimeError):
    pass


class SpireHost:
    def __init__(self, proc: subprocess.Popen, server: socket.socket, conn: socket.socket):
        self.proc, self._server, self._conn = proc, server, conn
        self._f = conn.makefile("rw", encoding="utf-8", newline="\n")
        self.decision: Decision | None = None
        self.checkpoint: str | None = None  # SL env: the latest node checkpoint the host sent
        self.end: dict | None = None  # run_end message once the run is over
        self.hello: dict | None = None

    @classmethod
    def start(cls, save: str | None = None, character: str | None = None, seed: int = 1, asc: int = 0, env: str = "SL",
              auto_win: str | None = None, extra: list[str] | None = None, clone: str | None = None) -> "SpireHost":
        """`clone`: a JSON file {"checkpoint", "trace"} (a fork point): the host replays it and the first decision is the one it was taken at."""
        path = "\0sts2py-%d-%s" % (os.getpid(), os.urandom(3).hex())
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(path)
        server.listen(1)
        dll = os.path.join(_SZ, "src/Sts2Ai.SimWorker/bin/Release/net9.0/Sts2Ai.SimWorker.dll")
        cmd = ["dotnet", dll, "run", "--n", "1", "--asc", str(asc), "--env", env, "--seed", str(seed), "--brain-socket", "@" + path[1:]]
        if clone:
            cmd += ["--from-clone", os.path.abspath(clone)]
        elif save:
            cmd += ["--from-save", os.path.abspath(save)]
        if character:
            cmd += ["--character", character]
        if auto_win:
            cmd += ["--auto-win", auto_win]
        cmd += extra or []
        penv = dict(os.environ)
        penv.setdefault("DOTNET_ROOT", "/usr/lib64/dotnet")
        penv.setdefault("STS2AI_ENGINE_LIB", os.path.join(_SZ, "engine/target/release/libsts2ai_engine.so"))
        penv.setdefault("STS2AI_ENGINE_SCHEMA", os.path.join(_SZ, "engine/crates/sts2ai-engine/src/gen/schema.json"))
        penv.setdefault("DOTNET_gcServer", "0")
        log = open(os.environ["SPIRE_HOST_LOG"], "ab") if os.environ.get("SPIRE_HOST_LOG") else subprocess.DEVNULL
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=penv, cwd=_SZ)
        server.settimeout(60)
        try:
            conn, _ = server.accept()
        except socket.timeout as e:
            proc.kill()
            raise HostError("the host never connected") from e
        conn.settimeout(120)
        host = cls(proc, server, conn)
        host._open()
        return host

    # --- wire -------------------------------------------------------------------------------------------------------

    def _send(self, msg: dict) -> None:
        msg["v"] = _WIRE_VERSION
        self._f.write(json.dumps(msg) + "\n")
        self._f.flush()

    def _recv(self) -> dict | None:
        line = self._f.readline()
        return json.loads(line) if line else None

    def _open(self) -> None:
        msg = self._recv()
        if not msg or msg["type"] != "hello":
            raise HostError(f"expected hello, got {msg}")
        self.hello = msg
        self._send({"type": "ready", "champion": "python", "shown": "python"})
        self._advance()

    def _advance(self) -> None:
        """Reads until the next decision (stored) or the run's end."""
        while True:
            msg = self._recv()
            if msg is None:
                self.decision = None
                self.end = self.end or {"reason": "closed"}
                return
            ty = msg["type"]
            if ty == "decision":
                self.decision = Decision(msg["decision_id"], msg["obs_hash"], msg["surface"], msg["obs"], msg["legal"],
                                         base64.b64decode(msg["export"]) if msg.get("export") else None, msg.get("trace"))
                return
            if ty == "node_checkpoint":
                self.checkpoint = msg["checkpoint"]
                continue
            if ty == "run_end":
                self.decision, self.end = None, msg
                return
            if ty == "error":
                raise HostError(msg.get("message"))
            # executed / node_checkpoint: nothing to do

    # --- driving ----------------------------------------------------------------------------------------------------

    def set_hp(self, hp: int) -> None:
        d = self.decision
        self._send({"type": "recommendation", "decision_id": d.id, "obs_hash": d.hash, "action": f"sethp:{int(hp)}", "brain": "python"})

    def shuffle(self, seed: int) -> None:
        """Not an action: shuffles the draw pile's order (System.Random(seed)); answered before the next act."""
        d = self.decision
        self._send({"type": "recommendation", "decision_id": d.id, "obs_hash": d.hash, "action": f"shuffle:{int(seed)}", "brain": "python"})

    def act(self, action: str) -> Decision | None:
        d = self.decision
        if d is None:
            raise HostError("the run is over")
        self._send({"type": "recommendation", "decision_id": d.id, "obs_hash": d.hash, "action": action, "brain": "python"})
        self._advance()
        return self.decision

    def abort(self, why: str = "done") -> None:
        """Ends the run from our side (the host fails it)."""
        try:
            self._send({"type": "error", "message": why})
        except OSError:
            pass
        self.close()

    def close(self) -> None:
        for c in (self._f, self._conn, self._server):
            try:
                c.close()
            except OSError:
                pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

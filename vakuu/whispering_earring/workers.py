"""The worker-process lifecycle every facility / evaluation shares.

Workers are *spawned* processes (never forked: they hold torch threads and a game host each). They end together: a worker stuck in a
native call is terminated, not waited for, so a Ctrl-C, an error or a normal finish all leave no process behind.
"""
from __future__ import annotations

import glob
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor


def make_pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))


def end_pool(pool: ProcessPoolExecutor) -> None:
    """Stop `pool` without waiting for its workers, then terminate them and every other child process."""
    procs = list((getattr(pool, "_processes", None) or {}).values())  # shutdown() forgets them
    pool.shutdown(wait=False, cancel_futures=True)
    for proc in procs + mp.active_children():
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            pass


def exit_now() -> None:
    """Leave without joining anything: the workers' native threads (torch, the engine) need not be waited for."""
    sys.stdout.flush()
    os._exit(0)


def act_saves(ckpt_dir: str, act: int) -> list[str]:
    """The act-boundary saves of act `act` in `ckpt_dir` (X_act2.json; the X_act2.ckpt.json sidecars are not saves)."""
    saves = [p for p in sorted(glob.glob(os.path.join(ckpt_dir, f"*_act{act}.json"))) if not p.endswith(".ckpt.json")]
    if not saves:
        raise SystemExit(f"no act-{act} saves in {ckpt_dir}")
    return saves

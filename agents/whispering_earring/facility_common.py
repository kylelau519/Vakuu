"""Shared self-play-facility infrastructure for the job-based design: FacilityConfig, the worker
orchestration base class (NewFacility), and the stderr relay / live status board its workers log through.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import atexit
import multiprocessing as mp
import os
import random
import shutil
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import sts2_path  # noqa: F401  (sets sys.path to sts2-cli/python before the imports below)
from workers import BufferWorker
from installation import ensure_setup
from utils import CONTROL_SEQUENCE_RE, CRASH_RE, MILESTONE_RE, WORKER_TAG_COLORS


def _tag_worker_stderr(label: str, worker_index: int, log_queue: mp.Queue, verbose: bool) -> None:
    """Sts2BufferSession loads Sts2Headless's CLR in-process (clr_bridge.load_clr),  Spwans a background thread to relay the worker's stderr through log_queue, tagging each line
 """
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def relay() -> None:
        crash_seen = False  # sticky once _CRASH_RE fires -- see _CRASH_RE's own comment
        with os.fdopen(read_fd) as pipe_reader:
            for line in pipe_reader:
                clean = CONTROL_SEQUENCE_RE.sub("", line)
                if not crash_seen and CRASH_RE.search(clean):
                    crash_seen = True
                if verbose or crash_seen or MILESTONE_RE.search(clean):
                    log_queue.put((label, worker_index, clean, crash_seen))

    threading.Thread(target=relay, daemon=True, name=f"stderr-relay-{label}-{worker_index}").start()

class _LiveStatus:
    """Fixed one-line-per-worker live status board for _log_writer_loop, 
    each worker gets a permanent row, redrawn in place.  """

    def __init__(self, labels: list[tuple[str, int]]):
        self.rows = {key: i for i, key in enumerate(labels)}
        self.n = len(labels)
        self.active = sys.stderr.isatty() and self.n > 0
        self._lock = threading.Lock()
        if self.active:
            for label, idx in labels:
                sys.stderr.write(f"{self._tag(label, idx)} (waiting...)\n")
            sys.stderr.flush()

    def update(self, label: str, worker_index: int, line: str) -> None:
        key = (label, worker_index)
        if not self.active or key not in self.rows:
            self.scroll(f"{self._tag(label, worker_index)} {line}")
            return
        with self._lock:
            up = self.n - self.rows[key]  # rows numbered top-down; cursor idles just below row n-1
            width = shutil.get_terminal_size(fallback=(120, 24)).columns
            avail = max(10, width - len(f"[{label} {worker_index}] ") - 1)
            msg = line.rstrip("\n")
            if len(msg) > avail:
                msg = msg[: avail - 1] + "…"
            # \x1b[{up}A/B: move up/down `up` rows without touching the terminal's scrollback --
            # \r + \x1b[2K then clears and rewrites just that row before returning to where every
            # scrolling write (this class's own `scroll()` included) expects the cursor to be.
            sys.stderr.write(f"\x1b[{up}A\r\x1b[2K{self._tag(label, worker_index)} {msg}\x1b[{up}B\r")
            sys.stderr.flush()

    def scroll(self, text: str) -> None:
        """Plain scrolling output, and a PERMANENT switch-off of the fixed rows (their cursor math only
        holds while nothing else scrolls the terminal). Only for crash dumps and one-off notices; routine
        progress belongs on the status row."""
        with self._lock:
            self.active = False
            sys.stderr.write(text if text.endswith("\n") else f"{text}\n")
            sys.stderr.flush()
            
    def loop(self, log_queue: mp.Queue, on_activity: Callable[[], None] | None = None) -> None:
        """Runs in a single background thread in the parent (Facility) process -- the only writer to
        the real terminal for every branch worker's relayed stderr.  """
        while True:
            item = log_queue.get()
            if item is None:
                return
            label, worker_index, line, is_crash = item
            if on_activity is not None:
                on_activity()
            if is_crash:
                self.scroll(f"{self._tag(label, worker_index)} {line.rstrip()}")
            else:
                self.update(label, worker_index, line)

    @staticmethod
    def _tag(label: str, worker_index: int) -> str:
        color = WORKER_TAG_COLORS[worker_index % len(WORKER_TAG_COLORS)]
        return f"\x1b[{color}m[{label} {worker_index}]\x1b[0m"

@dataclass
class FacilityConfig:
    characters: list[str] = field(default_factory=lambda: ["Ironclad"])
    ascension: int = 0
    seeds: list[str] | None = None  # runs cycle through these seeds in order (see utils.new_run); None = a fresh random seed per run
    forks_per_turn: int = 4  # alternative continuations played from every mainline combat turn
    dedup_forks: bool = False  # stop (and drop) a fork whose compared turn ends in a state an earlier member of its group already reached
    n_branch_workers: int = 4
    n_main_workers: int = 1  # concurrent main lines, sharing one branch-worker pool
    n_main_trials: int | None = 1  # stop after this many main-line trials *total*; None = forever
    raw_dir: str = "data/raw"
    verbose: bool = False
    stall_timeout_s: float = 300.0


def _die_with_parent(parent_pid: int) -> None:
    """If the facility process disappears for any reason (a second Ctrl-C, SIGKILL, a closed terminal,
    a crash), nothing else would ever stop this worker: it is in its own process group, so the terminal's
    signals miss it, and it would run on holding its memory for as long as its job lasts or forever if
    that job is stuck. A daemon thread polls for re-parenting and kills this whole process group
    (the worker and anything it spawned)."""
    def watch() -> None:
        while True:
            time.sleep(2)
            if os.getppid() != parent_pid:
                try:
                    os.killpg(os.getpid(), signal.SIGKILL)
                finally:
                    os._exit(1)

    threading.Thread(target=watch, daemon=True, name="die-with-parent").start()


def _worker_process(
    worker_index: int, job_queue, priority_job_queue, log_queue, result_queue, stop_event, verbose: bool,
) -> None:
    """Entry point of one worker process. The BufferWorker -- and so its CLR session, which can't be
    pickled across a spawn -- is built here, inside the child."""
    os.setpgid(0, 0)  # own process group, so stop()'s killpg takes its dotnet children too
    # Being in its own group means the terminal's Ctrl-C never reaches this process: shutdown is the
    # facility's job (stop()), so a stray SIGINT must not half-kill a worker mid-run either.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _die_with_parent(os.getppid())
    random.seed(os.urandom(16))  # a forked worker would otherwise replay its siblings' random choices
    _tag_worker_stderr("worker", worker_index, log_queue, verbose)
    BufferWorker(
        job_queue, priority_job_queue, log_queue, result_queue, worker_index, stop_event
    ).run()


class NewFacility(ABC):
    """A pool of BufferWorkers pulling jobs from two queues (priority first) and reporting
    (job.uid(), JobResult) on result_queue. Subclasses decide which jobs to submit and what to do
    with their results in run()."""

    def __init__(self, cfg: FacilityConfig):
        self.cfg = cfg
        self.job_queue: mp.Queue = mp.Queue()
        self.priority_job_queue: mp.Queue = mp.Queue()
        self.log_queue: mp.Queue = mp.Queue()
        self.result_queue: mp.Queue = mp.Queue()
        self.stop_event = mp.Event()
        self.worker_procs: list[mp.Process] = []
        self.log_thread: threading.Thread | None = None
        self.live: _LiveStatus | None = None  # built in start() -- see _LiveStatus
        self._interrupted = False
        self._stopping = False

    def _on_signal(self, signum, _frame) -> None:
        """Ctrl-C / SIGTERM / SIGHUP (closing the terminal). The first one unwinds run() as a
        KeyboardInterrupt, whose `finally: self.stop()` then kills the workers at once (an interrupted
        run's in-flight jobs are lost either way). A second one while that is already happening means
        "now": kill everything and leave."""
        if self._stopping:
            self._kill_all()
            os._exit(128 + signum)
        self._interrupted = True
        raise KeyboardInterrupt

    def _kill_all(self) -> None:
        """SIGKILLs every worker's process group. Only workers still alive: a reaped worker's pid may
        already belong to somebody else's process group by now."""
        for p in self.worker_procs:
            if p.is_alive():
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    p.kill()

    def start(self) -> None:
        ensure_setup()  # once, here -- not per-session, which would repeat the same checks
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, self._on_signal)
        atexit.register(self._kill_all)  # whatever way this process exits, no worker outlives it
        # One fixed row per worker plus a "status" row at the bottom for the facility's own running tally
        # (see status()). Routine progress goes there, never through live.scroll(): scroll() switches the
        # fixed rows off for good, which is for crash dumps and one-off notices only.
        labels = [("worker", i) for i in range(self.cfg.n_branch_workers)] + [("status", 0)]
        self.live = _LiveStatus(labels)
        self.log_thread = threading.Thread(
            target=self.live.loop, args=(self.log_queue,), daemon=True, name="log-writer",
        )
        self.log_thread.start()
        for i in range(self.cfg.n_branch_workers):
            p = mp.Process(
                target=_worker_process, name=f"worker_{i}", daemon=True,
                args=(i, self.job_queue, self.priority_job_queue, self.log_queue, self.result_queue,
                      self.stop_event, self.cfg.verbose),
            )
            p.start()
            self.worker_procs.append(p)

    def status(self, text: str) -> None:
        """Redraws the pinned status row in place (falls back to a plain line when stderr isn't a terminal)."""
        self.live.update("status", 0, text)

    def stop(self, timeout: float = 90) -> None:
        """Workers finish whatever is already queued, then exit (the None sentinels sit behind it).
        After an interrupt there is nothing worth waiting for: a short grace, then SIGKILL."""
        self._stopping = True
        if self._interrupted:
            timeout = min(timeout, 2)
        self.stop_event.set()
        for _ in self.worker_procs:
            self.job_queue.put(None)
        deadline = time.monotonic() + timeout
        for p in self.worker_procs:
            p.join(max(0.0, deadline - time.monotonic()))
        for p in self.worker_procs:
            if p.is_alive():
                self.live.scroll(f"{p.name} didn't stop within {timeout}s -- killing its process group")
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    p.kill()
                p.join()
        self.log_queue.put(None)
        if self.log_thread is not None:
            self.log_thread.join(5)

    @abstractmethod
    def run(self) -> None:
        """Start the pool, submit jobs, collect results, stop the pool."""


def add_common_cli_args(parser) -> None:
    """Flags shared by facility_claws.py's and facility_compass.py's CLIs."""
    parser.add_argument("--trials", type=int, default=None,
                         help="stop after this many main-line trials complete in total (default: run forever, Ctrl-C to stop)")
    parser.add_argument("--characters", nargs="+", default=["Ironclad"])
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", default=None,
                         help="a small pool of seeds: runs cycle through them in order, so progression "
                              "shows on a few fixed maps instead of a new one every run. Runs on the same seed "
                              "still differ through the policy's own sampling. Use one --characters with it "
                              "(default: a fresh random seed per run)")
    parser.add_argument("--branch-workers", type=int, default=4,
                         help="shared pool size -- serves branch jobs from every main worker, not "
                              "one dedicated set per main line (default: 4)")
    parser.add_argument("--main-workers", type=int, default=1,
                         help="concurrent main lines sharing the branch-worker pool above -- raise "
                              "this to keep branch workers fed between combats (default: 1)")
    parser.add_argument("--raw-dir", default="data/raw",
                         help="where trial JSON gets written -- resolved against the process's "
                              "cwd if relative (default: 'data/raw'), so pass an absolute path "
                              "when invoking this script from outside agents/whispering_earring/")
    parser.add_argument("--verbose", action="store_true",
                         help="relay every branch worker's [SIM]/[INFO] line (card plays, RNG "
                              "rolls, etc.) instead of just per-turn/per-room milestones "
                              "(default: milestones only)")

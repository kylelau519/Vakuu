import multiprocessing as mp
from queue import Empty

from buffer_env import Sts2BufferSession
from jobs import JobResult


class BufferWorker():
    '''A worker that operates on a buffer session.'''
    def __init__(self, job_queue: mp.Queue, priority_job_queue: mp.Queue, log_queue: mp.Queue,
                 result_queue: mp.Queue, worker_index: int, stop_event=None):
        self._work_station: Sts2BufferSession = Sts2BufferSession()
        self._worker_index: int = worker_index
        self._job_queue: mp.Queue = job_queue
        self._priority_job_queue: mp.Queue = priority_job_queue
        self._log_queue: mp.Queue = log_queue
        self._result_queue: mp.Queue = result_queue
        self._stop_event = stop_event

    def run(self):
        """Runs jobs until a None sentinel arrives or stop_event is set while idle."""
        while True:
            try:
                job = self.get_next_job()
            except Empty:
                if self._stop_event is not None and self._stop_event.is_set():
                    return
                continue
            if job is None:  # shutdown sentinel
                return
            try:
                result = job.execute(self._work_station)
            except Exception as e:
                self._log_queue.put(f"Worker {self._worker_index} encountered an error: {e}")
                result = JobResult(
                    main_id=job.main_id, outcome="error", action_log=list(self._work_station.action_log),
                    auxiliary_data={"error": f"{type(e).__name__}: {e}"},
                )
            self._result_queue.put((job.uid(), result))

    def get_next_job(self):
        """Priority queue first, then the normal one; raises queue.Empty if neither has a job."""
        try:
            return self._priority_job_queue.get_nowait()
        except Empty:
            return self._job_queue.get(timeout=0.2)

    def spawn_work_station(self):
        self._work_station = Sts2BufferSession()

    def close_work_station(self):
        self._work_station.close()

    def reset_work_station(self):
        self._work_station.logger = None

    def attach_logger(self, logger):
        self._work_station.logger = logger

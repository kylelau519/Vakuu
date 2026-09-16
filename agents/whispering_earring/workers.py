import multiprocessing as mp
from queue import Empty
import time
from buffer_env import Sts2BufferSession


class BufferWorker():
    '''A worker that operates on a buffer session.'''
    def __init__(self, job_queue: mp.Queue, priority_job_queue: mp.Queue, log_queue: mp.Queue, result_queue: mp.Queue, worker_index: int):
        self._work_station: Sts2BufferSession = Sts2BufferSession()
        self._worker_index: int = worker_index
        self._job_queue: mp.Queue = job_queue
        self._priority_job_queue: mp.Queue = priority_job_queue
        self._log_queue: mp.Queue = log_queue
        self._result_queue: mp.Queue = result_queue

    def run(self):
        while True:
            job = self.get_next_job()
            if job is None:
                time.sleep(0.1)
                continue
            try:
                result = job.execute(self._work_station)
                self._result_queue.put((job.uid(), result))
            except Exception as e:
                self._log_queue.put(f"Worker {self._worker_index} encountered an error: {e}")
                self._result_queue.put((job.uid(), ("error", str(e))))
              
    def get_next_job(self):
        try:
            item = self._priority_job_queue.get_nowait()
        except Empty:
            try:
                item = self._job_queue.get(timeout=0.2)
            except Empty:
                return None
        return item
    
    def spawn_work_station(self):
        self._work_station = Sts2BufferSession()
        
    def close_work_station(self):
        self._work_station.close()
        
    def reset_work_station(self):
        self._work_station.logger = None
        pass
    def attach_logger(self, logger):
        self._work_station.logger = logger
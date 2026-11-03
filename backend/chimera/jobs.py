"""Background jobs (training, video analysis, benchmarks) on a single worker thread."""

from __future__ import annotations

import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .db import Job, SessionLocal, utcnow

log = logging.getLogger(__name__)
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chimera-job")


class JobContext:
    def __init__(self, job_id: int):
        self.id = job_id
        self._last = 0.0
        self._lock = threading.Lock()

    def progress(self, frac: float, message: str = "", force: bool = False) -> None:
        now = time.monotonic()
        with self._lock:
            if not force and now - self._last < 0.5:
                return
            self._last = now
        with SessionLocal() as db:
            job = db.get(Job, self.id)
            if job:
                job.progress = float(max(0.0, min(1.0, frac)))
                job.message = message[:500]
                db.commit()


def submit(kind: str, params: dict[str, Any], fn: Callable[[JobContext], dict[str, Any]], scene_id: str | None = None) -> int:
    with SessionLocal() as db:
        job = Job(kind=kind, scene_id=scene_id, params=params)
        db.add(job)
        db.commit()
        job_id = job.id

    def run() -> None:
        ctx = JobContext(job_id)
        with SessionLocal() as db:
            j = db.get(Job, job_id)
            j.status = "running"
            db.commit()
        try:
            result = fn(ctx) or {}
            status, msg = "done", "completed"
        except Exception as e:
            log.exception("job %s failed", job_id)
            result, status, msg = {"traceback": traceback.format_exc()[-4000:]}, "error", str(e)
        with SessionLocal() as db:
            j = db.get(Job, job_id)
            j.status, j.message, j.result, j.finished_at = status, msg[:500], result, utcnow()
            if status == "done":
                j.progress = 1.0
            db.commit()

    _pool.submit(run)
    return job_id

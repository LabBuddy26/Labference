from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


FINAL_STATES = {"completed", "failed", "interrupted"}


def _now() -> str:
    return datetime.now().astimezone().isoformat()


class PersistentEvalQueue:
    """One-process FIFO worker with crash-visible, file-backed job state."""

    def __init__(self, jobs_root: str | Path, *, autostart: bool = True):
        self.jobs_root = Path(jobs_root).resolve()
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._pending: queue.Queue[str] = queue.Queue()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._recover()
        if autostart:
            self.start()

    def _path(self, job_id: str) -> Path:
        return self.jobs_root / f"{job_id}.json"

    def _load_path(self, path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _write(self, job: dict[str, Any]) -> None:
        path = self._path(str(job["job_id"]))
        temporary = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        for attempt in range(20):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.01)

    def _recover(self) -> None:
        queued: list[dict[str, Any]] = []
        for path in self.jobs_root.glob("*.json"):
            job = self._load_path(path)
            if not job:
                continue
            if job.get("status") == "running":
                job.update(
                    status="interrupted",
                    finished_at=_now(),
                    error="Evaluation server restarted while this job was running; it was not repeated automatically.",
                )
                self._write(job)
            elif job.get("status") == "queued":
                queued.append(job)
        for job in sorted(queued, key=lambda item: str(item.get("created_at") or "")):
            self._pending.put(str(job["job_id"]))

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._worker, name="eai4s-eval-fifo", daemon=True)
        self._thread.start()

    def enqueue(
        self,
        *,
        run_id: str,
        command: list[str],
        cwd: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not command or not all(isinstance(item, str) and item for item in command):
            raise ValueError("evaluation command must be a non-empty explicit argv list")
        job_id = uuid4().hex
        job = {
            "job_id": job_id,
            "run_id": str(run_id),
            "status": "queued",
            "command": list(command),
            "cwd": str(Path(cwd).resolve()),
            "metadata": metadata or {},
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "pid": None,
            "returncode": None,
            "error": None,
            "log_path": str(self.jobs_root / f"{job_id}.log"),
        }
        with self._lock:
            self._write(job)
            self._pending.put(job_id)
        return self.get(job_id) or job

    def get(self, job_or_run_id: str) -> dict[str, Any] | None:
        with self._lock:
            direct = self._load_path(self._path(job_or_run_id))
        if direct:
            return self._enrich(direct)
        matches = [job for job in self.list() if job.get("run_id") == job_or_run_id]
        return matches[0] if matches else None

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [self._load_path(path) for path in self.jobs_root.glob("*.json")]
        valid = [self._enrich(job) for job in jobs if job]
        valid.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return valid

    def _enrich(self, job: dict[str, Any]) -> dict[str, Any]:
        result = dict(job)
        result["log_tail"] = self.log_tail(job, max_chars=4000)
        return result

    def log_tail(self, job: dict[str, Any], *, max_chars: int = 4000) -> str:
        path = Path(str(job.get("log_path") or ""))
        try:
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                stream.seek(max(0, size - max_chars * 4), os.SEEK_SET)
                return stream.read().decode("utf-8", errors="replace")[-max_chars:]
        except OSError:
            return ""

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._pending.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._run_job(job_id)
            finally:
                self._pending.task_done()

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._load_path(self._path(job_id))
            if not job or job.get("status") != "queued":
                return
            job.update(status="running", started_at=_now())
            self._write(job)
        try:
            log_path = Path(str(job["log_path"]))
            with log_path.open("a", encoding="utf-8", buffering=1) as log:
                proc = subprocess.Popen(
                    list(job["command"]),
                    cwd=str(job["cwd"]),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                with self._lock:
                    job.update(pid=proc.pid)
                    self._write(job)
                returncode = proc.wait()
            with self._lock:
                job.update(
                    status="completed" if returncode == 0 else "failed",
                    returncode=returncode,
                    finished_at=_now(),
                )
                self._write(job)
        except Exception as exc:
            with self._lock:
                job.update(status="failed", finished_at=_now(), error=str(exc))
                self._write(job)

    def stop(self) -> None:
        self._stop.set()


__all__ = ["FINAL_STATES", "PersistentEvalQueue"]

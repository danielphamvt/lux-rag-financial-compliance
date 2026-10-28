"""Thread-safe background subprocess manager for evaluation benchmark runs."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_TIMEOUT_SEC = 1800


class EvaluationRunManager:
    """Thread-safe background subprocess manager for running the reasoning benchmark."""

    def __init__(
        self,
        workspace_root: Path = WORKSPACE_ROOT,
        timeout_sec: int = DEFAULT_RUN_TIMEOUT_SEC,
    ) -> None:
        self.workspace_root = workspace_root
        self.timeout_sec = timeout_sec
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._status: str = "idle"
        self._started_at: float | None = None
        self._ended_at: float | None = None
        self._exit_code: int | None = None
        self._pid: int | None = None
        self._logs: list[str] = []

    def is_running(self) -> bool:
        """Check whether an evaluation subprocess is currently active."""
        with self._lock:
            return self._is_running_unlocked()

    def _is_running_unlocked(self) -> bool:
        if self._proc is not None and self._proc.poll() is None:
            return True
        return False

    def get_status(self) -> dict[str, Any]:
        """Return current subprocess telemetry and status snapshot."""
        with self._lock:
            running = self._is_running_unlocked()
            elapsed = 0.0
            if self._started_at is not None:
                end_ref = time.time() if running else (self._ended_at or time.time())
                elapsed = round(max(0.0, end_ref - self._started_at), 2)
            return {
                "is_running": running,
                "status": "running" if running else self._status,
                "pid": self._pid if running else None,
                "started_at": datetime.fromtimestamp(self._started_at).isoformat() if self._started_at else None,
                "elapsed_sec": elapsed,
                "exit_code": self._exit_code,
                "log_count": len(self._logs),
            }

    def get_logs_since(self, index: int = 0) -> tuple[list[str], int]:
        """Return log lines starting from `index` and the new cursor position."""
        with self._lock:
            slice_lines = list(self._logs[index:])
            return slice_lines, len(self._logs)

    def start_rerun(self, command: list[str] | None = None) -> tuple[bool, dict[str, Any]]:
        """Launch benchmark subprocess in background if not already running.

        Args:
            command: Optional override command list (used for testing).

        Returns:
            Tuple of (started_new_process: bool, current_status_dict: dict).
        """
        with self._lock:
            if self._is_running_unlocked():
                return False, {
                    "is_running": True,
                    "status": "running",
                    "pid": self._pid,
                    "message": "An evaluation benchmark is already running.",
                }

            cmd = command or [
                sys.executable,
                "-W",
                "ignore",
                "-u",
                "-m",
                "evaluation.evaluate_agent_calculation",
                "--all",
            ]
            env = {
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "PYTHONWARNINGS": "ignore",
                "ENV_FILE": os.environ.get("ENV_FILE", ".env.test"),
            }
            self._logs = [
                f"[{datetime.now().strftime('%H:%M:%S')}] Starting benchmark: {' '.join(cmd)}"
            ]
            self._status = "running"
            self._started_at = time.time()
            self._ended_at = None
            self._exit_code = None

            proc = subprocess.Popen(
                cmd,
                cwd=str(self.workspace_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            self._proc = proc
            self._pid = proc.pid

            reader = threading.Thread(
                target=self._read_stdout_worker,
                args=(proc,),
                daemon=True,
                name="eval-stdout-reader",
            )
            reader.start()

            watchdog = threading.Thread(
                target=self._timeout_worker,
                args=(proc, self.timeout_sec),
                daemon=True,
                name="eval-timeout-watchdog",
            )
            watchdog.start()

        return True, self.get_status()

    def _read_stdout_worker(self, proc: subprocess.Popen[str]) -> None:
        try:
            if proc.stdout is not None:
                for raw_line in iter(proc.stdout.readline, ""):
                    line = raw_line.rstrip("\r\n")
                    with self._lock:
                        self._logs.append(line)
        except Exception as exc:
            with self._lock:
                self._logs.append(f"[ERROR] Log stream reader error: {exc}")
        finally:
            rc = proc.wait()
            with self._lock:
                self._ended_at = time.time()
                self._exit_code = rc
                if self._status == "running":
                    self._status = "completed" if rc == 0 else "failed"
                elapsed = round((self._ended_at - (self._started_at or self._ended_at)), 2)
                self._logs.append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Process finished with exit code {rc} "
                    f"({self._status.upper()} in {elapsed}s)."
                )

    def _timeout_worker(self, proc: subprocess.Popen[str], timeout_sec: int) -> None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.25)
        if proc.poll() is None:
            with self._lock:
                self._status = "timeout"
                self._logs.append(
                    f"[TIMEOUT] Benchmark exceeded {timeout_sec}s limit — terminating process."
                )
            self._terminate_process(proc)

    def cancel(self) -> dict[str, Any]:
        """Terminate the active evaluation subprocess safely."""
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                return self.get_status()
            self._status = "cancelled"
            self._logs.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] [CANCELLED] Termination requested by user."
            )
        self._terminate_process(proc)
        return self.get_status()

    @staticmethod
    def _terminate_process(proc: subprocess.Popen[str]) -> None:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        for _ in range(10):
            if proc.poll() is not None:
                return
            time.sleep(0.1)
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


EVAL_MANAGER = EvaluationRunManager()

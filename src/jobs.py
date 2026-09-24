"""Launch / monitor / stop detached pipeline jobs.

A "job" is one background run of `scripts/run_job.py`, started in its own process session so it
survives the launching UI session. State lives in results/_runs/: one `<run_id>.json` per job
plus a `latest.json` pointer. The UI uses these helpers; nothing here imports streamlit.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config import REPO_ROOT

RUNS_DIR = REPO_ROOT / "results" / "_runs"
ACTIVE_STATES = {"starting", "running"}
# A running job heartbeats every 5s; allow generous slack for a paused/slow machine.
HEARTBEAT_TIMEOUT = 25.0


def _latest_path() -> Path:
    return RUNS_DIR / "latest.json"


def record_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.json"


def log_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.log"


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def current() -> dict | None:
    """The most recently launched job's record (running or finished), or None."""
    latest = _read_json(_latest_path())
    if not latest:
        return None
    rec = _read_json(record_path(latest.get("run_id", "")))
    if rec:
        rec["log"] = str(log_path(rec["run_id"]))
    return rec


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _heartbeat_age(rec: dict) -> float | None:
    ls = rec.get("last_seen")
    if not ls:
        return None
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(ls)).total_seconds()
    except ValueError:
        return None


def is_running(rec: dict | None) -> bool:
    """A job is running only if its record says so AND it is recently alive.

    Liveness is judged primarily by a heartbeat the job stamps every few seconds, so a hard
    kill (kill -9, crash, reboot) — where the final status is never written and the process may
    linger as a zombie — is correctly detected as not-running once the heartbeat goes stale.
    """
    if not rec or rec.get("status") not in ACTIVE_STATES:
        return False
    age = _heartbeat_age(rec)
    if age is not None:
        return age < HEARTBEAT_TIMEOUT
    # No heartbeat yet (job just launched): brief grace based on pid / starting state.
    if rec.get("status") == "starting" and not rec.get("pid"):
        return True
    return pid_alive(rec.get("pid"))


def display_status(rec: dict | None) -> str:
    if not rec:
        return "none"
    if rec.get("status") in ACTIVE_STATES and not is_running(rec):
        return "interrupted"   # process vanished without recording an exit
    return rec.get("status", "unknown")


def launch(kind: str = "pipeline", *, stage: str = "all", fresh: bool = False,
           models: str = "", total: int = 20,
           areas: str = "MS-PS3,MS-LS1,MS-ESS2,MS-ETS1") -> str:
    """Start a detached job and return its run_id.

    `models` (comma-separated) scopes the per-model stages (benchmark, evaluate) to that
    subset of benchmark_models; empty means all configured models.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record_path(run_id).write_text(json.dumps(
        {"run_id": run_id, "kind": kind, "stage": stage, "models": models, "status": "starting",
         "pid": None, "started_at": started, "finished_at": None, "returncode": None},
        indent=2))
    _latest_path().write_text(json.dumps({"run_id": run_id}))

    cmd = [sys.executable, "scripts/run_job.py", "--run-id", run_id, "--kind", kind,
           "--stage", stage, "--total", str(total), "--areas", areas]
    if models:
        cmd += ["--models", models]
    if fresh:
        cmd.append("--fresh")

    with open(log_path(run_id), "w") as logf:
        subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True, env={**os.environ},
        )  # child inherits a dup of the fd; safe to close ours when the with-block exits
    return run_id


def stop(rec: dict | None) -> bool:
    """Signal a running job's process group to terminate. Returns True if a signal was sent."""
    if not rec or not is_running(rec):
        return False
    pid = rec.get("pid")
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        return True
    except (OSError, ProcessLookupError):
        return False


def tail_log(rec: dict | None, n: int = 300) -> str:
    if not rec:
        return ""
    p = log_path(rec["run_id"])
    if not p.exists():
        return ""
    lines = p.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])

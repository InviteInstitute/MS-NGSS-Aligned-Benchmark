"""Detached job runner — executes one pipeline action and records its status.

Launched (usually by the UI via src/jobs.py) in its own session so it keeps running after the
launching tab/session goes away. It writes a small JSON status record to results/_runs/<id>.json
(status, pid, timestamps, return code) that the UI polls, and stops promptly on SIGTERM while
leaving the pipeline's own checkpoints intact (so a stopped run resumes cleanly).

Not normally run by hand; see src/jobs.py and the UI's Run page.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RUNS_DIR = ROOT / "results" / "_runs"
HEARTBEAT_SECS = 5
_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _record_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.json"


def _load(run_id: str) -> dict:
    p = _record_path(run_id)
    return json.loads(p.read_text()) if p.exists() else {"run_id": run_id}


def _save(run_id: str, **fields) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with _LOCK:  # heartbeat thread and main thread both write the record
        rec = _load(run_id)
        rec.update(fields)
        # Atomic write (tmp + replace) so a UI poll never reads a half-written record.
        path = _record_path(run_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, indent=2))
        tmp.replace(path)


def _start_heartbeat(run_id: str) -> None:
    """Stamp `last_seen` periodically so the UI can detect a dead process even after a hard kill."""
    def _loop() -> None:
        while True:
            _save(run_id, last_seen=_now())
            time.sleep(HEARTBEAT_SECS)
    threading.Thread(target=_loop, daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Detached pipeline job runner.")
    p.add_argument("--run-id", required=True)
    p.add_argument("--kind", default="pipeline", choices=["pipeline", "quickgen", "smoke"])
    p.add_argument("--stage", default="all")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--models", default="",
                   help="Comma-separated benchmark_models subset for per-model stages.")
    p.add_argument("--total", type=int, default=20)
    p.add_argument("--areas", default="MS-PS3,MS-LS1,MS-ESS2,MS-ETS1")
    args = p.parse_args(argv)
    run_id = args.run_id

    _save(run_id, kind=args.kind, stage=args.stage, models=args.models, pid=os.getpid(),
          status="running", started_at=_load(run_id).get("started_at") or _now(),
          last_seen=_now(), finished_at=None, returncode=None)
    _start_heartbeat(run_id)

    def _on_term(signum, frame):  # noqa: ARG001
        _save(run_id, status="stopped", finished_at=_now(), returncode=143)
        print(f"\n[run_job] received signal {signum}; stopping.", flush=True)
        os._exit(143)  # force-exit so worker threads don't block shutdown; checkpoints persist

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    print(f"[run_job] {run_id} kind={args.kind} stage={args.stage} fresh={args.fresh}", flush=True)
    try:
        if args.kind == "smoke":
            from scripts.smoke_test import run as smoke_run
            smoke_run(use_mock=True)
        elif args.kind == "quickgen":
            from src.config import Config
            from src.generate_dataset import generate
            from scripts.quick_gen_test import _print, _split_for
            areas = [a.strip() for a in args.areas.split(",") if a.strip()]
            per_area = max(1, args.total // max(1, len(areas)))
            cfg = Config.load()
            cfg.raw["questions_per_area"] = per_area
            cfg.raw["question_type_split"] = _split_for(per_area)
            cfg.raw["paths"]["ground_truth"] = "data/ground_truth_sample.csv"
            df = generate(cfg, fresh=True, only_areas=areas)
            _print(df, cfg)   # show the generated questions in the job log for review
        else:
            from src.run_pipeline import run as pipeline_run
            models = [m.strip() for m in args.models.split(",") if m.strip()] or None
            pipeline_run(args.stage, fresh=args.fresh, models=models)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        _save(run_id, status="failed", finished_at=_now(), returncode=1)
        return 1
    _save(run_id, status="done", finished_at=_now(), returncode=0)
    print(f"[run_job] {run_id} done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

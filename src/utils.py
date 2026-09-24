"""Shared helpers: logging, CSV I/O, and checkpoint/resume.

Checkpoint/resume model
-----------------------
Every stage writes a CSV keyed by a stable id. A CheckpointWriter loads any existing rows
on construction, exposes the set of already-completed ids (so callers skip finished work),
and appends new rows, flushing periodically. Re-running an interrupted command therefore
resumes where it left off; pass `fresh=True` to start clean.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def backup_file(path: Path) -> Path | None:
    """Move an existing file into a sibling `_backups/` dir with a timestamped name, so a
    `fresh=True` wipe is always RECOVERABLE instead of a permanent delete.

    Returns the backup path (or None if the file didn't exist). The `_backups/` dirs are
    gitignored and safe to delete once you're sure you don't need the previous run. Backups
    live in a subdir so the pipeline's non-recursive `*.csv` scans never pick them up as data.
    """
    path = Path(path)
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup_dir = path.parent / "_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / f"{path.stem}.{stamp}{path.suffix}"
    path.replace(dest)   # atomic move within the same filesystem
    return dest


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def to_temp(value: object) -> float | None:
    """Coerce a config temperature to float, or None if unset/blank.

    None means "do not send a temperature" so the model uses its own default — important for
    models that reject an explicit temperature or require a specific value.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return float(value)


def to_int(value: object) -> int | None:
    """Coerce a config integer (e.g. max_tokens) to int, or None if unset/blank.

    None means "do not send the limit" so the model can produce its full output / reasoning.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return int(value)


def read_csv(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Expected input {path} not found. Run the prior pipeline stage first."
        )
    return pd.read_csv(path)


def load_judged(cfg) -> "pd.DataFrame | None":
    """Every judged ground-truth row (with `judge_<name>` columns), regardless of auto-drop.

    In auto-drop mode `ground_truth_validated.csv` keeps only unanimously-accepted rows, so judge
    *agreement/reliability* stats computed from it alone would be meaningless (~100% agreement).
    The auto-dropped rows are recorded (with their judge columns) in auto_dropped_groundtruth.csv;
    this concatenates both so agreement is computed over EVERY verdict. Returns None if there's no
    validated file yet. When auto-drop is off, this is just the validated set.
    """
    vp = Path(cfg.path("ground_truth_validated"))
    if not vp.exists():
        return None
    val = pd.read_csv(vp)
    dp = Path(cfg.path("human_review_dir")) / "auto_dropped_groundtruth.csv"
    if dp.exists():
        dropped = pd.read_csv(dp)
        if len(dropped):
            val = (pd.concat([val, dropped], ignore_index=True)
                   .drop_duplicates(subset=["question_id"], keep="first"))
    return val


class CheckpointWriter:
    """Append-only, resumable CSV writer keyed by one or more id columns."""

    def __init__(
        self,
        path: Path,
        key_cols: Sequence[str],
        columns: Sequence[str],
        *,
        flush_every: int = 25,
        fresh: bool = False,
    ) -> None:
        self.path = Path(path)
        self.key_cols = list(key_cols)
        self.columns = list(columns)
        self.flush_every = max(1, flush_every)
        self._buffer: list[dict] = []
        self._existing: pd.DataFrame
        ensure_parent(self.path)
        if fresh and self.path.exists():
            # Don't delete on --fresh: move the old file to _backups/ so a wipe is recoverable.
            dest = backup_file(self.path)
            if dest is not None:
                get_logger("checkpoint").info(
                    "fresh: backed up %s -> %s (delete _backups/ when you're sure).",
                    self.path, dest)
        if self.path.exists():
            self._existing = pd.read_csv(self.path)
        else:
            self._existing = pd.DataFrame(columns=self.columns)

    def _key(self, row: dict) -> tuple:
        return tuple(str(row[c]) for c in self.key_cols)

    @property
    def done_keys(self) -> set[tuple]:
        if self._existing.empty:
            return set()
        return {
            tuple(str(v) for v in vals)
            for vals in self._existing[self.key_cols].itertuples(index=False, name=None)
        }

    def is_done(self, row: dict) -> bool:
        return self._key(row) in self.done_keys

    def pending(self, rows: Iterable[dict]) -> list[dict]:
        done = self.done_keys
        return [r for r in rows if self._key(r) not in done]

    def add(self, row: dict) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        df = pd.DataFrame(self._buffer, columns=self.columns)
        header = not self.path.exists() or self.path.stat().st_size == 0
        df.to_csv(self.path, mode="a", header=header, index=False)
        self._buffer.clear()

    def close(self) -> pd.DataFrame:
        self.flush()
        return pd.read_csv(self.path) if self.path.exists() else pd.DataFrame(columns=self.columns)

    def __enter__(self) -> "CheckpointWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.flush()

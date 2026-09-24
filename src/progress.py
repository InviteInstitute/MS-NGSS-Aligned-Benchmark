"""Compute pipeline progress and usage stats from on-disk checkpoint files.

Because every stage writes its results incrementally (checkpoint/resume), the most reliable
progress signal is simply "rows completed / rows expected" read straight off those CSVs. This
works across restarts and disconnects and needs no in-memory run state, so the UI (or a CLI)
can report accurate progress even for a job started hours ago in another session.

CLI:  python -m src.progress         # print a progress + usage summary
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

from src.config import Config


@dataclass
class StageProgress:
    stage: str
    done: int
    total: int

    @property
    def pct(self) -> float:
        return 0.0 if self.total <= 0 else min(1.0, self.done / self.total)


def _count_rows(path: Path) -> int:
    """Fast data-row count (excludes header). 0 if the file is missing/empty."""
    if not Path(path).exists():
        return 0
    with open(path, "rb") as f:
        n = sum(1 for _ in f)
    return max(0, n - 1)


def _sum_rows(paths: list[Path]) -> int:
    return sum(_count_rows(p) for p in paths)


def stage_progress(cfg: Config | None = None) -> list[StageProgress]:
    cfg = cfg or Config.load(require_keys=False)
    qpa = int(cfg.get("questions_per_area", default=100))
    runs = int(cfg.get("runs_per_model", default=3))
    n_models = len(cfg.get("benchmark_models", default=[]) or [])
    n_gt_judges = len(cfg.get("gt_judges", default=[]) or [])
    mode = cfg.get("eval_mode", default="single")
    n_eval_judges = 1 if mode == "single" else len(cfg.get("eval_judges", default=[]) or [])

    gt_path = cfg.path("ground_truth")
    n_questions = _count_rows(gt_path) or 12 * qpa  # fall back to the target before generation

    resp_dir = cfg.path("responses_dir")
    resp_files = sorted(Path(resp_dir).glob("*.csv")) if Path(resp_dir).exists() else []
    benchmark_done = _sum_rows(resp_files)

    eval_dir = cfg.path("eval_dir")
    eval_long = sorted(Path(eval_dir).glob(f"_*_{mode}_long.csv")) if Path(eval_dir).exists() else []
    eval_done = _sum_rows(eval_long)

    gt_long = gt_path.parent / "_gt_judge_long.csv"

    return [
        StageProgress("generate", _count_rows(gt_path), 12 * qpa),
        StageProgress("validate", _count_rows(gt_long), n_questions * n_gt_judges),
        StageProgress("benchmark", benchmark_done, n_questions * runs * n_models),
        StageProgress("evaluate", eval_done, n_questions * runs * n_models * n_eval_judges),
    ]


def usage_stats(cfg: Config | None = None) -> dict:
    """Tokens consumed and API errors so far, summed across benchmark response files."""
    import pandas as pd

    cfg = cfg or Config.load(require_keys=False)
    resp_dir = cfg.path("responses_dir")
    files = sorted(Path(resp_dir).glob("*.csv")) if Path(resp_dir).exists() else []
    total_tokens = 0
    completion_tokens = 0
    reasoning_tokens = 0
    errors = 0
    responses = 0

    def _col_sum(df: "pd.DataFrame", col: str) -> int:
        if col not in df.columns:
            return 0
        return int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())

    for f in files:
        try:
            df = pd.read_csv(f, usecols=lambda c: c in {
                "total_tokens", "completion_tokens", "reasoning_tokens", "is_error"})
        except (ValueError, OSError):
            continue
        responses += len(df)
        total_tokens += _col_sum(df, "total_tokens")
        completion_tokens += _col_sum(df, "completion_tokens")
        reasoning_tokens += _col_sum(df, "reasoning_tokens")
        errors += _col_sum(df, "is_error")
    return {
        "responses": responses,
        "total_tokens": total_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "errors": errors,
    }


def summary(cfg: Config | None = None) -> dict:
    cfg = cfg or Config.load(require_keys=False)
    return {
        "stages": [asdict(s) | {"pct": round(s.pct, 4)} for s in stage_progress(cfg)],
        "usage": usage_stats(cfg),
    }


def main() -> int:
    cfg = Config.load(require_keys=False)
    print("Pipeline progress")
    for s in stage_progress(cfg):
        bar = "#" * int(s.pct * 20)
        print(f"  {s.stage:10s} [{bar:<20}] {s.pct*100:5.1f}%  ({s.done}/{s.total})")
    u = usage_stats(cfg)
    print(f"\nUsage: {u['responses']} responses, {u['total_tokens']:,} total tokens, "
          f"{u['reasoning_tokens']:,} reasoning, {u['errors']} API errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Readability helpers + Stage 1b dataset quality audit.

Readability functions wrap `textstat` (Flesch-Kincaid grade, Flesch reading-ease) and are
used both as a generation-time appropriateness gate and to score benchmark responses.

The audit (`run()` / CLI) reports, for the generated ground truth: per-area and per-type
counts, per-standard coverage, and the question readability distribution vs the target band.

CLI:  python -m src.quality_checks
"""
from __future__ import annotations

import textstat

from src.config import REPO_ROOT, Config
from src.standards import load_areas
from src.utils import get_logger, read_csv

log = get_logger("quality")


def fk_grade(text: str) -> float:
    """Flesch-Kincaid grade level. Returns NaN-safe 0.0 for empty/degenerate text."""
    if not text or not text.strip():
        return 0.0
    try:
        return float(textstat.flesch_kincaid_grade(text))
    except Exception:  # noqa: BLE001 - textstat can choke on odd input
        return 0.0


def reading_ease(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    try:
        return float(textstat.flesch_reading_ease(text))
    except Exception:  # noqa: BLE001
        return 0.0


def in_band(text: str, fk_range: tuple[float, float]) -> bool:
    lo, hi = fk_range
    return lo <= fk_grade(text) <= hi


def run(cfg: Config | None = None) -> dict:
    """Audit data/ground_truth.csv; write a report CSV; return a summary dict."""
    cfg = cfg or Config.load(require_keys=False)
    df = read_csv(cfg.path("ground_truth"))
    areas = {a.code: a for a in load_areas(cfg)}
    lo, hi = cfg.get("appropriateness", "fk_grade_range", default=[5.0, 9.0])

    df = df.copy()
    if "fk_grade" not in df.columns:
        df["fk_grade"] = df["question"].astype(str).map(fk_grade)

    rows = []
    for code, area in areas.items():
        sub = df[df["area"] == code]
        type_counts = sub["question_type"].value_counts().to_dict()
        std_counts = sub["standard"].value_counts().to_dict()
        covered = sum(1 for s in area.standards if std_counts.get(s.code, 0) > 0)
        in_band_frac = float(((sub["fk_grade"] >= lo) & (sub["fk_grade"] <= hi)).mean()) if len(sub) else 0.0
        rows.append({
            "area": code,
            "n_questions": int(len(sub)),
            "n_recall": int(type_counts.get("recall", 0)),
            "n_conceptual": int(type_counts.get("conceptual", 0)),
            "n_applied": int(type_counts.get("applied", 0)),
            "standards_total": len(area.standards),
            "standards_covered": covered,
            "mean_fk_grade": round(float(sub["fk_grade"].mean()), 2) if len(sub) else 0.0,
            "frac_in_band": round(in_band_frac, 3),
        })

    import pandas as pd

    report = pd.DataFrame(rows)
    out = cfg.path("reports_dir") / "dataset_quality.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)

    summary = {
        "total_questions": int(len(df)),
        "areas": int(report["area"].nunique()),
        "min_coverage": f"{report['standards_covered'].min()}/{report['standards_total'].max()}",
        "mean_frac_in_band": round(float(report["frac_in_band"].mean()), 3),
        "report": str(out.relative_to(REPO_ROOT)),
    }
    log.info("Dataset quality audit: %s", summary)
    # Warn on any area that under-delivered or drifted out of band.
    for r in rows:
        if r["n_questions"] < cfg.get("questions_per_area", default=100):
            log.warning("Area %s has only %d questions.", r["area"], r["n_questions"])
        if r["standards_covered"] < r["standards_total"]:
            log.warning("Area %s covers %d/%d standards.", r["area"],
                        r["standards_covered"], r["standards_total"])
    return summary


if __name__ == "__main__":
    run()

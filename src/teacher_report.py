"""Teacher-facing leaderboards: first-run vs. ensemble (majority-vote) accuracy per model.

Reads the evaluation outputs (results/eval/<model>_<mode>.csv) — reusing the analyze stage's
loaders so human corrections are honored — and writes simple, decision-oriented tables to
results/reports/ for a teacher choosing a model to support a given NGSS standard:

  teacher_leaderboard_overall.csv      models ranked overall
  teacher_leaderboard_by_area.csv      models ranked within each NGSS area
  teacher_leaderboard_by_standard.csv  models ranked within each standard (+ standard text)
  teacher_best_by_area.csv             quick pick: each area  -> all top-tied models, next tier, gap
  teacher_best_by_standard.csv         quick pick: each standard -> all top-tied models, next tier, gap

Two scores per model, both as 0-1 accuracy (a teacher reads 0.92 = 92%):
  first-run  = accuracy on run #1 only               (single-shot behaviour)
  ensemble   = majority vote across the model's runs, then accuracy  (self-consistency)

This is a purely ADDITIVE analysis: it adds no pipeline stage and MODIFIES NO existing files —
it only writes new teacher_*.csv reports. CLI:  python -m src.teacher_report [--mode single|triple]
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.analyze import _apply_human_overrides, _load_eval
from src.config import Config
from src.standards import load_areas
from src.utils import get_logger

log = get_logger("teacher_report")


def _first_run_and_ensemble(ev: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    """Per (model, *group): first-run accuracy, ensemble (majority-vote) accuracy, n_questions."""
    # First-run: run == 1 rows only, averaged over the group's questions.
    first = ev[ev["run"] == 1].groupby(["model", *group])["binary_score"].mean()

    # Ensemble: collapse each question across its runs by majority vote, then average.
    per_item = (ev.groupby(["model", *group, "question_id"])["binary_score"]
                .mean().ge(0.5).astype(int))
    ens = per_item.groupby(["model", *group]).mean()
    nq = per_item.groupby(["model", *group]).size()

    out = pd.DataFrame({"first_run_accuracy": first, "ensemble_accuracy": ens,
                        "n_questions": nq}).reset_index()
    out["first_run_accuracy"] = out["first_run_accuracy"].round(3)
    out["ensemble_accuracy"] = out["ensemble_accuracy"].round(3)
    out["n_questions"] = out["n_questions"].astype(int)
    return out


def _rank_within(df: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    """Rank models within each group by ensemble desc, then first-run desc, then name."""
    df = df.sort_values([*group, "ensemble_accuracy", "first_run_accuracy", "model"],
                        ascending=[*([True] * len(group)), False, False, True])
    df["rank"] = (df.groupby(group).cumcount() + 1) if group else range(1, len(df) + 1)
    return df.reset_index(drop=True)


def _tied_models(sub: pd.DataFrame, score: float) -> list[str]:
    """Models at exactly `score` (ensemble), ordered by first-run desc then name."""
    hit = sub[sub["ensemble_accuracy"] == score]
    return hit.sort_values(["first_run_accuracy", "model"],
                           ascending=[False, True])["model"].tolist()


def _best_pick(ranked: pd.DataFrame, key: str, carry: list[str]) -> pd.DataFrame:
    """One row per group: ALL models tied at the top ensemble score + the next tier + the gap.

    When several models tie at the top (common here — many standards are aced by multiple models),
    every tied model is listed rather than arbitrarily picking one. `runner_up_models` is the next
    distinct score tier (also listing all its ties). `ranked` is already in natural NGSS order, which
    groupby(sort=False) preserves.
    """
    rows = []
    for k, sub in ranked.groupby(key, sort=False):
        scores = sorted(sub["ensemble_accuracy"].unique(), reverse=True)  # rounded → exact ties group
        top = scores[0]
        best = _tied_models(sub, top)
        rec = {key: k}
        for c in carry:
            rec[c] = sub.iloc[0][c]          # carried labels are constant within the group
        rec["best_models"] = ", ".join(best)
        rec["n_tied"] = len(best)
        rec["ensemble_accuracy"] = top
        if len(scores) > 1:
            nxt = scores[1]
            rec["runner_up_models"] = ", ".join(_tied_models(sub, nxt))
            rec["runner_up_ensemble"] = nxt
            rec["gap"] = round(float(top - nxt), 3)
        else:
            rec["runner_up_models"] = rec["runner_up_ensemble"] = rec["gap"] = ""
        rows.append(rec)
    return pd.DataFrame(rows)


def teacher_report(cfg: Config | None = None, *, mode: str | None = None) -> dict[str, str]:
    """Write the teacher-facing leaderboard CSVs; return {filename: path} for those written."""
    cfg = cfg or Config.load(require_keys=False)
    mode = mode or cfg.get("eval_mode")
    ev = _load_eval(cfg, mode)                 # raises FileNotFoundError if Stage 4 hasn't run
    ev = _apply_human_overrides(cfg, ev)       # honor human label overrides + dropped items
    ev["run"] = pd.to_numeric(ev["run"], errors="coerce")

    areas = load_areas(cfg)
    area_title = {a.code: a.title for a in areas}
    standard_text = {s.code: s.text for a in areas for s in a.standards}
    standard_area = {s.code: a.code for a in areas for s in a.standards}
    area_ord = {a.code: i for i, a in enumerate(areas)}
    std_ord = {s.code: i for i, a in enumerate(areas) for s in a.standards}

    reports = cfg.path("reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def _write(name: str, df: pd.DataFrame) -> None:
        if df is None or len(df) == 0:
            return
        path = reports / name
        df.to_csv(path, index=False)
        written[name] = str(path)

    # Overall (micro accuracy over all questions).
    overall = _rank_within(_first_run_and_ensemble(ev, []), [])
    overall["ensemble_gain"] = (overall["ensemble_accuracy"] - overall["first_run_accuracy"]).round(3)
    _write("teacher_leaderboard_overall.csv",
           overall[["rank", "model", "first_run_accuracy", "ensemble_accuracy",
                    "ensemble_gain", "n_questions"]])

    # By area (ranked within each area, rows in natural NGSS area order).
    by_area = _rank_within(_first_run_and_ensemble(ev, ["area"]), ["area"])
    by_area["area_title"] = by_area["area"].map(area_title)
    by_area["_ord"] = by_area["area"].map(area_ord)
    by_area = by_area.sort_values(["_ord", "rank"]).drop(columns="_ord").reset_index(drop=True)
    _write("teacher_leaderboard_by_area.csv",
           by_area[["area", "area_title", "rank", "model", "first_run_accuracy",
                    "ensemble_accuracy", "n_questions"]])

    # By standard (ranked within each standard, rows in natural NGSS standard order).
    by_std = _rank_within(_first_run_and_ensemble(ev, ["standard"]), ["standard"])
    by_std["area"] = by_std["standard"].map(standard_area)
    by_std["standard_text"] = by_std["standard"].map(standard_text)
    by_std["_ord"] = by_std["standard"].map(std_ord)
    by_std = by_std.sort_values(["_ord", "rank"]).drop(columns="_ord").reset_index(drop=True)
    _write("teacher_leaderboard_by_standard.csv",
           by_std[["standard", "standard_text", "area", "rank", "model",
                   "first_run_accuracy", "ensemble_accuracy", "n_questions"]])

    # Quick-pick summaries (already in natural order from the ranked tables above).
    _write("teacher_best_by_area.csv", _best_pick(by_area, "area", ["area_title"]))
    _write("teacher_best_by_standard.csv", _best_pick(by_std, "standard", ["standard_text", "area"]))

    log.info("Teacher reports written: %s", sorted(written))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Teacher-facing leaderboards (first-run vs. ensemble).")
    parser.add_argument("--mode", choices=["single", "triple"], help="Override eval_mode.")
    args = parser.parse_args(argv)
    written = teacher_report(mode=args.mode)
    print("Wrote:", sorted(written) or "(nothing — no eval data?)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

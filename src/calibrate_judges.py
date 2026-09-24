"""Stage 4b — validate the LLM judges against human labels.

Two steps:
  export()  -> writes results/calibration/to_label.csv: a stratified (area x question_type)
               sample of ground-truth pairs and benchmark responses, each carrying the LLM
               judges' verdicts and an empty `human_score` column for a human to fill
               (1 = correct, 0 = incorrect).
  compute() -> once `human_score` is filled, writes results/calibration/judge_vs_human.csv:
               per-judge accuracy, Cohen's kappa, and false-correct/false-incorrect rates vs
               the human labels (overall and per source). This calibrates the instrument.

run() does export, then compute if human labels are present. CLI: python -m src.calibrate_judges
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.utils import get_logger, read_csv

log = get_logger("calibrate")


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa for two binary raters. NaN if undefined."""
    a, b = a.astype(int), b.astype(int)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    # expected agreement from marginals over classes {0,1}
    pe = 0.0
    for c in (0, 1):
        pe += (a == c).mean() * (b == c).mean()
    if pe == 1.0:
        return 1.0 if po == 1.0 else float("nan")
    return (po - pe) / (1 - pe)


def _gt_judge_long(cfg: Config) -> pd.DataFrame:
    """Ground-truth judge verdicts in long form: question_id, judge, llm_score, + context."""
    path = cfg.path("ground_truth_validated")
    if not path.exists():
        return pd.DataFrame()
    df = read_csv(path)
    judge_cols = [c for c in df.columns if c.startswith("judge_") and not c.endswith("_binary")]
    rows = []
    for _, r in df.iterrows():
        for jc in judge_cols:
            rows.append({
                "source": "groundtruth", "item_id": r["question_id"], "run": "",
                "area": r["area"], "question_type": r["question_type"],
                "question": r["question"], "reference_answer": r["reference_answer"],
                "response": "", "model": "", "judge": jc.replace("judge_", ""),
                "llm_score": r[jc],
            })
    return pd.DataFrame(rows)


def _eval_judge_long(cfg: Config) -> pd.DataFrame:
    """Response judge verdicts in long form from all eval files (single or triple)."""
    eval_dir = cfg.path("eval_dir")
    single_judge = cfg.get("eval_judge")
    rows = []
    for path in sorted(Path(eval_dir).glob("*_single.csv")) + sorted(Path(eval_dir).glob("*_triple.csv")):
        model = path.stem.rsplit("_", 1)[0]
        mode = path.stem.rsplit("_", 1)[1]
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            ctx = {"source": "response", "item_id": r["question_id"], "run": r.get("run", ""),
                   "area": r["area"], "question_type": r["question_type"],
                   "question": "", "reference_answer": "", "response": r.get("response", ""),
                   "model": model}
            if mode == "triple":
                for col in df.columns:
                    if col.startswith("judge_") and col.endswith("_binary"):
                        judge = col[len("judge_"):-len("_binary")]
                        rows.append({**ctx, "judge": judge, "llm_score": r[col]})
            else:
                rows.append({**ctx, "judge": single_judge, "llm_score": r.get("binary_score")})
    return pd.DataFrame(rows)


def export(cfg: Config | None = None) -> Path:
    cfg = cfg or Config.load(require_keys=False)
    per_cell = int(cfg.get("calibration", "sample_per_cell", default=3))
    rng = np.random.default_rng(42)

    long = pd.concat([_gt_judge_long(cfg), _eval_judge_long(cfg)], ignore_index=True)
    if long.empty:
        log.warning("No judge verdicts found yet (run Stage 2 / Stage 4 first).")
        out = cfg.path("calibration_dir") / "to_label.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(out, index=False)
        return out

    # One row per item with judges spread across llm_<judge> columns.
    item_keys = ["source", "item_id", "run", "model", "area", "question_type",
                 "question", "reference_answer", "response"]
    wide = long.pivot_table(index=item_keys, columns="judge", values="llm_score",
                            aggfunc="first").reset_index()
    judges = [c for c in wide.columns if c not in item_keys]
    wide = wide.rename(columns={j: f"llm_{j}" for j in judges})

    # Stratified sample by (source, area, question_type). An explicit loop preserves all
    # columns (groupby.apply can drop the grouping keys in some pandas versions).
    parts = [g.sample(min(len(g), per_cell), random_state=int(rng.integers(1_000_000)))
             for _, g in wide.groupby(["source", "area", "question_type"])]
    sampled = pd.concat(parts, ignore_index=True) if parts else wide.iloc[0:0]
    sampled["human_score"] = ""    # human fills: 1 = correct, 0 = incorrect
    sampled["human_notes"] = ""

    out = cfg.path("calibration_dir") / "to_label.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out, index=False)
    log.info("Wrote calibration sheet %s (%d items, judges: %s).",
             out, len(sampled), [j for j in judges])
    return out


def compute(cfg: Config | None = None) -> pd.DataFrame | None:
    cfg = cfg or Config.load(require_keys=False)
    path = cfg.path("calibration_dir") / "to_label.csv"
    if not path.exists():
        log.warning("No calibration sheet at %s; run export first.", path)
        return None
    df = pd.read_csv(path)
    if "human_score" not in df.columns:
        log.info("Calibration sheet has no human_score column; skipping.")
        return None
    df["human_score"] = pd.to_numeric(df["human_score"], errors="coerce")
    df = df.dropna(subset=["human_score"])
    if df.empty:
        log.info("Calibration sheet not yet labeled by a human; skipping agreement computation.")
        return None
    llm_cols = [c for c in df.columns if c.startswith("llm_")]

    rows = []
    for source in ["__all__", *sorted(df["source"].unique())]:
        sub = df if source == "__all__" else df[df["source"] == source]
        for col in llm_cols:
            judge = col[len("llm_"):]
            pair = sub[[col, "human_score"]].dropna()
            if pair.empty:
                continue
            llm = pair[col].astype(int)
            hum = pair["human_score"].astype(int)
            n = len(pair)
            acc = float((llm == hum).mean())
            # false-correct: judge said 1 (correct) but human said 0; vice versa
            fc = float(((llm == 1) & (hum == 0)).sum()) / max(1, int((hum == 0).sum()))
            fi = float(((llm == 0) & (hum == 1)).sum()) / max(1, int((hum == 1).sum()))
            rows.append({"source": source, "judge": judge, "n": n,
                         "accuracy": round(acc, 3),
                         "cohen_kappa": round(cohen_kappa(llm, hum), 3),
                         "false_correct_rate": round(fc, 3),
                         "false_incorrect_rate": round(fi, 3)})
    report = pd.DataFrame(rows)
    out = cfg.path("calibration_dir") / "judge_vs_human.csv"
    report.to_csv(out, index=False)
    log.info("Wrote judge-vs-human calibration %s:\n%s", out, report.to_string(index=False))
    return report


def run(cfg: Config | None = None) -> None:
    cfg = cfg or Config.load(require_keys=False)
    export(cfg)
    compute(cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 4b: judge<->human calibration.")
    parser.add_argument("--export", action="store_true", help="Only (re)write the labeling sheet.")
    parser.add_argument("--compute", action="store_true", help="Only compute agreement from labels.")
    args = parser.parse_args(argv)
    cfg = Config.load(require_keys=False)
    if args.export:
        export(cfg)
    elif args.compute:
        compute(cfg)
    else:
        run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

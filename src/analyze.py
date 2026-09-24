"""Stage 5 — compute metrics, statistics, and reports from the evaluation outputs.

Reads results/eval/<model>_<mode>.csv (+ response and ground-truth files) and writes a set
of report CSVs to results/reports/:

  accuracy_by_area.csv, accuracy_by_type.csv, accuracy_by_area_type.csv, accuracy_by_standard.csv
  leaderboard.csv          macro/micro accuracy, rubric, refusal rate, mean response reading level
  reading_level.csv        response Flesch-Kincaid aggregated over runs, area, type vs target band
  item_stats.csv           per-item fraction-correct, difficulty/discrimination, trivial/broken flags
  significance.csv         pairwise McNemar tests with Holm-Bonferroni correction
  judge_reliability.csv    Fleiss kappa + raw agreement for the judge panels (+ judge<->human if available)
  self_preference.csv      (triple mode) each judge's score delta on same-family vs other-family responses

Each report is guarded so the stage runs end-to-end even with partial data (e.g. a single-judge
smoke run). CLI:  python -m src.analyze [--mode single|triple]
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from src.config import Config
from src.utils import get_logger, load_judged

log = get_logger("analyze")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _load_eval(cfg: Config, mode: str) -> pd.DataFrame:
    """Concatenate per-model eval files for the active mode, adding a `model` column."""
    frames = []
    for path in sorted(Path(cfg.path("eval_dir")).glob(f"*_{mode}.csv")):
        model = path.stem.rsplit("_", 1)[0]
        df = pd.read_csv(path)
        df["model"] = model
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No eval files results/eval/*_{mode}.csv found. Run Stage 4 first."
        )
    out = pd.concat(frames, ignore_index=True)
    out["binary_score"] = pd.to_numeric(out["binary_score"], errors="coerce").fillna(0).astype(int)
    out["rubric_score"] = pd.to_numeric(out["rubric_score"], errors="coerce")
    out["is_refusal"] = pd.to_numeric(out.get("is_refusal", 0), errors="coerce").fillna(0).astype(int)
    return out


def _apply_human_overrides(cfg: Config, ev: pd.DataFrame) -> pd.DataFrame:
    """Fold filled-in human labels back into the evaluation table before scoring.

    - `results/human_review/flags_eval_<model>.csv`: where a human filled `human_score`, that
      authoritative 0/1 replaces the LLM `binary_score` for that (model, question_id, run).
    - `results/human_review/flags_groundtruth.csv`: any question a human marked incorrect
      (`human_score == 0`, i.e. the reference answer itself is wrong) is dropped from the
      benchmark entirely, since it is not a valid item to score models against.

    This is what makes "re-run analyze after labeling" actually change the leaderboard. Files
    that are absent or still unlabeled are silently ignored, so the stage runs before and after
    human input.
    """
    hr = cfg.path("human_review_dir")
    ev = ev.copy()

    # 1) Per-response corrections.
    n_resp = 0
    for path in sorted(Path(hr).glob("flags_eval_*.csv")):
        model = path.stem[len("flags_eval_"):]
        flags = pd.read_csv(path)
        if "human_score" not in flags.columns:
            continue
        flags = flags.copy()
        flags["human_score"] = pd.to_numeric(flags["human_score"], errors="coerce")
        flags = flags.dropna(subset=["human_score"])
        for _, r in flags.iterrows():
            mask = ((ev["model"] == model) & (ev["question_id"] == r["question_id"])
                    & (ev["run"] == r["run"]))
            if mask.any():
                ev.loc[mask, "binary_score"] = int(r["human_score"])
                n_resp += int(mask.sum())

    # 2) Drop ground-truth items a human rejected.
    gt_flags = Path(hr) / "flags_groundtruth.csv"
    dropped: set = set()
    if gt_flags.exists():
        gf = pd.read_csv(gt_flags)
        if "human_score" in gf.columns:
            gf["human_score"] = pd.to_numeric(gf["human_score"], errors="coerce")
            dropped = set(gf.loc[gf["human_score"] == 0, "question_id"])
    if dropped:
        before = len(ev)
        ev = ev[~ev["question_id"].isin(dropped)]
        log.info("Human review: dropped %d invalid ground-truth questions (%d eval rows).",
                 len(dropped), before - len(ev))
    if n_resp:
        log.info("Human review: overrode %d eval rows with human labels.", n_resp)
    return ev


def _wilson(correct: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    lo, hi = proportion_confint(correct, n, alpha=0.05, method="wilson")
    return round(float(lo), 4), round(float(hi), 4)


# --------------------------------------------------------------------------- #
# Accuracy tables
# --------------------------------------------------------------------------- #
def _accuracy_table(ev: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in ev.groupby(["model", *group]):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n = len(sub)
        correct = int(sub["binary_score"].sum())
        lo, hi = _wilson(correct, n)
        # run-to-run variance: accuracy per run, then std
        per_run = sub.groupby("run")["binary_score"].mean()
        rec = dict(zip(["model", *group], keys))
        rec.update({
            "n": n,
            "binary_accuracy": round(correct / n, 4) if n else 0.0,
            "wilson_low": lo, "wilson_high": hi,
            "rubric_mean": round(float(sub["rubric_score"].mean()), 4)
                if sub["rubric_score"].notna().any() else "",
            "refusal_rate": round(float(sub["is_refusal"].mean()), 4),
            "run_acc_std": round(float(per_run.std(ddof=0)), 4) if len(per_run) > 1 else 0.0,
        })
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["model", *group]).reset_index(drop=True)


def _leaderboard(ev: pd.DataFrame, responses_fk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, sub in ev.groupby("model"):
        micro = float(sub["binary_score"].mean())
        area_acc = sub.groupby("area")["binary_score"].mean()
        macro = float(area_acc.mean())
        rec = {
            "model": model,
            "macro_accuracy": round(macro, 4),     # equal weight per area (primary)
            "micro_accuracy": round(micro, 4),     # per-question
            "rubric_macro": round(float(sub.groupby("area")["rubric_score"].mean().mean()), 4)
                if sub["rubric_score"].notna().any() else "",
            "refusal_rate": round(float(sub["is_refusal"].mean()), 4),
            "n_items": int(len(sub)),
        }
        if not responses_fk.empty:
            sub_r = responses_fk[responses_fk["model"] == model]
            if "response_fk_grade" in sub_r.columns:
                fk = pd.to_numeric(sub_r["response_fk_grade"], errors="coerce")
                rec["mean_response_fk_grade"] = round(float(fk.mean()), 2) if len(fk) else ""
            for col, label in [("total_tokens", "mean_total_tokens"),
                               ("reasoning_tokens", "mean_reasoning_tokens")]:
                if col in sub_r.columns:
                    rec[label] = round(float(pd.to_numeric(sub_r[col], errors="coerce").mean()), 1)
        rows.append(rec)
    lb = pd.DataFrame(rows).sort_values("macro_accuracy", ascending=False).reset_index(drop=True)
    lb.insert(0, "rank", lb.index + 1)
    return lb


# --------------------------------------------------------------------------- #
# Reading level of responses
# --------------------------------------------------------------------------- #
def _reading_level(cfg: Config, mode: str) -> pd.DataFrame:
    lo, hi = cfg.get("appropriateness", "fk_grade_range", default=[5.0, 9.0])
    frames = []
    for path in sorted(Path(cfg.path("responses_dir")).glob("*.csv")):
        df = pd.read_csv(path)
        df["model"] = path.stem
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    resp = pd.concat(frames, ignore_index=True)
    resp["response_fk_grade"] = pd.to_numeric(resp["response_fk_grade"], errors="coerce")
    rows = []
    for (model, area, qtype), sub in resp.groupby(["model", "area", "question_type"]):
        mean_fk = float(sub["response_fk_grade"].mean())
        dist = 0.0 if lo <= mean_fk <= hi else min(abs(mean_fk - lo), abs(mean_fk - hi))
        rows.append({"model": model, "area": area, "question_type": qtype,
                     "mean_fk_grade": round(mean_fk, 2),
                     "mean_reading_ease": round(float(pd.to_numeric(
                         sub["response_reading_ease"], errors="coerce").mean()), 2),
                     "n": len(sub), "dist_from_band": round(dist, 2)})
    return pd.DataFrame(rows).sort_values(["model", "area", "question_type"]).reset_index(drop=True), resp


# --------------------------------------------------------------------------- #
# Token usage (actual provider-reported counts)
# --------------------------------------------------------------------------- #
_TOKEN_COLS = [("prompt_tokens", "input"), ("completion_tokens", "output"),
               ("reasoning_tokens", "reasoning"), ("total_tokens", "total")]


def _token_stats(responses: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    """Mean/SD (and sum) of actual token counts per model (optionally x area/type).

    Reasoning tokens are only meaningful when the provider reports them, so we also record
    `reasoning_available_frac` = fraction of responses with reasoning_tokens > 0.
    """
    df = responses.copy()
    for raw, _ in _TOKEN_COLS:
        df[raw] = pd.to_numeric(df.get(raw), errors="coerce").fillna(0)
    rows = []
    for keys, sub in df.groupby(["model", *group]):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(["model", *group], keys))
        rec["n"] = int(len(sub))
        for raw, label in _TOKEN_COLS:
            rec[f"{label}_mean"] = round(float(sub[raw].mean()), 1)
            rec[f"{label}_sd"] = round(float(sub[raw].std(ddof=1)), 1) if len(sub) > 1 else 0.0
        rec["total_sum"] = int(sub["total_tokens"].sum())
        rec["reasoning_available_frac"] = round(float((sub["reasoning_tokens"] > 0).mean()), 3)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["model", *group]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Item difficulty / discrimination
# --------------------------------------------------------------------------- #
def _item_stats(ev: pd.DataFrame) -> pd.DataFrame:
    # per-item, per-model accuracy (mean over runs)
    by_item_model = ev.groupby(["question_id", "area", "question_type", "model"])["binary_score"].mean()
    rows = []
    for (qid, area, qtype), sub in by_item_model.groupby(level=[0, 1, 2]):
        frac = float(sub.mean())   # fraction correct across models
        disc = float(sub.std(ddof=0))  # spread across models = discrimination
        rows.append({"question_id": qid, "area": area, "question_type": qtype,
                     "fraction_correct": round(frac, 3),
                     "discrimination": round(disc, 3),
                     "trivial": int(frac >= 0.95), "broken": int(frac <= 0.05)})
    return pd.DataFrame(rows).sort_values(["area", "fraction_correct"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Pairwise significance (McNemar + Holm-Bonferroni)
# --------------------------------------------------------------------------- #
def _significance(ev: pd.DataFrame) -> pd.DataFrame:
    # collapse runs to one 0/1 per (model, item) by majority
    item = (ev.groupby(["model", "question_id"])["binary_score"]
            .mean().ge(0.5).astype(int).reset_index())
    wide = item.pivot(index="question_id", columns="model", values="binary_score").dropna()
    models = list(wide.columns)
    if len(models) < 2 or wide.empty:
        return pd.DataFrame()
    rows = []
    for m1, m2 in itertools.combinations(models, 2):
        b = int(((wide[m1] == 1) & (wide[m2] == 0)).sum())   # m1 right, m2 wrong
        c = int(((wide[m1] == 0) & (wide[m2] == 1)).sum())   # m1 wrong, m2 right
        table = [[0, b], [c, 0]]
        try:
            p = float(mcnemar(table, exact=(b + c) < 25).pvalue)
        except Exception:  # noqa: BLE001
            p = float("nan")
        acc1, acc2 = float(wide[m1].mean()), float(wide[m2].mean())
        rows.append({"model_a": m1, "model_b": m2, "acc_a": round(acc1, 4),
                     "acc_b": round(acc2, 4), "a_better_b": int(acc1 > acc2),
                     "discordant_b": b, "discordant_c": c, "p_value": p})
    res = pd.DataFrame(rows)
    valid = res["p_value"].notna()
    res["p_holm"] = np.nan
    if valid.any():
        res.loc[valid, "p_holm"] = multipletests(res.loc[valid, "p_value"], method="holm")[1]
    res["significant_0.05"] = (res["p_holm"] < 0.05).astype("Int64")
    return res.sort_values("p_holm").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Judge reliability (Fleiss kappa) + judge<->human pass-through
# --------------------------------------------------------------------------- #
def fleiss_kappa(binary_matrix: np.ndarray) -> float:
    """Fleiss' kappa for N items x R raters of binary {0,1} labels."""
    n, r = binary_matrix.shape
    if n == 0 or r < 2:
        return float("nan")
    n1 = binary_matrix.sum(axis=1)         # count of "1" per item
    n0 = r - n1
    p_i = (n1 * (n1 - 1) + n0 * (n0 - 1)) / (r * (r - 1))
    p_bar = float(p_i.mean())
    p1 = float(n1.sum()) / (n * r)
    pe = p1 ** 2 + (1 - p1) ** 2
    if pe == 1.0:
        return 1.0 if p_bar == 1.0 else float("nan")
    return (p_bar - pe) / (1 - pe)


def _judge_reliability(cfg: Config, mode: str) -> pd.DataFrame:
    rows = []
    # Ground-truth panel — computed over EVERY judged row (incl. auto-dropped), so agreement/kappa
    # aren't skewed by auto-drop having removed exactly the disagreements.
    df = load_judged(cfg)
    if df is not None:
        jcols = [c for c in df.columns if c.startswith("judge_") and not c.endswith("_binary")
                 and not c.startswith("rationale")]
        jcols = [c for c in jcols if pd.api.types.is_numeric_dtype(df[c])]
        if len(jcols) >= 2:
            mat = df[jcols].dropna().to_numpy()
            # Recompute the flag (disagreement) rate from the verdicts rather than a stored column,
            # since auto-dropped rows don't carry human_review_flag.
            flag_rate = float((df[jcols].nunique(axis=1) > 1).mean()) if len(df) else 0.0
            rows.append({"panel": "ground_truth", "n_items": len(mat), "n_judges": len(jcols),
                         "fleiss_kappa": round(fleiss_kappa(mat), 3),
                         "raw_agreement": round(_raw_agreement(mat), 3),
                         "flag_rate": round(flag_rate, 3)})
    # Eval triple panels (per model)
    if mode == "triple":
        for path in sorted(Path(cfg.path("eval_dir")).glob("*_triple.csv")):
            model = path.stem.rsplit("_", 1)[0]
            df = pd.read_csv(path)
            jcols = [c for c in df.columns if c.startswith("judge_") and c.endswith("_binary")]
            if len(jcols) >= 2:
                mat = df[jcols].dropna().to_numpy()
                rows.append({"panel": f"eval:{model}", "n_items": len(mat), "n_judges": len(jcols),
                             "fleiss_kappa": round(fleiss_kappa(mat), 3),
                             "raw_agreement": round(_raw_agreement(mat), 3),
                             "flag_rate": round(float(df["human_review_flag"].mean()), 3)})
    return pd.DataFrame(rows)


def _raw_agreement(mat: np.ndarray) -> float:
    if len(mat) == 0:
        return float("nan")
    return float((mat.sum(axis=1) % mat.shape[1] == 0).mean())  # all-equal rows


# --------------------------------------------------------------------------- #
# Self-preference bias (triple mode, best-effort)
# --------------------------------------------------------------------------- #
def _self_preference(cfg: Config) -> pd.DataFrame:
    families = {k: cfg.get("models", k, "family", default="unknown") for k in cfg.get("models", default={})}
    rows = []
    for path in sorted(Path(cfg.path("eval_dir")).glob("*_triple.csv")):
        model = path.stem.rsplit("_", 1)[0]
        model_family = families.get(model, "unknown")
        df = pd.read_csv(path)
        for col in df.columns:
            if col.startswith("judge_") and col.endswith("_binary"):
                judge = col[len("judge_"):-len("_binary")]
                jfam = families.get(judge, "unknown")
                rows.append({"judge": judge, "judge_family": jfam,
                             "response_model": model, "response_family": model_family,
                             "same_family": int(jfam == model_family),
                             "mean_binary": float(pd.to_numeric(df[col], errors="coerce").mean())})
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    out = []
    for judge, sub in raw.groupby("judge"):
        same = sub[sub["same_family"] == 1]["mean_binary"].mean()
        other = sub[sub["same_family"] == 0]["mean_binary"].mean()
        out.append({"judge": judge,
                    "mean_score_same_family": round(float(same), 4) if pd.notna(same) else "",
                    "mean_score_other_family": round(float(other), 4) if pd.notna(other) else "",
                    "self_preference_delta": round(float(same - other), 4)
                        if pd.notna(same) and pd.notna(other) else ""})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def analyze(cfg: Config | None = None, *, mode: str | None = None) -> dict:
    cfg = cfg or Config.load(require_keys=False)
    mode = mode or cfg.get("eval_mode")
    reports = cfg.path("reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    ev = _load_eval(cfg, mode)
    ev = _apply_human_overrides(cfg, ev)

    written: dict[str, str] = {}

    def _write(name: str, df: pd.DataFrame) -> None:
        if df is None or len(df) == 0:
            return
        path = reports / name
        df.to_csv(path, index=False)
        written[name] = str(path)

    _write("accuracy_by_area.csv", _accuracy_table(ev, ["area"]))
    _write("accuracy_by_type.csv", _accuracy_table(ev, ["question_type"]))
    _write("accuracy_by_area_type.csv", _accuracy_table(ev, ["area", "question_type"]))
    _write("accuracy_by_standard.csv", _accuracy_table(ev, ["standard"]))

    rl = _reading_level(cfg, mode)
    reading_tbl, responses_df = (rl if isinstance(rl, tuple) else (pd.DataFrame(), pd.DataFrame()))
    _write("reading_level.csv", reading_tbl)
    _write("leaderboard.csv", _leaderboard(ev, responses_df))
    if not responses_df.empty:
        _write("token_usage_by_model.csv", _token_stats(responses_df, []))
        _write("token_usage_by_area.csv", _token_stats(responses_df, ["area"]))
        _write("token_usage_by_type.csv", _token_stats(responses_df, ["question_type"]))
    _write("item_stats.csv", _item_stats(ev))
    _write("significance.csv", _significance(ev))
    _write("judge_reliability.csv", _judge_reliability(cfg, mode))
    if mode == "triple":
        _write("self_preference.csv", _self_preference(cfg))

    # Fold in judge<->human calibration if it has been computed.
    jvh = cfg.path("calibration_dir") / "judge_vs_human.csv"
    if jvh.exists():
        written["judge_vs_human.csv"] = str(jvh)

    log.info("Stage 5 complete. Reports written: %s", sorted(written))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 5: analyze results.")
    parser.add_argument("--mode", choices=["single", "triple"], help="Override eval_mode.")
    args = parser.parse_args(argv)
    analyze(mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

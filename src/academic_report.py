"""Assemble a single, curated, academic Markdown report from the benchmark results.

Synthesizes the most important findings into one shareable narrative document
(results/reports/academic_report.md) with a linked table of contents, a numbered list of tables,
and appendices — rather than the ~20 scattered report CSVs. Recomputes everything fresh from the
eval + response files (reusing the analyze / teacher_report compute functions), so it is
self-consistent and independent of whether `analyze` was last run.

Purely additive: writes ONLY academic_report.md; no existing report, config, or data file changes.

CLI:  python -m src.academic_report [--mode single|triple]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.analyze import (_accuracy_table, _apply_human_overrides, _judge_reliability,
                         _leaderboard, _load_eval, _reading_level, _significance, _token_stats)
from src.config import Config
from src.standards import load_areas
from src.teacher_report import _best_pick, _first_run_and_ensemble, _rank_within
from src.utils import get_logger

log = get_logger("academic_report")

QTYPES = ["recall", "conceptual", "applied"]


# --------------------------------------------------------------------------- #
# Markdown helpers
# --------------------------------------------------------------------------- #
def _md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored Markdown table (links in cells pass through)."""
    if df is None or len(df) == 0:
        return "_(no data)_"
    cols = [str(c) for c in df.columns]

    def cell(v: object) -> str:
        s = "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
        return s.replace("|", "\\|").replace("\n", " ").strip()

    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


class _Doc:
    """Accumulates body Markdown while tracking sections (for the TOC) and numbered tables."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._sections: list[tuple[str, str]] = []
        self._tables: list[tuple[int, str, str]] = []
        self._tnum = 0

    def section(self, title: str, anchor: str) -> None:
        self._sections.append((title, anchor))
        self._parts.append(f'<a id="{anchor}"></a>\n\n## {title}\n')

    def sub(self, title: str, anchor: str | None = None) -> None:
        if anchor:
            self._parts.append(f'<a id="{anchor}"></a>')
        self._parts.append(f"### {title}\n")

    def text(self, s: str) -> None:
        self._parts.append(s + "\n")

    def table(self, df: pd.DataFrame, caption: str, *, list_it: bool = True,
              anchor: str | None = None) -> str:
        if list_it:
            self._tnum += 1
            a = anchor or f"table-{self._tnum}"
            self._tables.append((self._tnum, caption, a))
            label = f"**Table {self._tnum}. {caption}**"
        else:
            a = anchor or caption.lower().replace(" ", "-")
            label = f"**{caption}**"
        self._parts.append(f'<a id="{a}"></a>\n\n{label}\n')
        self._parts.append(_md(df) + "\n")
        return a

    def render(self, title: str, abstract: str) -> str:
        toc = ["## Contents", ""] + [f"- [{t}](#{a})" for t, a in self._sections] + [""]
        lot = ["## List of Tables", ""] + \
              [f"- [Table {n}. {cap}](#{a})" for n, cap, a in self._tables] + [""]
        head = [f"# {title}", "",
                f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}._", "",
                abstract, ""]
        return "\n".join(head + toc + lot + ["---", ""] + self._parts) + "\n"


# --------------------------------------------------------------------------- #
# Ensemble item statistics (recomputed — analyze._item_stats is a per-run mean)
# --------------------------------------------------------------------------- #
def _ensemble_item_stats(ev: pd.DataFrame) -> pd.DataFrame:
    """Per item: majority-vote (ensemble) collapse across runs first, then difficulty/discrimination."""
    per = (ev.groupby(["question_id", "area", "question_type", "model"])["binary_score"]
           .mean().ge(0.5).astype(int))
    rows = []
    for (qid, area, qtype), sub in per.groupby(level=[0, 1, 2]):
        frac = float(sub.mean())
        rows.append({"question_id": qid, "area": area, "question_type": qtype,
                     "fraction_correct": frac, "discrimination": float(sub.std(ddof=0)),
                     "trivial": int(frac >= 0.95), "broken": int(frac <= 0.05)})
    return pd.DataFrame(rows)


def _pivot_single_ens(fe: pd.DataFrame, col: str, order: list[str],
                      model_order: list[str]) -> pd.DataFrame:
    """fe (model, <col>, first_run_accuracy, ensemble_accuracy) -> model rows, per-value single/ens."""
    rows = []
    for model, sub in fe.groupby("model"):
        rec = {"model": model}
        m = sub.set_index(col)
        for v in order:
            rec[f"{v} (single)"] = m.loc[v, "first_run_accuracy"] if v in m.index else ""
            rec[f"{v} (ens)"] = m.loc[v, "ensemble_accuracy"] if v in m.index else ""
        rows.append(rec)
    df = pd.DataFrame(rows).set_index("model")
    df = df.reindex([m for m in model_order if m in df.index]).reset_index()
    return df


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #
def academic_report(cfg: Config | None = None, *, mode: str | None = None) -> Path:
    cfg = cfg or Config.load(require_keys=False)
    mode = mode or cfg.get("eval_mode")
    ev = _apply_human_overrides(cfg, _load_eval(cfg, mode))
    ev["run"] = pd.to_numeric(ev["run"], errors="coerce")

    areas = load_areas(cfg)
    area_title = {a.code: a.title for a in areas}
    std_text = {s.code: s.text for a in areas for s in a.standards}
    std_area = {s.code: a.code for a in areas for s in a.standards}
    area_codes = [a.code for a in areas]
    std_codes = [s.code for a in areas for s in a.standards]

    rl = _reading_level(cfg, mode)
    reading_tbl, responses_df = rl if isinstance(rl, tuple) else (pd.DataFrame(), pd.DataFrame())

    n_models = int(ev["model"].nunique())
    n_questions = int(ev["question_id"].nunique())
    runs = sorted(int(r) for r in ev["run"].dropna().unique())

    # Overall single/ensemble → canonical model order (best ensemble first).
    overall_fe = _rank_within(_first_run_and_ensemble(ev, []), [])
    overall_fe["ensemble_gain"] = (overall_fe["ensemble_accuracy"]
                                   - overall_fe["first_run_accuracy"]).round(3)
    model_order = overall_fe.sort_values("rank")["model"].tolist()

    doc = _Doc()

    # ---- Methods -----------------------------------------------------------
    doc.section("1. Methods", "methods")
    generator = cfg.get("generator_model", default="?")
    gt_judges = cfg.get("gt_judges", default=[]) or []
    eval_judge = cfg.get("eval_judge", default="?")
    split = cfg.get("question_type_split", default={}) or {}
    fk_band = (cfg.get("appropriateness", default={}) or {}).get("fk_grade_range", [5.0, 9.0])
    dedupe_thr = (cfg.get("dedupe", default={}) or {}).get("similarity_threshold", 88)

    def _rows(path):
        df = _safe_csv(path)
        return 0 if df is None else len(df)

    dropped_n = _rows(cfg.path("human_review_dir") / "auto_dropped_groundtruth.csv")
    rejects_n = _rows(cfg.path("reports_dir") / "generation_rejects.csv")
    generated_n = n_questions + dropped_n

    doc.text(
        f"**Dataset.** A ground-truth question–answer set was generated by **{generator}** across "
        f"the **12 NGSS middle-school science areas** ({len(std_codes)} performance-expectation "
        f"standards), targeting a balanced mix of question types "
        f"({', '.join(f'{k} {v}' for k, v in split.items()) or 'even split'}). Every candidate "
        f"passed three automatic gates before acceptance — near-duplicate removal (rapidfuzz "
        f"token-set ratio ≥ {dedupe_thr}), a middle-school readability band "
        f"(Flesch–Kincaid {fk_band[0]}–{fk_band[1]}), and an LLM appropriateness check — with "
        f"{rejects_n:,} candidates rejected and regenerated along the way.")
    doc.text(
        f"**Validation.** Each pair was scored independently by a 3-judge panel "
        f"({', '.join(gt_judges)}). Questions the panel doubted (disagreement or unanimous "
        f"rejection) were auto-dropped: **{generated_n:,} generated → {dropped_n} dropped → "
        f"{n_questions:,} validated** questions used for benchmarking.")
    doc.text(
        f"**Benchmark & evaluation.** **{n_models} candidate models** each answered every question "
        f"**{len(runs)} times** ({', '.join('run ' + str(r) for r in runs)}) under a standardized "
        f"zero-shot protocol (shared middle-school system prompt, no per-model tuning). Answers were "
        f"graded by a single judge (**{eval_judge}**, `eval_mode: {mode}`): binary correct/incorrect, "
        f"plus a 0/0.5/1 rubric for conceptual and applied items. Empty, errored, or refusing "
        f"responses are scored 0 and reported separately as the refusal rate.")
    doc.text(
        "**Scoring conventions used throughout this report.** For every model we report two scores: "
        "**single** — accuracy on **run #1 only** (one-shot behavior); and **ensemble** — accuracy "
        "after taking the **majority vote across the model's runs** (self-consistency). Accuracies "
        "are 0–1 proportions (0.92 = 92%). **Macro** accuracy weights each area equally; **micro** "
        "weights each question equally.")

    # ---- Overall leaderboard ----------------------------------------------
    doc.section("2. Overall performance", "overall")
    lb = _leaderboard(ev, responses_df)
    keep = ["model", "macro_accuracy", "micro_accuracy", "refusal_rate",
            "mean_response_fk_grade", "mean_total_tokens"]
    lb_small = lb[[c for c in keep if c in lb.columns]]
    overall_tbl = (overall_fe[["model", "first_run_accuracy", "ensemble_accuracy",
                               "ensemble_gain", "n_questions"]]
                   .merge(lb_small, on="model", how="left"))
    overall_tbl = overall_tbl.rename(columns={"first_run_accuracy": "single",
                                              "ensemble_accuracy": "ensemble"})
    overall_tbl = overall_tbl.set_index("model").reindex(model_order).reset_index()
    overall_tbl.insert(0, "rank", range(1, len(overall_tbl) + 1))
    doc.text("Models are ranked by ensemble accuracy. `single` and `ensemble` are defined in "
             "[Methods](#methods); `ensemble_gain` is the improvement from majority-voting the runs.")
    doc.table(overall_tbl, "Overall leaderboard — single vs. ensemble accuracy, with macro/micro "
                           "accuracy, refusal rate, reading level, and mean tokens, per model")

    # ---- Significance ------------------------------------------------------
    doc.section("3. Statistical significance of model differences", "significance")
    sig = _significance(ev)
    beats: set[tuple[str, str]] = set()
    if len(sig):
        for _, r in sig.iterrows():
            if int(pd.to_numeric(r.get("significant_0.05"), errors="coerce") or 0) == 1:
                w, l = ((r["model_a"], r["model_b"]) if int(r["a_better_b"]) == 1
                        else (r["model_b"], r["model_a"]))
                beats.add((w, l))
    if beats:
        mat = pd.DataFrame(index=model_order, columns=model_order, dtype=object)
        for ri in model_order:
            for ci in model_order:
                mat.loc[ri, ci] = "—" if ri == ci else ("✓" if (ri, ci) in beats else "·")
        mat_out = mat.reset_index().rename(columns={"index": "beats ↓ / vs →"})
        doc.text("Pairwise McNemar tests on run-collapsed verdicts, Holm–Bonferroni corrected across "
                 "all pairs. A **✓** means the row model **significantly** outperforms the column "
                 "model (p_holm < 0.05); **·** = not significant.")
        doc.table(mat_out, "Significant pairwise wins (row beats column, Holm-corrected McNemar)")
        wins = {m: sum(1 for (w, _l) in beats if w == m) for m in model_order}
        doc.text("**Significant wins per model:** "
                 + "; ".join(f"{m} beats {wins.get(m, 0)}" for m in model_order) + ".")
    else:
        doc.text("_No pairwise differences survived Holm correction (or insufficient data)._")

    # ---- By question type (summary) ---------------------------------------
    doc.section("4. Performance by question type", "qtype")
    qt_fe = _first_run_and_ensemble(ev, ["question_type"])
    qt_tbl = _pivot_single_ens(qt_fe, "question_type", QTYPES, model_order)
    doc.text("How each model handles the three cognitive levels — **recall**, **conceptual**, "
             "**applied** — as single/ensemble accuracy. Full model × area breakdowns for each "
             "type are in [Appendix C](#appendix-c).")
    doc.table(qt_tbl, "Accuracy by question type (single & ensemble), per model")
    best_qt = _best_pick(_rank_within(qt_fe.copy(), ["question_type"]), "question_type", [])
    doc.table(best_qt, "Best model(s) per question type (by ensemble)")

    # ---- By area -----------------------------------------------------------
    doc.section("5. Performance by area", "area")
    area_fe = _first_run_and_ensemble(ev, ["area"])
    area_mat = area_fe.pivot(index="model", columns="area", values="ensemble_accuracy")
    area_mat = area_mat.reindex(index=[m for m in model_order if m in area_mat.index],
                                columns=[a for a in area_codes if a in area_mat.columns])
    doc.text("Ensemble accuracy for every model across the 12 NGSS areas (see area titles below).")
    doc.table(area_mat.reset_index(), "Ensemble accuracy by area, per model")
    ranked_area = _rank_within(area_fe.copy(), ["area"])
    ranked_area["area_title"] = ranked_area["area"].map(area_title)
    ranked_area["_o"] = ranked_area["area"].map({c: i for i, c in enumerate(area_codes)})
    ranked_area = ranked_area.sort_values(["_o", "rank"]).drop(columns="_o")
    doc.table(_best_pick(ranked_area, "area", ["area_title"]),
              "Best model(s) per area (by ensemble), with runner-up and gap")

    # ---- Item psychometrics by area × type (ensemble) ---------------------
    doc.section("6. Item difficulty & discrimination (ensemble), by area × question type", "psychometrics")
    istats = _ensemble_item_stats(ev)
    agg = (istats.groupby(["area", "question_type"])
           .agg(n_items=("fraction_correct", "size"),
                mean_difficulty=("fraction_correct", "mean"),
                mean_discrimination=("discrimination", "mean"),
                pct_trivial=("trivial", "mean"), pct_broken=("broken", "mean"))
           .reset_index())
    for c in ["mean_difficulty", "mean_discrimination", "pct_trivial", "pct_broken"]:
        agg[c] = agg[c].round(3)
    agg["_o"] = agg["area"].map({c: i for i, c in enumerate(area_codes)})
    agg = agg.sort_values(["_o", "question_type"]).drop(columns="_o")
    doc.text("Computed on ensemble (majority-vote) item scores. **Difficulty** = mean fraction of "
             "models answering correctly (higher = easier). **Discrimination** = spread across models "
             "(higher = better separates strong/weak models). `trivial` = ≥95% correct, `broken` = "
             "≤5% correct — items carrying little signal.")
    doc.table(agg, "Item difficulty and discrimination by area × question type (ensemble)")

    # ---- Best model per standard (summary) --------------------------------
    doc.section("7. Best model per standard (summary)", "by-standard")
    std_fe = _first_run_and_ensemble(ev, ["standard"])
    ranked_std = _rank_within(std_fe.copy(), ["standard"])
    ranked_std["_o"] = ranked_std["standard"].map({c: i for i, c in enumerate(std_codes)})
    ranked_std = ranked_std.sort_values(["_o", "rank"]).drop(columns="_o")
    best_std = _best_pick(ranked_std, "standard", [])
    best_std["area"] = best_std["standard"].map(std_area)
    best_std["standard"] = best_std["standard"].map(lambda c: f"[{c}](#std-{c})")  # link to appendix A
    best_std = best_std[["standard", "area", "best_models", "n_tied", "ensemble_accuracy",
                         "runner_up_models", "gap"]]
    doc.text("The top model(s) for each of the 59 standards by ensemble accuracy. When several tie "
             "(common — many standards are aced by multiple models) all are listed with `n_tied` and "
             "the `gap` to the next tier. Each standard links to its full detail table in "
             "[Appendix A](#appendix-a).")
    doc.table(best_std, "Best model(s) per standard (by ensemble)")

    # ---- Self-consistency --------------------------------------------------
    doc.section("8. Run-to-run stability", "stability")
    acc_model = _accuracy_table(ev, [])
    stab = overall_fe[["model", "first_run_accuracy", "ensemble_accuracy", "ensemble_gain"]].copy()
    stab = stab.merge(acc_model[["model", "run_acc_std", "refusal_rate"]], on="model", how="left")
    stab = stab.rename(columns={"first_run_accuracy": "single", "ensemble_accuracy": "ensemble"})
    stab = stab.set_index("model").reindex(model_order).reset_index()
    doc.text("`run_acc_std` is the standard deviation of accuracy across the repeated runs (lower = "
             "more self-consistent); `ensemble_gain` is how much majority-voting improves over run 1.")
    doc.table(stab, "Run-to-run stability: single vs. ensemble, ensemble gain, and accuracy SD across runs")

    # ---- Reading level -----------------------------------------------------
    doc.section("9. Response reading level", "reading")
    if not responses_df.empty and "response_fk_grade" in responses_df.columns:
        fk = (responses_df.assign(fk=pd.to_numeric(responses_df["response_fk_grade"], errors="coerce"))
              .groupby("model")["fk"].mean().round(2))
        rd = fk.reindex([m for m in model_order if m in fk.index]).reset_index()
        rd.columns = ["model", "mean_response_fk_grade"]
        rd["target_band"] = f"{fk_band[0]}–{fk_band[1]}"
        doc.text("Mean Flesch–Kincaid grade of each model's responses vs. the target middle-school "
                 "band. Relevant because the intended audience is students.")
        doc.table(rd, "Mean response reading level (Flesch–Kincaid grade), per model")
    else:
        doc.text("_Response files not available; reading level omitted._")

    # ---- Compute cost ------------------------------------------------------
    doc.section("10. Compute cost (tokens)", "cost")
    if not responses_df.empty:
        tok = _token_stats(responses_df, [])
        keep = ["model", "input_mean", "output_mean", "reasoning_mean", "total_mean",
                "total_sum", "reasoning_available_frac"]
        tok = tok[[c for c in keep if c in tok.columns]]
        tok = tok.set_index("model").reindex([m for m in model_order if m in tok.index]).reset_index()
        doc.text("Provider-reported token usage (actual, not estimated). `total_sum` is the whole-run "
                 "total (a cost proxy); `reasoning_available_frac` is the share of responses for which "
                 "the provider reported reasoning tokens.")
        doc.table(tok, "Token usage per model (mean by category, run total, reasoning availability)")
    else:
        doc.text("_Response files not available; token usage omitted._")

    # ---- Judge reliability & validity -------------------------------------
    doc.section("11. Judge reliability & validity", "reliability")
    jr = _judge_reliability(cfg, mode)
    if jr is not None and len(jr):
        doc.table(jr, "Judge panel reliability — Fleiss' κ and raw agreement")
    doc.text("The ground-truth panel's inter-judge agreement is reported above (Fleiss' κ). "
             + ("Because evaluation used a **single** judge, there is no eval-panel reliability to "
                "report. " if mode == "single" else "")
             + ("A **judge↔human calibration** (per-judge accuracy and Cohen's κ vs. human labels) "
                "is **pending**: the calibration sample (`results/calibration/to_label.csv`) has not "
                "yet been human-labeled." if not (cfg.path("calibration_dir") / "judge_vs_human.csv").exists()
                else "Judge↔human calibration is available in `results/calibration/judge_vs_human.csv`."))

    # ---- Limitations -------------------------------------------------------
    doc.section("12. Limitations", "limitations")
    doc.text(
        "- **LLM-generated questions.** The item bank was written by an LLM; such questions may "
        "structurally favor LLM respondents, so absolute accuracies should be read with caution.\n"
        "- **Correctness only.** This measures answer correctness, not pedagogical quality, "
        "hallucination rate, or safety.\n"
        "- **Single evaluation judge.** Grading used one judge (`eval_mode: single`); a triple-judge "
        "panel would allow eval-side reliability estimates.\n"
        "- **Calibration pending.** Judge↔human agreement has not yet been established for this run.\n"
        "- **Ceiling effects / ties.** Many standards are answered correctly by several models, so "
        "'best model' is often a tie — treat the per-standard picks as 'any of these,' and decide on "
        "secondary factors (reading level, token cost, availability).")

    # ---- Appendix A: per-standard detail ----------------------------------
    doc.section("Appendix A — Per-standard detail (all models)", "appendix-a")
    doc.text("Full single & ensemble accuracy for every model, by question type, for each of the 59 "
             "standards. Jump to a standard: "
             + " · ".join(f"[{c}](#std-{c})" for c in std_codes) + ".")
    std_qt_fe = _first_run_and_ensemble(ev, ["standard", "question_type"])
    present_std = set(std_qt_fe["standard"].unique())
    for code in std_codes:
        if code not in present_std:
            continue
        sub = std_qt_fe[std_qt_fe["standard"] == code].drop(columns=["standard"])
        tbl = _pivot_single_ens(sub, "question_type", QTYPES, model_order)
        title = f"{code} — {std_text.get(code, '')}"
        doc.sub(title, anchor=f"std-{code}")
        doc.text(f"_Area {std_area.get(code, '')} ({area_title.get(std_area.get(code, ''), '')})._")
        doc.table(tbl, f"{code} — accuracy by model and question type (single & ensemble)",
                  list_it=False, anchor=f"tbl-std-{code}")

    # ---- Appendix C: question-type × area detail --------------------------
    doc.section("Appendix C — Question-type detail (model × area)", "appendix-c")
    doc.text("For each cognitive level, how every model performs across all 12 areas "
             "(single / ensemble accuracy).")
    at_fe = _first_run_and_ensemble(ev, ["question_type", "area"])
    for qt in QTYPES:
        sub = at_fe[at_fe["question_type"] == qt].drop(columns=["question_type"])
        if sub.empty:
            continue
        tbl = _pivot_single_ens(sub, "area", area_codes, model_order)
        doc.sub(f"{qt.capitalize()} questions", anchor=f"qtype-{qt}")
        doc.table(tbl, f"{qt.capitalize()} — accuracy by model and area (single & ensemble)")

    # ---- Appendix B: source data ------------------------------------------
    doc.section("Appendix B — Source data", "appendix-b")
    doc.text("All figures derive from the per-model evaluation files (`results/eval/*_" + str(mode)
             + ".csv`) and the standardized report CSVs in `results/reports/` "
             "(`leaderboard.csv`, `accuracy_by_*.csv`, `significance.csv`, `item_stats.csv`, "
             "`judge_reliability.csv`, `token_usage_by_*.csv`, `teacher_*.csv`). Human corrections, "
             "where present, are folded in from `results/human_review/`.")

    title = "NGSS Middle-School Science LLM Benchmark — Findings"
    abstract = (
        f"We benchmark **{n_models} large language models** on **{n_questions:,} validated "
        f"question–answer items** spanning the **12 NGSS middle-school science areas** "
        f"({len(std_codes)} performance-expectation standards), each answered **{len(runs)} times**. "
        "For every model we report **single-run** (one-shot) and **ensemble** (majority-vote across "
        "runs) accuracy, overall and broken down by area, question type, and individual standard, "
        "with pairwise significance testing, item psychometrics, judge reliability, response reading "
        "level, and compute cost. This document curates the most decision-relevant findings; raw "
        "tables are in [Appendix B](#appendix-b). Note the key caveat that questions are "
        "LLM-generated (see [Limitations](#limitations)).")

    out = cfg.path("reports_dir") / "academic_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc.render(title, abstract))
    log.info("Wrote academic report %s (%d tables).", out, doc._tnum)
    return out


def _safe_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path) if Path(path).exists() else None
    except Exception:  # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble the academic Markdown report.")
    parser.add_argument("--mode", choices=["single", "triple"], help="Override eval_mode.")
    args = parser.parse_args(argv)
    out = academic_report(mode=args.mode)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

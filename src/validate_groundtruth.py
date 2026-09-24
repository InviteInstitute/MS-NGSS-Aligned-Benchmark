"""Stage 2 — validate the ground-truth Q&A with an independent 3-judge LLM panel.

Each judge in `gt_judges` scores every pair 1 (reference answer correct) / 0, with a short
rationale, at temperature 0 and blind to the other judges. We then compute a majority label
and flag non-unanimous rows for human review.

If `validation.auto_drop_disagreements` is true, "doubted" questions (judges disagree, or all
judges mark the reference answer wrong) are DROPPED from the validated set instead of being
queued for human review — they're recorded in auto_dropped_groundtruth.csv, and ground_truth.csv
is left intact.

Outputs:
  data/ground_truth_validated.csv      ground-truth + judge_<key> cols + majority_label + human_review_flag
  results/human_review/flags_groundtruth.csv         non-unanimous rows, ready for a human to resolve
  results/human_review/auto_dropped_groundtruth.csv  (auto-drop mode) the doubted rows that were removed

Resumable via a long-format judging checkpoint. CLI:  python -m src.validate_groundtruth [--fresh]
"""
from __future__ import annotations

import argparse

import pandas as pd

from src import prompts
from src.concurrency import parallel_map
from src.config import Config
from src.llm_client import LLMClient, parse_json
from src.standards import load_areas
from src.utils import CheckpointWriter, get_logger, read_csv, to_int, to_temp

log = get_logger("validate")

LONG_COLUMNS = ["question_id", "judge", "score", "rationale"]


def _score_one(client: LLMClient, judge: str, row: pd.Series, std_text: dict[str, str],
               judge_temp: float | None, judge_max: int | None) -> dict:
    res = client.complete(
        judge,
        system=prompts.SYSTEM_PROMPTS["default"],
        user=prompts.gt_judge_prompt(
            str(row["question"]), str(row["reference_answer"]),
            str(row["standard"]), std_text.get(str(row["standard"]), ""),
        ),
        temperature=judge_temp, max_tokens=judge_max,
    )
    obj = parse_json(res.text) or {}
    score = obj.get("score")
    score = int(score) if score in (0, 1, "0", "1") else 0
    return {"question_id": row["question_id"], "judge": judge,
            "score": score, "rationale": str(obj.get("rationale", ""))[:300]}


def validate(cfg: Config | None = None, *, fresh: bool = False) -> pd.DataFrame:
    cfg = cfg or Config.load()
    client = LLMClient(cfg)
    judges = list(cfg.get("gt_judges"))
    workers = int(cfg.get("concurrency", "max_parallel_requests", default=8))
    flush_every = int(cfg.get("checkpoint", "flush_every", default=25))
    judge_temp = to_temp(cfg.get("defaults", "temperature_judge"))
    judge_max = to_int(cfg.get("defaults", "max_tokens_judge"))

    gt = read_csv(cfg.path("ground_truth"))
    std_text = {s.code: s.text for a in load_areas(cfg) for s in a.standards}
    long_path = cfg.path("ground_truth").parent / "_gt_judge_long.csv"
    writer = CheckpointWriter(long_path, key_cols=["question_id", "judge"],
                              columns=LONG_COLUMNS, flush_every=flush_every, fresh=fresh)

    tasks = [{"question_id": r["question_id"], "judge": j, "_row": r}
             for _, r in gt.iterrows() for j in judges]
    pending = writer.pending(tasks)
    log.info("Scoring %d / %d (question x judge) tasks (%d already done).",
             len(pending), len(tasks), len(tasks) - len(pending))

    def _task(t: dict) -> dict:
        return _score_one(client, t["judge"], t["_row"], std_text, judge_temp, judge_max)

    parallel_map(_task, pending, max_workers=workers, desc="gt-judges",
                 on_result=lambda i, t, r: writer.add(r))
    long_df = writer.close()

    # Pivot to wide: one column per judge.
    wide = long_df.pivot_table(index="question_id", columns="judge", values="score",
                               aggfunc="first").reset_index()
    judge_cols = [j for j in judges if j in wide.columns]
    rename = {j: f"judge_{j}" for j in judge_cols}
    wide = wide.rename(columns=rename)
    jcols = [f"judge_{j}" for j in judge_cols]

    rationale_wide = long_df.pivot_table(index="question_id", columns="judge",
                                         values="rationale", aggfunc="first").reset_index()
    rationale_wide = rationale_wide.rename(columns={j: f"rationale_{j}" for j in judge_cols})

    out = gt.merge(wide, on="question_id", how="left").merge(rationale_wide, on="question_id", how="left")
    out["majority_label"] = (out[jcols].sum(axis=1) >= (len(jcols) / 2.0)).astype(int)
    out["human_review_flag"] = (out[jcols].nunique(axis=1) > 1).astype(int)

    # "Doubted" = judges disagree OR unanimously reject the reference answer (majority_label == 0).
    # These are exactly the items the human-review queue targets.
    doubted = (out["human_review_flag"] == 1) | (out["majority_label"] == 0)

    if bool(cfg.get("validation", "auto_drop_disagreements", default=False)):
        validated = _auto_drop(cfg, out, doubted, jcols)
    else:
        validated = out
        _write_flags(cfg, out, jcols)

    validated.to_csv(cfg.path("ground_truth_validated"), index=False)
    log.info("Wrote %s (%d questions; %d flagged for human review).",
             cfg.path("ground_truth_validated"), len(validated),
             int(validated["human_review_flag"].sum()) if len(validated) else 0)
    return validated


def _auto_drop(cfg: Config, out: pd.DataFrame, doubted: pd.Series, jcols: list[str]) -> pd.DataFrame:
    """Auto-drop mode: exclude every doubted question from the validated (benchmark) set and
    record what was dropped. `ground_truth.csv` is left untouched; nothing is deleted."""
    dropped = out[doubted].copy()
    kept = out[~doubted].copy()

    # Always (re)write the record so it reflects THIS run — never a stale count from a prior one.
    keep_cols = ["question_id", "area", "standard", "question_type", "question",
                 "reference_answer", "majority_label", *jcols,
                 *[c for c in out.columns if c.startswith("rationale_")]]
    rec = dropped[[c for c in keep_cols if c in dropped.columns]].copy()
    rec["drop_reason"] = [
        "judge_disagreement" if f == 1 else "unanimous_incorrect"
        for f in dropped["human_review_flag"]
    ]
    path = cfg.path("human_review_dir") / "auto_dropped_groundtruth.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rec.to_csv(path, index=False)
    log.info("Auto-drop: removed %d doubted question(s) from the validated set; recorded in %s.",
             len(dropped), path)

    # Guard: warn loudly if auto-drop wiped an entire area (don't silently gut coverage).
    lost_areas = sorted(set(out["area"]) - set(kept["area"]))
    if lost_areas:
        log.warning("Auto-drop removed EVERY question from area(s): %s. Consider regenerating or "
                    "disabling auto_drop_disagreements.", lost_areas)
    return kept


def _write_flags(cfg: Config, out: pd.DataFrame, jcols: list[str]) -> None:
    flagged = out[out["human_review_flag"] == 1].copy()
    keep = ["question_id", "area", "standard", "question_type", "question",
            "reference_answer", *jcols]
    flagged = flagged[[c for c in keep if c in flagged.columns]]
    flagged["human_score"] = ""   # human fills: 1 if reference answer correct, else 0
    flagged["human_notes"] = ""
    path = cfg.path("human_review_dir") / "flags_groundtruth.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(path, index=False)
    log.info("Wrote human-review queue %s (%d rows).", path, len(flagged))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2: validate ground truth with judges.")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args(argv)
    validate(fresh=args.fresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

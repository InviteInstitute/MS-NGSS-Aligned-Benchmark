"""Stage 4 — score benchmark responses against the gold reference answers.

Config toggle `eval_mode`:
  - single : one judge (`eval_judge`) returns a binary 0/1 (and a rubric score for
             conceptual/applied items).
  - triple : each judge in `eval_judges` scores independently; we add per-judge binary
             columns, a majority binary_score, a mean rubric_score, and a human_review_flag
             when judges disagree on the binary verdict.

Refusal/non-answer/malformed policy (pre-registered): an empty, errored, or explicitly
refusing response is scored binary 0 / rubric 0 WITHOUT calling a judge, and tagged
is_refusal=1 so refusal rate is reported separately from wrong answers.

One CSV per model: results/eval/<model_key>_<mode>.csv. Resumable per (question_id, run, judge).

CLI:  python -m src.evaluate [--fresh] [--models ...] [--mode single|triple]
"""
from __future__ import annotations

import argparse
import re

import pandas as pd

from src import prompts
from src.concurrency import parallel_map
from src.config import Config
from src.llm_client import LLMClient, parse_json
from src.utils import CheckpointWriter, get_logger, read_csv, to_int, to_temp

log = get_logger("evaluate")

LONG_COLUMNS = ["question_id", "run", "judge", "binary", "rubric", "is_refusal", "rationale"]
_REFUSAL = re.compile(r"\b(i\s*(can('|no)?t|cannot|am unable|won'?t)|as an ai)\b", re.IGNORECASE)


def _is_refusal(response: str, is_error: int) -> bool:
    if int(is_error) == 1:
        return True
    text = (response or "").strip()
    if not text:
        return True
    return bool(_REFUSAL.match(text))


def _judge_response(client: LLMClient, judge: str, q: pd.Series, response: str,
                    use_rubric: bool, judge_temp: float | None, judge_max: int | None) -> dict:
    res = client.complete(
        judge, system=prompts.SYSTEM_PROMPTS["default"],
        user=prompts.eval_judge_prompt(str(q["question"]), str(q["reference_answer"]),
                                       response, str(q["question_type"]), use_rubric),
        temperature=judge_temp, max_tokens=judge_max,
    )
    obj = parse_json(res.text) or {}
    binary = obj.get("binary")
    binary = int(binary) if binary in (0, 1, "0", "1") else 0
    rubric = obj.get("rubric") if use_rubric else None
    rubric = float(rubric) if rubric in (0, 0.5, 1, "0", "0.5", "1") else (0.0 if use_rubric else None)
    return {"binary": binary, "rubric": rubric, "rationale": str(obj.get("rationale", ""))[:300]}


def evaluate_model(cfg: Config, client: LLMClient, model_key: str, gt: pd.DataFrame,
                   judges: list[str], mode: str, rubric_types: set[str],
                   *, fresh: bool = False) -> pd.DataFrame:
    resp_path = cfg.path("responses_dir") / f"{model_key}.csv"
    responses = read_csv(resp_path)
    gt_idx = gt.set_index("question_id")
    judge_temp = to_temp(cfg.get("defaults", "temperature_judge"))
    judge_max = to_int(cfg.get("defaults", "max_tokens_judge"))

    long_path = cfg.path("eval_dir") / f"_{model_key}_{mode}_long.csv"
    flush_every = int(cfg.get("checkpoint", "flush_every", default=25))
    workers = int(cfg.get("concurrency", "max_parallel_requests", default=8))
    writer = CheckpointWriter(long_path, key_cols=["question_id", "run", "judge"],
                              columns=LONG_COLUMNS, flush_every=flush_every, fresh=fresh)

    tasks = [{"question_id": r["question_id"], "run": r["run"], "judge": j, "_row": r}
             for _, r in responses.iterrows() for j in judges]
    pending = writer.pending(tasks)
    log.info("[%s/%s] %d / %d judging tasks pending.", model_key, mode, len(pending), len(tasks))

    def _task(t: dict) -> dict:
        row = t["_row"]
        qid = row["question_id"]
        if qid not in gt_idx.index:
            return {"question_id": qid, "run": t["run"], "judge": t["judge"],
                    "binary": 0, "rubric": "", "is_refusal": 1, "rationale": "no gold row"}
        q = gt_idx.loc[qid]
        use_rubric = str(q["question_type"]) in rubric_types
        if _is_refusal(str(row.get("response", "")), int(row.get("is_error", 0))):
            return {"question_id": qid, "run": t["run"], "judge": t["judge"],
                    "binary": 0, "rubric": (0.0 if use_rubric else ""), "is_refusal": 1,
                    "rationale": "refusal/empty/error"}
        scored = _judge_response(client, t["judge"], q, str(row["response"]), use_rubric,
                                 judge_temp, judge_max)
        return {"question_id": qid, "run": t["run"], "judge": t["judge"],
                "binary": scored["binary"],
                "rubric": (scored["rubric"] if use_rubric else ""),
                "is_refusal": 0, "rationale": scored["rationale"]}

    parallel_map(_task, pending, max_workers=workers, desc=f"eval {model_key}",
                 on_result=lambda i, t, r: writer.add(r))
    long_df = writer.close()
    return _assemble(cfg, model_key, mode, responses, long_df, judges)


def _assemble(cfg: Config, model_key: str, mode: str, responses: pd.DataFrame,
              long_df: pd.DataFrame, judges: list[str]) -> pd.DataFrame:
    base = responses[["question_id", "run", "area", "standard", "question_type", "response"]].copy()
    long_df = long_df.copy()
    long_df["binary"] = pd.to_numeric(long_df["binary"], errors="coerce")
    long_df["rubric"] = pd.to_numeric(long_df["rubric"], errors="coerce")
    long_df["is_refusal"] = pd.to_numeric(long_df["is_refusal"], errors="coerce").fillna(0).astype(int)

    grp = long_df.groupby(["question_id", "run"])
    agg = grp.agg(
        binary_score=("binary", lambda s: int(s.mean() >= 0.5)),
        rubric_score=("rubric", "mean"),
        is_refusal=("is_refusal", "max"),
        n_judge_verdicts=("binary", "nunique"),
    ).reset_index()
    if mode == "triple":
        agg["human_review_flag"] = (agg["n_judge_verdicts"] > 1).astype(int)
    else:
        agg["human_review_flag"] = 0
    agg = agg.drop(columns=["n_judge_verdicts"])

    out = base.merge(agg, on=["question_id", "run"], how="left")

    if mode == "triple":
        for j in judges:
            jb = long_df[long_df["judge"] == j][["question_id", "run", "binary"]]
            jb = jb.rename(columns={"binary": f"judge_{j}_binary"})
            out = out.merge(jb, on=["question_id", "run"], how="left")
    else:
        rat = long_df[["question_id", "run", "rationale"]]
        out = out.merge(rat, on=["question_id", "run"], how="left")

    out_path = cfg.path("eval_dir") / f"{model_key}_{mode}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    log.info("[%s/%s] wrote %s (%d rows, %d refusals).",
             model_key, mode, out_path, len(out), int(out["is_refusal"].sum()))

    if mode == "triple":
        _write_flags(cfg, model_key, out)
    return out


def _write_flags(cfg: Config, model_key: str, out: pd.DataFrame) -> None:
    flagged = out[out["human_review_flag"] == 1].copy()
    if flagged.empty:
        return
    flagged["human_score"] = ""   # human fills: 1 if response is correct, else 0
    flagged["human_notes"] = ""
    path = cfg.path("human_review_dir") / f"flags_eval_{model_key}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(path, index=False)
    log.info("[%s] wrote human-review queue %s (%d rows).", model_key, path, len(flagged))


def evaluate(cfg: Config | None = None, *, models: list[str] | None = None,
             mode: str | None = None, fresh: bool = False) -> None:
    cfg = cfg or Config.load()
    client = LLMClient(cfg)
    mode = mode or cfg.get("eval_mode")
    judges = [cfg.get("eval_judge")] if mode == "single" else list(cfg.get("eval_judges"))
    rubric_types = set(cfg.get("grading", "rubric_for", default=[]) or [])
    gt = read_csv(cfg.path("ground_truth"))
    model_keys = models or list(cfg.get("benchmark_models"))
    for key in model_keys:
        evaluate_model(cfg, client, key, gt, judges, mode, rubric_types, fresh=fresh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 4: evaluate benchmark responses.")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--models", help="Comma-separated subset of benchmark_models.")
    parser.add_argument("--mode", choices=["single", "triple"], help="Override eval_mode.")
    args = parser.parse_args(argv)
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    evaluate(models=models, mode=args.mode, fresh=args.fresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

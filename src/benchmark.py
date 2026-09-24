"""Stage 3 — benchmark candidate LLMs on the question set.

Every model in `benchmark_models` answers each question `runs_per_model` times using the
SAME standardized protocol (`benchmark_protocol`): identical temperature, max_tokens, the
shared `middle_school` system prompt, and zero-shot — so the comparison is fair. The model
sees the question only (no reference answer). Empty/error responses are kept (not dropped)
so refusals are scorable in Stage 4. Each response's reading level is computed at write time.

One CSV per model: results/responses/<model_key>.csv. Resumable per (question_id, run).

CLI:  python -m src.benchmark [--fresh] [--models gpt-4o,claude-opus]
"""
from __future__ import annotations

import argparse

import pandas as pd

from src import prompts
from src.concurrency import parallel_map
from src.config import Config
from src.llm_client import LLMClient
from src.quality_checks import fk_grade, reading_ease
from src.utils import CheckpointWriter, get_logger, read_csv, to_int, to_temp

log = get_logger("benchmark")

RESP_COLUMNS = [
    "question_id", "area", "standard", "question_type", "run",
    "response", "model_id", "finish_reason", "is_error",
    "response_fk_grade", "response_reading_ease",
    # Actual provider-reported token counts (not estimates):
    "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens",
]


def _gt_source(cfg: Config) -> pd.DataFrame:
    """Use validated ground truth if present, else the raw ground truth."""
    validated = cfg.path("ground_truth_validated")
    path = validated if validated.exists() else cfg.path("ground_truth")
    return read_csv(path)


def benchmark_model(cfg: Config, client: LLMClient, model_key: str, gt: pd.DataFrame,
                    *, fresh: bool = False) -> pd.DataFrame:
    proto = cfg.get("benchmark_protocol", default={}) or {}
    system = prompts.SYSTEM_PROMPTS[proto.get("system_prompt", "middle_school")]
    temperature = to_temp(proto.get("temperature"))   # None => omit; model uses its own default
    max_tokens = to_int(proto.get("max_tokens"))       # None => no output cap; model runs to completion
    runs = int(cfg.get("runs_per_model", default=3))
    workers = int(cfg.get("concurrency", "max_parallel_requests", default=8))
    flush_every = int(cfg.get("checkpoint", "flush_every", default=25))

    out_path = cfg.path("responses_dir") / f"{model_key}.csv"
    writer = CheckpointWriter(out_path, key_cols=["question_id", "run"],
                              columns=RESP_COLUMNS, flush_every=flush_every, fresh=fresh)

    tasks = [{"question_id": r["question_id"], "run": run, "_row": r}
             for _, r in gt.iterrows() for run in range(1, runs + 1)]
    pending = writer.pending(tasks)
    log.info("[%s] %d / %d responses pending.", model_key, len(pending), len(tasks))

    def _task(t: dict) -> dict:
        row = t["_row"]
        res = client.complete(model_key, system=system, user=str(row["question"]),
                              temperature=temperature, max_tokens=max_tokens)
        text = res.text
        return {
            "question_id": row["question_id"], "area": row["area"],
            "standard": row["standard"], "question_type": row["question_type"],
            "run": t["run"], "response": text, "model_id": res.model_id,
            "finish_reason": res.finish_reason, "is_error": int(not res.ok),
            "response_fk_grade": round(fk_grade(text), 2),
            "response_reading_ease": round(reading_ease(text), 2),
            "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
            "reasoning_tokens": res.reasoning_tokens, "total_tokens": res.total_tokens,
        }

    parallel_map(_task, pending, max_workers=workers, desc=f"bench {model_key}",
                 on_result=lambda i, t, r: writer.add(r))
    df = writer.close()
    errs = int(df["is_error"].sum()) if len(df) else 0
    log.info("[%s] wrote %s (%d rows, %d API errors).", model_key, out_path, len(df), errs)
    return df


def benchmark(cfg: Config | None = None, *, models: list[str] | None = None,
              fresh: bool = False) -> None:
    cfg = cfg or Config.load()
    client = LLMClient(cfg)
    gt = _gt_source(cfg)
    model_keys = models or list(cfg.get("benchmark_models"))
    for key in model_keys:
        benchmark_model(cfg, client, key, gt, fresh=fresh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 3: benchmark candidate models.")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--models", help="Comma-separated subset of benchmark_models.")
    args = parser.parse_args(argv)
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    benchmark(models=models, fresh=args.fresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

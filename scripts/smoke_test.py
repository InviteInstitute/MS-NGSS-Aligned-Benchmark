"""Quick end-to-end pipeline test on a tiny slice.

Runs generate -> quality -> validate -> benchmark -> evaluate -> calibrate -> analyze with a
small questions_per_area, one benchmarked model, and one run, then asserts that every expected
output CSV exists with the right shape.

Modes:
  --mock   (default OFF)  patch the LLM client with deterministic canned responses, so the
           pipeline can be verified OFFLINE with no API keys or cost. Used in CI / dev.
  (no flag)               run against your real configured endpoint using .env keys.

Usage:
  python scripts/smoke_test.py --mock      # offline wiring check
  python scripts/smoke_test.py             # live check against your endpoint

SANDBOXED: all outputs are redirected into `_smoke_scratch/` (see _tiny_cfg / SMOKE_SCRATCH).
The smoke test — and the pytest suite that drives it — never read or overwrite the real
`data/ground_truth*.csv` or `results/`. run() aborts if that redirect is ever broken.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config                       # noqa: E402
from src.llm_client import LLMResult               # noqa: E402

# A smoke test is a wiring check — it must NEVER touch the real dataset or results. All of its
# outputs go into this disposable scratch tree (relative to the repo root), applied in _tiny_cfg
# for BOTH mock and live runs. Kept as a module constant so run()'s safety guard can verify it.
SMOKE_SCRATCH = "_smoke_scratch"

_COUNTER = itertools.count(1)


def _mock_complete(self, model_key, *, user, system=None, temperature=0.0, max_tokens=1024):
    """Deterministic stand-in for a real chat completion, keyed off the prompt's purpose."""
    n = next(_COUNTER)
    if "Return ONLY valid JSON in exactly this shape" in user and '"items"' in user:
        # generation: emit lexically distinct, in-band Q&A pairs (so dedupe doesn't collapse them)
        nouns = ["neutron", "proton", "electron", "molecule", "atom", "ion", "isotope",
                 "compound", "element", "nucleus", "crystal", "polymer", "mineral", "cell",
                 "gene", "fossil", "current", "magnet", "wave", "photon", "glacier", "comet"]
        verbs = ["form", "move", "change", "combine", "transfer energy", "react", "vibrate",
                 "attract", "repel", "dissolve", "expand", "cool down"]
        items = []
        for k in range(6):
            noun = nouns[(n * 7 + k * 3) % len(nouns)]
            verb = verbs[(n * 5 + k) % len(verbs)]
            q = f"How does a {noun} {verb} when conditions around it shift during this lesson?"
            a = (f"A {noun} can {verb} because tiny particles follow clear rules. In simple "
                 f"words, energy and matter behave in ways students can observe and measure.")
            items.append({"question": q, "answer": a})
        import json
        return LLMResult(json.dumps({"items": items}), model_key, "mock")
    if "appropriate for a middle-school" in user:
        return LLMResult('{"appropriate": true, "reason": "ok"}', model_key, "mock")
    if "validating a ground-truth" in user:
        return LLMResult('{"score": 1, "rationale": "looks correct"}', model_key, "mock")
    if "grading a candidate answer" in user:
        # vary by judge so triple mode produces some disagreement to exercise flagging
        binary = 0 if (model_key.endswith("mini") and n % 5 == 0) else 1
        return LLMResult(f'{{"binary": {binary}, "rubric": 1, "rationale": "ok"}}', model_key, "mock")
    # benchmark answer (plain middle-school text). Emit non-zero usage; pretend the "mini"
    # model is a reasoning model that reports reasoning tokens, to exercise token analytics.
    reasoning = 8 if model_key.endswith("mini") else 0
    prompt_tokens, completion_tokens = 20, 30
    return LLMResult(
        "A neutron is a tiny particle inside the center of an atom. It has no electric charge, "
        "and it helps hold the atom's nucleus together with the protons.",
        model_key, "mock", finish_reason="stop",
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        reasoning_tokens=reasoning, total_tokens=prompt_tokens + completion_tokens + reasoning,
    )


def _tiny_cfg(use_mock: bool) -> Config:
    cfg = Config.load(require_keys=not use_mock)
    raw = cfg.raw

    # SAFETY (load-bearing): redirect every OUTPUT path into the disposable scratch tree so a
    # smoke run — mock or live — can never read from or overwrite the real ground truth / results.
    # `_gt_judge_long.csv` is written next to `ground_truth`, so it follows automatically.
    # Inputs (standards_yaml) intentionally keep pointing at the real files. This is the single
    # choke point: both smoke_run() and the pytest fixture obtain their Config from here.
    raw["paths"] = {
        **(raw.get("paths") or {}),
        "ground_truth": f"{SMOKE_SCRATCH}/data/ground_truth.csv",
        "ground_truth_validated": f"{SMOKE_SCRATCH}/data/ground_truth_validated.csv",
        "responses_dir": f"{SMOKE_SCRATCH}/results/responses",
        "eval_dir": f"{SMOKE_SCRATCH}/results/eval",
        "calibration_dir": f"{SMOKE_SCRATCH}/results/calibration",
        "human_review_dir": f"{SMOKE_SCRATCH}/results/human_review",
        "reports_dir": f"{SMOKE_SCRATCH}/results/reports",
    }
    (ROOT / SMOKE_SCRATCH / "data").mkdir(parents=True, exist_ok=True)

    raw["questions_per_area"] = 6
    raw["question_type_split"] = {"recall": 2, "conceptual": 2, "applied": 2}
    raw["runs_per_model"] = 1
    raw["appropriateness"]["fk_grade_range"] = [0.0, 100.0]       # don't fight FK in a smoke test

    if use_mock:
        # Self-contained mock registry + roles, so the wiring check never depends on the user's
        # current config (e.g. an empty benchmark_models list). The client is patched anyway.
        raw["models"] = {
            "mock-fast": {"base_url": "http://mock/v1", "api_key_env": "MOCK_KEY",
                          "model": "mock-fast-v1", "family": "mockA"},
            "mock-mini": {"base_url": "http://mock/v1", "api_key_env": "MOCK_KEY",
                          "model": "mock-mini-v1", "family": "mockB"},
        }
        raw["generator_model"] = "mock-fast"
        raw["gt_judges"] = ["mock-fast", "mock-mini"]
        raw["benchmark_models"] = ["mock-fast", "mock-mini"]   # exercise pairwise stats
        raw["eval_mode"] = "single"
        raw["eval_judge"] = "mock-fast"
        raw["eval_judges"] = ["mock-fast", "mock-mini"]
        cfg._resolve_models(require_keys=False)   # rebuild the resolved-model cache for the new registry
    else:
        # Live smoke against the user's real endpoint: use their roles, but cap benchmark models
        # and fall back to the registry if they left benchmark_models empty.
        bm = list(raw.get("benchmark_models") or [])
        raw["benchmark_models"] = bm[:2] or list((raw.get("models") or {}).keys())[:1]
    return cfg


def _assert(path: Path, label: str) -> None:
    assert path.exists(), f"MISSING expected output: {label} ({path})"
    import pandas as pd
    df = pd.read_csv(path)
    print(f"  ok  {label:30s} {path.name:32s} rows={len(df)}")


def run(use_mock: bool) -> int:
    cfg = _tiny_cfg(use_mock)

    # Defense in depth: refuse to run if outputs are NOT sandboxed (e.g. if the redirect above is
    # ever broken). Better to abort a smoke test than to overwrite the real dataset again.
    scratch_root = ROOT / SMOKE_SCRATCH
    for name in ("ground_truth", "ground_truth_validated", "responses_dir", "eval_dir",
                 "calibration_dir", "human_review_dir", "reports_dir"):
        if scratch_root not in cfg.path(name).parents:
            raise SystemExit(
                f"SAFETY ABORT: smoke output paths.{name} ({cfg.path(name)}) is outside the "
                f"scratch dir {scratch_root}. Refusing to run so real data isn't overwritten.")

    ctx = mock.patch("src.llm_client.LLMClient.complete", _mock_complete) if use_mock else _nullctx()
    with ctx:
        from src.generate_dataset import generate
        from src.quality_checks import run as quality_run
        from src.validate_groundtruth import validate
        from src.benchmark import benchmark
        from src.evaluate import evaluate
        from src.calibrate_judges import run as calibrate_run
        from src.analyze import analyze

        generate(cfg, fresh=True)
        quality_run(cfg)
        validate(cfg, fresh=True)
        benchmark(cfg, fresh=True)
        evaluate(cfg, fresh=True)
        calibrate_run(cfg)
        analyze(cfg)

    print("\nVerifying outputs:")
    _assert(cfg.path("ground_truth"), "ground truth")
    _assert(cfg.path("ground_truth_validated"), "validated ground truth")
    first_model = cfg.get("benchmark_models")[0]
    mode = cfg.get("eval_mode")
    _assert(cfg.path("responses_dir") / f"{first_model}.csv", "responses")
    _assert(cfg.path("eval_dir") / f"{first_model}_{mode}.csv", "evaluation")
    _assert(cfg.path("calibration_dir") / "to_label.csv", "calibration sheet")
    _assert(cfg.path("reports_dir") / "leaderboard.csv", "leaderboard")
    _assert(cfg.path("reports_dir") / "accuracy_by_area_type.csv", "accuracy by area x type")
    _assert(cfg.path("reports_dir") / "reading_level.csv", "reading level")
    print("\nSMOKE TEST PASSED")
    return 0


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tiny end-to-end pipeline smoke test.")
    parser.add_argument("--mock", action="store_true",
                        help="Use canned LLM responses (offline, no API keys).")
    args = parser.parse_args(argv)
    return run(use_mock=args.mock)


if __name__ == "__main__":
    raise SystemExit(main())

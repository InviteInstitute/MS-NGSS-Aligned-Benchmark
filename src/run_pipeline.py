"""Pipeline orchestrator — run one stage or the whole thing.

Stages (in order):
  generate  -> quality -> validate -> benchmark -> evaluate -> calibrate -> analyze

Examples:
  python -m src.run_pipeline --stage all
  python -m src.run_pipeline --stage benchmark
  python -m src.run_pipeline --stage all --fresh
"""
from __future__ import annotations

import argparse

from src.config import Config
from src.utils import get_logger

log = get_logger("pipeline")

STAGE_ORDER = ["generate", "quality", "validate", "benchmark", "evaluate", "calibrate", "analyze"]


def _run_stage(stage: str, cfg: Config, *, fresh: bool, models: list[str] | None = None) -> None:
    log.info("=== stage: %s ===", stage)
    if stage == "generate":
        from src.generate_dataset import generate
        generate(cfg, fresh=fresh)
    elif stage == "quality":
        from src.quality_checks import run
        run(cfg)
    elif stage == "validate":
        from src.validate_groundtruth import validate
        validate(cfg, fresh=fresh)
    elif stage == "benchmark":
        from src.benchmark import benchmark
        benchmark(cfg, models=models, fresh=fresh)
    elif stage == "evaluate":
        from src.evaluate import evaluate
        evaluate(cfg, models=models, fresh=fresh)
    elif stage == "calibrate":
        from src.calibrate_judges import run
        run(cfg)
    elif stage == "analyze":
        from src.analyze import analyze
        analyze(cfg)
    else:
        raise ValueError(f"unknown stage {stage!r}")


# Named stage groups (also accepted by --stage).
GROUPS = {
    "all": STAGE_ORDER,
    "groundtruth": ["generate", "quality", "validate"],   # build the dataset only
    "benchmark_all": ["benchmark", "evaluate", "calibrate", "analyze"],  # all benchmark models
}
# Stages that actually call the model APIs (and so require keys).
_NEEDS_KEYS = {"generate", "validate", "benchmark", "evaluate"}


def resolve_stages(stage: str) -> list[str]:
    """Map a --stage value to an ordered stage list.

    Accepts a group name ('all', 'groundtruth', 'benchmark_all'), a single stage, or a
    comma-separated list ('benchmark,evaluate,analyze').
    """
    if stage in GROUPS:
        return list(GROUPS[stage])
    requested = [s.strip() for s in stage.split(",") if s.strip()]
    unknown = [s for s in requested if s not in STAGE_ORDER]
    if unknown:
        raise ValueError(f"unknown stage(s): {unknown}. Valid: {STAGE_ORDER} or groups {list(GROUPS)}")
    return requested


def run(stage: str = "all", *, fresh: bool = False, models: list[str] | None = None) -> None:
    """Run one or more stages. `models` (if given) scopes the per-model stages
    (benchmark, evaluate) to that subset of benchmark_models; other stages ignore it.
    """
    stages = resolve_stages(stage)
    cfg = Config.load(require_keys=any(s in _NEEDS_KEYS for s in stages))
    for s in stages:
        _run_stage(s, cfg, fresh=fresh, models=models)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the benchmark pipeline.")
    parser.add_argument("--stage", default="all",
                        help="Stage, comma-separated stages, or a group "
                             f"({', '.join(GROUPS)}). Single stages: {', '.join(STAGE_ORDER)}.")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore checkpoints and recompute the run stages.")
    parser.add_argument("--models",
                        help="Comma-separated subset of benchmark_models to scope the "
                             "per-model stages (benchmark, evaluate) to.")
    args = parser.parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    run(args.stage, fresh=args.fresh, models=models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

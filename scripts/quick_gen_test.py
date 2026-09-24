"""Quick generation test — produce a small batch of questions (default 20) and print them.

Runs ONLY Stage 1 (generation + the appropriateness/dedupe filters) on a few areas, writes to
a SEPARATE file (data/ground_truth_sample.csv, so your real ground truth is untouched), then
prints every question grouped by area so you can eyeball quality and standard-alignment.

Examples:
  python scripts/quick_gen_test.py                       # 20 Qs across 4 areas, live (needs .env keys)
  python scripts/quick_gen_test.py --total 12 --areas MS-PS3,MS-LS1,MS-ESS1
  python scripts/quick_gen_test.py --mock                # offline wiring check, no keys/cost
  python scripts/quick_gen_test.py --no-llm-check        # skip the LLM appropriateness gate (faster/cheaper)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config                       # noqa: E402
from src.generate_dataset import generate           # noqa: E402

# A spread across all four disciplines for a representative eyeball.
DEFAULT_AREAS = ["MS-PS3", "MS-LS1", "MS-ESS2", "MS-ETS1"]
SAMPLE_OUT = "data/ground_truth_sample.csv"


def _split_for(n: int) -> dict[str, int]:
    """Even-ish recall/conceptual/applied split summing to n."""
    base, rem = divmod(n, 3)
    counts = {"recall": base, "conceptual": base, "applied": base}
    for t in list(counts)[:rem]:
        counts[t] += 1
    return counts


def run(total: int, areas: list[str], use_mock: bool, llm_check: bool) -> int:
    per_area, rem = divmod(total, len(areas))
    if per_area == 0:
        print(f"--total {total} is too small for {len(areas)} areas; use fewer areas.")
        return 1
    if rem:
        print(f"note: {total} not divisible by {len(areas)} areas; generating "
              f"{per_area} each = {per_area * len(areas)}.")

    cfg = Config.load(require_keys=not use_mock)
    cfg.raw["questions_per_area"] = per_area
    cfg.raw["question_type_split"] = _split_for(per_area)
    cfg.raw["paths"]["ground_truth"] = SAMPLE_OUT      # write to the sample file, not real GT
    cfg.raw.setdefault("appropriateness", {})["llm_check"] = llm_check

    ctx = mock.patch("src.llm_client.LLMClient.complete", _mock) if use_mock else _nullctx()
    with ctx:
        df = generate(cfg, fresh=True, only_areas=areas)

    _print(df, cfg)
    print(f"\nWrote {len(df)} questions to {SAMPLE_OUT}. "
          f"Review them, then delete the sample file when done.")
    return 0


def _print(df, cfg) -> None:
    key = cfg.get("generator_model")
    model_id = cfg.model(key).model
    print("\n" + "=" * 80)
    print(f"GENERATED {len(df)} QUESTIONS  (generator key '{key}' -> model '{model_id}')")
    print("=" * 80)
    for area, sub in df.groupby("area"):
        print(f"\n### {area}  ({len(sub)} questions)")
        for _, r in sub.iterrows():
            print(f"\n  [{r['standard']} | {r['question_type']} | FK grade {r['fk_grade']}]")
            print(f"  Q: {r['question']}")
            print(f"  A: {r['reference_answer']}")
    # quick summary
    print("\n" + "-" * 80)
    print("Summary by type:", df["question_type"].value_counts().to_dict())
    print("Mean FK grade   :", round(float(df["fk_grade"].mean()), 2))
    print("Standards covered:", df["standard"].nunique())


# --- offline mock (mirrors scripts/smoke_test.py so this runs without keys) -----------------
def _mock(self, model_key, *, user, system=None, temperature=None, max_tokens=None):
    from scripts.smoke_test import _mock_complete
    return _mock_complete(self, model_key, user=user, system=system,
                          temperature=temperature, max_tokens=max_tokens)


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Quick small-batch generation test.")
    p.add_argument("--total", type=int, default=20, help="Total questions to generate (default 20).")
    p.add_argument("--areas", default=",".join(DEFAULT_AREAS),
                   help="Comma-separated DCI area codes (default: a spread across disciplines).")
    p.add_argument("--mock", action="store_true", help="Offline canned responses (no keys/cost).")
    p.add_argument("--no-llm-check", action="store_true",
                   help="Skip the LLM appropriateness gate (FK band + dedupe still apply).")
    args = p.parse_args(argv)
    areas = [a.strip() for a in args.areas.split(",") if a.strip()]
    return run(args.total, areas, use_mock=args.mock, llm_check=not args.no_llm_check)


if __name__ == "__main__":
    raise SystemExit(main())

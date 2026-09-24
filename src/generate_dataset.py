"""Stage 1 — generate the ground-truth question/answer dataset.

For every area we target `questions_per_area` questions split across recall/conceptual/applied
(`question_type_split`) and distributed across the area's performance-expectation standards.
Each candidate question passes a level-appropriateness filter (Flesch-Kincaid band + optional
LLM gate) and an embedding-free dedupe before it is accepted. Short cells are backfilled over
several rounds. Generation/appropriateness calls are parallelized; acceptance is deterministic.

Reaching quota: a (standard, type) slot that keeps failing the gates (e.g. only so many distinct
in-band questions exist for a narrow standard) is declared "stuck" after `stuck_patience`
consecutive zero-accept rounds; its remaining need is REDISTRIBUTED to sibling standards in the
same area+type so the AREA still meets its per-type quota. The run only finishes short for an
area+type if every one of its standards is exhausted.

Resumable: rows are appended to ground_truth.csv keyed by question_id, so re-running continues
from the existing file. Use --fresh to start over.

CLI:  python -m src.generate_dataset [--fresh]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from src import prompts
from src.concurrency import parallel_map
from src.config import Config
from src.dedupe import normalize
from src.llm_client import LLMClient, parse_json
from src.quality_checks import fk_grade
from src.standards import Area, load_areas
from src.utils import CheckpointWriter, get_logger, to_int, to_temp
from rapidfuzz import fuzz

log = get_logger("generate")

GT_COLUMNS = [
    "question_id", "area", "standard", "question_type",
    "question", "reference_answer", "generator_model", "fk_grade",
]
DEFAULT_MAX_ROUNDS = 20
DEFAULT_OVERGEN = 1.6   # request ~60% more than needed per cell to survive filtering
DEFAULT_STUCK_PATIENCE = 2   # zero-accept rounds for a slot before redistributing its need


@dataclass
class Cell:
    area: str
    standard: str
    qtype: str
    need: int


def _plan_area(area: Area, qpa: int, split: dict[str, int]) -> dict[tuple[str, str], int]:
    """Target counts per (standard, question_type) for one area."""
    targets: dict[tuple[str, str], int] = {}
    std_codes = [s.code for s in area.standards]
    for qtype, type_total in split.items():
        # spread this type's quota across the area's standards (remainder front-loaded)
        base, extra = divmod(type_total, len(std_codes))
        for i, code in enumerate(std_codes):
            targets[(code, qtype)] = base + (1 if i < extra else 0)
    return targets


def _have_counts(df: pd.DataFrame) -> dict[tuple[str, str, str], int]:
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for _, r in df.iterrows():
        counts[(r["area"], r["standard"], r["question_type"])] += 1
    return counts


def generate(cfg: Config | None = None, *, fresh: bool = False,
             only_areas: list[str] | None = None) -> pd.DataFrame:
    cfg = cfg or Config.load()
    client = LLMClient(cfg)
    gen_model = cfg.get("generator_model")           # registry key, used to make the API call
    gen_model_id = cfg.model(gen_model).model         # ACTUAL model id that ran (for the label)
    areas = load_areas(cfg)
    if only_areas:
        wanted = set(only_areas)
        areas = [a for a in areas if a.code in wanted]
        missing = wanted - {a.code for a in areas}
        if missing:
            raise ValueError(f"Unknown area code(s): {sorted(missing)}")
    qpa = int(cfg.get("questions_per_area", default=100))
    split = cfg.get("question_type_split")
    threshold = float(cfg.get("dedupe", "similarity_threshold", default=88))
    fk_lo, fk_hi = cfg.get("appropriateness", "fk_grade_range", default=[5.0, 9.0])
    llm_check = bool(cfg.get("appropriateness", "llm_check", default=True))
    flush_every = int(cfg.get("checkpoint", "flush_every", default=25))
    workers = int(cfg.get("concurrency", "max_parallel_requests", default=8))
    gen_temp = to_temp(cfg.get("defaults", "temperature_generation"))
    judge_temp = to_temp(cfg.get("defaults", "temperature_judge"))
    gen_max = to_int(cfg.get("defaults", "max_tokens_generation"))
    judge_max = to_int(cfg.get("defaults", "max_tokens_judge"))
    max_rounds = int(cfg.get("generation", "max_rounds", default=DEFAULT_MAX_ROUNDS))
    overgen = float(cfg.get("generation", "overgen", default=DEFAULT_OVERGEN))
    stuck_patience = int(cfg.get("generation", "stuck_patience", default=DEFAULT_STUCK_PATIENCE))

    area_by_code = {a.code: a for a in areas}
    targets = {a.code: _plan_area(a, qpa, split) for a in areas}

    writer = CheckpointWriter(
        cfg.path("ground_truth"), key_cols=["question_id"], columns=GT_COLUMNS,
        flush_every=flush_every, fresh=fresh,
    )
    existing = writer.close()  # current state on disk
    have = _have_counts(existing)

    # Per-area: accepted normalized questions (for dedupe) and next id index.
    accepted_norms: dict[str, list[str]] = defaultdict(list)
    next_idx: dict[str, int] = defaultdict(lambda: 1)
    for _, r in existing.iterrows():
        accepted_norms[r["area"]].append(normalize(str(r["question"])))
        try:
            n = int(str(r["question_id"]).rsplit("-", 1)[-1])
            next_idx[r["area"]] = max(next_idx[r["area"]], n + 1)
        except ValueError:
            pass

    rejects: list[dict] = []
    capped: set[tuple[str, str, str]] = set()          # slots declared unfillable (won't be retried)
    stall: dict[tuple[str, str, str], int] = defaultdict(int)   # consecutive zero-accept rounds per slot

    def _need(area: str, std: str, qtype: str) -> int:
        return targets[area][(std, qtype)] - have[(area, std, qtype)]

    def _redistribute(area: str, std: str, qtype: str) -> None:
        """Cap a stuck slot at what it has and move its remaining need onto fillable sibling
        standards in the same area+type, so the area still reaches its per-type quota."""
        shortfall = _need(area, std, qtype)
        targets[area][(std, qtype)] = have[(area, std, qtype)]   # stop requesting this slot
        capped.add((area, std, qtype))
        if shortfall <= 0:
            return
        siblings = [s for (s, qt) in targets[area]
                    if qt == qtype and s != std and (area, s, qtype) not in capped]
        if not siblings:
            log.warning("Area %s %s/%s stuck at %d; no sibling standard left to absorb %d — "
                        "area+type will finish short.", area, std, qtype,
                        have[(area, std, qtype)], shortfall)
            return
        for i in range(shortfall):                              # spread the need round-robin
            targets[area][(siblings[i % len(siblings)], qtype)] += 1
        log.info("Area %s %s/%s stuck at %d; redistributed %d need to sibling standard(s): %s.",
                 area, std, qtype, have[(area, std, qtype)], shortfall, siblings)

    for rnd in range(1, max_rounds + 1):
        cells = [
            Cell(area, std, qtype, _need(area, std, qtype))
            for area in area_by_code
            for (std, qtype) in list(targets[area])
            if _need(area, std, qtype) > 0 and (area, std, qtype) not in capped
        ]
        if not cells:
            break
        log.info("Round %d: %d slots need filling (%d questions).",
                 rnd, len(cells), sum(c.need for c in cells))

        def _gen(cell: Cell) -> tuple[Cell, list[dict]]:
            area = area_by_code[cell.area]
            std = next(s for s in area.standards if s.code == cell.standard)
            n_req = max(cell.need, int(round(cell.need * overgen)))
            res = client.complete(
                gen_model,
                system=prompts.SYSTEM_PROMPTS["default"],
                user=prompts.generation_prompt(area.title, std.code, std.text, cell.qtype, n_req),
                temperature=gen_temp,
                max_tokens=gen_max,
            )
            obj = parse_json(res.text) or {}
            items = obj.get("items", []) if isinstance(obj, dict) else []
            return cell, [it for it in items if isinstance(it, dict) and it.get("question")]

        gen_results = parallel_map(_gen, cells, max_workers=workers, desc=f"generate r{rnd}")

        # Flatten candidates; apply the free FK-band gate immediately.
        candidates: list[dict] = []
        for cell, items in gen_results:
            for it in items:
                q = str(it.get("question", "")).strip()
                a = str(it.get("answer", "")).strip()
                if not q or not a:
                    continue
                fk = fk_grade(q)
                if not (fk_lo <= fk <= fk_hi):
                    rejects.append({"area": cell.area, "standard": cell.standard,
                                    "question_type": cell.qtype, "question": q,
                                    "reason": f"fk_grade={fk:.1f}_out_of_band", "round": rnd})
                    continue
                candidates.append({"area": cell.area, "standard": cell.standard,
                                   "question_type": cell.qtype, "question": q,
                                   "answer": a, "fk_grade": round(fk, 2)})

        # Optional LLM appropriateness gate (parallelized).
        if llm_check and candidates:
            def _gate(c: dict) -> bool:
                area = area_by_code[c["area"]]
                res = client.complete(
                    gen_model, system=prompts.SYSTEM_PROMPTS["default"],
                    user=prompts.appropriateness_prompt(c["question"], area.title),
                    temperature=judge_temp, max_tokens=judge_max,
                )
                obj = parse_json(res.text) or {}
                return bool(obj.get("appropriate", True))  # default-accept if judge unparsable

            verdicts = parallel_map(_gate, candidates, max_workers=workers, desc=f"appropriate r{rnd}")
            kept = []
            for c, ok in zip(candidates, verdicts):
                if ok:
                    kept.append(c)
                else:
                    rejects.append({**{k: c[k] for k in ("area", "standard", "question_type", "question")},
                                    "reason": "llm_inappropriate", "round": rnd})
            candidates = kept

        # Deterministic acceptance: dedupe within area, respect remaining per-slot need.
        accepted_by_cell: dict[tuple[str, str, str], int] = defaultdict(int)
        for c in candidates:
            area, std, qtype = c["area"], c["standard"], c["question_type"]
            if (area, std, qtype) in capped or _need(area, std, qtype) <= 0:
                continue
            norm = normalize(c["question"])
            if any(fuzz.token_set_ratio(norm, prev) >= threshold for prev in accepted_norms[area]):
                rejects.append({"area": area, "standard": std, "question_type": qtype,
                                "question": c["question"], "reason": "duplicate", "round": rnd})
                continue
            qid = f"{area}-{next_idx[area]:03d}"
            next_idx[area] += 1
            writer.add({
                "question_id": qid, "area": area, "standard": std, "question_type": qtype,
                "question": c["question"], "reference_answer": c["answer"],
                "generator_model": gen_model_id, "fk_grade": c["fk_grade"],
            })
            accepted_norms[area].append(norm)
            have[(area, std, qtype)] += 1
            accepted_by_cell[(area, std, qtype)] += 1
        writer.flush()
        log.info("Round %d accepted %d questions.", rnd, sum(accepted_by_cell.values()))

        # Track stalled slots; redistribute any that made no progress for `stuck_patience` rounds.
        # This replaces the old global "stop at first zero-accept round" break: instead of abandoning
        # the whole run, we only give up on individual slots (and hand their quota to siblings).
        for cell in cells:
            key = (cell.area, cell.standard, cell.qtype)
            if _need(*key) <= 0:
                stall[key] = 0
            elif accepted_by_cell[key] > 0:
                stall[key] = 0          # progress; keep trying this slot
            else:
                stall[key] += 1
                if stall[key] >= stuck_patience:
                    _redistribute(*key)

    df = writer.close()
    if rejects:
        rej_path = cfg.path("reports_dir") / "generation_rejects.csv"
        rej_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rejects).to_csv(rej_path, index=False)
        log.info("Logged %d rejected candidates to %s", len(rejects), rej_path)

    # Final coverage: report per-area shortfalls and the grand total against quota.
    final_have = _have_counts(df)
    short_areas = 0
    for area in area_by_code:
        got = sum(v for (a, _, _), v in final_have.items() if a == area)
        if got < qpa:
            short_areas += 1
            log.warning("Area %s ended with %d/%d questions (every standard for some type exhausted).",
                        area, got, qpa)
    target_total = qpa * len(area_by_code)
    if len(df) < target_total:
        log.warning("Stage 1 finished SHORT of quota: %d/%d questions (%d area(s) under target).",
                    len(df), target_total, short_areas)
    else:
        log.info("Stage 1 reached quota: %d/%d questions.", len(df), target_total)
    log.info("Stage 1 complete: %d questions in %s", len(df), cfg.path("ground_truth"))
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 1: generate ground-truth Q&A.")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing output and regenerate.")
    args = parser.parse_args(argv)
    generate(fresh=args.fresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

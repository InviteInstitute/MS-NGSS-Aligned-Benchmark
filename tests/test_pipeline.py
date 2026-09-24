"""Schema / value-domain checks for the pipeline outputs.

Runs the offline mock pipeline once (no API keys, no cost) and asserts that every stage's
CSV has the expected columns and that score/flag values fall in their allowed domains. Also
unit-tests the embedding-free dedupe and the kappa helpers.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.config import Config
from src.dedupe import dedupe
from src.analyze import fleiss_kappa
from src.calibrate_judges import cohen_kappa
import numpy as np
from scripts.smoke_test import run as smoke_run, _tiny_cfg


@pytest.fixture(scope="module")
def cfg() -> Config:
    smoke_run(use_mock=True)          # generate all outputs once for the module
    return _tiny_cfg(use_mock=True)   # same mock config the run used (models, mode, paths)


def test_ground_truth_schema(cfg: Config) -> None:
    df = pd.read_csv(cfg.path("ground_truth"))
    for col in ["question_id", "area", "standard", "question_type", "question",
                "reference_answer", "generator_model", "fk_grade"]:
        assert col in df.columns
    assert set(df["question_type"]).issubset({"recall", "conceptual", "applied"})
    assert df["question_id"].is_unique
    assert len(df) > 0


def test_validated_flags_binary(cfg: Config) -> None:
    df = pd.read_csv(cfg.path("ground_truth_validated"))
    assert set(df["human_review_flag"].unique()).issubset({0, 1})
    assert set(df["majority_label"].unique()).issubset({0, 1})


def test_eval_value_domains(cfg: Config) -> None:
    mode = cfg.get("eval_mode")
    model = cfg.get("benchmark_models")[0]
    df = pd.read_csv(cfg.path("eval_dir") / f"{model}_{mode}.csv")
    assert set(pd.to_numeric(df["binary_score"]).unique()).issubset({0, 1})
    assert set(pd.to_numeric(df["is_refusal"]).unique()).issubset({0, 1})
    rub = pd.to_numeric(df["rubric_score"], errors="coerce").dropna()
    assert rub.isin([0.0, 0.5, 1.0]).all()


def test_reading_level_present(cfg: Config) -> None:
    model = cfg.get("benchmark_models")[0]
    df = pd.read_csv(cfg.path("responses_dir") / f"{model}.csv")
    assert "response_fk_grade" in df.columns
    assert "response_reading_ease" in df.columns


def test_actual_token_columns(cfg: Config) -> None:
    model = cfg.get("benchmark_models")[0]
    df = pd.read_csv(cfg.path("responses_dir") / f"{model}.csv")
    for col in ["prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"]:
        assert col in df.columns


def test_leaderboard_columns(cfg: Config) -> None:
    df = pd.read_csv(cfg.path("reports_dir") / "leaderboard.csv")
    for col in ["rank", "model", "macro_accuracy", "micro_accuracy", "refusal_rate"]:
        assert col in df.columns
    assert (df["macro_accuracy"].between(0, 1)).all()


def test_token_usage_report(cfg: Config) -> None:
    df = pd.read_csv(cfg.path("reports_dir") / "token_usage_by_model.csv")
    for col in ["model", "input_mean", "input_sd", "output_mean", "output_sd",
                "reasoning_mean", "reasoning_sd", "total_mean", "total_sd",
                "total_sum", "reasoning_available_frac"]:
        assert col in df.columns
    # the mock "mini" model reports reasoning tokens; at least one model should show > 0
    assert (df["reasoning_mean"] > 0).any()


def test_dedupe_drops_near_duplicates() -> None:
    qs = ["What is a neutron?", "what is a neutron", "How do plants make food?"]
    kept, dropped = dedupe(qs, threshold=88)
    assert len(kept) == 2 and len(dropped) == 1


def test_human_overrides_change_scores(cfg: Config) -> None:
    from src.analyze import _apply_human_overrides
    ev = pd.DataFrame({
        "model": ["m", "m", "m"],
        "question_id": ["Q1", "Q2", "Q3"],
        "run": [1, 1, 1],
        "binary_score": [1, 1, 0],
        "area": ["A", "A", "A"], "standard": ["S", "S", "S"],
        "question_type": ["recall", "recall", "recall"],
        "rubric_score": [float("nan")] * 3, "is_refusal": [0, 0, 0],
    })
    hr = cfg.path("human_review_dir")
    hr.mkdir(parents=True, exist_ok=True)
    # human says Q1's response is actually wrong (override 1 -> 0)
    pd.DataFrame([{"question_id": "Q1", "run": 1, "human_score": 0}]).to_csv(
        hr / "flags_eval_m.csv", index=False)
    # human says Q2 is an invalid ground-truth item -> drop it
    pd.DataFrame([{"question_id": "Q2", "human_score": 0}]).to_csv(
        hr / "flags_groundtruth.csv", index=False)
    out = _apply_human_overrides(cfg, ev)
    assert int(out.loc[out["question_id"] == "Q1", "binary_score"].iloc[0]) == 0
    assert "Q2" not in set(out["question_id"])           # dropped
    assert "Q3" in set(out["question_id"])               # untouched
    # cleanup so other tests/fixtures aren't affected
    (hr / "flags_eval_m.csv").unlink()
    (hr / "flags_groundtruth.csv").unlink()


def test_auto_drop_disagreements(tmp_path) -> None:
    """auto_drop_disagreements: validate() drops doubted questions (disagreement or unanimous
    incorrect), keeps only unanimously-accepted ones, and records what it dropped."""
    import json
    from unittest import mock as _mock
    import src.validate_groundtruth as vg
    from src.llm_client import LLMResult

    cfg = _tiny_cfg(use_mock=True)  # mock judges: mock-fast, mock-mini
    cfg.raw.setdefault("validation", {})["auto_drop_disagreements"] = True
    # Isolate all I/O to tmp_path so the shared fixture files are untouched.
    cfg.raw["paths"]["ground_truth"] = str(tmp_path / "gt.csv")
    cfg.raw["paths"]["ground_truth_validated"] = str(tmp_path / "gtv.csv")
    cfg.raw["paths"]["human_review_dir"] = str(tmp_path / "hr")

    pd.DataFrame([
        {"question_id": "MS-PS1-001", "area": "MS-PS1", "standard": "MS-PS1-1",
         "question_type": "recall", "question": "KEEP marker — what is an atom?",
         "reference_answer": "A unit of matter.", "generator_model": "mock", "fk_grade": 6.0},
        {"question_id": "MS-PS1-002", "area": "MS-PS1", "standard": "MS-PS1-1",
         "question_type": "recall", "question": "DISAGREE marker — what is a molecule?",
         "reference_answer": "Two or more atoms bonded.", "generator_model": "mock", "fk_grade": 6.0},
        {"question_id": "MS-PS1-003", "area": "MS-PS1", "standard": "MS-PS1-1",
         "question_type": "recall", "question": "WRONG marker — what is a proton?",
         "reference_answer": "A totally wrong answer.", "generator_model": "mock", "fk_grade": 6.0},
    ]).to_csv(cfg.path("ground_truth"), index=False)

    def fake_complete(self, model_key, *, user, system=None, temperature=None, max_tokens=None):
        if "KEEP marker" in user:
            score = 1                                   # both judges accept -> keep
        elif "DISAGREE marker" in user:
            score = 1 if model_key == "mock-fast" else 0  # judges split -> drop
        else:
            score = 0                                   # both reject -> drop (unanimous wrong)
        return LLMResult(json.dumps({"score": score, "rationale": "r"}), model_key, "mock")

    with _mock.patch("src.llm_client.LLMClient.complete", fake_complete):
        out = vg.validate(cfg, fresh=True)

    assert set(out["question_id"]) == {"MS-PS1-001"}          # only unanimous-correct survives
    rec = pd.read_csv(cfg.path("human_review_dir") / "auto_dropped_groundtruth.csv")
    assert set(rec["question_id"]) == {"MS-PS1-002", "MS-PS1-003"}
    reasons = dict(zip(rec["question_id"], rec["drop_reason"]))
    assert reasons["MS-PS1-002"] == "judge_disagreement"
    assert reasons["MS-PS1-003"] == "unanimous_incorrect"

    # Judge-agreement stats must be computed over ALL judged rows (incl. auto-dropped), not just the
    # single retained row — otherwise agreement is a meaningless 100%.
    from src.utils import load_judged
    from src.analyze import _judge_reliability
    judged = load_judged(cfg)
    assert len(judged) == 3                                    # full judged set, not just retained
    rel = _judge_reliability(cfg, "single")
    gt_panel = rel[rel["panel"] == "ground_truth"].iloc[0]
    assert gt_panel["n_items"] == 3                            # reliability over all 3 verdicts
    assert gt_panel["flag_rate"] == round(1 / 3, 3)            # exactly the 1 disagreement of 3


def test_teacher_report(tmp_path) -> None:
    """first-run vs. ensemble scoring + best-pick: the run-1 winner and the ensemble winner differ,
    proving the two columns measure different things and the best-pick uses ensemble."""
    import src.teacher_report as tr

    cfg = _tiny_cfg(use_mock=True)
    cfg.raw["paths"]["eval_dir"] = str(tmp_path / "eval")
    cfg.raw["paths"]["reports_dir"] = str(tmp_path / "reports")
    cfg.raw["paths"]["human_review_dir"] = str(tmp_path / "hr")
    (tmp_path / "eval").mkdir()

    def eval_rows(per_q_runs: dict) -> pd.DataFrame:
        out = []
        for q, runs in per_q_runs.items():
            for i, s in enumerate(runs, start=1):
                out.append({"question_id": q, "run": i, "area": "MS-PS1",
                            "standard": "MS-PS1-1", "question_type": "recall",
                            "binary_score": s, "rubric_score": "", "is_refusal": 0})
        return pd.DataFrame(out)

    # Models A and C: wrong on run 1 but right 2/3 runs -> first_run 0.0, ensemble 1.0 (they TIE)
    eval_rows({"Q1": (0, 1, 1), "Q2": (0, 1, 1)}).to_csv(tmp_path / "eval" / "Model A_single.csv", index=False)
    eval_rows({"Q1": (0, 1, 1), "Q2": (0, 1, 1)}).to_csv(tmp_path / "eval" / "Model C_single.csv", index=False)
    # Model B: right on run 1 but only 1/3 runs -> first_run 1.0, ensemble 0.0
    eval_rows({"Q1": (1, 0, 0), "Q2": (1, 0, 0)}).to_csv(tmp_path / "eval" / "Model B_single.csv", index=False)

    tr.teacher_report(cfg)
    reports = tmp_path / "reports"

    overall = pd.read_csv(reports / "teacher_leaderboard_overall.csv").set_index("model")
    assert overall.loc["Model A", "first_run_accuracy"] == 0.0
    assert overall.loc["Model A", "ensemble_accuracy"] == 1.0
    assert overall.loc["Model B", "first_run_accuracy"] == 1.0
    assert overall.loc["Model B", "ensemble_accuracy"] == 0.0

    best = pd.read_csv(reports / "teacher_best_by_standard.csv")
    row = best[best["standard"] == "MS-PS1-1"].iloc[0]
    # BOTH tied top models are listed, not just one
    assert set(row["best_models"].split(", ")) == {"Model A", "Model C"}
    assert int(row["n_tied"]) == 2
    assert row["runner_up_models"] == "Model B"
    assert float(row["ensemble_accuracy"]) == 1.0 and float(row["gap"]) == 1.0
    assert isinstance(row["standard_text"], str) and len(row["standard_text"]) > 0  # friendly label


def test_academic_report(tmp_path) -> None:
    """The academic report assembles from eval files: writes only academic_report.md, includes the
    linked TOC / List of Tables, all key sections, both appendices, and degrades gracefully when
    responses/validated/calibration data are absent."""
    import src.academic_report as ar

    cfg = _tiny_cfg(use_mock=True)
    for key, sub in {"eval_dir": "eval", "reports_dir": "reports", "responses_dir": "responses",
                     "human_review_dir": "hr", "calibration_dir": "calib"}.items():
        cfg.raw["paths"][key] = str(tmp_path / sub)
    cfg.raw["paths"]["ground_truth_validated"] = str(tmp_path / "gtv.csv")
    (tmp_path / "eval").mkdir()

    def make(base: int) -> pd.DataFrame:
        rows, qn = [], 0
        for std in ["MS-PS1-1", "MS-PS1-2"]:
            for qt in ["recall", "conceptual", "applied"]:
                for _ in range(2):
                    qn += 1
                    for run in (1, 2, 3):
                        rows.append({"question_id": f"{std}-{qt[:3]}-{qn}", "run": run,
                                     "area": "MS-PS1", "standard": std, "question_type": qt,
                                     "binary_score": base, "rubric_score": "", "is_refusal": 0})
        return pd.DataFrame(rows)

    make(1).to_csv(tmp_path / "eval" / "Model X_single.csv", index=False)   # all correct
    make(0).to_csv(tmp_path / "eval" / "Model Y_single.csv", index=False)   # all wrong

    out = ar.academic_report(cfg)
    md = out.read_text()

    # Only the report was written — no existing report clobbered/created.
    assert {p.name for p in (tmp_path / "reports").glob("*")} == {"academic_report.md"}
    for marker in ["## Contents", "## List of Tables", "## 1. Methods", "## 2. Overall performance",
                   "Appendix A", "Appendix C", "Model X", "Model Y",
                   'id="std-MS-PS1-1"', 'id="qtype-recall"', "single", "ensemble"]:
        assert marker in md, marker


def test_kappa_helpers() -> None:
    a = pd.Series([1, 1, 0, 0])
    assert cohen_kappa(a, a) == 1.0
    mat = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 0]])
    assert -1.0 <= fleiss_kappa(mat) <= 1.0

"""All prompt text — editable from the UI without touching code.

Templates live as code defaults below and can be overridden by `data/prompts.yaml` (which the
UI's Prompts tab writes). Placeholders use `$name` (string.Template) rather than `.format`, so
the literal JSON braces in the prompts don't need escaping and are safe to edit by hand.

Sections:
  system_prompts   : system prompts keyed by name (benchmark_protocol.system_prompt)
  question_types   : definitions of recall / conceptual / applied
  templates        : the generation / appropriateness / judge prompt bodies

Public builders: generation_prompt, appropriateness_prompt, gt_judge_prompt, eval_judge_prompt.

CLI:  python -m src.prompts --export   # write current effective prompts to data/prompts.yaml
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from string import Template

import yaml

from src.config import REPO_ROOT

PROMPTS_PATH = REPO_ROOT / "data" / "prompts.yaml"

# --------------------------------------------------------------------------- #
# Code defaults (used when data/prompts.yaml is absent or a key is missing)
# --------------------------------------------------------------------------- #
_DEFAULTS: dict[str, dict[str, str]] = {
    "system_prompts": {
        "middle_school": (
            "You are a helpful science tutor answering a question from a middle-school student "
            "(roughly ages 11-14, grades 6-8). Answer accurately and concisely, and write your "
            "explanation at a reading level appropriate for a middle-school learner: use plain "
            "language, short sentences, and define any necessary terms. Do not include "
            "disclaimers or meta-commentary."
        ),
        "default": "You are a helpful, accurate science assistant.",
    },
    "question_types": {
        "recall": (
            "RECALL: a short factual or vocabulary question with a single, unambiguous correct "
            "answer (e.g. 'What is a neutron?'). The answer is a fact or definition."
        ),
        "conceptual": (
            "CONCEPTUAL: a 'how' or 'why' question that asks the student to explain a "
            "relationship or mechanism (e.g. 'Why do gases spread out to fill a container?'). "
            "The answer is a short explanation."
        ),
        "applied": (
            "APPLIED: a question that asks the student to apply the idea to a concrete, everyday "
            "scenario or simple prediction (e.g. 'If you heat a sealed balloon, what happens to "
            "the gas inside and why?'). The answer applies the concept to the situation."
        ),
    },
    "templates": {
        "generation": (
            "You are creating a benchmark of questions a middle-school student (grades 6-8) "
            "might ask while learning science aligned to the Next Generation Science Standards.\n\n"
            "Science area: $area_title\n"
            "Standard $standard_code: $standard_text\n\n"
            "Write exactly $n DISTINCT questions of the following type:\n"
            "$question_type_def\n\n"
            "Requirements:\n"
            "- Each question must be something a curious middle-schooler would genuinely ask "
            "while learning this standard, and must be answerable with a short, factual "
            "reference answer.\n"
            "- Phrase questions in plain, middle-school-level language.\n"
            "- Provide a concise, correct reference answer (1-3 sentences) for each.\n"
            "- Avoid opinion, ambiguous, or multi-part questions. Avoid duplicates and "
            "near-duplicates.\n\n"
            "Return ONLY valid JSON in exactly this shape:\n"
            '{"items": [{"question": "...", "answer": "..."}, ...]}'
        ),
        "appropriateness": (
            "Decide whether the following question is appropriate for a middle-school student "
            '(grades 6-8) learning about "$area_title".\n\n'
            "Reject the question if it is: too advanced (high-school/college level), too trivial "
            "(below grade 6), off-topic for the area, ambiguous, or not answerable with a clear "
            "factual answer.\n\n"
            "Question: $question\n\n"
            'Return ONLY valid JSON: {"appropriate": true or false, "reason": "<short reason>"}'
        ),
        "gt_judge": (
            "You are validating a ground-truth question-and-answer pair for a middle-school "
            "science benchmark.\n\n"
            "Standard $standard_code: $standard_text\n"
            "Question: $question\n"
            "Proposed reference answer: $reference_answer\n\n"
            "Is the reference answer CORRECT and a complete enough answer to the question for "
            "this standard?\n"
            "Score 1 if the reference answer is correct, 0 if it is incorrect, misleading, or "
            "inadequate.\n\n"
            'Return ONLY valid JSON: {"score": 0 or 1, "rationale": "<one sentence>"}'
        ),
        "eval_judge_base": (
            "You are grading a candidate answer to a middle-school science question against a "
            "known-correct reference answer. Judge only correctness of the science content; "
            "ignore style, length, and the fact that wording differs from the reference.\n\n"
            "Question ($question_type): $question\n"
            "Reference (correct) answer: $reference_answer\n"
            "Candidate answer: $candidate_response\n\n"
            "Give a BINARY score: 1 if the candidate answer is correct (agrees with the "
            "reference and would be marked right), 0 if it is incorrect, off-topic, refuses, or "
            "is empty.$rubric_block\n"
            "Return ONLY valid JSON: $fields"
        ),
        "eval_judge_rubric_block": (
            "\nAlso give a 3-level rubric score: 1 = fully correct, 0.5 = partially correct "
            "(some correct content but incomplete or with a minor error), 0 = incorrect.\n"
        ),
        "eval_judge_fields_plain": '{"binary": 0 or 1, "rationale": "<one sentence>"}',
        "eval_judge_fields_rubric": (
            '{"binary": 0 or 1, "rubric": 0, 0.5, or 1, "rationale": "<one sentence>"}'
        ),
    },
}

# Placeholders each template must keep (used by the UI to validate edits).
REQUIRED_PLACEHOLDERS: dict[str, list[str]] = {
    "generation": ["area_title", "standard_code", "standard_text", "question_type_def", "n"],
    "appropriateness": ["area_title", "question"],
    "gt_judge": ["standard_code", "standard_text", "question", "reference_answer"],
    "eval_judge_base": ["question_type", "question", "reference_answer", "candidate_response",
                        "rubric_block", "fields"],
}


def _load() -> dict[str, dict[str, str]]:
    merged = deepcopy(_DEFAULTS)
    if PROMPTS_PATH.exists():
        data = yaml.safe_load(PROMPTS_PATH.read_text()) or {}
        for section in ("system_prompts", "question_types", "templates"):
            if isinstance(data.get(section), dict):
                merged[section].update(data[section])
    return merged


_P = _load()
SYSTEM_PROMPTS: dict[str, str] = _P["system_prompts"]
QUESTION_TYPES: dict[str, str] = _P["question_types"]
_T: dict[str, str] = _P["templates"]


def effective_templates() -> dict[str, dict[str, str]]:
    """The current merged prompts (defaults overlaid with data/prompts.yaml) — for the UI."""
    return deepcopy(_P)


def save_templates(data: dict[str, dict[str, str]]) -> None:
    """Persist edited prompts to data/prompts.yaml (the UI calls this)."""
    PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROMPTS_PATH.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True,
                                           width=100))


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def generation_prompt(area_title: str, standard_code: str, standard_text: str,
                      question_type: str, n: int) -> str:
    return Template(_T["generation"]).safe_substitute(
        area_title=area_title, standard_code=standard_code, standard_text=standard_text,
        question_type_def=QUESTION_TYPES[question_type], n=n)


def appropriateness_prompt(question: str, area_title: str) -> str:
    return Template(_T["appropriateness"]).safe_substitute(question=question, area_title=area_title)


def gt_judge_prompt(question: str, reference_answer: str, standard_code: str,
                    standard_text: str) -> str:
    return Template(_T["gt_judge"]).safe_substitute(
        question=question, reference_answer=reference_answer,
        standard_code=standard_code, standard_text=standard_text)


def eval_judge_prompt(question: str, reference_answer: str, candidate_response: str,
                      question_type: str, use_rubric: bool) -> str:
    rubric_block = _T["eval_judge_rubric_block"] if use_rubric else ""
    fields = _T["eval_judge_fields_rubric"] if use_rubric else _T["eval_judge_fields_plain"]
    return Template(_T["eval_judge_base"]).safe_substitute(
        question_type=question_type, question=question, reference_answer=reference_answer,
        candidate_response=candidate_response, rubric_block=rubric_block, fields=fields)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prompt utilities.")
    parser.add_argument("--export", action="store_true",
                        help="Write the current effective prompts to data/prompts.yaml.")
    args = parser.parse_args(argv)
    if args.export:
        save_templates(effective_templates())
        print(f"Wrote {PROMPTS_PATH.relative_to(REPO_ROOT)}")
    else:
        print("system prompts:", list(SYSTEM_PROMPTS))
        print("question types:", list(QUESTION_TYPES))
        print("templates:", list(_T))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

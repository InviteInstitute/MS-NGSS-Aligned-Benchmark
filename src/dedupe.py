"""Embedding-free near-duplicate detection for generated questions.

Uses rapidfuzz token-set ratio on normalized text. Two questions are near-duplicates if
their similarity is >= the configured threshold (0-100). Greedy: questions are added in
order and a candidate is rejected if it is too similar to any already-accepted question.
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "to", "in", "on",
    "and", "or", "for", "with", "what", "why", "how", "do", "does", "did", "can", "could",
    "would", "it", "this", "that", "as", "at", "by", "from", "you", "your",
}


def normalize(text: str) -> str:
    tokens = [t for t in _WORD.findall(text.lower()) if t not in _STOP]
    return " ".join(tokens)


def is_duplicate(candidate: str, accepted_norms: list[str], threshold: float) -> bool:
    norm = normalize(candidate)
    if not norm:
        return False
    return any(fuzz.token_set_ratio(norm, prev) >= threshold for prev in accepted_norms)


def dedupe(questions: list[str], threshold: float) -> tuple[list[int], list[int]]:
    """Return (kept_indices, dropped_indices) for a list of question strings."""
    kept: list[int] = []
    dropped: list[int] = []
    accepted_norms: list[str] = []
    for i, q in enumerate(questions):
        if is_duplicate(q, accepted_norms, threshold):
            dropped.append(i)
        else:
            kept.append(i)
            accepted_norms.append(normalize(q))
    return kept, dropped

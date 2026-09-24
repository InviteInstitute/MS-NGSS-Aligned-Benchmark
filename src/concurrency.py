"""Bounded parallel map with a progress bar.

Stages call `parallel_map(fn, items, max_workers=...)` to fan per-item API work across a
thread pool. Threads (not async) keep the codebase simple and work fine because the work is
I/O-bound on network calls. Results preserve input order.

Task functions are expected to be total — they should return a row/result rather than raise,
since the LLM client already converts API errors into ok=False results. A genuinely
unexpected exception is allowed to propagate (fail loud) rather than be silently swallowed.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Sequence, TypeVar

from tqdm import tqdm

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    fn: Callable[[T], R],
    items: Sequence[T] | Iterable[T],
    *,
    max_workers: int = 8,
    desc: str = "",
    on_result: Callable[[int, T, R], None] | None = None,
) -> list[R]:
    """Apply `fn` to each item across up to `max_workers` threads.

    If `on_result` is provided it is called (in completion order) as
    `on_result(index, item, result)` — handy for checkpointing each result as it lands.
    Returns results in the original input order.
    """
    items = list(items)
    results: list[R | None] = [None] * len(items)
    if not items:
        return []
    workers = max(1, int(max_workers))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for future in tqdm(
            as_completed(future_to_idx), total=len(items), desc=desc or "working", unit="item"
        ):
            idx = future_to_idx[future]
            result = future.result()
            results[idx] = result
            if on_result is not None:
                on_result(idx, items[idx], result)
    return results  # type: ignore[return-value]

"""
parallel.py

Thread-pool helpers for I/O-bound LLM evaluation loops.

Use --workers > 1 on cloud API providers (OpenAI, Groq, …) to keep several
requests in flight. CSV appends and API-key rotation are locked elsewhere so
resume/checkpointing stays safe.

No CLI arguments — pair with utilities.cli.add_workers_arg.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_parallel(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    workers: int = 1,
    on_result: Optional[Callable[[R], None]] = None,
) -> List[R]:
    """
    Map fn over items.

    workers <= 1 runs sequentially (same order as input).
    workers > 1 uses a thread pool; results are yielded as they complete
    (order not preserved). Exceptions from fn propagate immediately.
    """
    item_list = list(items)
    if not item_list:
        return []

    workers = max(1, int(workers))
    if workers == 1 or len(item_list) == 1:
        results: List[R] = []
        for item in item_list:
            result = fn(item)
            results.append(result)
            if on_result is not None:
                on_result(result)
        return results

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, item) for item in item_list]
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            if on_result is not None:
                on_result(result)
    return results

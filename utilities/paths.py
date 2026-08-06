"""
paths.py

Resolve the Evaluation Lab repo root and dataset directories from any script.

Functions
---------
  project_root(from_file=None)
      Absolute path to the repo root (directory containing utilities/ + benchmarks/).
  ensure_project_on_path(from_file=None)
      Insert that root on sys.path so `import utilities` works.
  dataset_dir(benchmark, *subpaths, from_file=None)
      Path under datasets/<benchmark>/… (e.g. dataset_dir("MMLU", "data", "data")).

No CLI arguments — library helper only.
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def project_root(from_file: Optional[str] = None) -> str:
    """
    Absolute path to the evaluation_lab repo root.

    If from_file is given (typically __file__ of a caller), walks up until a
    directory containing both `utilities/` and `benchmarks/` is found.
    Otherwise uses this package's parent directory.
    """
    if from_file is None:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    search_dir = os.path.dirname(os.path.abspath(from_file))
    while True:
        if os.path.isdir(os.path.join(search_dir, "utilities")) and os.path.isdir(
            os.path.join(search_dir, "benchmarks")
        ):
            return search_dir
        parent = os.path.dirname(search_dir)
        if parent == search_dir:
            # Fallback: three levels up from typical benchmarks/<X>/experiments/ script
            return os.path.abspath(os.path.join(os.path.dirname(from_file), "..", "..", ".."))
        search_dir = parent


def ensure_project_on_path(from_file: Optional[str] = None) -> str:
    """Insert the repo root on sys.path so `import utilities` works from scripts."""
    root = project_root(from_file)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def dataset_dir(benchmark: str, *subpaths: str, from_file: Optional[str] = None) -> str:
    """
    Path under datasets/<benchmark>/.

    Example:
        dataset_dir("MMLU", "data", "data")
        -> <root>/datasets/MMLU/data/data
    """
    return os.path.join(project_root(from_file), "datasets", benchmark, *subpaths)

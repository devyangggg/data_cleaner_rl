"""Feature extraction from OpenEnv observations."""

from __future__ import annotations

from typing import Any

import numpy as np


def featurize_observation(observation: dict[str, Any], max_steps: int = 10) -> np.ndarray:
    """Convert observation into a fixed-size numeric feature vector."""
    step_count = float(observation.get("step_count", 0))
    step_ratio = step_count / float(max_steps)

    diff_items = observation.get("diff_items", []) or []
    dtype_mismatch = sum(1 for item in diff_items if item.get("type") == "dtype_mismatch")
    null_mismatch = sum(1 for item in diff_items if item.get("type") == "null_count_mismatch")
    value_mismatch = sum(1 for item in diff_items if item.get("type") == "value_mismatch")
    missing_cols = sum(1 for item in diff_items if item.get("type") == "missing_column")
    unexpected_cols = sum(1 for item in diff_items if item.get("type") == "unexpected_column")

    revenue_dtype_bad = 0.0
    units_null_bad = 0.0
    for item in diff_items:
        if item.get("type") == "dtype_mismatch" and item.get("column") == "revenue":
            revenue_dtype_bad = 1.0
        if item.get("type") == "null_count_mismatch" and item.get("column") == "units_sold":
            units_null_bad = 1.0

    summary = str(observation.get("diff_summary", "")).lower()
    date_signal = 1.0 if "value mismatch" in summary else 0.0
    terminal = 1.0 if observation.get("is_terminal", False) else 0.0

    return np.asarray(
        [
            step_ratio,
            float(dtype_mismatch),
            float(null_mismatch),
            float(value_mismatch),
            float(missing_cols),
            float(unexpected_cols),
            revenue_dtype_bad,
            units_null_bad,
            date_signal,
            terminal,
        ],
        dtype=np.float32,
    )

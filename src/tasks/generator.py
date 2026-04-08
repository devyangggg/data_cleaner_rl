"""Procedural task generation for evaluating generalization."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _base_dataframe(rng: np.random.RandomState, rows: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"ID-{i:04d}" for i in range(rows)],
            "value": rng.uniform(10, 1000, size=rows).round(2),
            "count": rng.randint(1, 100, size=rows).astype(float),
            "category": rng.choice(["A", "B", "C"], size=rows),
        }
    )


def generate_task(seed: int, difficulty: str) -> dict[str, Any]:
    """Generate a novel broken task from a seed and difficulty."""
    rng = np.random.RandomState(seed)
    expected_df = _base_dataframe(rng)
    broken_df = expected_df.copy()
    bug_types: list[str] = []

    if difficulty == "easy":
        choices = ["dtype_mismatch", "single_null_column", "wrong_column_name"]
        selected = [rng.choice(choices)]
    elif difficulty == "medium":
        choices = ["dtype_mismatch", "null_values", "bad_join_key", "wrong_aggregation"]
        selected = list(rng.choice(choices, size=2, replace=False))
    elif difficulty == "hard":
        choices = [
            "duplicate_rows",
            "mixed_dtypes",
            "cascading_join_error",
            "dtype_mismatch",
            "null_values",
        ]
        selected = list(rng.choice(choices, size=3, replace=False))
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    for bug in selected:
        bug_types.append(str(bug))
        if bug == "dtype_mismatch":
            broken_df["value"] = broken_df["value"].astype(str)
        elif bug in ("single_null_column", "null_values"):
            idx = rng.choice(len(broken_df), size=max(5, len(broken_df) // 10), replace=False)
            broken_df.loc[idx, "count"] = np.nan
            expected_df.loc[idx, "count"] = 0.0
        elif bug == "wrong_column_name":
            broken_df = broken_df.rename(columns={"category": "segment"})
        elif bug == "duplicate_rows":
            dup = broken_df.sample(n=8, random_state=rng).copy()
            broken_df = pd.concat([broken_df, dup], ignore_index=True)
        elif bug == "mixed_dtypes":
            idx = rng.choice(len(broken_df), size=8, replace=False)
            broken_df.loc[idx, "count"] = broken_df.loc[idx, "count"].astype(int).astype(str)
        elif bug == "cascading_join_error":
            broken_df["id"] = broken_df["id"].str.replace("ID-", "IDX-", regex=False)
        elif bug == "wrong_aggregation":
            broken_df["value"] = (broken_df["value"] * 1.1).round(2)
            expected_df["value"] = expected_df["value"].round(2)
        elif bug == "bad_join_key":
            broken_df["category"] = np.nan

    description = f"Procedural {difficulty} task with bugs: {', '.join(bug_types)}"
    return {
        "broken_df": broken_df,
        "expected_df": expected_df,
        "bug_types": bug_types,
        "description": description,
    }

"""Deterministic reward computation for the Pipeline Debug Environment."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_reward(
    current_df: pd.DataFrame,
    expected_df: pd.DataFrame,
    action_result: str,
    step: int,
    max_steps: int,
) -> float:
    """Compute the reward after an agent action.

    Breakdown
    ---------
    - Schema score: 0.0 – 0.4  (fraction of correctly-typed columns)
    - Value score:  0.0 – 0.4  (fraction of rows with all values matching)
    - Efficiency:   0.0 – 0.2  (remaining steps ratio)
    - Penalties:    -0.05 for invalid, -0.02 for no_change
    """
    total_columns = len(expected_df.columns)
    if total_columns == 0:
        schema_score = 0.4
    else:
        matching_cols = 0
        for col in expected_df.columns:
            if col in current_df.columns:
                if str(current_df[col].dtype) == str(expected_df[col].dtype):
                    matching_cols += 1
        schema_score = (matching_cols / total_columns) * 0.4

    # Value score — a row is correct if ALL values match within tolerance 1e-2
    total_rows = len(expected_df)
    if total_rows == 0:
        value_score = 0.4
    else:
        correct_rows = 0
        # Align columns: only score columns present in expected
        shared_cols = [c for c in expected_df.columns if c in current_df.columns]
        if len(shared_cols) != len(expected_df.columns) or len(current_df) != len(expected_df):
            value_score = 0.0
        else:
            for i in range(total_rows):
                row_ok = True
                for col in shared_cols:
                    exp_val = expected_df[col].iloc[i]
                    try:
                        cur_val = current_df[col].iloc[i]
                    except (IndexError, KeyError):
                        row_ok = False
                        break

                    # Compare with tolerance for numerics
                    try:
                        if pd.isna(exp_val) and pd.isna(cur_val):
                            continue
                        if pd.isna(exp_val) or pd.isna(cur_val):
                            row_ok = False
                            break
                        if isinstance(exp_val, (int, float, np.integer, np.floating)):
                            if abs(float(cur_val) - float(exp_val)) > 1e-2:
                                row_ok = False
                                break
                        else:
                            if str(cur_val) != str(exp_val):
                                row_ok = False
                                break
                    except (ValueError, TypeError):
                        if str(cur_val) != str(exp_val):
                            row_ok = False
                            break

                if row_ok:
                    correct_rows += 1
            value_score = (correct_rows / total_rows) * 0.4

    # Efficiency score
    efficiency_score = ((max_steps - step) / max_steps) * 0.2

    total = schema_score + value_score + efficiency_score

    # Penalties
    if action_result == "invalid":
        total -= 0.05
    if action_result == "no_change":
        total -= 0.02

    return round(total, 4)

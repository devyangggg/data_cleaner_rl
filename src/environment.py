"""PipelineDebugEnv — the core RL environment for data pipeline debugging."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.graders.grader import compute_reward
from src.models import Action, Observation
from src.tasks import TASK_REGISTRY


# Commands and their required parameter keys
VALID_COMMANDS: dict[str, list[str]] = {
    "cast_column": ["column", "dtype"],
    "fill_nulls": ["column", "value"],
    "fix_date_format": ["column", "format"],
    "fix_join": ["left_key", "right_key"],
    "drop_duplicates": ["subset"],
    "apply_transform": ["column", "expression"],
    "revert_step": [],
}


def _diff_summary(current_df: pd.DataFrame, expected_df: pd.DataFrame) -> str:
    """Build a human-readable diff summary."""
    diffs: list[str] = []

    # Row count
    if len(current_df) != len(expected_df):
        diffs.append(f"Row count: {len(current_df)} (expected {len(expected_df)})")

    # Column differences
    cur_cols = set(current_df.columns)
    exp_cols = set(expected_df.columns)
    if cur_cols != exp_cols:
        missing = exp_cols - cur_cols
        extra = cur_cols - exp_cols
        if missing:
            diffs.append(f"Missing columns: {sorted(missing)}")
        if extra:
            diffs.append(f"Extra columns: {sorted(extra)}")

    # Dtype mismatches
    for col in expected_df.columns:
        if col in current_df.columns:
            if str(current_df[col].dtype) != str(expected_df[col].dtype):
                diffs.append(
                    f"Column '{col}' dtype: {current_df[col].dtype} "
                    f"(expected {expected_df[col].dtype})"
                )

    # Null counts
    for col in expected_df.columns:
        if col in current_df.columns:
            cur_nulls = int(current_df[col].isna().sum())
            exp_nulls = int(expected_df[col].isna().sum())
            if cur_nulls != exp_nulls:
                diffs.append(
                    f"Column '{col}' nulls: {cur_nulls} (expected {exp_nulls})"
                )

    # Value mismatches sample
    shared_cols = [c for c in expected_df.columns if c in current_df.columns]
    if len(current_df) == len(expected_df) and shared_cols:
        mismatch_count = 0
        for col in shared_cols:
            try:
                for i in range(len(expected_df)):
                    ev = expected_df[col].iloc[i]
                    cv = current_df[col].iloc[i]
                    if pd.isna(ev) and pd.isna(cv):
                        continue
                    if pd.isna(ev) or pd.isna(cv):
                        mismatch_count += 1
                        continue
                    if isinstance(ev, (int, float, np.integer, np.floating)):
                        if abs(float(cv) - float(ev)) > 1e-2:
                            mismatch_count += 1
                    else:
                        if str(cv) != str(ev):
                            mismatch_count += 1
            except (ValueError, TypeError):
                mismatch_count += 1
        if mismatch_count > 0:
            diffs.append(f"Value mismatches: {mismatch_count} cells differ")

    return "; ".join(diffs) if diffs else "Dataframes match perfectly"


class PipelineDebugEnv:
    """OpenEnv-compliant environment for data pipeline debugging."""

    def __init__(self, task_id: str = "easy") -> None:
        self.task_id = task_id
        self.task = TASK_REGISTRY[task_id]()

        self.current_df: pd.DataFrame = pd.DataFrame()
        self.expected_df: pd.DataFrame = pd.DataFrame()
        self.prev_df: pd.DataFrame | None = None
        self.step_count: int = 0
        self.last_action_result: str = "success"
        self.last_score: float = 0.0
        self.done: bool = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""
        self.task = TASK_REGISTRY[self.task_id]()
        self.current_df = self.task.get_broken_df()
        self.expected_df = self.task.get_expected_df()
        self.prev_df = None
        self.step_count = 0
        self.last_action_result = "success"
        self.last_score = 0.0
        self.done = False
        return self._make_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Execute one action and return (observation, reward, terminated, info)."""
        if self.done:
            return self._make_observation(), self.last_score, True, {"message": "Environment already terminated"}

        self.step_count += 1

        # Validate command
        if action.command not in VALID_COMMANDS:
            self.last_action_result = "invalid"
        else:
            required = VALID_COMMANDS[action.command]
            if not all(k in action.parameters for k in required):
                self.last_action_result = "invalid"
            else:
                # Save state for revert
                old_df = self.current_df.copy()

                # Apply command
                try:
                    self._apply_command(action)
                except Exception:
                    self.last_action_result = "invalid"
                else:
                    # Check if anything changed
                    if old_df.equals(self.current_df):
                        self.last_action_result = "no_change"
                        self.prev_df = old_df
                    else:
                        self.last_action_result = "success"
                        self.prev_df = old_df

        # Compute reward
        reward = compute_reward(
            self.current_df,
            self.expected_df,
            self.last_action_result,
            self.step_count,
            self.task.max_steps,
        )
        delta = round(reward - self.last_score, 4)
        self.last_score = reward

        # Check termination
        terminated = False
        if self._dataframes_match():
            terminated = True
        elif self.step_count >= self.task.max_steps:
            terminated = True
        self.done = terminated

        info = {
            "reward": reward,
            "delta": delta,
            "reason": self._reward_reason(),
            "step": self.step_count,
            "max_steps": self.task.max_steps,
        }

        return self._make_observation(), reward, terminated, info

    def state(self) -> dict[str, Any]:
        """Return raw environment state for debugging."""
        return {
            "task_id": self.task_id,
            "step": self.step_count,
            "done": self.done,
            "last_score": self.last_score,
            "current_df_shape": list(self.current_df.shape),
            "expected_df_shape": list(self.expected_df.shape),
            "current_df_dtypes": {
                col: str(self.current_df[col].dtype) for col in self.current_df.columns
            },
            "current_df_preview": self.current_df.head(5).to_dict(orient="records"),
        }

    # ------------------------------------------------------------------
    # Command application
    # ------------------------------------------------------------------

    def _apply_command(self, action: Action) -> None:
        params = action.parameters

        if action.command == "cast_column":
            col = params["column"]
            dtype = params["dtype"]
            if col not in self.current_df.columns:
                raise ValueError(f"Column '{col}' not found")
            self.current_df[col] = self.current_df[col].astype(dtype)

        elif action.command == "fill_nulls":
            col = params["column"]
            value = params["value"]
            if col not in self.current_df.columns:
                raise ValueError(f"Column '{col}' not found")
            self.current_df[col] = self.current_df[col].fillna(value)

        elif action.command == "fix_date_format":
            col = params["column"]
            fmt = params["format"]
            if col not in self.current_df.columns:
                raise ValueError(f"Column '{col}' not found")
            self.current_df[col] = pd.to_datetime(
                self.current_df[col], format=fmt, dayfirst=True
            ).dt.strftime("%Y-%m-%d")

        elif action.command == "fix_join":
            left_key = params["left_key"]
            right_key = params["right_key"]
            # Re-execute the join using source tables from the task
            if not hasattr(self.task, "orders_df") or not hasattr(self.task, "customers_df"):
                raise ValueError("fix_join only available for tasks with source tables")
            self.current_df = self.task.orders_df.merge(
                self.task.customers_df,
                left_on=left_key,
                right_on=right_key,
                how="left",
            )

        elif action.command == "drop_duplicates":
            subset = params["subset"]
            self.current_df = self.current_df.drop_duplicates(subset=subset).reset_index(drop=True)

        elif action.command == "apply_transform":
            col = params["column"]
            expression = params["expression"]
            if col not in self.current_df.columns:
                raise ValueError(f"Column '{col}' not found")
            # expression uses 'x' as the cell variable and 'row' for the row
            # Safe eval with restricted namespace
            self.current_df[col] = self.current_df.apply(
                lambda row: eval(  # noqa: S307
                    expression,
                    {"__builtins__": {}},
                    {"x": row[col], "row": row, "pd": pd, "np": np},
                ),
                axis=1,
            )

        elif action.command == "revert_step":
            if self.prev_df is not None:
                self.current_df = self.prev_df.copy()
                self.prev_df = None
            else:
                raise ValueError("No previous state to revert to")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_observation(self) -> Observation:
        preview_df = self.current_df.head(5).copy()
        # Convert NaN to None for JSON serialization
        preview = preview_df.where(preview_df.notna(), None).to_dict(orient="records")

        return Observation(
            task_id=self.task_id,
            step=self.step_count,
            schema={col: str(self.current_df[col].dtype) for col in self.current_df.columns},
            preview=preview,
            expected_schema={col: str(self.expected_df[col].dtype) for col in self.expected_df.columns},
            diff_summary=_diff_summary(self.current_df, self.expected_df),
            last_action_result=self.last_action_result,
        )

    def _dataframes_match(self) -> bool:
        """Check if current_df matches expected_df within tolerance."""
        if list(self.current_df.columns) != list(self.expected_df.columns):
            return False
        if len(self.current_df) != len(self.expected_df):
            return False
        for col in self.expected_df.columns:
            if str(self.current_df[col].dtype) != str(self.expected_df[col].dtype):
                return False

        for col in self.expected_df.columns:
            for i in range(len(self.expected_df)):
                ev = self.expected_df[col].iloc[i]
                cv = self.current_df[col].iloc[i]
                if pd.isna(ev) and pd.isna(cv):
                    continue
                if pd.isna(ev) or pd.isna(cv):
                    return False
                if isinstance(ev, (int, float, np.integer, np.floating)):
                    if abs(float(cv) - float(ev)) > 1e-2:
                        return False
                else:
                    if str(cv) != str(ev):
                        return False
        return True

    def _reward_reason(self) -> str:
        if self.last_action_result == "invalid":
            return "Invalid command or missing parameters"
        if self.last_action_result == "no_change":
            return "Action produced no change"
        if self._dataframes_match():
            return "All bugs fixed — dataframe matches expected output"
        return "Progress made toward expected output"

"""PipelineDebugEnv — the core RL environment for data pipeline debugging."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
from typing import Any

import numpy as np
import pandas as pd

from src.graders.grader import compute_reward
from src.models import EpisodeState, PipelineAction, PipelineObservation
from src.tasks import TASK_REGISTRY


# Commands and their required parameter keys
VALID_COMMANDS: dict[str, list[str]] = {
    "cast_column": ["column", "dtype"],
    "fill_nulls": ["column", "value"],
    "fix_date_format": ["column", "format"],
    "fix_join": ["left_key", "right_key"],
    "drop_duplicates": ["subset"],
    "apply_transform": ["column"],
    "revert_step": [],
    "rename_column": ["old_name", "new_name"],
    "drop_column": ["column"],
    "sort_values": ["by"],
}


def _diff_summary(current_df: pd.DataFrame, expected_df: pd.DataFrame) -> str:
    """Build a consistent, parseable diff summary for zero-shot agents."""
    diffs: list[str] = []

    # Row count
    if len(current_df) != len(expected_df):
        diffs.append(
            f"Row count mismatch: current {len(current_df)}, expected {len(expected_df)}."
        )

    # Column differences
    cur_cols = set(current_df.columns)
    exp_cols = set(expected_df.columns)
    if cur_cols != exp_cols:
        missing = exp_cols - cur_cols
        extra = cur_cols - exp_cols
        if missing:
            diffs.append(
                "Missing columns: " + ", ".join(f"'{c}'" for c in sorted(missing)) + "."
            )
        if extra:
            diffs.append(
                "Unexpected columns: " + ", ".join(f"'{c}'" for c in sorted(extra)) + "."
            )

    # Dtype mismatches
    for col in expected_df.columns:
        if col in current_df.columns:
            if str(current_df[col].dtype) != str(expected_df[col].dtype):
                diffs.append(
                    f"Column '{col}': dtype is {current_df[col].dtype}, "
                    f"expected {expected_df[col].dtype}."
                )

    # Null counts
    for col in expected_df.columns:
        if col in current_df.columns:
            cur_nulls = int(current_df[col].isna().sum())
            exp_nulls = int(expected_df[col].isna().sum())
            if cur_nulls != exp_nulls:
                diffs.append(
                    f"Column '{col}': null count is {cur_nulls}, expected {exp_nulls}."
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
            diffs.append(f"Value mismatch: {mismatch_count} cell values differ from expected.")

    return " ".join(diffs) if diffs else "Dataframes match perfectly."


def _diff_items(current_df: pd.DataFrame, expected_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build machine-readable diffs for robust agent parsing."""
    items: list[dict[str, Any]] = []

    if len(current_df) != len(expected_df):
        items.append(
            {
                "type": "row_count_mismatch",
                "current": len(current_df),
                "expected": len(expected_df),
            }
        )

    cur_cols = set(current_df.columns)
    exp_cols = set(expected_df.columns)
    missing = sorted(exp_cols - cur_cols)
    extra = sorted(cur_cols - exp_cols)
    for col in missing:
        items.append({"type": "missing_column", "column": col})
    for col in extra:
        items.append({"type": "unexpected_column", "column": col})

    for col in expected_df.columns:
        if col in current_df.columns:
            cur_dtype = str(current_df[col].dtype)
            exp_dtype = str(expected_df[col].dtype)
            if cur_dtype != exp_dtype:
                items.append(
                    {
                        "type": "dtype_mismatch",
                        "column": col,
                        "current": cur_dtype,
                        "expected": exp_dtype,
                    }
                )

    for col in expected_df.columns:
        if col in current_df.columns:
            cur_nulls = int(current_df[col].isna().sum())
            exp_nulls = int(expected_df[col].isna().sum())
            if cur_nulls != exp_nulls:
                items.append(
                    {
                        "type": "null_count_mismatch",
                        "column": col,
                        "current": cur_nulls,
                        "expected": exp_nulls,
                    }
                )

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
            items.append({"type": "value_mismatch", "cells": mismatch_count})

    return items


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
        self.last_rationale: str = "Environment reset"
        self.last_score: float = 0.0
        self.done: bool = False
        self.episode_id: str = ""
        self.started_at: str = ""

        self.command_registry: dict[str, Any] = {
            "cast_column": self._cmd_cast_column,
            "fill_nulls": self._cmd_fill_nulls,
            "fix_date_format": self._cmd_fix_date_format,
            "fix_join": self._cmd_fix_join,
            "drop_duplicates": self._cmd_drop_duplicates,
            "apply_transform": self._cmd_apply_transform,
            "revert_step": self._cmd_revert_step,
            "rename_column": self._cmd_rename_column,
            "drop_column": self._cmd_drop_column,
            "sort_values": self._cmd_sort_values,
        }

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> PipelineObservation:
        """Reset the environment and return the initial observation."""
        self.task = TASK_REGISTRY[self.task_id]()
        self.current_df = self.task.get_broken_df()
        self.expected_df = self.task.get_expected_df()
        self.prev_df = None
        self.step_count = 0
        self.last_action_result = "success"
        self.last_rationale = "Environment reset"
        self.last_score = 0.0
        self.done = False
        self.episode_id = str(uuid4())
        self.started_at = datetime.now(timezone.utc).isoformat()
        return self._make_observation()

    def step(self, action: PipelineAction) -> tuple[PipelineObservation, float, bool, dict[str, Any]]:
        """Execute one action and return (observation, reward, terminated, info)."""
        if self.done:
            return self._make_observation(), self.last_score, True, {"message": "Environment already terminated"}

        self.step_count += 1

        # Validate command
        if action.command not in VALID_COMMANDS:
            self.last_action_result = "invalid"
        else:
            required = VALID_COMMANDS[action.command]
            if not all(k in action.params for k in required):
                self.last_action_result = "invalid"
                self.last_rationale = "Missing required parameters"
            else:
                # Save state for revert
                old_df = self.current_df.copy()

                # Apply command
                try:
                    self._apply_command(action)
                except Exception as exc:
                    self.last_action_result = "invalid"
                    self.last_rationale = f"Command failed: {exc}"
                else:
                    # Check if anything changed
                    if old_df.equals(self.current_df):
                        self.last_action_result = "no_change"
                        self.prev_df = old_df
                        self.last_rationale = "Action executed but dataframe did not change"
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
            "delta_reward": delta,
            "reason": self._reward_reason(),
            "action_result": self.last_action_result,
            "rationale": self.last_rationale,
            "step": self.step_count,
            "max_steps": self.task.max_steps,
        }

        return self._make_observation(), reward, terminated, info

    def state(self) -> dict[str, Any]:
        """Return raw environment state for debugging."""
        episode = EpisodeState(
            episode_id=self.episode_id,
            task_id=self.task_id,
            step_count=self.step_count,
            total_reward=self.last_score,
            started_at=self.started_at,
            done=self.done,
        )
        payload = episode.model_dump()
        payload.update(
            {
                "current_df_shape": list(self.current_df.shape),
                "expected_df_shape": list(self.expected_df.shape),
                "current_df_dtypes": {
                    col: str(self.current_df[col].dtype) for col in self.current_df.columns
                },
                "current_df_preview": self.current_df.head(5).to_dict(orient="records"),
            }
        )
        return payload

    # ------------------------------------------------------------------
    # Command application
    # ------------------------------------------------------------------

    def _apply_command(self, action: PipelineAction) -> None:
        handler = self.command_registry.get(action.command)
        if handler is None:
            raise ValueError(f"Unknown command: {action.command}")
        handler(action.params)

    def _cmd_cast_column(self, params: dict[str, Any]) -> None:
        col = params["column"]
        dtype = params["dtype"]
        if col not in self.current_df.columns:
            raise ValueError(f"Column '{col}' not found")
        before = str(self.current_df[col].dtype)
        self.current_df[col] = self.current_df[col].astype(dtype)
        self.last_rationale = f"cast_column applied: {col} dtype changed from {before} to {self.current_df[col].dtype}"

    def _cmd_fill_nulls(self, params: dict[str, Any]) -> None:
        col = params["column"]
        value = params["value"]
        if col not in self.current_df.columns:
            raise ValueError(f"Column '{col}' not found")
        before = int(self.current_df[col].isna().sum())
        self.current_df[col] = self.current_df[col].fillna(value)
        after = int(self.current_df[col].isna().sum())
        self.last_rationale = f"fill_nulls applied: {col} nulls reduced from {before} to {after}"

    def _cmd_fix_date_format(self, params: dict[str, Any]) -> None:
        col = params["column"]
        fmt = params["format"]
        if col not in self.current_df.columns:
            raise ValueError(f"Column '{col}' not found")
        self.current_df[col] = pd.to_datetime(
            self.current_df[col], format=fmt, dayfirst=True
        ).dt.strftime("%Y-%m-%d")
        self.last_rationale = f"fix_date_format applied: {col} parsed with format {fmt}"

    def _cmd_fix_join(self, params: dict[str, Any]) -> None:
        left_key = params["left_key"]
        right_key = params["right_key"]
        if not hasattr(self.task, "orders_df") or not hasattr(self.task, "customers_df"):
            raise ValueError("fix_join only available for tasks with source tables")
        self.current_df = self.task.orders_df.merge(
            self.task.customers_df,
            left_on=left_key,
            right_on=right_key,
            how="left",
        )
        self.last_rationale = f"fix_join applied: merged orders using {left_key} -> {right_key}"

    def _cmd_drop_duplicates(self, params: dict[str, Any]) -> None:
        subset = params["subset"]
        before = len(self.current_df)
        self.current_df = self.current_df.drop_duplicates(subset=subset).reset_index(drop=True)
        after = len(self.current_df)
        self.last_rationale = f"drop_duplicates applied: rows reduced from {before} to {after}"

    def _cmd_apply_transform(self, params: dict[str, Any]) -> None:
        col = params["column"]
        if col not in self.current_df.columns:
            raise ValueError(f"Column '{col}' not found")

        if "expression" in params:
            expr = str(params["expression"]).strip()
            expected_expr = "row['converted_amount'] / 1.23 if row['currency'] == 'USD' else x"
            expected_expr_alt = 'row["converted_amount"] / 1.23 if row["currency"] == "USD" else x'
            if expr not in (expected_expr, expected_expr_alt):
                raise ValueError("Unsupported expression for apply_transform")
            self.current_df[col] = self.current_df.apply(
                lambda row: row[col] / 1.23 if row["currency"] == "USD" else row[col],
                axis=1,
            )
            self.last_rationale = "apply_transform applied: corrected USD rows by reversing duplicated FX conversion"
            return

        op = params.get("op", "identity")
        value = params.get("value")
        cond_col = params.get("condition_column")
        cond_eq = params.get("condition_equals")

        mask = pd.Series([True] * len(self.current_df))
        if cond_col is not None:
            if cond_col not in self.current_df.columns:
                raise ValueError(f"Column '{cond_col}' not found")
            mask = self.current_df[cond_col] == cond_eq

        if op == "div":
            self.current_df.loc[mask, col] = self.current_df.loc[mask, col] / float(value)
        elif op == "mul":
            self.current_df.loc[mask, col] = self.current_df.loc[mask, col] * float(value)
        elif op == "add":
            self.current_df.loc[mask, col] = self.current_df.loc[mask, col] + float(value)
        elif op == "sub":
            self.current_df.loc[mask, col] = self.current_df.loc[mask, col] - float(value)
        elif op == "round":
            self.current_df.loc[mask, col] = self.current_df.loc[mask, col].round(int(value))
        else:
            raise ValueError(f"Unsupported operation: {op}")

        self.last_rationale = f"apply_transform applied: operation '{op}' executed on column {col}"

    def _cmd_revert_step(self, params: dict[str, Any]) -> None:  # noqa: ARG002
        if self.prev_df is not None:
            self.current_df = self.prev_df.copy()
            self.prev_df = None
            self.last_rationale = "revert_step applied: restored previous dataframe state"
        else:
            raise ValueError("No previous state to revert to")

    def _cmd_rename_column(self, params: dict[str, Any]) -> None:
        old_name = params["old_name"]
        new_name = params["new_name"]
        if old_name not in self.current_df.columns:
            raise ValueError(f"Column '{old_name}' not found")
        self.current_df = self.current_df.rename(columns={old_name: new_name})
        self.last_rationale = f"rename_column applied: {old_name} -> {new_name}"

    def _cmd_drop_column(self, params: dict[str, Any]) -> None:
        col = params["column"]
        if col not in self.current_df.columns:
            raise ValueError(f"Column '{col}' not found")
        self.current_df = self.current_df.drop(columns=[col])
        self.last_rationale = f"drop_column applied: removed {col}"

    def _cmd_sort_values(self, params: dict[str, Any]) -> None:
        by = params["by"]
        ascending = bool(params.get("ascending", True))
        self.current_df = self.current_df.sort_values(by=by, ascending=ascending).reset_index(drop=True)
        self.last_rationale = f"sort_values applied: sorted by {by}, ascending={ascending}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_observation(self) -> PipelineObservation:
        preview_df = self.current_df.head(5).copy()
        # Convert NaN to None for JSON serialization
        preview = preview_df.where(preview_df.notna(), None).to_dict(orient="records")

        return PipelineObservation(
            task_id=self.task_id,
            current_schema={col: str(self.current_df[col].dtype) for col in self.current_df.columns},
            expected_schema={col: str(self.expected_df[col].dtype) for col in self.expected_df.columns},
            preview_rows=preview,
            diff_summary=_diff_summary(self.current_df, self.expected_df),
            diff_items=_diff_items(self.current_df, self.expected_df),
            step_count=self.step_count,
            is_terminal=self.done,
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

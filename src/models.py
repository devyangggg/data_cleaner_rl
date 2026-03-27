"""Pydantic models for the Pipeline Debug Environment."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """What the agent sees after each step."""

    task_id: str
    step: int
    schema: dict[str, str] = Field(
        description="Column name → dtype string for the current dataframe"
    )
    preview: list[dict[str, Any]] = Field(
        description="First 5 rows of the current dataframe"
    )
    expected_schema: dict[str, str] = Field(
        description="Column name → dtype string for the expected dataframe"
    )
    diff_summary: str = Field(
        description="Human-readable summary of differences between current and expected"
    )
    last_action_result: str = Field(
        default="success",
        description="One of: success, invalid, no_change",
    )


class Action(BaseModel):
    """A single fix command issued by the agent."""

    command: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class Reward(BaseModel):
    """Reward signal returned after each step."""

    score: float = Field(ge=0.0, le=1.0)
    delta: float = Field(description="Change from last step's score")
    reason: str

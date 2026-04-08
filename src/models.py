"""Pydantic models for OpenEnv-compatible pipeline debugging APIs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class PipelineAction(BaseModel):
    """Single command emitted by an agent."""

    command: str
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def support_legacy_parameters_key(cls, data: Any) -> Any:
        if isinstance(data, dict) and "params" not in data and "parameters" in data:
            data = dict(data)
            data["params"] = data.get("parameters", {})
        return data


class PipelineObservation(BaseModel):
    """Structured observation returned by reset/step."""

    task_id: str
    current_schema: dict[str, str]
    expected_schema: dict[str, str]
    preview_rows: list[dict[str, Any]]
    diff_summary: str
    diff_items: list[dict[str, Any]] = Field(default_factory=list)
    step_count: int
    is_terminal: bool
    last_action_result: str = "success"


class StepResult(BaseModel):
    """OpenEnv-style step response."""

    observation: PipelineObservation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class EpisodeState(BaseModel):
    """Serializable state snapshot for debugging and clients."""

    episode_id: str
    task_id: str
    step_count: int
    total_reward: float
    started_at: str
    done: bool


class CurriculumStats(BaseModel):
    current_level: str
    recent_history: list[dict[str, Any]] = Field(default_factory=list)

"""FastAPI server exposing the Pipeline Debug Environment."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query

from src.environment import PipelineDebugEnv
from src.models import Action, Observation

app = FastAPI(
    title="Pipeline Debug Environment",
    version="0.1.0",
    description="OpenEnv-compliant RL environment for data pipeline debugging",
)

# Single global environment instance
env: PipelineDebugEnv | None = None


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for HF Space ping."""
    return {"status": "ok"}


@app.post("/reset", response_model=Observation)
def reset(task_id: str = Query(default="easy")) -> Observation:
    """Reset the environment with the given task and return initial observation."""
    global env
    if task_id not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")
    env = PipelineDebugEnv(task_id=task_id)
    return env.reset()


@app.post("/step")
def step(action: Action) -> dict[str, Any]:
    """Execute an action and return observation, reward, terminated, truncated, info."""
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    observation, reward, terminated, info = env.step(action)

    return {
        "observation": observation.model_dump(),
        "reward": reward,
        "terminated": terminated,
        "truncated": False,
        "info": info,
    }


@app.get("/state")
def get_state() -> dict[str, Any]:
    """Return the current raw environment state."""
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return env.state()

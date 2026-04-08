"""Canonical FastAPI server exposing the OpenEnv-compatible API."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from src.curriculum import CurriculumManager
from src.environment import VALID_COMMANDS, PipelineDebugEnv
from src.models import PipelineAction, PipelineObservation, StepResult
from src.tasks import TASK_REGISTRY

PORT = int(os.environ.get("PORT", 7860))

app = FastAPI(
    title="Data Cleaner RL Environment",
    version="1.0.0",
    description="OpenEnv-compatible RL environment for data pipeline debugging",
)

env: PipelineDebugEnv | None = None
curriculum = CurriculumManager()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return {
        "name": "data-cleaner-rl",
        "version": "1.0.0",
        "description": "RL environment for debugging broken pandas data pipelines",
        "tasks": list(TASK_REGISTRY.keys()),
        "action_space": {
            "type": "discrete_parameterized",
            "commands": [
                {"name": k, "required_params": v} for k, v in VALID_COMMANDS.items()
            ],
        },
    }


@app.get("/schema")
def schema() -> dict[str, Any]:
    return {
        "action": PipelineAction.model_json_schema(),
        "observation": PipelineObservation.model_json_schema(),
        "step_result": StepResult.model_json_schema(),
    }


@app.get("/mcp")
def mcp() -> dict[str, Any]:
    return {
        "name": "data-cleaner-rl",
        "tools": [
            {
                "name": "step",
                "description": "Apply one dataframe repair action",
                "input_schema": PipelineAction.model_json_schema(),
            },
            {
                "name": "reset",
                "description": "Reset an episode for a given task",
                "input_schema": {
                    "type": "object",
                    "properties": {"task_id": {"type": "string"}},
                },
            },
            {
                "name": "state",
                "description": "Read current environment state",
                "input_schema": {"type": "object", "properties": {}},
            },
        ],
    }


@app.get("/web", response_class=HTMLResponse)
def web() -> str:
    return """
    <html><body>
      <h2>Data Cleaner RL Environment</h2>
      <p>Use <code>/reset</code>, <code>/step</code>, and <code>/state</code> endpoints.</p>
      <p>Try <a href='/metadata'>/metadata</a> and <a href='/schema'>/schema</a>.</p>
    </body></html>
    """


@app.post("/reset", response_model=PipelineObservation)
def reset(task_id: str = Query(default="easy")) -> PipelineObservation:
    global env
    if task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")
    env = PipelineDebugEnv(task_id=task_id)
    return env.reset()


@app.post("/curriculum/reset")
def curriculum_reset() -> dict[str, Any]:
    global env
    next_task = curriculum.next_task()
    env = PipelineDebugEnv(task_id=next_task)
    observation = env.reset()
    return {
        "observation": observation.model_dump(),
        "info": {
            "selected_task_id": next_task,
            "curriculum": curriculum.get_stats(),
        },
    }


@app.post("/step", response_model=StepResult)
def step(action: PipelineAction) -> StepResult:
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    observation, reward, done, info = env.step(action)
    if done:
        curriculum.record(env.task_id, env._dataframes_match())

    return StepResult(
        observation=observation,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state")
def state() -> dict[str, Any]:
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return env.state()


def main() -> None:
    import uvicorn

    uvicorn.run("src.server:app", host="0.0.0.0", port=PORT, reload=False)


if __name__ == "__main__":
    main()

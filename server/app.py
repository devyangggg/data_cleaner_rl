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
    return {"status": "healthy"}  # changed from "ok" to "healthy"


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return {
        "name": "pipeline-debug-env",
        "description": "An RL environment where an agent debugs broken data pipelines by issuing fix commands",
        "version": "0.1.0",
        "tasks": ["easy", "medium", "hard"],
    }


@app.get("/schema")
def schema() -> dict[str, Any]:
    return {
        "action": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "parameters": {"type": "object"},
            },
        },
        "observation": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "step": {"type": "integer"},
                "schema": {"type": "object"},
                "preview": {"type": "array"},
                "expected_schema": {"type": "object"},
                "diff_summary": {"type": "string"},
                "last_action_result": {"type": "string"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "step": {"type": "integer"},
                "done": {"type": "boolean"},
            },
        },
    }


@app.post("/mcp")
def mcp(payload: dict[str, Any] = {}) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": payload.get("id", 1),
        "result": {
            "name": "pipeline-debug-env",
            "description": "Data pipeline debugging RL environment",
        },
    }


@app.post("/reset", response_model=Observation)
def reset(task_id: str = Query(default="easy")) -> Observation:
    global env
    if task_id not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")
    env = PipelineDebugEnv(task_id=task_id)
    return env.reset()


@app.post("/step")
def step(action: Action) -> dict[str, Any]:
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
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return env.state()


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
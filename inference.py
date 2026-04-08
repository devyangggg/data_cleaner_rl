"""LLM-driven inference runner for Data Cleaner RL."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
HF_TOKEN = os.environ.get("HF_TOKEN", "hf-no-key")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
MAX_STEPS_DEFAULT = int(os.environ.get("MAX_STEPS", "30"))

COMMANDS = {
    "cast_column": ["column", "dtype"],
    "fill_nulls": ["column", "value"],
    "fix_date_format": ["column", "format"],
    "fix_join": ["left_key", "right_key"],
    "drop_duplicates": ["subset"],
    "apply_transform": ["column", "expression"],
    "revert_step": [],
    "rename_column": ["old_name", "new_name"],
    "drop_column": ["column"],
    "sort_values": ["by", "ascending"],
}

SYSTEM_PROMPT = (
    "You are a data pipeline repair agent. "
    "Return only one valid JSON object with keys 'command' and 'params'. "
    "No markdown, no prose."
)

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def _extract_action(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])

    command = parsed["command"]
    params = parsed.get("params", parsed.get("parameters", {}))
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    return {"command": command, "params": params}


def call_llm(observation: dict[str, Any], total_reward: float, retries: int = 2) -> dict[str, Any]:
    payload = {
        "task_id": observation.get("task_id"),
        "diff_items": observation.get("diff_items", []),
        "diff_summary": observation.get("diff_summary"),
        "current_schema": observation.get("current_schema", {}),
        "expected_schema": observation.get("expected_schema", {}),
        "step_count": observation.get("step_count", 0),
        "current_reward": round(total_reward, 4),
        "available_commands": COMMANDS,
        "response_format": {"command": "<name>", "params": {"key": "value"}},
    }
    user_prompt = (
        "Given this observation, choose exactly one next repair action.\n"
        "Use diff_items as the primary signal because it is structured; use diff_summary only as backup context.\n"
        "Return only JSON object with keys command and params.\n"
        f"Observation:\n{json.dumps(payload, indent=2)}"
    )

    last_error: Exception | None = None
    for _ in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = resp.choices[0].message.content or ""
            return _extract_action(text)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"LLM call failed after retries: {last_error}")


def run_task(task_id: str) -> dict[str, Any]:
    reset = httpx.post(f"{ENV_URL}/reset", params={"task_id": task_id}, timeout=30)
    reset.raise_for_status()
    observation = reset.json()

    total_reward = 0.0
    steps = 0
    done = False

    print(f"\n=== TASK {task_id.upper()} ===")

    while not done and steps < MAX_STEPS_DEFAULT:
        action = call_llm(observation, total_reward)
        step_resp = httpx.post(f"{ENV_URL}/step", json=action, timeout=30)
        step_resp.raise_for_status()
        result = step_resp.json()

        observation = result["observation"]
        reward = float(result["reward"])
        done = bool(result["done"])
        info = result.get("info", {})
        total_reward = reward
        steps += 1

        print(
            f"step={steps:02d} action={action} reward={reward:.4f} "
            f"result={info.get('action_result', observation.get('last_action_result'))} "
            f"rationale={info.get('rationale', '')}"
        )

    success = bool(observation.get("is_terminal", done) and "match" in observation.get("diff_summary", "").lower())
    if observation.get("diff_summary") == "Dataframes match perfectly":
        success = True

    return {
        "task": task_id,
        "success": success,
        "total_reward": round(total_reward, 4),
        "steps": steps,
    }


def main() -> None:
    results = []
    for task in ["easy", "medium", "hard"]:
        results.append(run_task(task))

    print("\n=== FINAL SUMMARY ===")
    for row in results:
        status = "SUCCESS" if row["success"] else "FAIL"
        print(
            f"task={row['task']:<6} status={status:<7} "
            f"reward={row['total_reward']:.4f} steps={row['steps']}"
        )


if __name__ == "__main__":
    main()

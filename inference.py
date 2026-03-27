"""Inference script — runs a rule-based agent through all 3 pipeline debugging tasks."""

from __future__ import annotations

import json
import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

TASKS = ["easy", "medium", "hard"]

RULE_BASED_ACTIONS = {
    "easy": [
        {"command": "cast_column",     "parameters": {"column": "revenue", "dtype": "float"}},
        {"command": "fix_date_format", "parameters": {"column": "date", "format": "%d-%m-%Y"}},
        {"command": "fill_nulls",      "parameters": {"column": "units_sold", "value": 0}},
    ],
    "medium": [
        {"command": "fix_join",        "parameters": {"left_key": "customer_code", "right_key": "customer_id"}},
        {"command": "drop_duplicates", "parameters": {"subset": ["order_id"]}},
    ],
    "hard": [
        {"command": "apply_transform", "parameters": {
            "column": "converted_amount",
            "expression": "row['converted_amount'] / 1.23 if row['currency'] == 'USD' else x"
        }},
    ],
}


def run_task(task_id: str) -> float:
    print(f"\n{'='*60}")
    print(f"  TASK: {task_id.upper()}")
    print(f"{'='*60}")

    resp = httpx.post(f"{ENV_BASE_URL}/reset", params={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    obs = resp.json()

    actions = RULE_BASED_ACTIONS[task_id]
    terminated = False
    final_score = 0.0
    step = 0

    while not terminated:
        if step < len(actions):
            action = actions[step]
        else:
            break

        print(f"\n  Step {step + 1} | Action: {json.dumps(action)}")

        try:
            resp = httpx.post(
                f"{ENV_BASE_URL}/step",
                json=action,
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            print(f"  [Env Error] {e}")
            break

        obs = result["observation"]
        reward = result["reward"]
        terminated = result["terminated"]
        info = result["info"]

        print(
            f"  -> reward={reward:.4f} "
            f"result={obs['last_action_result']} | "
            f"{info.get('reason', '')}"
        )

        final_score = reward
        step += 1

    print(f"\n  Final score for {task_id}: {final_score:.4f}")
    return final_score


def main() -> None:
    scores: dict[str, float] = {}
    start_time = time.time()

    for task_id in TASKS:
        score = run_task(task_id)
        scores[task_id] = score

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print("  BASELINE RESULTS")
    print(f"{'='*60}")
    for task_id, score in scores.items():
        print(f"  {task_id:>8s}: {score:.4f}")
    print(f"  {'average':>8s}: {sum(scores.values()) / len(scores):.4f}")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
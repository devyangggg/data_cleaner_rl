"""Benchmark harness for heuristic and LLM agents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.environment import PipelineDebugEnv
from src.models import PipelineAction

SEEDS = [42, 43, 44, 45, 46]
TASKS = ["easy", "medium", "hard"]

HEURISTIC_ACTIONS = {
    "easy": [
        {"command": "cast_column", "params": {"column": "revenue", "dtype": "float64"}},
        {"command": "fix_date_format", "params": {"column": "date", "format": "%d-%m-%Y"}},
        {"command": "fill_nulls", "params": {"column": "units_sold", "value": 0}},
    ],
    "medium": [
        {"command": "fix_join", "params": {"left_key": "customer_code", "right_key": "customer_id"}},
        {"command": "drop_duplicates", "params": {"subset": ["order_id"]}},
    ],
    "hard": [
        {
            "command": "apply_transform",
            "params": {
                "column": "converted_amount",
                "expression": "row['converted_amount'] / 1.23 if row['currency'] == 'USD' else x",
            },
        }
    ],
}


def _is_success(env: PipelineDebugEnv) -> bool:
    return env._dataframes_match()  # noqa: SLF001


def _heuristic_action(task_id: str, step: int) -> dict[str, Any] | None:
    actions = HEURISTIC_ACTIONS[task_id]
    if step >= len(actions):
        return None
    return actions[step]


def _llm_action(observation: dict[str, Any], total_reward: float) -> tuple[dict[str, Any] | None, str | None]:
    try:
        from inference import call_llm
        return call_llm(observation, total_reward), None
    except Exception as exc:
        return None, str(exc)


def run_episode(task_id: str, agent: str, seed: int) -> dict[str, Any]:
    env = PipelineDebugEnv(task_id=task_id)
    obs = env.reset().model_dump()

    done = False
    steps = 0
    max_steps = env.task.max_steps
    reward = 0.0
    llm_error: str | None = None

    while not done and steps < max_steps:
        if agent == "heuristic":
            action_dict = _heuristic_action(task_id, steps)
        else:
            action_dict, llm_error = _llm_action(obs, reward)

        if action_dict is None:
            break

        action = PipelineAction(**action_dict)
        obs_obj, reward, done, _info = env.step(action)
        obs = obs_obj.model_dump()
        steps += 1

    out = {
        "seed": seed,
        "task": task_id,
        "success": _is_success(env),
        "total_reward": round(float(reward), 4),
        "steps_taken": steps,
    }
    if llm_error:
        out["llm_error"] = llm_error
    return out


def summarize(rows: list[dict[str, Any]], agent: str) -> tuple[str, list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    lines = [
        "| Task   | Agent     | Success Rate | Avg Reward | Avg Steps |",
        "|--------|-----------|--------------|------------|-----------|",
    ]

    for task in TASKS:
        task_rows = [r for r in rows if r["task"] == task]
        success_count = sum(1 for r in task_rows if r["success"])
        avg_reward = sum(r["total_reward"] for r in task_rows) / len(task_rows)
        avg_steps = sum(r["steps_taken"] for r in task_rows) / len(task_rows)
        summary_row = {
            "task": task,
            "agent": agent,
            "success_rate": f"{success_count}/{len(task_rows)} ({(success_count/len(task_rows))*100:.0f}%)",
            "avg_reward": round(avg_reward, 4),
            "avg_steps": round(avg_steps, 2),
        }
        summary.append(summary_row)
        lines.append(
            f"| {task:<6} | {agent:<9} | {summary_row['success_rate']:<12} | "
            f"{summary_row['avg_reward']:<10} | {summary_row['avg_steps']:<9} |"
        )

    return "\n".join(lines), summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Data Cleaner RL agents")
    parser.add_argument("--agent", choices=["heuristic", "llm"], default="heuristic")
    args = parser.parse_args()

    runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        for task in TASKS:
            runs.append(run_episode(task, args.agent, seed))

    markdown, summary = summarize(runs, args.agent)
    print(markdown)
    if args.agent == "llm":
        errors = [str(r["llm_error"]) for r in runs if r.get("llm_error")]
        if errors:
            unique = sorted(set(errors))
            print("\nLLM errors detected:")
            for err in unique[:5]:
                print(f"- {err}")

    out = {
        "agent": args.agent,
        "seeds": SEEDS,
        "runs": runs,
        "summary": summary,
    }
    Path("benchmark_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

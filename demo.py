"""Simple Gradio UI for interacting with the environment."""

from __future__ import annotations

import json
import os
from typing import Any

import gradio as gr
import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical

from rl.action_space import EASY_ACTION_TEMPLATES
from rl.features import featurize_observation
from rl.policy import PolicyNet
from src.environment import PipelineDebugEnv
from src.models import PipelineAction

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

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


def format_df_preview(obs: dict[str, Any]) -> pd.DataFrame:
    rows = obs.get("preview_rows", [])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def format_action_label(command: str, params: dict[str, Any]) -> str:
    if command == "cast_column":
        col = params.get("column", "?")
        dtype = params.get("dtype", "?")
        return f"cast_column({col} -> {dtype})"
    if command == "fill_nulls":
        col = params.get("column", "?")
        value = params.get("value", "?")
        return f"fill_nulls({col} = {value})"
    if command == "fix_join":
        left_key = params.get("left_key", "?")
        right_key = params.get("right_key", "?")
        return f"fix_join({left_key} -> {right_key})"
    if command == "rename_column":
        old = params.get("old_name", "?")
        new = params.get("new_name", "?")
        return f"rename_column({old} -> {new})"
    rendered = ", ".join(f"{k}={params[k]}" for k in sorted(params.keys()))
    return f"{command}({rendered})" if rendered else command


def format_timeline(lines: list[str]) -> str:
    return "\n".join(lines) if lines else "No actions yet."


def _trajectory_line(step_num: int, command: str, params: dict[str, Any], delta: float, reward: float) -> str:
    action_label = format_action_label(command, params)
    return (
        f"Step {step_num} | {action_label} | "
        f"Delta reward: {delta:+.2f} | Running total: {reward:.2f}"
    )


def reset_env(task_id: str) -> tuple[pd.DataFrame, str, str, str, list[str], str]:
    r = httpx.post(f"{ENV_URL}/reset", params={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    obs = r.json()
    trajectory: list[str] = []
    return (
        format_df_preview(obs),
        obs.get("diff_summary", ""),
        "0",
        "Ready",
        trajectory,
        format_timeline(trajectory),
    )


def take_action(command: str, params_json: str, trajectory: list[str]) -> tuple[pd.DataFrame, str, str, str, list[str], str]:
    params = json.loads(params_json or "{}")
    if not isinstance(params, dict):
        raise gr.Error("Params must be a JSON object")

    r = httpx.post(
        f"{ENV_URL}/step",
        json={"command": command, "params": params},
        timeout=30,
    )
    r.raise_for_status()
    result = r.json()
    obs = result["observation"]
    info = result.get("info", {})
    reward = float(result.get("reward", 0.0))
    step_num = int(obs.get("step_count", 0))
    delta = float(info.get("delta_reward", 0.0))
    trajectory_line = _trajectory_line(step_num, command, params, delta, reward)
    new_trajectory = [*trajectory, trajectory_line]
    status = "Done!" if result.get("done") else "Continue"

    return (
        format_df_preview(obs),
        obs.get("diff_summary", ""),
        str(reward),
        status,
        new_trajectory,
        format_timeline(new_trajectory),
    )


def autoplay_heuristic(task_id: str) -> tuple[pd.DataFrame, str, str, str, list[str], str]:
    r = httpx.post(f"{ENV_URL}/reset", params={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    obs = r.json()
    trajectory: list[str] = []
    reward = 0.0
    done = False

    for action in HEURISTIC_ACTIONS.get(task_id, []):
        if done:
            break
        step_resp = httpx.post(
            f"{ENV_URL}/step",
            json=action,
            timeout=30,
        )
        step_resp.raise_for_status()
        result = step_resp.json()

        obs = result["observation"]
        info = result.get("info", {})
        reward = float(result.get("reward", 0.0))
        done = bool(result.get("done", False))
        step_num = int(obs.get("step_count", 0))
        delta = float(info.get("delta_reward", 0.0))
        trajectory.append(
            _trajectory_line(step_num, action.get("command", "?"), action.get("params", {}), delta, reward)
        )

    if done:
        status = "Done! (heuristic autoplay)"
    else:
        status = "Heuristic stopped (no more scripted actions)"

    return (
        format_df_preview(obs),
        obs.get("diff_summary", ""),
        str(reward),
        status,
        trajectory,
        format_timeline(trajectory),
    )


def _make_training_plot(rewards: list[float], success: list[float]):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    if rewards:
        x = np.arange(1, len(rewards) + 1)
        axes[0].plot(x, rewards, label="reward", alpha=0.45)
        window = min(10, len(rewards))
        ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        axes[0].plot(np.arange(window, len(rewards) + 1), ma, label=f"reward_ma{window}")
    axes[0].set_title("Reward per Episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Final reward")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="lower right")

    if success:
        x = np.arange(1, len(success) + 1)
        axes[1].plot(x, success, label="success", alpha=0.35)
        window = min(10, len(success))
        ma = np.convolve(success, np.ones(window) / window, mode="valid")
        axes[1].plot(np.arange(window, len(success) + 1), ma, label=f"success_ma{window}")
    axes[1].set_title("Success per Episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    return fig


def _run_training_episode(env: PipelineDebugEnv, policy: PolicyNet, gamma: float):
    obs = env.reset().model_dump()
    done = False
    last_reward = 0.0
    log_probs: list[torch.Tensor] = []
    deltas: list[float] = []

    while not done and len(deltas) < env.task.max_steps:
        feats = featurize_observation(obs, max_steps=env.task.max_steps)
        x = torch.from_numpy(feats).unsqueeze(0)
        logits = policy(x)
        dist = Categorical(logits=logits)
        action_idx = int(dist.sample().item())

        action = PipelineAction(**EASY_ACTION_TEMPLATES[action_idx])
        obs_obj, reward, done, _ = env.step(action)
        obs = obs_obj.model_dump()

        log_probs.append(dist.log_prob(torch.tensor(action_idx)))
        deltas.append(float(reward - last_reward))
        last_reward = float(reward)

    returns = []
    g = 0.0
    for r in reversed(deltas):
        g = r + gamma * g
        returns.insert(0, g)
    returns_t = torch.tensor(returns, dtype=torch.float32)
    if len(returns_t) > 1:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

    success = env._dataframes_match()  # noqa: SLF001
    return last_reward, success, log_probs, returns_t


def train_rl_live(episodes: int, lr: float, gamma: float, hidden: int, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = PipelineDebugEnv(task_id="easy")
    policy = PolicyNet(input_dim=10, hidden_dim=hidden, num_actions=len(EASY_ACTION_TEMPLATES))
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    rewards: list[float] = []
    success: list[float] = []

    for ep in range(1, episodes + 1):
        final_reward, ok, log_probs, returns_t = _run_training_episode(env, policy, gamma)

        loss = torch.tensor(0.0)
        for lp, ret in zip(log_probs, returns_t):
            loss = loss - lp * ret

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards.append(float(final_reward))
        success.append(1.0 if ok else 0.0)

        if ep == 1 or ep % 5 == 0 or ep == episodes:
            fig = _make_training_plot(rewards, success)
            avg_r = float(np.mean(rewards[-10:]))
            avg_s = float(np.mean(success[-10:]))
            summary = (
                f"Episode {ep}/{episodes} | "
                f"avg_reward_last10={avg_r:.4f} | success_last10={avg_s:.2f}"
            )
            log_text = "\n".join(
                [
                    f"ep={i+1:03d} reward={rewards[i]:.4f} success={int(success[i])}"
                    for i in range(max(0, len(rewards) - 15), len(rewards))
                ]
            )
            yield fig, summary, log_text


with gr.Blocks(title="Data Cleaner RL Demo") as demo:
    gr.Markdown("## Data Pipeline Debugger - RL Environment")
    with gr.Row():
        task_dd = gr.Dropdown(["easy", "medium", "hard"], label="Task", value="easy")
        reset_btn = gr.Button("Reset episode")
        autoplay_btn = gr.Button("Auto-play heuristic")
    with gr.Row():
        df_display = gr.Dataframe(label="Current dataframe (preview)", interactive=False)
        diff_display = gr.Textbox(label="Diff summary", lines=10)
    with gr.Row():
        cmd_input = gr.Textbox(label="Command", placeholder="cast_column")
        params_input = gr.Textbox(
            label="Params (JSON)",
            placeholder='{"column": "revenue", "dtype": "float64"}',
        )
        step_btn = gr.Button("Take action")
    with gr.Row():
        reward_out = gr.Textbox(label="Reward")
        status_out = gr.Textbox(label="Status")
    trajectory_state = gr.State([])
    trajectory_view = gr.Textbox(label="Agent trajectory timeline", lines=12)

    gr.Markdown("## RL Training (Easy Task)")
    with gr.Row():
        episodes_in = gr.Slider(20, 300, value=100, step=10, label="Episodes")
        lr_in = gr.Number(value=0.001, label="Learning rate")
        gamma_in = gr.Number(value=0.99, label="Gamma")
        hidden_in = gr.Slider(16, 256, value=64, step=16, label="Hidden size")
        seed_in = gr.Number(value=42, label="Seed", precision=0)
    with gr.Row():
        train_btn = gr.Button("Train RL (Live Plot)")
    rl_plot = gr.Plot(label="Training curves (real-time)")
    rl_summary = gr.Textbox(label="Training summary")
    rl_log = gr.Textbox(label="Recent episode log", lines=12)

    reset_btn.click(
        reset_env,
        inputs=[task_dd],
        outputs=[df_display, diff_display, reward_out, status_out, trajectory_state, trajectory_view],
    )
    step_btn.click(
        take_action,
        inputs=[cmd_input, params_input, trajectory_state],
        outputs=[df_display, diff_display, reward_out, status_out, trajectory_state, trajectory_view],
    )
    autoplay_btn.click(
        autoplay_heuristic,
        inputs=[task_dd],
        outputs=[df_display, diff_display, reward_out, status_out, trajectory_state, trajectory_view],
    )
    train_btn.click(
        train_rl_live,
        inputs=[episodes_in, lr_in, gamma_in, hidden_in, seed_in],
        outputs=[rl_plot, rl_summary, rl_log],
    )


if __name__ == "__main__":
    demo.queue().launch()

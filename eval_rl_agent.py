"""Evaluate trained easy-task RL policy against random baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rl.action_space import EASY_ACTION_TEMPLATES
from rl.features import featurize_observation
from rl.policy import PolicyNet
from src.environment import PipelineDebugEnv
from src.models import PipelineAction


def run_policy_episode(env: PipelineDebugEnv, policy: PolicyNet | None, random_actions: bool = False) -> tuple[float, bool, int]:
    obs = env.reset().model_dump()
    done = False
    reward = 0.0
    steps = 0

    while not done and steps < env.task.max_steps:
        if random_actions:
            idx = int(np.random.randint(0, len(EASY_ACTION_TEMPLATES)))
        else:
            feats = featurize_observation(obs, max_steps=env.task.max_steps)
            x = torch.from_numpy(feats).unsqueeze(0)
            logits = policy(x)
            idx = int(torch.argmax(logits, dim=-1).item())

        action = PipelineAction(**EASY_ACTION_TEMPLATES[idx])
        obs_obj, reward, done, _info = env.step(action)
        obs = obs_obj.model_dump()
        steps += 1

    success = env._dataframes_match()  # noqa: SLF001
    return float(reward), success, steps


def evaluate(episodes: int, checkpoint: str) -> dict:
    env = PipelineDebugEnv(task_id="easy")
    policy = PolicyNet(input_dim=10, hidden_dim=64, num_actions=len(EASY_ACTION_TEMPLATES))
    state_dict = torch.load(checkpoint, map_location="cpu")
    policy.load_state_dict(state_dict)
    policy.eval()

    rl_rewards: list[float] = []
    rl_success: list[float] = []
    rl_steps: list[int] = []

    rnd_rewards: list[float] = []
    rnd_success: list[float] = []
    rnd_steps: list[int] = []

    for _ in range(episodes):
        r, s, st = run_policy_episode(env, policy=policy, random_actions=False)
        rl_rewards.append(r)
        rl_success.append(1.0 if s else 0.0)
        rl_steps.append(st)

        r2, s2, st2 = run_policy_episode(env, policy=None, random_actions=True)
        rnd_rewards.append(r2)
        rnd_success.append(1.0 if s2 else 0.0)
        rnd_steps.append(st2)

    out = {
        "episodes": episodes,
        "rl": {
            "avg_reward": float(np.mean(rl_rewards)),
            "success_rate": float(np.mean(rl_success)),
            "avg_steps": float(np.mean(rl_steps)),
        },
        "random": {
            "avg_reward": float(np.mean(rnd_rewards)),
            "success_rate": float(np.mean(rnd_success)),
            "avg_steps": float(np.mean(rnd_steps)),
        },
    }

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    eval_path = outputs_dir / "eval_easy.json"
    eval_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"saved eval: {eval_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate easy RL policy")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default="outputs/easy_policy.pt")
    args = parser.parse_args()
    evaluate(episodes=args.episodes, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()

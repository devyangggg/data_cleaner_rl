"""Train a lightweight RL policy on the easy task and log reward curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from rl.action_space import EASY_ACTION_TEMPLATES
from rl.features import featurize_observation
from rl.policy import PolicyNet
from src.environment import PipelineDebugEnv
from src.models import PipelineAction


def run_episode(
    env: PipelineDebugEnv,
    policy: PolicyNet,
    max_steps: int,
    training: bool,
) -> tuple[float, bool, list[torch.Tensor], list[float], int]:
    obs = env.reset().model_dump()
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []
    done = False

    last_score = 0.0
    steps = 0

    while not done and steps < max_steps:
        feats = featurize_observation(obs, max_steps=env.task.max_steps)
        x = torch.from_numpy(feats).unsqueeze(0)
        logits = policy(x)
        dist = Categorical(logits=logits)
        action_idx = int(dist.sample().item()) if training else int(torch.argmax(logits, dim=-1).item())

        action_payload = EASY_ACTION_TEMPLATES[action_idx]
        action = PipelineAction(**action_payload)
        obs_obj, reward, done, _info = env.step(action)
        obs = obs_obj.model_dump()

        delta = float(reward - last_score)
        last_score = float(reward)
        rewards.append(delta)
        log_probs.append(dist.log_prob(torch.tensor(action_idx)))
        steps += 1

    success = env._dataframes_match()  # noqa: SLF001
    return float(last_score), success, log_probs, rewards, steps


def train(total_episodes: int, lr: float, gamma: float, hidden_dim: int, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = PipelineDebugEnv(task_id="easy")
    input_dim = 10
    num_actions = len(EASY_ACTION_TEMPLATES)

    policy = PolicyNet(input_dim=input_dim, hidden_dim=hidden_dim, num_actions=num_actions)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    episode_rewards: list[float] = []
    episode_success: list[float] = []
    episode_steps: list[int] = []

    for ep in range(1, total_episodes + 1):
        final_reward, success, log_probs, deltas, steps = run_episode(
            env=env,
            policy=policy,
            max_steps=env.task.max_steps,
            training=True,
        )

        returns: list[float] = []
        g = 0.0
        for r in reversed(deltas):
            g = r + gamma * g
            returns.insert(0, g)

        returns_t = torch.tensor(returns, dtype=torch.float32)
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        loss = torch.tensor(0.0)
        for lp, ret in zip(log_probs, returns_t):
            loss = loss - lp * ret

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(final_reward)
        episode_success.append(1.0 if success else 0.0)
        episode_steps.append(steps)

        if ep % 10 == 0 or ep == 1:
            avg_r = float(np.mean(episode_rewards[-10:]))
            avg_s = float(np.mean(episode_success[-10:]))
            print(f"episode={ep:03d} avg_reward_last10={avg_r:.4f} success_last10={avg_s:.2f}")

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    ckpt_path = outputs_dir / "easy_policy.pt"
    torch.save(policy.state_dict(), ckpt_path)

    metrics = {
        "episodes": total_episodes,
        "avg_reward_last10": float(np.mean(episode_rewards[-10:])),
        "success_rate_last10": float(np.mean(episode_success[-10:])),
        "avg_steps_last10": float(np.mean(episode_steps[-10:])),
        "reward_curve": episode_rewards,
        "success_curve": episode_success,
        "steps_curve": episode_steps,
        "checkpoint": str(ckpt_path),
    }

    metrics_path = outputs_dir / "train_metrics_easy.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved checkpoint: {ckpt_path}")
    print(f"saved metrics: {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train simple RL policy on easy task")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        total_episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        hidden_dim=args.hidden,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

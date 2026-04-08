"""Plot reward and success curves from training metrics JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def moving_avg(values: list[float], window: int = 10) -> list[float]:
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start : i + 1]
        out.append(sum(window_vals) / len(window_vals))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RL training curves")
    parser.add_argument("--metrics", type=str, default="outputs/train_metrics_easy.json")
    parser.add_argument("--out", type=str, default="outputs/reward_curve_easy.png")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    rewards = [float(x) for x in metrics.get("reward_curve", [])]
    success = [float(x) for x in metrics.get("success_curve", [])]

    r_ma = moving_avg(rewards, window=10)
    s_ma = moving_avg(success, window=10)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(rewards, alpha=0.35, label="reward")
    axes[0].plot(r_ma, linewidth=2, label="reward_ma10")
    axes[0].set_title("Episode Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Final reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(success, alpha=0.35, label="success")
    axes[1].plot(s_ma, linewidth=2, label="success_ma10")
    axes[1].set_title("Episode Success")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success (0/1)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved plot: {out_path}")


if __name__ == "__main__":
    main()

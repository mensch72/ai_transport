"""Plot decision-step rewards and episode rewards."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback


def plot_reward_analysis(
    step_records,
    episode_records,
    output_dir,
    title_prefix,
):
    """Save three reward figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not step_records:
        return

    def save_figure(fig, filename):
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)

    # 1. Step reward curve.
    # Draw each episode separately: do not connect across resets.
    episode_groups = {}
    for record in step_records:
        episode_groups.setdefault(record["episode"], []).append(record)

    fig, ax = plt.subplots(figsize=(10, 4))

    for records in episode_groups.values():
        ax.plot(
            [record["global_step"] for record in records],
            [record["scaled_reward"] for record in records],
            color="tab:blue",
            linewidth=0.8,
            marker=".",
            markersize=3,
        )

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set(
        title=f"{title_prefix} Step Reward Curve",
        xlabel="Cumulative decision steps",
        ylabel="Scaled step reward",
    )
    ax.grid(alpha=0.25)
    save_figure(fig, "step_reward_curve.png")

    # 2. Histogram of individual step rewards.
    rewards = [record["scaled_reward"] for record in step_records]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rewards, bins=50, edgecolor="white")
    ax.set(
        title=f"{title_prefix} Step Reward Histogram",
        xlabel="Scaled step reward",
        ylabel="Number of decision steps",
    )
    save_figure(fig, "step_reward_histogram.png")

    # 3. Total reward for each completed episode.
    fig, ax = plt.subplots(figsize=(10, 4))

    if episode_records:
        episodes = [record["episode"] for record in episode_records]
        episode_rewards = [
            record["scaled_reward"] for record in episode_records
        ]

        ax.plot(
            episodes,
            episode_rewards,
            marker="o",
            markersize=3,
            linewidth=1,
            label="Episode reward",
        )

        window = min(20, len(episode_rewards))
        moving_mean = [
            sum(episode_rewards[max(0, i - window + 1):i + 1])
            / len(episode_rewards[max(0, i - window + 1):i + 1])
            for i in range(len(episode_rewards))
        ]

        ax.plot(
            episodes,
            moving_mean,
            linewidth=2,
            label=f"Moving mean (up to {window} episodes)",
        )
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No completed episodes",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set(
        title=f"{title_prefix} Episode Reward Curve",
        xlabel="Episode",
        ylabel="Scaled episode reward",
    )
    ax.grid(alpha=0.25)
    save_figure(fig, "episode_reward_curve.png")


class RewardAnalysisCallback(BaseCallback):
    """Record rewards during single-environment DQN training."""

    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.step_records = []
        self.episode_records = []

        self.episode = 1
        self.episode_step = 0
        self.episode_reward = 0.0

    def _on_training_start(self):
        if self.training_env.num_envs != 1:
            raise ValueError(
                "This reward callback currently supports one environment."
            )

    def _on_step(self):
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])

        self.episode_step += 1
        self.episode_reward += reward

        self.step_records.append({
            "global_step": self.num_timesteps,
            "episode": self.episode,
            "step": self.episode_step,
            "scaled_reward": reward,
        })

        if done:
            self.episode_records.append({
                "episode": self.episode,
                "scaled_reward": self.episode_reward,
            })

            self.episode += 1
            self.episode_step = 0
            self.episode_reward = 0.0

        return True

    def _on_training_end(self):
        plot_reward_analysis(
            self.step_records,
            self.episode_records,
            self.output_dir,
            title_prefix="DQN Training",
        )
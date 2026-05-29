import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from accessibility_equity.policies import HeuristicRoutingHumanPolicy
from accessibility_equity.wrappers import REWARD_SCALE, create_dqn_env
from accessibility_equity.rewards import compute_scenario_utility_bounds
from accessibility_equity.visualization import save_scenario_figure

from stable_baselines3.common.monitor import Monitor


def make_env(cfg, output_dir=None, seed=None, monitor=True, render_mode=None):
    """
    Create a small single-vehicle environment for DQN smoke training.
    """
    if monitor and output_dir is None:
        raise ValueError("output_dir is required when moitor=True")
    train_monitor_path = output_dir / "train.monitor.csv"

    env = create_dqn_env(
        num_humans=cfg.env.scenario.num_humans,
        num_vehicles=cfg.env.scenario.num_vehicles,
        num_nodes=cfg.env.scenario.num_nodes,
        seed=seed,
        human_speeds=[cfg.env.mobility.human_walking_speed_kmh] * cfg.env.scenario.num_humans,
        vehicle_speeds=[cfg.env.mobility.vehicle_speed_kmh] * cfg.env.scenario.num_vehicles,
        human_policy_class=HeuristicRoutingHumanPolicy,
        human_policy_kwargs={"p_wait": 0.5},
        network_kwargs={"speed_mean": cfg.env.mobility.edge_speed_kmh},
        reward_config=cfg.env.reward,
        mobility_config=cfg.env.mobility,
        max_steps=cfg.dqn.max_steps,
        render_mode=render_mode,
        use_action_masking=True,
        decision_mode=cfg.dqn.decision_mode,
    )
    if monitor:
        output_dir.mkdir(parents=True, exist_ok=True)
        return Monitor(env, filename=str(train_monitor_path))
    return env


def _utility_legend_label(reward_config):
    utility_mode = reward_config.utility_mode
    if utility_mode == "sum_accessibility":
        formula = r"$U(s_t)=\sum_h X_h(s_t)$"
    elif utility_mode == "equity":
        formula = (
            rf"$U(s_t)=-(\sum_h \max(X_h(s_t),{reward_config.epsilon:g})^"
            rf"{{-{reward_config.xi:g}}})^{{{reward_config.eta:g}}}$"
        )
    else:
        formula = "unknown utility formula"
    return formula


def plot_training_rewards(cfg, monitor_path, output_path):
    """
    Plot episode reward against cumulative training timesteps from monitor logs.
    """
    if not monitor_path.exists():
        print(f"Training monitor file not found: '{monitor_path}'")
        return

    episode_rewards = []
    episode_lengths = []
    with monitor_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
        if not first_line.startswith("#"):
            handle.seek(0)
        reader = csv.DictReader(handle)
        for row in reader:
            episode_rewards.append(float(row["r"]))
            episode_lengths.append(int(float(row["l"])))

    if not episode_rewards:
        print(f"No training episodes found in '{monitor_path}'")
        return

    timesteps = []
    running_steps = 0
    for length in episode_lengths:
        running_steps += length
        timesteps.append(running_steps)

    window = min(20, len(episode_rewards))
    moving_mean = []
    for index in range(len(episode_rewards)):
        start = max(0, index - window + 1)
        window_values = episode_rewards[start : index + 1]
        moving_mean.append(sum(window_values) / len(window_values))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(
        timesteps,
        episode_rewards,
        color="#9ecae1",
        linewidth=1.2,
        marker="o",
        markersize=3,
        label="episode reward",
    )
    ax.plot(
        timesteps,
        moving_mean,
        color="#1d4ed8",
        linewidth=2.0,
        label=f"moving mean ({window} eps)",
    )
    ax.plot([], [], color="none", label=_utility_legend_label(cfg.env.reward))
    ax.set_title(f"DQN Training Reward Curve (scaled x{REWARD_SCALE:g})")
    ax.set_xlabel("training timesteps")
    ax.set_ylabel("scaled episode reward")
    ax.grid(True, color="#e5e7eb", linewidth=0.6)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Training reward curve saved to '{output_path}'")

def save_training_scenario_diagnostics(cfg, output_dir):
    """
    Save the exact configured scenario picture for this training run.
    """
    scenario_image_path = output_dir / "scenario.png"
    preview_env = make_env(cfg, output_dir, seed=cfg.env.scenario.seed, monitor=False, render_mode=None)
    try:
        scenario = preview_env.base_env.scenario
        utility_bounds = compute_scenario_utility_bounds(
            graph=scenario.network,
            population_size=cfg.env.scenario.num_humans,
            reward_config=cfg.env.reward,
        )
        footer_lines = [
            (
                "Utility bounds without time: "
                f"lower={utility_bounds['utility_lower_bound']:.3g} "
                f"uses X_h={utility_bounds['min_accessibility']:.3f} "
                f"from node {utility_bounds['min_accessibility_node']}; "
                f"upper={utility_bounds['utility_upper_bound']:.3g} "
                f"uses X_h={utility_bounds['max_accessibility']:.3f} "
                f"from node {utility_bounds['max_accessibility_node']}"
            )
        ]
        save_scenario_figure(
            scenario,
            scenario_image_path,
            show_node_labels=True,
            show_edge_length_table=True,
            footer_lines=footer_lines,
            utility_bounds=utility_bounds,
        )
        print(f"Scenario image saved to '{scenario_image_path}'")
    finally:
        preview_env.close()

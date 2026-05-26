"""
Minimal example: Train DQN for single-vehicle accessibility-equity control.

This example intentionally starts with one vehicle because standard DQN expects
a single discrete action. Multi-vehicle DQN needs a separate action design.
"""

from collections import Counter
import csv
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from accessibility_equity.config import DEFAULT_EXPERIMENT_CONFIG, save_experiment_config
from accessibility_equity.policies import HeuristicRoutingHumanPolicy
from accessibility_equity.rewards import compute_scenario_utility_bounds
from accessibility_equity.visualization import render_episode_frame_array, save_scenario_figure
from accessibility_equity.wrappers import REWARD_SCALE, create_dqn_env
from accessibility_equity.heuristics import TSPVehicleAgent

try:
    from accessibility_equity.algorithms import MaskedDQN as DQN
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    print("ERROR: stable-baselines3 not installed.")
    print("Install with: pip install stable-baselines3")
    raise SystemExit(1)


EXPERIMENT_CONFIG = DEFAULT_EXPERIMENT_CONFIG
SCENARIO_CONFIG = EXPERIMENT_CONFIG.scenario
MOBILITY_CONFIG = EXPERIMENT_CONFIG.mobility
REWARD_CONFIG = EXPERIMENT_CONFIG.reward
DQN_CONFIG = EXPERIMENT_CONFIG.dqn
DECISION_MODE = DQN_CONFIG.decision_mode
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dqn"


def _safe_name(value):
    if not value:
        return ""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _allocate_experiment_dir(root):
    """
    Allocate a fresh DQN experiment directory such as experiment_001.

    A new folder is used for each run so config, model, plots, videos, and
    scenario diagnostics from different runs do not overwrite each other.
    """
    root.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^experiment_(\d{3,})(?:_|$)")
    existing_numbers = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match:
            existing_numbers.append(int(match.group(1)))
    next_number = max(existing_numbers, default=0) + 1
    label = _safe_name(DQN_CONFIG.output_run_name)
    suffix = f"_{label}" if label else ""
    return root / f"experiment_{next_number:03d}{suffix}"


OUTPUT_DIR = _allocate_experiment_dir(OUTPUT_ROOT)
MODEL_DIR = OUTPUT_DIR / "models"
VIDEO_DIR = OUTPUT_DIR / "videos"
SCENARIO_IMAGE_PATH = OUTPUT_DIR / "scenario.png"
LOG_PATH = OUTPUT_DIR / "evaluation_summary.txt"
EVALUATION_EPISODES_SUMMARY_CSV_PATH = OUTPUT_DIR / "evaluation_episodes_summary.csv"
TRAIN_MONITOR_PATH = OUTPUT_DIR / "train.monitor.csv"
TRAIN_CURVE_PATH = OUTPUT_DIR / "training_reward_curve.png"
DECISION_REPLAY_VIDEO_PATH = VIDEO_DIR / "dqn_decision_replay_all_episodes.mp4"
MODEL_PATH = MODEL_DIR / "accessibility_equity_dqn_single_vehicle"
CONFIG_PATH = OUTPUT_DIR / "config.json"
SAVE_MODEL = DQN_CONFIG.save_model
DEFAULT_MAX_STEPS = DQN_CONFIG.max_steps
DEFAULT_EVAL_EPISODES = DQN_CONFIG.eval_episodes
DEFAULT_EVAL_SEED = DQN_CONFIG.eval_seed
TOTAL_TIMESTEPS = DQN_CONFIG.total_timesteps
NUM_HUMANS = SCENARIO_CONFIG.num_humans
NUM_VEHICLES = SCENARIO_CONFIG.num_vehicles
NUM_NODES = SCENARIO_CONFIG.num_nodes
SAVE_DECISION_REPLAY_VIDEO = DQN_CONFIG.save_decision_replay_video
DECISION_REPLAY_FPS = DQN_CONFIG.decision_replay_fps


def make_env(seed=None, monitor=True, render_mode=None):
    """
    Create a small single-vehicle environment for DQN smoke training.
    """
    env = create_dqn_env(
        num_humans=NUM_HUMANS,
        num_vehicles=NUM_VEHICLES,
        num_nodes=NUM_NODES,
        seed=seed,
        human_speeds=[MOBILITY_CONFIG.human_walking_speed_kmh] * NUM_HUMANS,
        vehicle_speeds=[MOBILITY_CONFIG.vehicle_speed_kmh] * NUM_VEHICLES,
        human_policy_class=HeuristicRoutingHumanPolicy,
        human_policy_kwargs={"p_wait": 0.5},
        network_kwargs={"speed_mean": MOBILITY_CONFIG.edge_speed_kmh},
        poi_kwargs=SCENARIO_CONFIG.poi_kwargs,
        reward_config=REWARD_CONFIG,
        mobility_config=MOBILITY_CONFIG,
        max_steps=DEFAULT_MAX_STEPS,
        render_mode=render_mode,
        use_action_masking=True,
        decision_mode=DECISION_MODE,
    )
    if monitor:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return Monitor(env, filename=str(TRAIN_MONITOR_PATH))
    return env


def save_training_scenario_diagnostics():
    """
    Save the exact configured scenario picture for this training run.
    """
    preview_env = make_env(seed=SCENARIO_CONFIG.seed, monitor=False, render_mode=None)
    try:
        scenario = preview_env.base_env.scenario
        utility_bounds = compute_scenario_utility_bounds(
            graph=scenario.network,
            population_size=SCENARIO_CONFIG.num_humans,
            reward_config=REWARD_CONFIG,
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
            SCENARIO_IMAGE_PATH,
            show_node_labels=True,
            show_edge_length_table=True,
            footer_lines=footer_lines,
            utility_bounds=utility_bounds,
        )
        print(f"Scenario image saved to '{SCENARIO_IMAGE_PATH}'")
    finally:
        preview_env.close()


def plot_training_rewards(monitor_path, output_path):
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
    ax.plot([], [], color="none", label=_utility_legend_label(REWARD_CONFIG))
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


def _select_action(policy_name, model, env, obs):
    """
    Choose one action for the requested evaluation policy.
    """
    if policy_name == "dqn":
        action, _state = model.predict(
            obs,
            deterministic=True,
            action_masks=env.action_masks(),
        )
        return int(action)
    if policy_name == "tsp":
        return model.get_action(obs)
    if policy_name == "random":
        return int(env.action_space.sample())
    if policy_name == "always_pass":
        return 0
    raise ValueError(f"Unknown evaluation policy: {policy_name}")


def _open_video_writer(video_path, fps=2):
    video_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        return imageio.get_writer(str(video_path), fps=fps), video_path
    except Exception as exc:
        gif_path = video_path.with_suffix(".gif")
        try:
            return imageio.get_writer(str(gif_path), mode="I", fps=fps), gif_path
        except Exception as gif_exc:
            print(
                f"Could not open decision replay video writer for '{video_path}' "
                f"or '{gif_path}': {exc}; {gif_exc}"
            )
            return None, None


def _action_description(env, action_value):
    if int(action_value) <= 0:
        return "pass / clear destination"
    target_index = int(action_value) - 1
    if 0 <= target_index < len(env.node_order):
        return f"go to node {env.node_order[target_index]}"
    return f"unknown action {action_value}"


def _format_float(value):
    if value is None:
        return "NA"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "NA"


def _append_decision_replay_frame(
    writer,
    env,
    policy_name,
    episode_index,
    step_index,
    seed,
    action_value=None,
    info=None,
    reward=None,
):
    info = info or {}
    if action_value is None:
        decision_text = "initial state before first DQN decision"
    else:
        decision_text = f"action={action_value} ({_action_description(env, action_value)})"

    title_parts = [
        f"{policy_name.upper()} replay | episode={episode_index + 1} | step={step_index} | seed={seed}",
        decision_text,
    ]
    if reward is not None:
        title_parts.append(
            "impact: "
            f"reward={_format_float(reward)}, "
            f"raw={_format_float(info.get('raw_reward', reward))}, "
            f"U={_format_float(info.get('current_utility'))} -> {_format_float(info.get('next_utility'))}, "
            f"A={_format_float(info.get('current_total_accessibility'))} -> {_format_float(info.get('next_total_accessibility'))}"
        )

    frame = render_episode_frame_array(
        scenario=env.base_env.scenario,
        agent_positions=dict(env.base_env.env.agent_positions),
        human_destinations=dict(env.base_env.env.human_destinations),
        vehicle_destinations=dict(env.base_env.env.vehicle_destinations),
        title="\n".join(title_parts),
    )
    writer.append_data(frame)


def _format_boarding_debug_line(policy_name, episode_index, dqn_step, record):
    """
    Format one boarding-phase diagnostic record for console and log output.
    """
    return (
        f"[boarding-debug] policy={policy_name} "
        f"episode={episode_index + 1} "
        f"dqn_step={dqn_step} "
        f"underlying_step={record.get('dqn_underlying_step')} "
        f"env_step={record.get('step_count')} "
        f"time={float(record.get('real_time') or 0.0):.1f} "
        f"human={record.get('human')} "
        f"human_node={record.get('human_node')} "
        f"vehicle_nodes={record.get('vehicle_nodes')} "
        f"vehicle_destinations={record.get('vehicle_destinations')} "
        f"same_node_vehicles={record.get('same_node_vehicles')} "
        f"human_aboard_before={record.get('human_aboard_before')} "
        f"human_aboard_after={record.get('human_aboard_after')} "
        f"human_boarding_action={record.get('human_boarding_action')} "
        f"human_action_space_n={record.get('human_action_space_n')} "
        f"reason={record.get('human_action_reason')}"
    )


def _current_vehicle_node(env):
    position = env.base_env.env.agent_positions.get(env.vehicle_agent)
    return env._position_to_node(position)


def _run_policy_episode(
    policy_name,
    model,
    episode_index,
    seed,
    print_step_details=False,
    detail_steps=8,
    replay_writer=None,
):
    """
    Run one evaluation episode for a named policy.
    """
    env = make_env(
        seed=seed,
        monitor=False,
        render_mode=None,
    )
    if policy_name == 'tsp':
        model = TSPVehicleAgent(env)
    obs, _info = env.reset(seed=seed)
    terminated = False
    truncated = False
    episode_reward = 0.0
    raw_episode_reward = 0.0
    episode_steps = 0
    action_counts = Counter()
    invalid_action_count = 0
    total_action_count = 0
    start_node = _current_vehicle_node(env)
    end_node = start_node
    visited_nodes = set()
    if start_node is not None:
        visited_nodes.add(start_node)
    if replay_writer is not None:
        _append_decision_replay_frame(
            writer=replay_writer,
            env=env,
            policy_name=policy_name,
            episode_index=episode_index,
            step_index=0,
            seed=seed,
        )

    while not (terminated or truncated):
        action_value = _select_action(policy_name, model, env, obs)
        action_counts[action_value] += 1
        obs, reward, terminated, truncated, info = env.step(action_value)
        end_node = _current_vehicle_node(env)
        if end_node is not None:
            visited_nodes.add(end_node)
        total_action_count += 1
        if not bool(info.get("dqn_action_was_valid", True)):
            invalid_action_count += 1
        episode_reward += float(reward)
        raw_episode_reward += float(info.get("raw_reward", reward))
        episode_steps += 1
        for record in info.get("dqn_boarding_debug_records", []):
            line = _format_boarding_debug_line(
                policy_name=policy_name,
                episode_index=episode_index,
                dqn_step=episode_steps,
                record=record,
            )
            print(line)

        if print_step_details and episode_steps <= detail_steps:
            print(
                f"  step={episode_steps:03d} "
                f"action={action_value:02d} "
                f"valid_action={bool(info.get('dqn_action_was_valid', True))} "
                f"scaled_reward={float(reward): .6f} "
                f"raw_reward={float(info.get('raw_reward', reward)): .6f} "
                f"current_U={info.get('current_utility')} "
                f"current_U_norm={info.get('current_normalized_utility')} "
                f"next_U={info.get('next_utility')} "
                f"current_A={info.get('current_total_accessibility')} "
                f"next_A={info.get('next_total_accessibility')}"
            )
        if replay_writer is not None:
            _append_decision_replay_frame(
                writer=replay_writer,
                env=env,
                policy_name=policy_name,
                episode_index=episode_index,
                step_index=episode_steps,
                seed=seed,
                action_value=action_value,
                info=info,
                reward=reward,
            )

    env.close()

    return {
        "reward": episode_reward,
        "raw_reward": raw_episode_reward,
        "steps": episode_steps,
        "action_counts": action_counts,
        "invalid_action_count": invalid_action_count,
        "total_action_count": total_action_count,
        "start_node": start_node,
        "end_node": end_node,
        "unique_nodes_visited": len(visited_nodes),
        "visited_nodes": sorted(visited_nodes, key=str),
    }


def evaluate_model(model, episodes=3, seed=100):
    """
    Run policy comparison after training.
    """
    log_lines = []
    episode_summary_rows = []

    def emit(line=""):
        print(line)
        log_lines.append(line)

    print("\n" + "=" * 70)
    print("Evaluating trained DQN model against baselines")
    print("=" * 70)
    emit("=" * 70)
    emit("Evaluating trained DQN model against baselines")
    emit(f"Rewards reported below are scaled by REWARD_SCALE={REWARD_SCALE:g}.")
    emit("Use raw_reward for the original unscaled accessibility-equity reward.")
    emit("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    policy_names = ["dqn", "random", "always_pass", "tsp"]
    replay_writer = None
    replay_video_path = None
    if SAVE_DECISION_REPLAY_VIDEO:
        replay_writer, replay_video_path = _open_video_writer(
            DECISION_REPLAY_VIDEO_PATH,
            fps=DECISION_REPLAY_FPS,
        )

    try:
        for policy_name in policy_names:
            emit("")
            emit("-" * 70)
            emit(f"Policy: {policy_name}")
            emit("-" * 70)

            episode_rewards = []
            episode_raw_rewards = []
            action_counts = Counter()
            invalid_action_count = 0
            total_action_count = 0

            for episode in range(episodes):
                result = _run_policy_episode(
                    policy_name=policy_name,
                    model=model,
                    episode_index=episode,
                    seed=seed + episode,
                    print_step_details=(episode == 0),
                    detail_steps=8,
                    replay_writer=replay_writer if policy_name == "dqn" else None,
                )
                episode_rewards.append(result["reward"])
                episode_raw_rewards.append(result["raw_reward"])
                action_counts.update(result["action_counts"])
                invalid_action_count += result["invalid_action_count"]
                total_action_count += result["total_action_count"]

                invalid_rate = (
                    result["invalid_action_count"] / result["total_action_count"]
                    if result["total_action_count"]
                    else 0.0
                )
                emit(
                    f"Episode {episode + 1}: "
                    f"scaled_reward={result['reward']:.6f}, "
                    f"raw_reward={result['raw_reward']:.6f}, "
                    f"steps={result['steps']}, "
                    f"invalid_actions={result['invalid_action_count']}/"
                    f"{result['total_action_count']} ({invalid_rate:.2%})"
                )

                most_common_action = None
                if result["action_counts"]:
                    most_common_action = result["action_counts"].most_common(1)[0][0]
                episode_summary_rows.append(
                    {
                        "policy": policy_name,
                        "episode": episode + 1,
                        "reward": result["reward"],
                        "raw_reward": result["raw_reward"],
                        "steps": result["steps"],
                        "invalid_actions": result["invalid_action_count"],
                        "total_actions": result["total_action_count"],
                        "most_common_action": most_common_action,
                        "unique_nodes_visited": result["unique_nodes_visited"],
                        "start_node": result["start_node"],
                        "end_node": result["end_node"],
                        "visited_nodes": " ".join(str(node) for node in result["visited_nodes"]),
                    }
                )

            mean_reward = sum(episode_rewards) / len(episode_rewards)
            mean_raw_reward = sum(episode_raw_rewards) / len(episode_raw_rewards)
            invalid_action_rate = (
                invalid_action_count / total_action_count
                if total_action_count
                else 0.0
            )
            emit(f"Mean evaluation scaled reward: {mean_reward:.6f}")
            emit(f"Mean evaluation raw reward: {mean_raw_reward:.6f}")
            emit(
                f"Invalid action rate: {invalid_action_count}/"
                f"{total_action_count} ({invalid_action_rate:.2%})"
            )
            emit(f"Most common actions: {action_counts.most_common(10)}")
    finally:
        if replay_writer is not None:
            replay_writer.close()
            emit(f"Decision replay video saved to: {replay_video_path}")

    LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    with EVALUATION_EPISODES_SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "policy",
            "episode",
            "reward",
            "raw_reward",
            "steps",
            "invalid_actions",
            "total_actions",
            "most_common_action",
            "unique_nodes_visited",
            "start_node",
            "end_node",
            "visited_nodes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_summary_rows)
    print(f"Evaluation summary saved to '{LOG_PATH}'")
    print(f"Evaluation episode summary CSV saved to '{EVALUATION_EPISODES_SUMMARY_CSV_PATH}'")


def main():
    """
    Run a small DQN training job.
    """
    print("=" * 70)
    print("Training DQN for Single-Vehicle Accessibility-Equity Control")
    print(f"Decision mode: {DECISION_MODE}")
    print(f"Scenario: humans={NUM_HUMANS}, vehicles={NUM_VEHICLES}, nodes={NUM_NODES}")
    print(
        "Scenario config: "
        f"seed={SCENARIO_CONFIG.seed}, "
        f"central_count={SCENARIO_CONFIG.central_count}"
    )
    print(
        "Mobility config: "
        f"human={MOBILITY_CONFIG.human_walking_speed_kmh:g} km/h, "
        f"vehicle={MOBILITY_CONFIG.vehicle_speed_kmh:g} km/h, "
        f"edge={MOBILITY_CONFIG.edge_speed_kmh:g} km/h"
    )
    print(
        "Reward config: "
        f"utility_mode={REWARD_CONFIG.utility_mode}, "
        f"beta={REWARD_CONFIG.beta:g}, alpha={REWARD_CONFIG.alpha:g}, "
        f"xi={REWARD_CONFIG.xi:g}, eta={REWARD_CONFIG.eta:g}, "
        f"normalize_utility={REWARD_CONFIG.normalize_utility}, "
        f"clip_normalized_utility={REWARD_CONFIG.clip_normalized_utility}, "
        f"normalized_reward_scale={REWARD_CONFIG.normalized_reward_scale:g}"
    )
    print(
        "DQN output config: "
        f"save_model={DQN_CONFIG.save_model}, "
        f"save_decision_replay_video={DQN_CONFIG.save_decision_replay_video}, "
        f"decision_replay_fps={DQN_CONFIG.decision_replay_fps}"
    )
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"NOTE: DQN rewards are scaled by REWARD_SCALE={REWARD_SCALE:g}.")
    print("      Monitor curves and scaled_reward logs are not raw rewards.")
    print("=" * 70)

    save_experiment_config(EXPERIMENT_CONFIG, CONFIG_PATH)
    print(f"Resolved config saved to '{CONFIG_PATH}'")
    save_training_scenario_diagnostics()
    if TRAIN_MONITOR_PATH.exists():
        TRAIN_MONITOR_PATH.unlink()
    env = make_env(seed=SCENARIO_CONFIG.seed, monitor=True, render_mode=None)

    model = DQN(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=DQN_CONFIG.learning_rate,
        buffer_size=DQN_CONFIG.buffer_size,
        learning_starts=DQN_CONFIG.learning_starts,
        batch_size=DQN_CONFIG.batch_size,
        gamma=DQN_CONFIG.gamma,
        train_freq=DQN_CONFIG.train_freq,
        target_update_interval=DQN_CONFIG.target_update_interval,
        exploration_fraction=DQN_CONFIG.exploration_fraction,
        exploration_final_eps=DQN_CONFIG.exploration_final_eps,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    if SAVE_MODEL:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_PATH)
    env.close()
    plot_training_rewards(TRAIN_MONITOR_PATH, TRAIN_CURVE_PATH)
    if TRAIN_MONITOR_PATH.exists():
        TRAIN_MONITOR_PATH.unlink()
    if SAVE_MODEL:
        print(f"Model saved to '{MODEL_PATH.with_suffix('.zip')}'")
    else:
        print("Model file not saved (SAVE_MODEL=False).")

    evaluate_model(
        model,
        episodes=DEFAULT_EVAL_EPISODES,
        seed=DEFAULT_EVAL_SEED,
    )


if __name__ == "__main__":
    main()

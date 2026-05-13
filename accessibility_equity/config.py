"""
Project-wide experiment configuration.

This is the single default source for scenario, mobility, reward, and DQN
training parameters. Scripts may still accept command-line overrides for
one-off inspection, but project defaults should live here.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ScenarioConfig:
    num_nodes: int = 6
    num_humans: int = 10
    num_vehicles: int = 1
    seed: int = 42
    central_count: int = 1

    @property
    def poi_kwargs(self) -> Dict[str, Any]:
        return {"central_count": self.central_count}

    def with_overrides(
        self,
        num_nodes: Optional[int] = None,
        num_humans: Optional[int] = None,
        num_vehicles: Optional[int] = None,
        seed: Optional[int] = None,
        central_count: Optional[int] = None,
    ) -> "ScenarioConfig":
        return ScenarioConfig(
            num_nodes=self.num_nodes if num_nodes is None else int(num_nodes),
            num_humans=self.num_humans if num_humans is None else int(num_humans),
            num_vehicles=self.num_vehicles if num_vehicles is None else int(num_vehicles),
            seed=self.seed if seed is None else int(seed),
            central_count=self.central_count if central_count is None else int(central_count),
        )


@dataclass(frozen=True)
class MobilityConfig:
    human_walking_speed_kmh: float = 3.0
    vehicle_speed_kmh: float = 30.0
    edge_speed_kmh: float = 30.0


@dataclass(frozen=True)
class RewardConfig:
    utility_mode: str = "equity"#"sum_accessibility"，"equity"
    beta: float = 1.0
    alpha: float = 0.8
    xi: float = 1.0
    eta: float = 2.0
    epsilon: float = 1e-6
    reward_mode: str = "time_integrated_utility"
    normalize_utility: bool = False
    clip_normalized_utility: bool = False
    normalized_reward_scale: float = 100.0


@dataclass(frozen=True)
class DQNConfig:
    decision_mode: str = "routing"
    output_run_name: Optional[str] = None
    max_steps: int = 100
    eval_episodes: int = 10
    eval_seed: int = 42
    save_model: bool = True
    total_timesteps: int = 20_000
    learning_rate: float = 1e-4
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    batch_size: int = 32
    gamma: float = 0.99
    train_freq: int = 4
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.2
    exploration_final_eps: float = 0.05
    save_decision_replay_video: bool = False
    decision_replay_fps: int = 2


@dataclass(frozen=True)
class ExperimentConfig:
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    mobility: MobilityConfig = field(default_factory=MobilityConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def with_overrides(
        self,
        scenario: Optional[ScenarioConfig] = None,
        mobility: Optional[MobilityConfig] = None,
        reward: Optional[RewardConfig] = None,
        dqn: Optional[DQNConfig] = None,
    ) -> "ExperimentConfig":
        return replace(
            self,
            scenario=self.scenario if scenario is None else scenario,
            mobility=self.mobility if mobility is None else mobility,
            reward=self.reward if reward is None else reward,
            dqn=self.dqn if dqn is None else dqn,
        )


def _read_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


def _read_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}.") from exc


def _read_str_env(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}.")


def load_experiment_config_from_env() -> ExperimentConfig:
    """
    Load shared defaults from environment variables.

    Scenario:
        AE_NUM_NODES, AE_NUM_HUMANS, AE_NUM_VEHICLES, AE_SEED,
        AE_CENTRAL_COUNT

    Mobility:
        AE_HUMAN_WALKING_SPEED_KMH, AE_VEHICLE_SPEED_KMH,
        AE_EDGE_SPEED_KMH

    Reward:
        AE_REWARD_UTILITY_MODE, AE_REWARD_BETA, AE_REWARD_ALPHA,
        AE_REWARD_XI, AE_REWARD_ETA, AE_REWARD_EPSILON, AE_REWARD_MODE,
        AE_REWARD_NORMALIZE_UTILITY, AE_REWARD_CLIP_NORMALIZED_UTILITY,
        AE_REWARD_NORMALIZED_SCALE

    DQN:
        AE_DQN_DECISION_MODE, AE_DQN_OUTPUT_RUN_NAME, AE_DQN_MAX_STEPS,
        AE_DQN_EVAL_EPISODES, AE_DQN_EVAL_SEED, AE_DQN_SAVE_MODEL,
        AE_DQN_TOTAL_TIMESTEPS, AE_DQN_LEARNING_RATE, AE_DQN_BUFFER_SIZE,
        AE_DQN_LEARNING_STARTS, AE_DQN_BATCH_SIZE, AE_DQN_GAMMA,
        AE_DQN_TRAIN_FREQ, AE_DQN_TARGET_UPDATE_INTERVAL,
        AE_DQN_EXPLORATION_FRACTION, AE_DQN_EXPLORATION_FINAL_EPS,
        AE_DQN_SAVE_DECISION_REPLAY_VIDEO, AE_DQN_DECISION_REPLAY_FPS
    """
    defaults = ExperimentConfig()
    scenario = ScenarioConfig(
        num_nodes=_read_int_env("AE_NUM_NODES", defaults.scenario.num_nodes),
        num_humans=_read_int_env("AE_NUM_HUMANS", defaults.scenario.num_humans),
        num_vehicles=_read_int_env("AE_NUM_VEHICLES", defaults.scenario.num_vehicles),
        seed=_read_int_env("AE_SEED", defaults.scenario.seed),
        central_count=_read_int_env("AE_CENTRAL_COUNT", defaults.scenario.central_count),
    )
    mobility = MobilityConfig(
        human_walking_speed_kmh=_read_float_env(
            "AE_HUMAN_WALKING_SPEED_KMH",
            defaults.mobility.human_walking_speed_kmh,
        ),
        vehicle_speed_kmh=_read_float_env(
            "AE_VEHICLE_SPEED_KMH",
            defaults.mobility.vehicle_speed_kmh,
        ),
        edge_speed_kmh=_read_float_env(
            "AE_EDGE_SPEED_KMH",
            defaults.mobility.edge_speed_kmh,
        ),
    )
    reward = RewardConfig(
        utility_mode=_read_str_env("AE_REWARD_UTILITY_MODE", defaults.reward.utility_mode),
        beta=_read_float_env("AE_REWARD_BETA", defaults.reward.beta),
        alpha=_read_float_env("AE_REWARD_ALPHA", defaults.reward.alpha),
        xi=_read_float_env("AE_REWARD_XI", defaults.reward.xi),
        eta=_read_float_env("AE_REWARD_ETA", defaults.reward.eta),
        epsilon=_read_float_env("AE_REWARD_EPSILON", defaults.reward.epsilon),
        reward_mode=_read_str_env("AE_REWARD_MODE", defaults.reward.reward_mode),
        normalize_utility=_read_bool_env(
            "AE_REWARD_NORMALIZE_UTILITY",
            defaults.reward.normalize_utility,
        ),
        clip_normalized_utility=_read_bool_env(
            "AE_REWARD_CLIP_NORMALIZED_UTILITY",
            defaults.reward.clip_normalized_utility,
        ),
        normalized_reward_scale=_read_float_env(
            "AE_REWARD_NORMALIZED_SCALE",
            defaults.reward.normalized_reward_scale,
        ),
    )
    dqn = DQNConfig(
        decision_mode=_read_str_env(
            "AE_DQN_DECISION_MODE",
            os.environ.get("DQN_DECISION_MODE", defaults.dqn.decision_mode),
        ),
        output_run_name=os.environ.get(
            "AE_DQN_OUTPUT_RUN_NAME",
            os.environ.get("DQN_OUTPUT_RUN_NAME", defaults.dqn.output_run_name),
        ),
        max_steps=_read_int_env("AE_DQN_MAX_STEPS", defaults.dqn.max_steps),
        eval_episodes=_read_int_env("AE_DQN_EVAL_EPISODES", defaults.dqn.eval_episodes),
        eval_seed=_read_int_env("AE_DQN_EVAL_SEED", scenario.seed),
        save_model=_read_bool_env("AE_DQN_SAVE_MODEL", defaults.dqn.save_model),
        total_timesteps=_read_int_env("AE_DQN_TOTAL_TIMESTEPS", defaults.dqn.total_timesteps),
        learning_rate=_read_float_env("AE_DQN_LEARNING_RATE", defaults.dqn.learning_rate),
        buffer_size=_read_int_env("AE_DQN_BUFFER_SIZE", defaults.dqn.buffer_size),
        learning_starts=_read_int_env("AE_DQN_LEARNING_STARTS", defaults.dqn.learning_starts),
        batch_size=_read_int_env("AE_DQN_BATCH_SIZE", defaults.dqn.batch_size),
        gamma=_read_float_env("AE_DQN_GAMMA", defaults.dqn.gamma),
        train_freq=_read_int_env("AE_DQN_TRAIN_FREQ", defaults.dqn.train_freq),
        target_update_interval=_read_int_env(
            "AE_DQN_TARGET_UPDATE_INTERVAL",
            defaults.dqn.target_update_interval,
        ),
        exploration_fraction=_read_float_env(
            "AE_DQN_EXPLORATION_FRACTION",
            defaults.dqn.exploration_fraction,
        ),
        exploration_final_eps=_read_float_env(
            "AE_DQN_EXPLORATION_FINAL_EPS",
            defaults.dqn.exploration_final_eps,
        ),
        save_decision_replay_video=_read_bool_env(
            "AE_DQN_SAVE_DECISION_REPLAY_VIDEO",
            defaults.dqn.save_decision_replay_video,
        ),
        decision_replay_fps=_read_int_env(
            "AE_DQN_DECISION_REPLAY_FPS",
            defaults.dqn.decision_replay_fps,
        ),
    )
    return ExperimentConfig(
        scenario=scenario,
        mobility=mobility,
        reward=reward,
        dqn=dqn,
    )


def save_experiment_config(config: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


DEFAULT_EXPERIMENT_CONFIG = load_experiment_config_from_env()
DEFAULT_SCENARIO_CONFIG = DEFAULT_EXPERIMENT_CONFIG.scenario


def load_scenario_config_from_env() -> ScenarioConfig:
    """Backward-compatible scenario-only config loader."""
    return load_experiment_config_from_env().scenario

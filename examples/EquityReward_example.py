from pathlib import Path
import sys
import networkx as nx
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from accessibility_equity.config import (
    DEFAULT_EXPERIMENT_CONFIG,
    save_experiment_config,
)
from accessibility_equity.policies import HeuristicRoutingHumanPolicy
from accessibility_equity.rewards import compute_scenario_utility_bounds
from accessibility_equity.visualization import (
    render_episode_frame_array,
    save_scenario_figure,
)
from accessibility_equity.wrappers import REWARD_SCALE, create_transport_env
from accessibility_equity.wrappers import REWARD_SCALE, create_dqn_env
from accessibility_equity.heuristics import TSPVehicleAgent
from accessibility_equity.rewards.efficient_equity_reward import EquityReward 

EXPERIMENT_CONFIG = DEFAULT_EXPERIMENT_CONFIG
SCENARIO_CONFIG = EXPERIMENT_CONFIG.scenario
MOBILITY_CONFIG = EXPERIMENT_CONFIG.mobility
REWARD_CONFIG = EXPERIMENT_CONFIG.reward

NUM_HUMANS = SCENARIO_CONFIG.num_humans
NUM_VEHICLES = SCENARIO_CONFIG.num_vehicles
NUM_NODES = SCENARIO_CONFIG.num_nodes

equity_reward = EquityReward(
    REWARD_CONFIG.beta, 
    REWARD_CONFIG.alpha, 
    REWARD_CONFIG.xi, 
    REWARD_CONFIG.eta, 
    MOBILITY_CONFIG.human_walking_speed_kmh,
    MOBILITY_CONFIG.vehicle_speed_kmh)

env = create_dqn_env(
    num_humans=NUM_HUMANS,
    num_vehicles=NUM_VEHICLES,
    num_nodes=NUM_NODES,
    human_policy_class=HeuristicRoutingHumanPolicy,
    seed=42,
    reward_function=equity_reward.reward
)

equity_reward.initialize(
    env.base_env.env.network, 
    env.base_env.vehicle_agents,
    env.base_env.human_agents,
    )

#env.base_env.env.start_video_recording()

obs, info = env.reset()
# env.base_env.env.enable_rendering("graphical")

tsp_agent = TSPVehicleAgent(env)

for _ in range(1):
    action = tsp_agent.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    # env.base_env.env.render()

# env.base_env.env.save_video("test.mp4", fps=5)

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

EXPERIMENT_CONFIG = DEFAULT_EXPERIMENT_CONFIG
SCENARIO_CONFIG = EXPERIMENT_CONFIG.scenario
MOBILITY_CONFIG = EXPERIMENT_CONFIG.mobility
REWARD_CONFIG = EXPERIMENT_CONFIG.reward

NUM_HUMANS = SCENARIO_CONFIG.num_humans
NUM_VEHICLES = SCENARIO_CONFIG.num_vehicles
NUM_NODES = SCENARIO_CONFIG.num_nodes


env = create_dqn_env(
    num_humans=NUM_HUMANS,
    num_vehicles=NUM_VEHICLES,
    num_nodes=NUM_NODES,
    human_policy_class=HeuristicRoutingHumanPolicy,
    seed=0,
    render_mode="human",
)

env.base_env.env.start_video_recording()

obs, info = env.reset()
env.base_env.env.enable_rendering("graphical")

tsp_agent = TSPVehicleAgent(env)

for _ in range(100):
    action = tsp_agent.get_action(obs)
    print(action)
    obs, reward, terminated, truncated, info = env.step(action)
    env.base_env.env.render()

env.base_env.env.save_video("test.mp4", fps=5)

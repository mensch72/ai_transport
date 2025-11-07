# ai_transport

A PettingZoo environment for multi-agent transport systems, following the [Parallel API](https://pettingzoo.farama.org/api/parallel/).

## Overview

This environment simulates a transport network with two types of agents:
- **Humans**: Agents with a `speed` attribute
- **Vehicles**: Agents with `speed`, `capacity`, and `fuel_use` attributes

The environment operates on a network represented as a NetworkX directed graph with:
- Node attribute: `name`
- Edge attributes: `length`, `speed`, `capacity`

## Installation

```bash
pip install -e .
```

For development with testing:
```bash
pip install -e ".[dev]"
```

## Environment State

The environment maintains the following state components:

1. **real_time**: A real-valued continuous time (distinct from discrete time steps)
2. **Agent positions**: Each agent has a position which is either:
   - A node in the network, or
   - A tuple `(edge, coordinate)` where `coordinate` is between 0 and `edge.length`
3. **Vehicle destinations**: Each vehicle has a destination which is either:
   - `None` (no current destination), or
   - A node in the network
4. **Human aboard status**: Each human has an `aboard` status which is either:
   - `None` (not aboard any vehicle), or
   - The ID of a vehicle (aboard that vehicle)
5. **Step type**: The current step type, which determines which agents can take which actions:
   - `routing`: Vehicles at nodes can set their destination
   - `unboarding`: Humans aboard vehicles at nodes can unboard
   - `boarding`: Humans at nodes can board vehicles at the same node
   - `departing`: Vehicles at nodes can depart into outgoing edges; humans at nodes (not aboard) can walk into outgoing edges

## Action Spaces

Action spaces are dynamic and depend on the current `step_type` and agent state:

### Routing Step
- **Vehicles at nodes**: Can set destination to `None` or any node (N+1 actions where N is number of nodes)
- **All other agents**: Can only pass (1 action)

### Unboarding Step
- **Humans aboard vehicles at nodes**: Can pass or unboard (2 actions)
- **All other agents**: Can only pass (1 action)

### Boarding Step
- **Humans at nodes (not aboard)**: Can pass or board any vehicle at the same node (M+1 actions where M is number of vehicles at that node)
- **All other agents**: Can only pass (1 action)

### Departing Step
- **Vehicles at nodes**: Can pass or depart into any outgoing edge (E+1 actions where E is number of outgoing edges)
- **Humans at nodes (not aboard)**: Can pass or walk into any outgoing edge (E+1 actions)
- **All other agents**: Can only pass (1 action)

**Note**: Agents on edges (not at nodes) can only pass in all step types.

## Usage

### Basic Example

```python
from ai_transport import parallel_env
import networkx as nx

# Create a custom network
G = nx.DiGraph()
G.add_node(0, name="Station_A")
G.add_node(1, name="Station_B")
G.add_edge(0, 1, length=10.0, speed=5.0, capacity=10)

# Create environment
env = parallel_env(
    num_humans=2,
    num_vehicles=1,
    network=G
)

# Reset environment
observations, infos = env.reset()

# Take a step
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
observations, rewards, terminations, truncations, infos = env.step(actions)
```

### Custom Configuration

```python
env = parallel_env(
    render_mode="human",
    num_humans=3,
    num_vehicles=2,
    network=G,
    human_speed=1.5,
    vehicle_speed=3.0,
    vehicle_capacity=5,
    vehicle_fuel_use=1.2
)
```

## Step Logic

The environment processes actions and updates state based on the current `step_type`:

### Routing Step
- **State Changes**: Vehicles at nodes can change their `vehicle_destination` to `None` or any node
- **Time**: Real time does NOT advance
- **Example**: Vehicle sets destination to node 2

### Unboarding Step
- **State Changes**: Humans aboard vehicles at nodes can change their `aboard` status to `None`
- **Time**: Real time does NOT advance
- **Example**: Human unboards from vehicle

### Boarding Step
- **State Changes**: Humans at nodes attempt to board vehicles at the same node
- **Capacity Constraint**: Humans are processed in random order; only board if vehicle not full (humans aboard < capacity)
- **Time**: Real time does NOT advance
- **Example**: Two humans try to board a vehicle with capacity 2; both succeed

### Departing Step
- **State Changes**: 
  1. Agents at nodes that don't pass move to `(chosen_edge, 0.0)`
  2. For all agents on edges, compute remaining duration: `(edge_length - coordinate) / speed`
  3. Find minimum duration `delta_t` and advance `real_time` by that amount
  4. Move all agents on edges: `new_coord = coord + speed * delta_t`
  5. Agents reaching edge end (coord ≈ length) move to target node
  6. Humans aboard vehicles have their position synchronized with their vehicle
- **Speed**: Vehicles use edge speed; humans use their own speed
- **Time**: Real time ADVANCES by minimum remaining duration
- **Example**: Vehicle and human depart; vehicle reaches destination first, human still on edge

See `examples/step_logic_demo.py` for a complete demonstration.

## Package Structure

```
ai_transport/
├── ai_transport/          # Main package
│   ├── __init__.py       # Package initialization
│   └── envs/             # Environment modules
│       ├── __init__.py
│       └── transport_env.py  # Main environment implementation
├── tests/                # Test suite
│   └── test_transport_env.py
├── examples/             # Usage examples
│   └── basic_example.py
├── pyproject.toml        # Package configuration
└── README.md            # This file
```

## API

The environment follows the PettingZoo Parallel API. Key methods:

- `env.reset(seed=None, options=None)`: Reset the environment
- `env.step(actions)`: Take a step with the given actions
- `env.render()`: Render the current state (if render_mode is set)
- `env.close()`: Clean up resources

## Development

Run tests:
```bash
pytest tests/
```

Run examples:
```bash
python examples/basic_example.py
python examples/action_spaces_demo.py
python examples/step_logic_demo.py
```

## Note

Observations and rewards are not yet implemented. The environment currently returns placeholder zero observations and rewards. These will be specified and implemented in future updates.

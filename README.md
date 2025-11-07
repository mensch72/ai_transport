# ai_transport

PettingZoo parallel environment for AI-governed public transport simulation.

## Overview

This package provides a multi-agent reinforcement learning environment for simulating AI-controlled public transport systems. Multiple AI agents control different vehicles (buses) to efficiently transport passengers across a network of stops.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from ai_transport import parallel_env

# Create the environment
env = parallel_env(
    num_buses=3,
    num_stops=10,
    max_passengers_per_vehicle=50,
    max_passengers_per_stop=20,
    render_mode="human"
)

# Reset the environment
observations, infos = env.reset(seed=42)

# Run simulation
for step in range(100):
    # Sample random actions for all agents
    actions = {
        agent: env.action_space(agent).sample()
        for agent in env.agents
    }
    
    # Execute step
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    # Render
    env.render()
    
    if any(terminations.values()) or any(truncations.values()):
        break

env.close()
```

## Environment Details

### Agents
- Multiple bus agents (e.g., "bus_0", "bus_1", "bus_2")
- Each agent controls one vehicle in the transport network

### Actions
Each agent can choose from 3 discrete actions:
- `0`: Move to next stop (pick up/drop off passengers)
- `1`: Wait at current stop (pick up more passengers)
- `2`: Skip stop (express route, move 2 stops ahead)

### Observations
Each agent receives a 5-dimensional observation vector:
- Current stop position
- Number of passengers on vehicle
- Number of passengers waiting at current stop
- Number of passengers waiting at next stop
- Current timestep

### Rewards
Agents receive rewards based on:
- Picking up passengers (+0.5 per passenger)
- Dropping off passengers (+1.0 per passenger)
- Penalties for waiting (-0.1) or skipping stops (-0.2)
- Penalties for overcrowded stops (-0.1 per crowded stop)

### Episode Termination
Episodes terminate after 200 timesteps (truncation).

## PettingZoo Parallel API Compliance

This environment fully implements the PettingZoo Parallel API as described in the [official documentation](https://pettingzoo.farama.org/api/parallel/).

Key features:
- `parallel_env()`: Create a parallel environment instance
- `reset()`: Reset environment and return observations
- `step(actions)`: Execute actions for all agents simultaneously
- `observation_spaces`: Dict of observation spaces for each agent
- `action_spaces`: Dict of action spaces for each agent

## Running Tests

```bash
pytest test_env.py -v
```

## Example

Run the included example script:

```bash
python example.py
```

## License

MIT

# Task Specification

## Project Goal

This project studies how an autonomous vehicle fleet can be controlled to
improve urban accessibility in a more equitable way. The task is formulated on
top of `ai_transport`, but the reinforcement learning objective follows the
accessibility-equity perspective described in `1.Model Formulation.docx`.

More specifically, the project aims to use vehicle routing and fleet control as
the intervention mechanism for reducing residential accessibility inequality.

## Learning Setting

- The learning setting is **single-agent reinforcement learning**.
- The RL agent is a **centralized supervisor**.
- The supervisor controls **all vehicles jointly**.
- Humans are **not learning agents**. They are treated as part of the
  environment and follow rule-based behavior.

## Environment Layers

The project is organized into four layers:

- **scenario**
  - defines how the network, hierarchy, POIs, and initial human positions are initialized
- **environment**
  - reuses `ai_transport` to define state transitions and transport mechanics
- **policy**
  - defines rule-based human behavior and baseline vehicle behavior
- **wrapper**
  - exposes the simulator as a single-agent RL task

## Initialization

At reset time, the project keeps the base `ai_transport` vehicle initialization
logic, but overrides the human initialization rule:

- POI and residential area labels are first assigned on the network
- humans are then placed only on nodes labeled as `residential`
- humans are initialized as not aboard any vehicle

This makes the initial state consistent with the current scenario design, where
residential areas are the starting locations of human agents.

## Observation

At the current stage, the wrapper keeps the observation structure close to
`ai_transport.wrappers.gym_wrapper`:

- step type
- current simulation time
- vehicle positions
- human positions

This is only a first version. Future project-specific observation features are
expected to include:

- hierarchy information
- POI-related information
- accessibility or demand-related aggregates

## Action

Only vehicles are controlled by the RL agent.

The current wrapper follows the same centralized fleet-control idea already used
in `ai_transport.wrappers.gym_wrapper`:

- one external action is provided for each vehicle
- the wrapper combines these vehicle actions with internally generated human actions

## Reward

The long-term project objective is to define reward through:

- accessibility at the individual level
- fairness-aware aggregation at the population level

At the current stage, the default wrapper reward is already connected to the
project reward modules:

- location-based individual accessibility under the current transport system
- inequality-sensitive population utility
- step reward based on utility improvement

The intended interpretation is:

- individual accessibility is attached to the human's effective reachable
  opportunities from a relevant location
- vehicle routes influence accessibility by changing effective travel costs
- fairness is evaluated with residential inequality reduction in mind, not as
  route richness alone

## Current Implementation Boundary

Already grounded in `ai_transport`:

- simulator dynamics
- centralized fleet-control wrapper skeleton
- baseline human and vehicle policy interfaces

Already implemented in the project layer:

- POI distribution over Voronoi-based spatial partitions
- initial human placement on residential nodes
- a first route-influenced accessibility metric
- equity-aware reward aggregation

Still project-specific and not yet fully implemented:

- richer human target generation beyond initial placement
- transition from route-based approximation toward fully location-based accessibility
- richer project-specific observations beyond positions and step type

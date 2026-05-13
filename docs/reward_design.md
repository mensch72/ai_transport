# Reward Design

This document describes the current accessibility and equity reward design used
in `ai_transport_accessibility_equity`.

## Design Principle

The reward is built in two layers:

1. individual accessibility value
2. population-level inequality-sensitive aggregation

This follows the formulation in `1.Model Formulation.docx`, while adapting the
definition of accessibility to the transport setting used in this project.

## Individual Accessibility

For each human `h`, the individual locational value is defined as:

```text
X_h(s_t) = sum_{c=1}^M a_c ( sum_{j: c_j = c} exp(-beta d_j) )^alpha
```

where:

- `a_c > 0` is the importance weight of POI category `c`
- `beta > 0` controls time-distance decay
- `alpha in (0, 1)` captures diminishing returns within each category

### Interpretation in This Project

In this project, `X_h(s_t)` should be understood as a
location-based accessibility value:

- each human has a relevant location, with residential origin as the main
  equity reference point
- accessibility is defined by the opportunities that can be reached from that
  location under the current transport system
- vehicle routing is not the accessibility value itself
- instead, vehicle routing changes the effective cost of reaching POIs and
  therefore changes accessibility

So the transport mechanism affects accessibility through the effective travel
time term:

```text
d_hj(s_t)
```

where `d_hj(s_t)` is the effective travel time for human `h` to access POI `j`
under the current system state.

This is not a geometric distance. It is a time-based accessibility cost.

In the intended model, it should reflect the current transport system,
including factors such as:

- the human's current location
- available vehicles
- vehicle routes
- riding time
- walking time before or after vehicle use

In the current implementation, this effective travel time is still
approximated using the network structure and vehicle route information.

This means routes matter because they can improve or worsen access to valuable
opportunities such as:

- workplaces
- supermarkets
- healthcare facilities

## Population-Level Equity Utility

After computing `X_h(s_t)` for all humans, the population-level utility is
defined as:

```text
U(s_t) = - (sum_h X_h(s_t)^(-xi))^eta
```

where:

- `xi` controls inequality aversion inside the aggregation
- `eta` controls the outer transformation
- In the default training configuration, `eta = 2` so that better
  accessibility produces a larger utility value for RL maximization.

### Interpretation

This aggregation is designed to give greater weight to improvements for poorly
served individuals. Since low-accessibility humans have smaller `X_h`, the term
`X_h^(-xi)` becomes larger for them, making the population utility more
sensitive to deprivation.

In the intended project interpretation, this is not only route fairness. The
goal is residential equity:

- humans from weaker or more peripheral residential contexts should be harder
  to neglect
- route planning is the intervention mechanism
- residential accessibility inequality reduction is the welfare objective

## Reward for Reinforcement Learning

There are two closely related quantities:

### 1. State Utility

`U(s_t)` is the utility of the current state. It can be used as:

- an analysis metric
- a debugging signal
- a state-based training reward

In the current training setup, this state utility is used directly as the
reward returned by the Gym wrapper:

```text
r_t = U(s_{t+1})
```

The post-action state `s_{t+1}` is used because Gym rewards are attached to the
transition after applying the action.

### 2. Utility-Improvement Reward

An alternative training reward is:

```text
r_t = U(s_{t+1}) - U(s_t)
```

Optionally, an operating-cost term can be added:

```text
r_t = U(s_{t+1}) - U(s_t) - lambda * operating_cost_t
```

This step reward measures whether the current action improves the
equity-sensitive accessibility objective. It remains available in the reward
module, but it is not the default wrapper reward.

If the exact sign convention of `U(s_t)` makes better states numerically
smaller, the training reward should use the corresponding direction-consistent
improvement term while still preserving the same welfare ordering.

## Current Implementation Mapping

The current code splits the reward logic into two modules:

- [`accessibility.py`](C:/Users/yingyuel/Desktop/codex_aitranport/ai_transport_accessibility_equity/ai_transport_accessibility_equity/rewards/accessibility.py)
  - currently computes a route-influenced approximation to individual accessibility
  - should evolve toward location-based accessibility under the current transport system
  - now uses a minimal effective travel time approximation based on edge length and speed

- [`equity_reward.py`](C:/Users/yingyuel/Desktop/codex_aitranport/ai_transport_accessibility_equity/ai_transport_accessibility_equity/rewards/equity_reward.py)
  - aggregates individual values into `U(s_t)`
  - computes the training reward from utility differences

## Notes

- The current implementation uses the sign-correct form
  `U(s_t) = - (sum_h X_h(s_t)^(-xi))^eta`, with default `eta = 2`.
- The sum inside the power is positive, and the leading negative sign is outside
  the power. This avoids complex values while preserving the intended welfare
  ordering.

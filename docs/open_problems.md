# Date

2026-04-21

# Problem1

The current reward does not yet fully match the residential equity goal.

The intended research goal of this project is:

- before intervention, humans living in remote or peripheral residential areas
  have lower accessibility than humans living in central residential areas
- after intervention by an AI-controlled vehicle fleet, the accessibility gap
  between remote residents and central residents should become smaller
- if that gap becomes smaller, the transport system should be considered more
  equal and should receive a higher reward

At the current stage, the project already contains:

- residential node assignment
- center / intermediate / peripheral hierarchy
- human initialization on residential nodes
- a location-based but route-influenced accessibility approximation
- an inequality-sensitive population aggregation function

However, the reward still does not fully encode the intended residential equity
goal.

The main gap is:

- the current reward is not yet explicitly tied to the human's residential
  origin (`home_node`)
- therefore, it does not yet directly prioritize reducing the accessibility gap
  between remote residents and central residents

This means the current model can improve accessibility in a general sense, but
it cannot yet be said with confidence that it is optimizing residential
accessibility equality.

# How To Solve

The reward should explicitly incorporate residential origin.

A good next step is:

- use `home_node` as an explicit human attribute throughout the reward pipeline
- compare accessibility outcomes across residential groups
- keep the current inequality-averse aggregation in its original form for now

Possible implementation direction:

1. direct group-gap reduction objective

For example:

- compute average accessibility for central residents
- compute average accessibility for peripheral residents
- reward the system when the gap is reduced

This direction may be closer to the research goal statement.

# Date

2026-04-22

# Problem1

There is a risk of double counting if remote residents are given an extra
residential weight on top of already having lower accessibility values.

If low accessibility already reflects the disadvantage of living in remote or
peripheral residential areas, then adding an extra group weight may count the
same disadvantage twice.

# How To Solve

Current preferred idea:

- do not add an extra weight to remote residents
- instead, directly compare the accessibility gap between central and
  peripheral residents
- reward the system when this gap becomes smaller

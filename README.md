# ai_transport_accessibility_equity

`ai_transport_accessibility_equity` is a reinforcement learning project built on
top of `ai_transport`.

The design principle is to keep the transport simulator from `ai_transport` as
intact as possible while adding new layers for:

- accessibility- and equity-oriented task design
- scenario generation for networks, spatial hierarchy, POIs, and initial human placement
- centralized single-agent control of multiple vehicles
- reward definitions based on accessibility and fairness

Current status:
- the package structure is in place
- environment, wrapper, and policy modules already reuse `ai_transport` logic
- scenario network construction and hierarchy assignment now have first working versions
- accessibility/equity reward modules and some scenario modules still need project-specific content

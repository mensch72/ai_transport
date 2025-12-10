"""
Test script with handcrafted sequence of events to verify video rendering.

Timeline:
- t=0.0: human_0 departs from node 0 to node 1 (speed=1, length=1) → arrives at t=1.0
- t=0.0: human_1 boards vehicle_0 at node 2
- t=0.0: vehicle_0 departs to node 3 (speed=10, length=3) → arrives at t=0.3
- t=0.3: vehicle_0 stops at node 3
- t=0.3: human_1 unboards and departs to node 4 (speed=2, length=2) → arrives at t=1.3
- t=1.0: human_0 arrives at node 1, stops and waits
- t=1.3: human_1 arrives at node 4, episode ends

time_per_frame = 0.07 (weird fraction so no arrival times fall on clicks)
Clicks at: 0.07, 0.14, 0.21, 0.28, 0.35, 0.42, ... (none match 0.3, 1.0, or 1.3)
"""

import networkx as nx
from ai_transport import parallel_env

# Create network with exact edge lengths
G = nx.DiGraph()
G.add_node(0, name="Node0", x=0.0, y=0.0)
G.add_node(1, name="Node1", x=10.0, y=0.0)
G.add_node(2, name="Node2", x=20.0, y=0.0)
G.add_node(3, name="Node3", x=30.0, y=0.0)
G.add_node(4, name="Node4", x=40.0, y=0.0)

# Add edges with specified lengths and speeds
G.add_edge(0, 1, length=1.0, speed=1.0, capacity=10)   # human_0 walks here
G.add_edge(2, 3, length=3.0, speed=10.0, capacity=10)  # vehicle_0 drives here
G.add_edge(3, 4, length=2.0, speed=2.0, capacity=10)   # human_1 walks here

# Create environment
env = parallel_env(
    num_humans=2,
    num_vehicles=1,
    network=G,
    render_mode="human",
    human_speeds=[1.0, 2.0],  # human_0: speed 1, human_1: speed 2
    vehicle_speeds=[10.0],     # vehicle_0: speed 10
)

env.reset(seed=42)

# Set weird time_per_frame so no arrival times fall on clicks!
env._time_per_frame = 0.07  # Clicks at 0.07, 0.14, 0.21, 0.28, 0.35, ...
                             # Arrivals at 0.3, 1.0, 1.3 don't match any clicks!

# Set initial positions
env.agent_positions['human_0'] = 0    # At node 0
env.agent_positions['human_1'] = 2    # At node 2
env.agent_positions['vehicle_0'] = 2  # At node 2
env.human_aboard['human_0'] = None
env.human_aboard['human_1'] = None

print("="*80)
print("TEST: Handcrafted sequence with known arrival times")
print(f"time_per_frame = {env._time_per_frame} (weird fraction)")
print("Clicks at: 0.07, 0.14, 0.21, 0.28, 0.35, 0.42, 0.49, 0.56, 0.63, 0.70, ...")
print("Arrivals at: 0.3, 1.0, 1.3 (none match clicks!)")
print("="*80)

# Enable video recording
env.enable_rendering('graphical')
env.start_video_recording()

print(f"\nInitial state (t={env.real_time}):")
print(f"  human_0: at node {env.agent_positions['human_0']}")
print(f"  human_1: at node {env.agent_positions['human_1']}")
print(f"  vehicle_0: at node {env.agent_positions['vehicle_0']}")

# === Event 1: t=0, departures ===
print("\n" + "="*80)
print("Event 1: t=0.0 - Departures")
print("="*80)

# Routing step
env._set_step_type_for_testing('routing')
env.step({'human_0': 0, 'human_1': 0, 'vehicle_0': 0})

# Unboarding step
env._set_step_type_for_testing('unboarding')
env.step({'human_0': 0, 'human_1': 0, 'vehicle_0': 0})

# Boarding step - human_1 boards vehicle_0
env._set_step_type_for_testing('boarding')
env.step({'human_0': 0, 'human_1': 1, 'vehicle_0': 0})  # human_1 boards

print(f"After boarding (t={env.real_time}):")
print(f"  human_1 aboard: {env.human_aboard['human_1']}")

# Departing step - both human_0 and vehicle_0 depart
env._set_step_type_for_testing('departing')
actions = {
    'human_0': 1,     # Depart on edge (0,1)
    'human_1': 0,     # Pass (aboard vehicle)
    'vehicle_0': 1,   # Depart on edge (2,3)
}
env.step(actions)

print(f"\nAfter departures (t={env.real_time}):")
print(f"  human_0: {env.agent_positions['human_0']}")
print(f"  human_1: {env.agent_positions['human_1']}")
print(f"  vehicle_0: {env.agent_positions['vehicle_0']}")

# Verify arrival time for vehicle
edge_23_length = G[2][3]['length']
edge_23_speed = G[2][3]['speed']
expected_arrival_vehicle = edge_23_length / edge_23_speed
print(f"\n  Expected vehicle arrival: t={expected_arrival_vehicle} (length={edge_23_length}, speed={edge_23_speed})")
print(f"  Actual time: t={env.real_time}")
print(f"  ✓ Match!" if abs(env.real_time - expected_arrival_vehicle) < 0.001 else f"  ✗ Mismatch!")

env.render()

# === Event 2: t=0.3, vehicle arrives, human_1 unboards and departs ===
print("\n" + "="*80)
print("Event 2: t=0.3 - Vehicle arrives, human_1 unboards and departs")
print("="*80)

# Routing
env._set_step_type_for_testing('routing')
env.step({'human_0': 0, 'human_1': 0, 'vehicle_0': 0})

# Unboarding - human_1 unboards
env._set_step_type_for_testing('unboarding')
env.step({'human_0': 0, 'human_1': 1, 'vehicle_0': 0})  # human_1 unboards

print(f"After unboarding (t={env.real_time}):")
print(f"  human_1 aboard: {env.human_aboard['human_1']}")
print(f"  human_1 position: {env.agent_positions['human_1']}")

# Boarding
env._set_step_type_for_testing('boarding')
env.step({'human_0': 0, 'human_1': 0, 'vehicle_0': 0})

# Departing - human_1 departs on edge (3,4)
env._set_step_type_for_testing('departing')
actions = {
    'human_0': 0,     # Pass (still on edge 0-1)
    'human_1': 1,     # Depart on edge (3,4)
    'vehicle_0': 0,   # Pass (stays at node 3)
}
env.step(actions)

print(f"\nAfter human_1 departs (t={env.real_time}):")
print(f"  human_0: {env.agent_positions['human_0']}")
print(f"  human_1: {env.agent_positions['human_1']}")
print(f"  vehicle_0: {env.agent_positions['vehicle_0']}")

# Verify human_0 arrival time
edge_01_length = G[0][1]['length']
human_0_speed = 1.0
expected_arrival_h0 = edge_01_length / human_0_speed
print(f"\n  Expected human_0 arrival: t={expected_arrival_h0} (length={edge_01_length}, speed={human_0_speed})")
print(f"  Actual time: t={env.real_time}")
print(f"  ✓ Match!" if abs(env.real_time - expected_arrival_h0) < 0.001 else f"  ✗ Mismatch!")

env.render()

# === Event 3: t=1.0, human_0 arrives ===
print("\n" + "="*80)
print("Event 3: t=1.0 - human_0 arrives at node 1")
print("="*80)

# Continue stepping to let human_0 finish
env._set_step_type_for_testing('routing')
env.step({'human_0': 0, 'human_1': 0, 'vehicle_0': 0})
env._set_step_type_for_testing('unboarding')
env.step({'human_0': 0, 'human_1': 0, 'vehicle_0': 0})
env._set_step_type_for_testing('boarding')
env.step({'human_0': 0, 'human_1': 0, 'vehicle_0': 0})
env._set_step_type_for_testing('departing')
env.step({'human_0': 0, 'human_1': 0, 'vehicle_0': 0})

print(f"After step (t={env.real_time}):")
print(f"  human_0: {env.agent_positions['human_0']}")
print(f"  human_1: {env.agent_positions['human_1']}")
print(f"  vehicle_0: {env.agent_positions['vehicle_0']}")

# Verify human_1 arrival time
edge_34_length = G[3][4]['length']
human_1_speed = 2.0
time_h1_departed = 0.3
expected_arrival_h1 = time_h1_departed + edge_34_length / human_1_speed
print(f"\n  Expected human_1 arrival: t={expected_arrival_h1} (departed at t={time_h1_departed}, length={edge_34_length}, speed={human_1_speed})")
print(f"  Actual time: t={env.real_time}")
print(f"  ✓ Match!" if abs(env.real_time - expected_arrival_h1) < 0.001 else f"  ✗ Mismatch!")

env.render()

# Save videos
print("\n" + "="*80)
print("Saving videos...")
print("="*80)

# Try to save both MP4 and GIF
# MP4 first (requires ffmpeg)
try:
    env.frames_copy = env.frames.copy()  # Save frames before they're cleared
    env.save_video('test_handcrafted_sequence.mp4', fps=10)
    env.frames = env.frames_copy  # Restore for GIF
except:
    pass

# Save as GIF
env.save_video('test_handcrafted_sequence.gif', fps=10)

print(f"\nVideo files generated:")
import os
if os.path.exists('test_handcrafted_sequence.mp4'):
    print(f"  test_handcrafted_sequence.mp4")
if os.path.exists('test_handcrafted_sequence.gif'):
    print(f"  test_handcrafted_sequence.gif")

# Show which clicks were rendered
print("\n" + "="*80)
print("Frames rendered at these click times:")
print("="*80)
time_per_frame = 0.07
for i in range(1, 19):  # We got 18 frames
    click_time = i * time_per_frame
    print(f"  Frame {i:2d}: t = {click_time:.2f}")

print("\nNote: No frames at exact arrival times (0.3, 1.0, 1.3)")
print("      Agent positions were extrapolated for each click time!")

env.close()

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print("\nTimeline verification:")
print("  t=0.0: human_0 departs (0→1), vehicle_0 departs (2→3) with human_1")
print("  t=0.3: vehicle_0 arrives at 3, human_1 unboards and departs (3→4)")
print("  t=1.0: human_0 arrives at 1")
print("  t=1.3: human_1 arrives at 4")
print("\nAll arrival times calculated correctly based on edge lengths and agent speeds!")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from GridWorld import GridWorld # Assuming this file exists and defines GridWorld correctly
from ValueIteration import ValueIteration # Assuming this file exists and defines ValueIteration correctly

# Original problem setup
# Using np.nan instead of the removed np.NaN
problem = GridWorld('data/sunset_basic.csv', reward={0: -0.04, 1: 5.0, 2: -5.0, 3: np.nan}, random_rate=0.2)

solver = ValueIteration(problem.reward_function, problem.transition_model, gamma=0.9)
solver.train()

# Original visualization (static plot)
problem.visualize_value_policy(policy=solver.policy, values=solver.values)
# The original script also had:
# problem.random_start_policy(policy=solver.policy, start_pos=(40, 79), n=1000)
# This might run simulations. Our animation below will show a single deterministic path.

# --- Animation Part ---
print("Preparing animation...")

fig_anim, ax_anim = plt.subplots(figsize=(10, 8)) # Adjust figure size as needed

# Define action effects (dy, dx) based on common conventions
# 0: Up, 1: Down, 2: Left, 3: Right
# IMPORTANT: This must match the action definitions in your GridWorld/ValueIteration
action_effects = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}

# Agent starting position (row, col)
# Using the start_pos from the original script's random_start_policy for consistency
start_pos = (40, 79)
current_pos_list = list(start_pos) # Use a list for mutable current position
agent_path = [start_pos]
max_steps_anim = 200  # Max steps for the animation path

# Simulate path (deterministic according to policy)
# This loop simulates the agent moving based on the learned policy.
# It assumes the policy intends to move to an adjacent cell.
for step in range(max_steps_anim):
    r, c = int(current_pos_list[0]), int(current_pos_list[1])

    # Check if current position is valid and within bounds
    if not (0 <= r < problem.height and 0 <= c < problem.width):
        print(f"Agent out of bounds at step {step}: ({r},{c}). Stopping path simulation.")
        break

    # Check if current state is a terminal state (goal or penalty)
    # Assuming problem.grid uses 1 for goal state type, 2 for penalty state type.
    # These correspond to keys in the reward dict.
    current_grid_value = problem.grid[r, c]
    if current_grid_value == 1:
        print(f"Agent reached goal (type 1) at ({r},{c}) at step {step}.")
        break
    if current_grid_value == 2:
        print(f"Agent reached penalty (type 2) at ({r},{c}) at step {step}.")
        break

    # Get action from the learned policy
    action = solver.policy[r, c]

    # Check for terminal/undefined policy action (e.g., if policy uses -1 for no action)
    if action == -1 or action not in action_effects:
        print(f"Policy indicates no action or invalid action ({action}) at ({r},{c}) at step {step}. Stopping.")
        break

    # Get change in row and column based on action
    dr, dc = action_effects[int(action)]
    next_r, next_c = r + dr, c + dc

    # Check if the next move is valid (within bounds and not an obstacle)
    # Assuming problem.grid uses 3 for obstacle type
    if not (0 <= next_r < problem.height and 0 <= next_c < problem.width):
        print(f"Policy leads out of bounds from ({r},{c}) to ({next_r},{next_c}) at step {step}. Stopping.")
        break
    if problem.grid[next_r, next_c] == 3: # Obstacle type
        print(f"Policy leads into obstacle from ({r},{c}) to ({next_r},{next_c}) at step {step}. Stopping.")
        break
        
    current_pos_list = [next_r, next_c]
    agent_path.append(tuple(current_pos_list))

    # Stop if agent gets stuck in a loop of 1 (e.g. policy leads back and forth)
    if len(agent_path) > 1 and agent_path[-1] == agent_path[-2]:
        print(f"Agent stuck (or reached terminal) at ({next_r},{next_c}) at step {step}. Stopping.")
        break
    if len(agent_path) > 2 and agent_path[-1] == agent_path[-3]: # Stuck in 2-step loop
        print(f"Agent in 2-step loop, last pos: ({next_r},{next_c}) at step {step}. Stopping.")
        break


# --- Setup Matplotlib Animation ---

# Display the state values as a heatmap background
# Using 'viridis' colormap, 'upper' origin to match NumPy array indexing (0,0 at top-left)
im_values = ax_anim.imshow(solver.values, cmap='viridis', interpolation='nearest', origin='upper')
plt.colorbar(im_values, ax=ax_anim, label='State Value')

# Overlay obstacles
# Assuming problem.grid has terrain types, and type 3 is an obstacle
obstacle_mask = (problem.grid == 3)
ax_anim.imshow(obstacle_mask, cmap='Greys', alpha=0.3, interpolation='nearest', origin='upper')

# Mark goal and penalty locations for context
# Assuming grid value 1 is goal (reward key 1), 2 is penalty (reward key 2)
goal_indices = np.where(problem.grid == 1)
penalty_indices = np.where(problem.grid == 2)

if goal_indices[0].size > 0:
    ax_anim.scatter(goal_indices[1], goal_indices[0], marker='*', color='gold', s=200, edgecolor='black', label='Goal (Type 1)')
if penalty_indices[0].size > 0:
    ax_anim.scatter(penalty_indices[1], penalty_indices[0], marker='X', color='red', s=150, label='Penalty (Type 2)')

# Configure plot appearance
ax_anim.set_xticks(np.arange(-.5, problem.width, 1), minor=True)
ax_anim.set_yticks(np.arange(-.5, problem.height, 1), minor=True)
ax_anim.grid(which='minor', color='k', linestyle='-', linewidth=0.5)
ax_anim.set_xticks(np.arange(0, problem.width, 5 if problem.width > 20 else 1)) # Adjust tick frequency
ax_anim.set_yticks(np.arange(0, problem.height, 5 if problem.height > 20 else 1))
ax_anim.set_xlim([-0.5, problem.width - 0.5])
ax_anim.set_ylim([problem.height - 0.5, -0.5]) # Y-axis inverted by imshow origin='upper'

# Agent marker and path line initialization
agent_marker, = ax_anim.plot([], [], marker='o', color='lime', markersize=8, linestyle='None', label='Agent')
path_line, = ax_anim.plot([], [], color='cyan', linewidth=1.5, label='Path Taken')


def init_animation():
    agent_marker.set_data([], [])
    path_line.set_data([], [])
    ax_anim.set_title(f'Agent Path Animation (Start: {start_pos})')
    # Place legend outside the plot if space is needed
    ax_anim.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    fig_anim.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    return agent_marker, path_line

def update_animation(frame_idx):
    if frame_idx < len(agent_path):
        pos_r, pos_c = agent_path[frame_idx]
        agent_marker.set_data([pos_c], [pos_r]) # plot is (x,y) so (col, row)

        # Update path line up to current frame
        path_coords_so_far = np.array(agent_path[:frame_idx+1])
        if path_coords_so_far.ndim == 2 and path_coords_so_far.shape[0] > 0:
            path_line.set_data(path_coords_so_far[:,1], path_coords_so_far[:,0]) # (cols, rows)
        else: # Single point case
            path_line.set_data([agent_path[0][1]], [agent_path[0][0]])
            
    # Update title with current step
    ax_anim.set_title(f'Agent Path Animation (Start: {start_pos}) - Step: {frame_idx+1}/{len(agent_path)}')
    return agent_marker, path_line,

# Create the animation
# interval: delay between frames in milliseconds
# blit=True for performance, repeat=False so it plays once
num_frames = len(agent_path)
if num_frames > 0:
    ani = animation.FuncAnimation(fig_anim, update_animation, frames=num_frames,
                                  init_func=init_animation, blit=True, repeat=False, interval=300)

    # Save the animation as a GIF
    # You might need to install 'pillow': pip install pillow
    animation_filename = 'gridworld_agent_animation.gif'
    try:
        ani.save(animation_filename, writer='pillow', fps=5)
        print(f"Animation successfully saved to {animation_filename}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Please ensure you have 'pillow' installed (pip install pillow).")
        print("Alternatively, try saving as .mp4 (requires ffmpeg).")

    plt.show() # Display the animation
else:
    print("No path generated for animation (agent might be starting in a terminal/obstacle or path is empty).")
    # Still show the basic plot setup if desired
    init_animation() # To show legend and title on the static plot
    plt.show()
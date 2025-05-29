import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from time import time

# Assuming GridWorld.py is in the same directory
from GridWorld import GridWorld

class AnimatableGridWorld(GridWorld):
    def __init__(self, filename, reward, random_rate, time_limit=1000):
        super().__init__(filename, reward, random_rate, time_limit)

    def execute_policy_for_animation(self, policy, start_pos, max_steps_override=None):
        path_states = []
        path_coords = []

        s = self.get_state_from_pos(start_pos)
        initial_r, initial_c = self.get_pos_from_state(s)
        
        if self.map[initial_r, initial_c] == 3:
            print(f"Error: Agent start position {start_pos} is an obstacle.")
            return float('-inf'), [], [], True 

        path_states.append(s)
        path_coords.append(self.get_pos_from_state(s))

        current_reward_val = self.reward_function[s]
        total_reward = current_reward_val

        start_sim_time = int(round(time() * 1000))
        overtime = False
        
        max_s = max_steps_override if max_steps_override is not None else self.num_states * 2 
        if max_s == 0 : max_s = 1 # Ensure at least one step if path_coords already has start

        step_count = 0
        while (self.reward.get(1.0) is not None and current_reward_val != self.reward[1.0]) and \
              (self.reward.get(2.0) is not None and current_reward_val != self.reward[2.0]) and \
              step_count < max_s:
            
            if not (0 <= s < self.num_states and 0 <= policy[s] < self.num_actions):
                print(f"Error: Invalid state {s} or action {policy[s]}. Stopping path generation.")
                overtime = True
                break

            s_prime = np.random.choice(self.num_states, p=self.transition_model[s, policy[s]])
            
            path_states.append(s_prime)
            path_coords.append(self.get_pos_from_state(s_prime))
            
            current_reward_val = self.reward_function[s_prime]
            total_reward += current_reward_val
            s = s_prime
            step_count += 1

            cur_sim_time = int(round(time() * 1000)) - start_sim_time
            if cur_sim_time > self.time_limit:
                overtime = True
                print("Simulation overtime during agent path generation.")
                break
        
        if step_count >= max_s and not overtime:
            print(f"Agent reached max steps ({max_s}) during path generation.")

        final_reward = float('-inf') if overtime and not path_coords else total_reward # Avoid -inf if path started
        return final_reward, path_states, path_coords, overtime

def generate_random_enemy_path(grid_world_obj, start_pos, num_steps):
    """Generates a simple random walk path for an enemy."""
    enemy_path_coords = []
    if not start_pos: return []

    s = grid_world_obj.get_state_from_pos(start_pos)
    r_start, c_start = start_pos

    if grid_world_obj.map[r_start, c_start] == 3: # Obstacle
        print(f"Warning: Enemy start position {start_pos} is an obstacle. Enemy will not move.")
        # Keep enemy static at its invalid start or a default valid spot if needed.
        # For now, just return its start pos repeated if it's on an obstacle.
        return [start_pos] * num_steps 
    
    enemy_path_coords.append((r_start,c_start))

    for i in range(num_steps -1): # num_steps includes the start position
        current_r, current_c = grid_world_obj.get_pos_from_state(s)
        possible_next_positions = []

        # Try N, E, S, W moves
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            next_r, next_c = current_r + dr, current_c + dc
            if 0 <= next_r < grid_world_obj.num_rows and \
               0 <= next_c < grid_world_obj.num_cols and \
               grid_world_obj.map[next_r, next_c] != 3: # Not an obstacle
                possible_next_positions.append((next_r, next_c))
        
        if possible_next_positions:
            next_pos = possible_next_positions[np.random.randint(len(possible_next_positions))]
            s = grid_world_obj.get_state_from_pos(next_pos)
            enemy_path_coords.append(next_pos)
        else: # Stuck
            enemy_path_coords.append(grid_world_obj.get_pos_from_state(s)) # Stay in the same place
            
    return enemy_path_coords


def animate_all_movements(grid_world_obj, agent_path_coords, enemy_path_coords, fig_size=(8, 6), interval=200):
    if not agent_path_coords:
        print("Agent path is empty, cannot animate.")
        return

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.axis('off')

    num_rows, num_cols = grid_world_obj.num_rows, grid_world_obj.num_cols
    world_map = grid_world_obj.map
    unit = max(1, min(fig_size[1] // num_rows, fig_size[0] // num_cols))

    # Draw grid and map features (same as before)
    for i in range(num_cols + 1):
        ax.plot([i * unit, i * unit], [0, num_rows * unit], color='black' if i == 0 or i == num_cols else 'grey', linestyle='-' if i == 0 or i == num_cols else 'dashed', alpha=0.7)
    for i in range(num_rows + 1):
        ax.plot([0, num_cols * unit], [i * unit, i * unit], color='black' if i == 0 or i == num_rows else 'grey', linestyle='-' if i == 0 or i == num_rows else 'dashed', alpha=0.7)

    for r_idx in range(num_rows):
        for c_idx in range(num_cols):
            y, x = (num_rows - 1 - r_idx) * unit, c_idx * unit
            facecolor, alpha = 'white', 0.6
            if world_map[r_idx, c_idx] == 3: facecolor = 'black'
            elif world_map[r_idx, c_idx] == 2: facecolor = 'salmon' # Static trap color
            elif world_map[r_idx, c_idx] == 1: facecolor = 'lightgreen' # Goal color
            if world_map[r_idx, c_idx] != 0:
                 rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor=facecolor, alpha=alpha)
                 ax.add_patch(rect)

    # Agent (blue)
    agent_r_init, agent_c_init = agent_path_coords[0]
    agent_circle = patches.Circle(((agent_c_init + 0.5) * unit, (num_rows - 1 - agent_r_init + 0.5) * unit), radius=unit * 0.3, color='blue', alpha=0.9, zorder=10)
    ax.add_patch(agent_circle)

    # Enemy (red)
    enemy_circle = None
    if enemy_path_coords:
        enemy_r_init, enemy_c_init = enemy_path_coords[0]
        enemy_circle = patches.Circle(((enemy_c_init + 0.5) * unit, (num_rows - 1 - enemy_r_init + 0.5) * unit), radius=unit * 0.3, color='red', alpha=0.9, zorder=9)
        ax.add_patch(enemy_circle)
    
    title = ax.text(0.5, 1.05, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes, ha="center", zorder=11)

    # Determine the number of frames for the animation (longest path)
    num_frames = len(agent_path_coords)
    if enemy_path_coords:
        num_frames = max(num_frames, len(enemy_path_coords))


    def update(frame_num):
        # Update Agent
        if frame_num < len(agent_path_coords):
            agent_r, agent_c = agent_path_coords[frame_num]
            agent_circle.center = ((agent_c + 0.5) * unit, (num_rows - 1 - agent_r + 0.5) * unit)
        
        # Update Enemy
        if enemy_circle and frame_num < len(enemy_path_coords):
            enemy_r, enemy_c = enemy_path_coords[frame_num]
            enemy_circle.center = ((enemy_c + 0.5) * unit, (num_rows - 1 - enemy_r + 0.5) * unit)
        elif enemy_circle and enemy_path_coords: # Enemy path shorter, keep at last known pos
             enemy_r, enemy_c = enemy_path_coords[-1]
             enemy_circle.center = ((enemy_c + 0.5) * unit, (num_rows - 1 - enemy_r + 0.5) * unit)


        title.set_text(f'Step: {frame_num + 1}/{num_frames}')
        
        elements_to_return = [agent_circle, title]
        if enemy_circle:
            elements_to_return.append(enemy_circle)
        return tuple(elements_to_return)

    ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                  interval=interval, blit=True, repeat=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    map_file = "sunset_basic.csv"
    try:
        with open(map_file, 'x') as f:
            f.write("0,0,0,0,1\n0,3,3,0,0\n0,0,0,3,2\n0,3,0,0,0\n0,0,0,0,0\n") # 5x5 map
            print(f"Created sample map: '{map_file}'")
    except FileExistsError:
        print(f"Using existing map: '{map_file}'")
    except IOError as e:
        print(f"Error with map file: {e}. Please create '{map_file}' manually.")
        exit()

    rewards = { 0.0: -0.1, 1.0: 100.0, 2.0: -50.0, 3.0: 0.0 }
    rand_rate = 0.15
    t_limit = 3000 
    agent_max_steps = 30 # Shorter for random policy visualization

    animation_speed_ms = 250
    figure_dimensions = (7, 7)

    try:
        anim_gw = AnimatableGridWorld(map_file, rewards, rand_rate, t_limit)
    except Exception as e:
        print(f"FATAL Error initializing GridWorld: {e}")
        exit()

    # --- Agent ---
    agent_policy = anim_gw.generate_random_policy()
    agent_start_r, agent_start_c = 0, 0 # Top-left
    if anim_gw.map[agent_start_r, agent_start_c] == 3: # If start is obstacle
        print(f"Agent start ({agent_start_r},{agent_start_c}) is obstacle. Finding new start.")
        # Basic fallback: find first non-obstacle cell
        for r in range(anim_gw.num_rows):
            for c in range(anim_gw.num_cols):
                if anim_gw.map[r,c] != 3:
                    agent_start_r, agent_start_c = r,c
                    break
            if anim_gw.map[agent_start_r, agent_start_c] != 3: break
    agent_start_pos = (agent_start_r, agent_start_c)
    
    print(f"Agent starting at: {agent_start_pos}")
    _, _, agent_coords_path, _ = anim_gw.execute_policy_for_animation(
        agent_policy, agent_start_pos, max_steps_override=agent_max_steps
    )
    if not agent_coords_path: # If agent path generation failed critically at start
        print("Agent path could not be generated. Exiting.")
        exit()


    # --- Enemy ---
    enemy_start_r, enemy_start_c = anim_gw.num_rows - 1, anim_gw.num_cols - 1 # Bottom-right
    if anim_gw.map[enemy_start_r, enemy_start_c] == 3: # If start is obstacle
         print(f"Enemy start ({enemy_start_r},{enemy_start_c}) is obstacle. Finding new start.")
         for r in range(anim_gw.num_rows-1, -1, -1): # search from bottom up
            for c in range(anim_gw.num_cols-1, -1, -1):
                if anim_gw.map[r,c] != 3:
                    enemy_start_r, enemy_start_c = r,c
                    break
            if anim_gw.map[enemy_start_r, enemy_start_c] != 3: break
    enemy_start_pos = (enemy_start_r, enemy_start_c)

    print(f"Enemy starting at: {enemy_start_pos}")
    # Enemy path length can be tied to agent's path length or a fixed number
    enemy_path_length = len(agent_coords_path) # Make enemy move for same duration as agent
    enemy_coords_path = generate_random_enemy_path(anim_gw, enemy_start_pos, enemy_path_length)


    if not agent_coords_path:
        print("No agent path generated, cannot animate.")
    else:
        print(f"Agent path: {len(agent_coords_path)} steps. Enemy path: {len(enemy_coords_path)} steps.")
        animate_all_movements(
            anim_gw, agent_coords_path, enemy_coords_path,
            fig_size=figure_dimensions, interval=animation_speed_ms
        )
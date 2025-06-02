import numpy as np

from GridWorld import GridWorld
from ValueIteration import ValueIteration

#problem = GridWorld('data/sunset_basic.csv', reward={0: -0.04, 1: 5.0, 2: -5.0, 3: np.NaN}, random_rate=0.2)
#problem = GridWorld('data/sunset_heatmap.csv', reward={0: -0.04, 1: 10.0, 2: -10.0, 3: np.NaN, 4:-0.1}, random_rate=0.2)
#problem = GridWorld('data/defender_sunset.csv', reward={0: -0.04, 1: 10.0, 2: -10.0, 3: np.NaN, 4:-0.1}, random_rate=0.2)
problem = GridWorld('data/defender_sunset_heatmap.csv', reward={0: -0.04, 1: 10.0, 2: -10.0, 3: np.NaN, 4:-0.1}, random_rate=0.2)
problem.plot_map(fig_size=(8, 6))

solver = ValueIteration(problem.reward_function, problem.transition_model, gamma=0.9)
solver.train()

problem.plot_policy(policy=solver.policy, fig_size=(8, 6))
#problem.visualize_value_policy2(policy=solver.policy, values=solver.values, fig_size=(16,16))
problem.random_start_policy(policy=solver.policy, start_pos=(3, 40), n=1000)

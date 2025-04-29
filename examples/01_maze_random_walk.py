import numpy as np
import matplotlib.pyplot as plt

from sampling_planners.envs.plane_env import PlaneEnv, PlaneObstacleShapes

# 1) define workspace limits
x_lim = (0.0, 10.0)
y_lim = (0.0, 5.0)
# 2) define obstacles: circle + box (capsule future support commented)
obstacles_type = [
    PlaneObstacleShapes.CIRCLE,
    PlaneObstacleShapes.BOX,
    PlaneObstacleShapes.CAPSULE,  # future support
]
obstacles_data = [
    np.array([3.0, 2.5, 1.0]),  # circle: (x=3,y=2.5), r=1
    np.array([6.0, 1.0, 8.0, 3.0]),  # box: x0=6,y0=1 → x1=8,y1=3
    np.array([1.0, 4.0, 4.0, 4.5, 0.5]),  # capsule: x0,y0,x1,y1,r
]

# 3) construct environment
env = PlaneEnv(x_lim, y_lim, obstacles_type=obstacles_type, obstacles_data=obstacles_data, min_obstacle_distance=0.1)

# 4) test API
start = env.draw_random_state()
goal = env.draw_random_state()
print(f"Start: {start}, valid? {env.is_state_valid(start)}")
print(f" Goal: {goal}, valid? {env.is_state_valid(goal)}")
print(f"Distance start -> goal: {env.distance(start, goal):.3f}")

# 5) build trajectory
T = 50
traj = np.linspace(start, goal, T)

# 6) build one growing tree
tree_nodes = [[env.draw_random_state()] for _ in range(T)]
tree_edges = [[(0, 1)]] + [[(0, i), (1, i)] for i in range(T)]

# 7) visualize single tree
env.visualize(start, goal, tree_nodes, tree_edges)

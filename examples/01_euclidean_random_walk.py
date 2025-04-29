import numpy as np
from sampling_planners.envs.euclidean_env import EuclideanEnv, EuclidObstacleShapes

# 1) workspace
x_lim = (0, 10)
y_lim = (0, 8)
z_lim = (0, 5)
# 2) obstacles: sphere, box, capsule
obs_type = [EuclidObstacleShapes.SPHERE, EuclidObstacleShapes.BOX, EuclidObstacleShapes.CAPSULE]
obs_data = [
    np.array([4, 4, 2, 1]),  # sphere (x,y,z,r)
    np.array([6, 1, 1, 8, 4, 3]),  # box (x0,y0,z0,x1,y1,z1)
    np.array([1, 6, 1, 3, 2, 4, 0.5]),  # capsule (a_xyz, b_xyz, r)
]
# 3) env
env = EuclideanEnv(x_lim, y_lim, z_lim, obs_type, obs_data, min_obstacle_distance=0.1)
# 4) test
start = env.draw_random_state()
goal = env.draw_random_state()
print(f"Start: {start}, valid? {env.is_state_valid(start)}")
print(f" Goal: {goal}, valid? {env.is_state_valid(goal)}")
print(f"Distance: {env.distance(start, goal):.3f}")
# 5) straight trajectory
T = 60
traj = np.linspace(start, goal, T)
# 6) build one growing tree

tree_nodes = [[env.draw_random_state()] for i in range(T)]
tree_edges = [[(0, i)] for i in range(T)]
# 7) visualize single tree
env.visualize(start, goal, tree_nodes, tree_edges)

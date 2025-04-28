import numpy as np
import matplotlib.pyplot as plt

from sampling_planners.envs.plane_env import PlaneTask, PlaneObstacleShapes

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
env = PlaneTask(x_lim, y_lim, obstacles_type=obstacles_type, obstacles_data=obstacles_data, min_obstacle_distance=0.1)

# 4) test API
start = env.draw_random_state()
goal = env.draw_random_state()
print(f"Start: {start}, valid? {env.is_state_valid(start)}")
print(f" Goal: {goal}, valid? {env.is_state_valid(goal)}")
print(f"Distance start→goal: {env.distance(start, goal):.3f}")

# 5) build a dummy straight‐line trajectory
T = 50
traj = np.linspace(start, goal, T)

# 6) build a “tree” of random valid samples that grows over time
tree = []
for i in range(T):
    pts = np.vstack([env.draw_random_state() for _ in range(i + 1)])
    tree.append(pts)

# 7) visualize
env.visualize(traj, tree)

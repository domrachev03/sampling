import argparse

import numpy as np

from sampling_planners.envs.euclidean_env import EuclideanEnv, EuclidObstacleShapes

from sampling_planners.envs.plane_env import PlaneObstacleShapes, PlaneEnv
from sampling_planners.planners.primitive_planner import PrimitiveTreePlanner
from sampling_planners.planners.rrt_planner import RRTPlanner


def run_2d(choose_nearest: bool = False, planner_type: str = "primitive"):
    x_lim, y_lim = (0, 10), (0, 5)
    obs_types = [PlaneObstacleShapes.CIRCLE, PlaneObstacleShapes.BOX]
    obs_data = [
        np.array([3.0, 2.5, 1.0]),  # circle: x,y,r
        np.array([6.0, 1.0, 8.0, 3.0]),  # box: x0,y0,x1,y1
    ]
    env = PlaneEnv(x_lim, y_lim, obs_types, obs_data, min_obstacle_distance=0.1)

    start, goal = env.draw_random_state(), env.draw_random_state()
    print(choose_nearest)

    # select planner
    if planner_type == "rrt":
        planner = RRTPlanner(start, goal, env, sol_threshold=0.2, extend_step=0.05)
    else:
        planner = PrimitiveTreePlanner(
            start, goal, env, sol_threshold=0.2, extend_step=0.05, choose_nearest=choose_nearest
        )

    N_iter = 100
    for _ in range(N_iter):
        planner.step()
        # while not planner.step():
        pass
    print(f"Found solution: {planner.opt_path} with distance {planner.min_dist:.3f}")
    planner.visualize()


def run_3d(choose_nearest: bool = False, planner_type: str = "primitive"):
    x_lim, y_lim, z_lim = (0, 3), (0, 3), (0, 3)
    obs_types = [EuclidObstacleShapes.SPHERE, EuclidObstacleShapes.BOX]
    obs_data = [
        np.array([2.0, 2.0, 2.0, 0.8]),
        np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0]),
    ]
    env = EuclideanEnv(
        x_lim,
        y_lim,
        z_lim,
        obs_types,
        obs_data,
        min_obstacle_distance=0.1,
    )

    start, goal = env.draw_random_state(), env.draw_random_state()
    print(choose_nearest)

    # select planner
    if planner_type == "rrt":
        planner = RRTPlanner(start, goal, env, sol_threshold=0.2, extend_step=0.2)
    else:
        planner = PrimitiveTreePlanner(
            start, goal, env, sol_threshold=0.2, extend_step=0.2, choose_nearest=choose_nearest
        )

    N_iter = 500
    for _ in range(N_iter):
        planner.step()
        pass
    print(f"Found solution: {planner.opt_path} with distance {planner.min_dist:.3f}")
    planner.visualize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", choices=["2d", "3d"], default="3d")
    parser.add_argument("--planner", choices=["primitive", "rrt"], default="rrt")
    parser.add_argument("--choose-nearest", action="store_true", default=True)
    args = parser.parse_args()

    if args.dim == "2d":
        run_2d(choose_nearest=args.choose_nearest, planner_type=args.planner)
    else:
        run_3d(choose_nearest=args.choose_nearest, planner_type=args.planner)

import argparse

import numpy as np

from sampling_planners.envs.euclidean_env import EuclideanEnv, EuclidObstacleShapes
from sampling_planners.envs.plane_env import PlaneEnv, PlaneObstacleShapes
from sampling_planners.planners.primitive_planner import PrimitiveTreePlanner
from sampling_planners.planners.rrt_planner import RRTPlanner
from sampling_planners.planners.rrt_star_planner import RrtStarPlanner


def run_2d(
    choose_nearest: bool = False,
    planner_type: str = "primitive",
    sigma: float = 1.0,
    p_sample_goal: float = 0.0,
    use_global_cost: bool = False,
    skip_extend_in_choose_parent: bool = False,
):
    x_lim, y_lim = (0, 10), (0, 5)
    obs_types = [PlaneObstacleShapes.CIRCLE, PlaneObstacleShapes.BOX]
    obs_data = [
        np.array([3.0, 2.5, 1.0]),  # circle: x,y,r
        np.array([6.0, 1.0, 8.0, 3.0]),  # box: x0,y0,x1,y1
    ]
    env = PlaneEnv(x_lim, y_lim, obs_types, obs_data, min_obstacle_distance=0.1)

    start, goal = env.draw_random_state(), env.draw_random_state()

    # select planner
    if planner_type == "primitive":
        planner = PrimitiveTreePlanner(
            start, goal, env, sol_threshold=0.2, extend_step=0.2, choose_nearest=choose_nearest
        )
    elif planner_type == "rrt":
        planner = RRTPlanner(start, goal, env, sol_threshold=0.2, extend_step=0.2, p_sample_goal=p_sample_goal)
    else:  # rrt_star
        planner = RrtStarPlanner(
            start,
            goal,
            env,
            sol_threshold=0.2,
            extend_step=0.2,
            sigma=sigma,
            p_sample_goal=p_sample_goal,
            use_global_cost=use_global_cost,
            skip_extend_in_choose_parent=skip_extend_in_choose_parent,
        )

    N_iter = 500
    for _ in range(N_iter):
        planner.step()
        # while not planner.step():
        pass
    print(f"Found solution: {planner.opt_path} with distance {planner.min_dist:.3f}")
    planner.visualize()


def run_3d(
    choose_nearest: bool = False,
    planner_type: str = "primitive",
    sigma: float = 1.0,
    p_sample_goal: float = 0.0,
    use_global_cost: bool = False,
    skip_extend_in_choose_parent: bool = False,
):
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
    if planner_type == "primitive":
        planner = PrimitiveTreePlanner(
            start, goal, env, sol_threshold=0.2, extend_step=0.1, choose_nearest=choose_nearest
        )
    elif planner_type == "rrt":
        planner = RRTPlanner(start, goal, env, sol_threshold=0.2, extend_step=0.1, p_sample_goal=p_sample_goal)
    else:  # rrt_star
        planner = RrtStarPlanner(
            start,
            goal,
            env,
            sol_threshold=0.2,
            extend_step=0.1,
            sigma=sigma,
            use_global_cost=use_global_cost,
            skip_extend_in_choose_parent=skip_extend_in_choose_parent,
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
    parser.add_argument(
        "--planner",
        choices=["primitive", "rrt", "rrt_star"],
        default="rrt",
        help="Which planner to run",
    )
    parser.add_argument("--choose-nearest", action="store_true", default=True)
    parser.add_argument(
        "--sigma",
        type=float,
        default=5.0,
        help="Rewiring radius sigma for RRT*",
    )
    parser.add_argument("--p-sample-goal", type=float, default=0.1, help="Probability of sampling the goal in RRT")
    parser.add_argument(
        "--use-global-cost",
        action="store_true",
        default=False,
        help="Flag to use global cost in RRT*",
    )
    parser.add_argument(
        "--skip-extend-in-choose-parent",
        action="store_true",
        default=True,
        help="Flag to skip extension in choose-parent step for RRT*",
    )
    args = parser.parse_args()

    if args.dim == "2d":
        run_2d(
            choose_nearest=args.choose_nearest,
            planner_type=args.planner,
            sigma=args.sigma,
            p_sample_goal=args.p_sample_goal,
            use_global_cost=args.use_global_cost,
            skip_extend_in_choose_parent=args.skip_extend_in_choose_parent,
        )
    else:
        run_3d(
            choose_nearest=args.choose_nearest,
            planner_type=args.planner,
            sigma=args.sigma,
            p_sample_goal=args.p_sample_goal,
            use_global_cost=args.use_global_cost,
            skip_extend_in_choose_parent=args.skip_extend_in_choose_parent,
        )

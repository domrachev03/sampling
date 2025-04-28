from enum import Enum

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle

from .base_env import BaseEnv


class PlaneObstacleShapes(Enum):
    CIRCLE = 1
    BOX = 2
    CAPSULE = 3


# class PlaneTask(BaseEnv):
class PlaneTask(BaseEnv):
    def __init__(
        self,
        x_lim: tuple[float, float],
        y_lim: tuple[float, float],
        obstacles_type: list[PlaneObstacleShapes],
        obstacles_data: list[np.ndarray],
        min_obstacle_distance: float = 0.0,
    ):
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.obstacles_type = obstacles_type
        self.obstacles_data = obstacles_data
        self.min_obstacle_distance = min_obstacle_distance

    def _signed_distance(self, state: np.ndarray, obs_id: int) -> float:
        obs_type = self.obstacles_type[obs_id]
        match obs_type:
            case PlaneObstacleShapes.CIRCLE:
                center = self.obstacles_data[obs_id][:2]
                radius = self.obstacles_data[obs_id][2]
                return np.linalg.norm(state - center) - radius
            case PlaneObstacleShapes.BOX:
                # compute signed distance to axis-aligned rectangle
                x0, y0, x1, y1 = self.obstacles_data[obs_id]
                px, py = state
                # outwards gap along each axis
                dx = max(x0 - px, 0.0, px - x1)
                dy = max(y0 - py, 0.0, py - y1)
                outside_dist = np.hypot(dx, dy)
                if dx == 0.0 and dy == 0.0:
                    # inside: negative distance to nearest edge
                    inside_dist = min(px - x0, x1 - px, py - y0, y1 - py)
                    return -inside_dist
                return outside_dist
            case PlaneObstacleShapes.CAPSULE:
                x0, y0, x1, y1, r = self.obstacles_data[obs_id]
                a = np.array([x0, y0])
                b = np.array([x1, y1])
                ab = b - a
                # project point onto AB, clamp to [0,1]
                t = np.dot(state - a, ab) / np.dot(ab, ab)
                t = np.clip(t, 0.0, 1.0)
                closest = a + t * ab
                # signed distance = dist(point, segment) - radius
                return np.linalg.norm(state - closest) - r

            case _:
                raise ValueError("Unknown obstacle type")

    def is_state_valid(self, state):
        if not (self.x_lim[0] <= state[0] <= self.x_lim[1]) or not (self.y_lim[0] <= state[1] <= self.y_lim[1]):
            return False

        for i in range(len(self.obstacles_type)):
            if self._signed_distance(state, i) < self.min_obstacle_distance:
                return False

        return True

    def distance(self, s1: np.ndarray, s2: np.ndarray) -> float:
        return np.linalg.norm(s1 - s2)

    def draw_random_state(self):
        while True:
            state = np.random.uniform([self.x_lim[0], self.y_lim[0]], [self.x_lim[1], self.y_lim[1]])
            if self.is_state_valid(state):
                return state

    def visualize(self, traj: np.ndarray, trees: list[list[np.ndarray]], filename: str = "") -> None:
        """
        Animate the growing trees and the execution of the trajectory.
        traj: shape (T,2)
        trees: list of length N; trees[j][i].shape = (Ni,2)
        """
        fig, ax = plt.subplots()
        ax.set_aspect("equal", "box")
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)

        # draw static obstacles
        for obs_type, data in zip(self.obstacles_type, self.obstacles_data):
            if obs_type is PlaneObstacleShapes.CIRCLE:
                x, y, r = data
                patch = Circle((x, y), r, color="k", alpha=0.4)
                ax.add_patch(patch)

            elif obs_type is PlaneObstacleShapes.BOX:
                x0, y0, x1, y1 = data
                patch = Rectangle((x0, y0), x1 - x0, y1 - y0, color="k", alpha=0.4)
                ax.add_patch(patch)

            # optional capsule drawing: two end‐caps + rectangle
            elif obs_type is PlaneObstacleShapes.CAPSULE:
                x0, y0, x1, y1, r = data
                # main rect
                dx = x1 - x0
                dy = y1 - y0
                length = np.hypot(dx, dy)
                angle = np.degrees(np.arctan2(dy, dx))
                rect = Rectangle((0, -r), length, 2 * r, color="k", alpha=0.4)
                t = mpl.transforms.Affine2D().rotate_deg(angle).translate(x0, y0)
                rect.set_transform(t + ax.transData)
                ax.add_patch(rect)
                ax.add_patch(Circle((x0, y0), r, color="k", alpha=0.4))
                ax.add_patch(Circle((x1, y1), r, color="k", alpha=0.4))

        T = len(traj)
        N = len(trees)
        # one line per tree
        tree_lines = [ax.plot([], [], color="gray", lw=1)[0] for _ in range(N)]
        (traj_line,) = ax.plot([], [], color="blue", lw=2)
        (traj_point,) = ax.plot([], [], "ro", ms=5)

        def init():
            for ln in tree_lines:
                ln.set_data([], [])
            traj_line.set_data([], [])
            traj_point.set_data([], [])
            return tree_lines + [traj_line, traj_point]

        def update(i):
            for j, tr in enumerate(trees):
                pts = tr[i]
                tree_lines[j].set_data(pts[:, 0], pts[:, 1])
            traj_line.set_data(traj[: i + 1, 0], traj[: i + 1, 1])
            traj_point.set_data([traj[i, 0]], [traj[i, 1]])
            return tree_lines + [traj_line, traj_point]

        anim = FuncAnimation(
            fig,
            update,
            frames=T,
            init_func=init,
            blit=True,
            interval=100,
        )
        if filename:
            anim.save(filename, fps=30, extra_args=["-vcodec", "libx264"])
        else:
            plt.show()

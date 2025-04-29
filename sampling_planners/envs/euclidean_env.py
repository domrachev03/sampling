from typing import Sequence
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .base_env import BaseEnv


class EuclidObstacleShapes(Enum):
    SPHERE = 1
    BOX = 2
    CAPSULE = 3


class EuclideanEnv(BaseEnv):
    def __init__(
        self,
        x_lim: tuple[float, float],
        y_lim: tuple[float, float],
        z_lim: tuple[float, float],
        obstacles_type: list[EuclidObstacleShapes],
        obstacles_data: list[np.ndarray],
        min_obstacle_distance: float = 0.0,
    ):
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.z_lim = z_lim
        self.obstacles_type = obstacles_type
        self.obstacles_data = obstacles_data
        self.min_obstacle_distance = min_obstacle_distance

    def _signed_distance(self, state: np.ndarray, i: int) -> float:
        t = self.obstacles_type[i]
        d = self.obstacles_data[i]
        match t:
            case EuclidObstacleShapes.SPHERE:
                center = d[:3]
                r = d[3]
                return np.linalg.norm(state - center) - r
            case EuclidObstacleShapes.BOX:
                x0, y0, z0, x1, y1, z1 = d
                px, py, pz = state
                dx = max(x0 - px, 0, px - x1)
                dy = max(y0 - py, 0, py - y1)
                dz = max(z0 - pz, 0, pz - z1)
                out = np.linalg.norm([dx, dy, dz])
                if dx == dy == dz == 0:
                    ins = min(px - x0, x1 - px, py - y0, y1 - py, pz - z0, z1 - pz)
                    return -ins
                return out
            case EuclidObstacleShapes.CAPSULE:
                a = d[:3]
                b = d[3:6]
                r = d[6]
                ab = b - a
                tproj = np.dot(state - a, ab) / np.dot(ab, ab)
                tproj = np.clip(tproj, 0, 1)
                closest = a + tproj * ab
                return np.linalg.norm(state - closest) - r
            case _:
                raise ValueError("Unknown shape")

    def is_state_valid(self, state: np.ndarray) -> bool:
        if not (
            self.x_lim[0] <= state[0] <= self.x_lim[1]
            and self.y_lim[0] <= state[1] <= self.y_lim[1]
            and self.z_lim[0] <= state[2] <= self.z_lim[1]
        ):
            return False
        for i in range(len(self.obstacles_type)):
            if self._signed_distance(state, i) < self.min_obstacle_distance:
                return False
        return True

    def distance(self, s1: np.ndarray, s2: np.ndarray) -> float:
        return np.linalg.norm(s1 - s2)

    def draw_random_state(self) -> np.ndarray:
        while True:
            pt = np.random.uniform(
                [self.x_lim[0], self.y_lim[0], self.z_lim[0]],
                [self.x_lim[1], self.y_lim[1], self.z_lim[1]],
            )
            if self.is_state_valid(pt):
                return pt

    def visualize(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        tree_nodes: Sequence[Sequence[np.ndarray]],
        tree_edges: Sequence[Sequence[tuple[int, int]]],
        highlighted_path: Sequence[Sequence[tuple[int, int]] | None] | None = None,
        filename: str = "",
    ):
        """Visualize the sampling tree and found trajectory, if any.

        This method creates a visualization of the environment, resulted tree in this trajectory, and
        the found path, if any.

        Args:
            start (np.ndarray): starting state of the trajectory.

            goal (np.ndarray): goal state of the trajectory.

            tree_nodes (Sequence[ Sequence[np.ndarray]]): nodes of the tree.
            Nodes sequence is a Sequence[Sequence[np.ndarray]]. Nodes are only added to the tree,
            hence each entry in the sequence represents only new nodes added to the tree during one iteration.
            The overall list of nodes in a tree is prefix sum of all lists in Sequence.
            Here, node is a point in 3D space.

            tree_edges (Sequence[tuple[int, int]]): edges of the trees.
            Each edge is a tuple of (node1, node2), and each iteration contains Sequence[tuple[int, int]].
            Unlike in the trajectory, the edges could dissappear from the tree between iterations.

            highlighted_path (Sequence[Sequence[int] | None] | None): hightlighted path in the tree.
            If highlighted path is present, it is a list of edges in the tree. Otherwise, it is None.
            Defaults to None, meaning no path highlighted.

            filename (str, optional): name of the file where to save the result of visualization.
            Defaults to "", meaning no visualization.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.set_zlim(self.z_lim)

        # draw static obstacles
        for t, d in zip(self.obstacles_type, self.obstacles_data):
            if t is EuclidObstacleShapes.SPHERE:
                c, r = d[:3], d[3]
                u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
                x = c[0] + r * np.cos(u) * np.sin(v)
                y = c[1] + r * np.sin(u) * np.sin(v)
                z = c[2] + r * np.cos(v)
                ax.plot_surface(x, y, z, color="k", alpha=0.3)
            elif t is EuclidObstacleShapes.BOX:
                x0, y0, z0, x1, y1, z1 = d
                # define 8 corners of the box
                C = np.array(
                    [
                        [x0, y0, z0],
                        [x1, y0, z0],
                        [x1, y1, z0],
                        [x0, y1, z0],
                        [x0, y0, z1],
                        [x1, y0, z1],
                        [x1, y1, z1],
                        [x0, y1, z1],
                    ]
                )
                # six faces as lists of four vertices
                faces = [
                    [C[0], C[1], C[2], C[3]],  # bottom
                    [C[4], C[5], C[6], C[7]],  # top
                    [C[0], C[1], C[5], C[4]],  # front
                    [C[2], C[3], C[7], C[6]],  # back
                    [C[1], C[2], C[6], C[5]],  # right
                    [C[0], C[3], C[7], C[4]],  # left
                ]
                box = Poly3DCollection(faces, facecolors="k", alpha=0.3)
                ax.add_collection3d(box)
            elif t is EuclidObstacleShapes.CAPSULE:
                a, b, r = d[:3], d[3:6], d[6]
                v = b - a
                L = np.linalg.norm(v)
                v_unit = v / L
                # build orthonormal basis {v_unit, n1, n2}
                arb = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(v_unit, arb)) > 0.9:
                    arb = np.array([0.0, 1.0, 0.0])
                n1 = np.cross(v_unit, arb)
                n1 /= np.linalg.norm(n1)
                n2 = np.cross(v_unit, n1)
                # cylinder surface
                theta = np.linspace(0, 2 * np.pi, 20)
                z_lin = np.linspace(0, L, 20)
                Θ, Zcyl = np.meshgrid(theta, z_lin)
                Xc = a[0] + v_unit[0] * Zcyl + r * np.cos(Θ) * n1[0] + r * np.sin(Θ) * n2[0]
                Yc = a[1] + v_unit[1] * Zcyl + r * np.cos(Θ) * n1[1] + r * np.sin(Θ) * n2[1]
                Zc = a[2] + v_unit[2] * Zcyl + r * np.cos(Θ) * n1[2] + r * np.sin(Θ) * n2[2]
                ax.plot_surface(Xc, Yc, Zc, color="k", alpha=0.3)
                # spherical end‐caps
                us, vs = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
                Xs1 = a[0] + r * np.cos(us) * np.sin(vs)
                Ys1 = a[1] + r * np.sin(us) * np.sin(vs)
                Zs1 = a[2] + r * np.cos(vs)
                ax.plot_surface(Xs1, Ys1, Zs1, color="k", alpha=0.3)
                Xs2 = b[0] + r * np.cos(us) * np.sin(vs)
                Ys2 = b[1] + r * np.sin(us) * np.sin(vs)
                Zs2 = b[2] + r * np.cos(vs)
                ax.plot_surface(Xs2, Ys2, Zs2, color="k", alpha=0.3)

        # now single tree
        T = len(tree_nodes)
        traj = np.linspace(start, goal, T)

        tree_line = ax.plot([], [], [], "o", color="gray", lw=1)[0]
        edge_lines: list = []
        highlight_lines: list = []
        (traj_line,) = ax.plot([], [], [], "b-", lw=2)
        (traj_point,) = ax.plot([], [], [], "ro")

        def init():
            tree_line.set_data([], [])
            tree_line.set_3d_properties([])
            for ln in edge_lines + highlight_lines:
                ln.remove()
            edge_lines.clear()
            highlight_lines.clear()
            traj_line.set_data([], [])
            traj_line.set_3d_properties([])
            traj_point.set_data([], [])
            traj_point.set_3d_properties([])
            return [tree_line, traj_line, traj_point]

        def update(i):
            # concatenate all nodes up to i
            nodes_i = np.vstack(tree_nodes[: i + 1])
            tree_line.set_data(nodes_i[:, 0], nodes_i[:, 1])
            tree_line.set_3d_properties(nodes_i[:, 2])

            # clear old edges
            for ln in edge_lines + highlight_lines:
                ln.remove()
            edge_lines.clear()
            highlight_lines.clear()

            # draw all edges at iteration i
            for u, v in tree_edges[i]:
                segment = nodes_i[[u, v], :]
                print(segment)
                ln = ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="gray", lw=1)[0]
                edge_lines.append(ln)
            # optional highlight
            if highlighted_path and highlighted_path[i]:
                for u, v in highlighted_path[i]:
                    segment = nodes_i[[u, v], :]
                    hl = ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="red", lw=2)[0]
                    highlight_lines.append(hl)
            print("-" * 50)
            # update traj
            traj_line.set_data(traj[: i + 1, 0], traj[: i + 1, 1])
            traj_line.set_3d_properties(traj[: i + 1, 2])
            traj_point.set_data([traj[i, 0]], [traj[i, 1]])
            traj_point.set_3d_properties([traj[i, 2]])

            return [tree_line, traj_line, traj_point] + edge_lines + highlight_lines

        anim = FuncAnimation(fig, update, frames=T, init_func=init, blit=False, interval=100)
        if filename:
            anim.save(filename, fps=30, extra_args=["-vcodec", "libx264"])
        else:
            plt.show()

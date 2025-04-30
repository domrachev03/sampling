import heapq  # new import
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from sampling_planners.envs.base_env import BaseEnv
from time import perf_counter


class BaseTreePlanner(ABC):
    def __init__(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        env: BaseEnv,
        sol_threshold: float = 1e-3,
        extend_step: float = 0.1,
    ):
        self.env = env
        self.start = start
        self.goal = goal
        self.sol_threshold = sol_threshold
        self.extend_step = extend_step

        # Initialize the tree
        self.reset()

    def reset(self):
        self.tree_nodes: np.ndarray = self.start.reshape(1, -1)  # Shape (n_nodes, env_dim)
        self.tree_edges: np.ndarray = np.empty(
            (0, 2), dtype=np.int32
        )  # Integers (n_edges, 2) -- first node, second node

        self.tree_nodes_history: list[np.ndarray] = [self.start.copy()]  # List of arrays (n_nodes, env_dim)
        self.tree_edges_history: list[np.ndarray] = [self.tree_edges.copy()]  # # List of arrays (n_edges, 2)
        self.solution_history: list[np.ndarray | None] = [None]  # List of arrays (n_nodes, env_dim)

        self.distance_from_start: np.ndarray = np.zeros(1)
        self.parent_node: np.ndarray = -np.ones(1, dtype=int)

        self.min_dist = np.inf
        self.opt_path = None
        self.n_steps = 0

    def step(self) -> bool:
        n_nodes_last = self.n_nodes

        candidates = self.step_body()
        if self.n_nodes == n_nodes_last:
            return self.min_dist != np.inf
        self.n_steps += 1

        candidates = candidates if candidates is not None else np.arange(self.n_nodes)
        sol_nodes = [node for node in candidates if self.is_solution(node)]
        if len(sol_nodes) != 0:
            closest_sol_node = sol_nodes[np.argmin(self.distance_from_start[sol_nodes])]
            path_to_sol = [closest_sol_node]
            cur_node = closest_sol_node.copy()
            while self.parent_node[cur_node] != -1:
                path_to_sol.append(self.parent_node[cur_node])
                cur_node = self.parent_node[cur_node]
            path = path_to_sol[::-1]
            self.opt_path = np.array(list(zip(path[:-1], path[1:])), dtype=int)
            self.min_dist = self.distance_from_start[closest_sol_node].copy()

        sol_found = self.min_dist != np.inf

        if sol_found:
            self.solution_history.append(self.opt_path.copy())
        else:
            self.solution_history.append(None)

        self.tree_nodes_history.append(self.tree_nodes[n_nodes_last:])
        self.tree_edges_history.append(self.tree_edges.copy())

        return sol_found

    @abstractmethod
    def step_body(self) -> np.ndarray | None:
        pass

    def sample_state(self) -> np.ndarray:
        return self.env.draw_random_state()

    def add_nodes(self, new_states: np.ndarray) -> np.ndarray:
        new_nodes = np.arange(self.n_nodes, self.n_nodes + len(new_states))
        self.tree_nodes = np.vstack((self.tree_nodes, new_states))
        return new_nodes

    def add_edges(self, new_edges: Sequence[tuple[int, int]] | np.ndarray) -> None:
        new_edges = np.array(new_edges, dtype=np.int32)
        # Sort the edges to ensure nodes are in increasing order
        new_edges = np.sort(new_edges, axis=1)
        self.tree_edges = np.vstack((self.tree_edges, new_edges))

    def remove_edges(self, edges: Sequence[tuple[int, int]] | np.ndarray) -> None:
        edges = np.array(edges, dtype=np.int32)
        # Sort the edges to ensure nodes are in increasing order
        edges = np.sort(edges, axis=1)
        # normalize input to a set of tuples for fast lookup
        edges_to_remove = {tuple(e) for e in edges.tolist()}

        # filter out any edge that appears in edges_to_remove
        filtered = [edge for edge in self.tree_edges if tuple(edge) not in edges_to_remove]
        self.tree_edges = np.array(filtered, dtype=self.tree_edges.dtype)

    def is_solution(self, node: int) -> bool:
        state = self.tree_nodes[node]
        return self.env.distance(state, self.goal) < self.sol_threshold

    def visualize(self, filename: str = "", show: bool = True):
        return self.env.visualize(
            self.start,
            self.goal,
            self.tree_nodes_history,
            self.tree_edges_history,
            self.solution_history,
            filename=filename,
            show=show,
        )

    def nearest_node(self, state: np.ndarray) -> int:
        distances = np.linalg.norm(self.tree_nodes - state, axis=1)
        return np.argmin(distances)

    def find_near(self, state: np.ndarray, radius: float) -> np.ndarray:
        distances = np.linalg.norm(self.tree_nodes - state, axis=1)
        return np.argwhere(distances < radius).ravel()

    def extend(self, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
        # move from s1 toward s2 in fixed‐length steps
        direction = s2 - s1
        dist = np.linalg.norm(direction)
        if dist == 0 or self.extend_step <= 0:
            return np.empty((0, s1.shape[0]))
        unit_dir = direction / dist
        new_states = []
        step_dist = 0
        while step_dist < dist:
            step_dist += self.extend_step
            if step_dist > dist:
                step_dist = dist
            state = s1 + unit_dir * step_dist
            if not self.env.is_state_valid(state):
                break
            new_states.append(state)
        return np.array(new_states)

    @property
    def n_nodes(self) -> int:
        return len(self.tree_nodes)

    @property
    def n_edges(self) -> int:
        return len(self.tree_edges)

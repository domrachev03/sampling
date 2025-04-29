import heapq  # new import
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from sampling_planners.envs.base_env import BaseEnv


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

        self.min_dist = np.inf
        self.opt_path = None
        self.n_steps = 0

        # adjacency: list of (neighbor, weight) lists
        self.adj: list[list[tuple[int, float]]] = [[]]

    @abstractmethod
    def step(self) -> bool:
        self.n_steps += 1

    @abstractmethod
    def extend(self, state: np.ndarray) -> np.ndarray:
        pass

    def sample_state(self) -> np.ndarray:
        return self.env.draw_random_state()

    def add_nodes(self, new_states: np.ndarray) -> np.ndarray:
        new_nodes = np.arange(len(self.tree_nodes), len(self.tree_nodes) + len(new_states))
        self.tree_nodes = np.vstack((self.tree_nodes, new_states))
        self.tree_nodes_history.append(new_states.copy())

        # extend adjacency list with empty neighbors
        for _ in new_nodes:
            self.adj.append([])

        return new_nodes

    def add_edges(self, new_edges: np.ndarray) -> None:
        self.tree_edges = np.vstack((self.tree_edges, new_edges))
        self.tree_edges_history.append(new_edges.copy())

        # update adjacency incrementally
        for u, v in new_edges.tolist():
            w = self.env.distance(self.tree_nodes[u], self.tree_nodes[v])
            self.adj[u].append((v, w))
            self.adj[v].append((u, w))

    def is_solution(self, node: int) -> bool:
        state = self.tree_nodes[node]
        return self.env.distance(state, self.goal) < self.sol_threshold

    def check_solution(self, nodes: np.ndarray | None = None) -> bool:
        nodes = nodes if nodes is None else np.arange(len(self.tree_nodes))
        sol_nodes = [node for node in nodes if self.is_solution(node)]
        if len(sol_nodes) != 0:
            costs, paths = self.djikstra(0, sol_nodes)
            opt_sol = np.argmin(costs)
            opt_dist = costs[opt_sol]
            opt_path = paths[opt_sol]
            if self.min_dist > opt_dist:
                self.min_dist = opt_dist
                self.opt_path = opt_path
            self.solution_history.append(opt_path)
            return True
        else:
            self.solution_history.append(None)
            return False

    def visualize(self, filename: str = ""):
        self.env.visualize(
            self.start,
            self.goal,
            self.tree_nodes_history,
            self.tree_edges_history,
            self.solution_history,
            filename=filename,
        )

    def nearest_node(self, state: np.ndarray) -> int:
        distances = np.linalg.norm(self.tree_nodes - state, axis=1)
        return np.argmin(distances)

    def djikstra(self, node1: int, node2: Sequence[int] | int) -> tuple[np.ndarray, list[np.ndarray]]:
        """Depth-first search to find the smallest path between one node and a set of other nodes.

        This implementation uses a set to keep track of unvisited nodes, and returns both shortest distances
        and paths as arrays of edge tuples (u,v).

        Args:
            node1 (int): starting node
            node2 (Sequence[int] | int): target nodes

        Returns:
            tuple[np.ndarray, list[np.ndarray]]: costs for each target and list of edge‐arrays
        """
        # normalize targets
        targets = [node2] if isinstance(node2, int) else list(node2)

        # use prebuilt adjacency
        n = len(self.tree_nodes)
        adj = self.adj

        # lists for faster indexing
        dist: list[float] = [np.inf] * n
        prev: list[int | None] = [None] * n
        dist[node1] = 0.0

        to_find = set(targets)
        heap = [(0.0, node1)]
        while heap and to_find:
            cd, u = heapq.heappop(heap)
            if cd > dist[u]:
                continue
            if u in to_find:
                to_find.remove(u)
            for v, w in adj[u]:
                nd = cd + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

        # collect results
        costs = np.array([dist[t] for t in targets])
        paths: list[np.ndarray] = []
        for t in targets:
            # reconstruct node sequence
            seq = []
            curr = t
            while curr is not None:
                seq.append(curr)
                curr = prev[curr]
            seq.reverse()
            # build edges array
            if len(seq) < 2:
                paths.append(np.empty((0, 2), dtype=int))
            else:
                paths.append(np.array(list(zip(seq[:-1], seq[1:])), dtype=int))
        return costs, paths

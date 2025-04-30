import numpy as np

from sampling_planners.envs.base_env import BaseEnv
from sampling_planners.planners.base_planner import BaseTreePlanner


class RrtStarPlanner(BaseTreePlanner):
    def __init__(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        env: BaseEnv,
        sol_threshold: float = 1e-3,
        extend_step: float = 0.1,
        sigma: float = 2.0,
    ):
        super().__init__(start, goal, env, sol_threshold, extend_step=extend_step)
        self.sigma = sigma

        self.distance_from_start: np.ndarray = np.zeros(1)
        self.parent_node: np.ndarray = -np.ones(1, dtype=int)
        self.iter_count = 0

    def step_body(self) -> bool:
        self.iter_count += 1
        x_rand = self.sample_state()
        nodes_near = self.near(x_rand)
        closest_parent_node, sigma_min = self.choose_parent(nodes_near, x_rand)
        if closest_parent_node == -1:
            return self.min_dist != np.inf
        new_node = self.add_nodes([x_rand]).item()
        self.distance_from_start = np.concatenate([self.distance_from_start, [sigma_min]])
        self.parent_node = np.concatenate([self.parent_node, [closest_parent_node]])

        self.add_edges([(closest_parent_node, new_node)])
        self.rewire(new_node, nodes_near, closest_parent_node)

        sol_nodes = [node for node in np.arange(self.n_nodes) if self.is_solution(node)]
        if len(sol_nodes) == 0:
            return False
        closest_sol_node = sol_nodes[np.argmin(self.distance_from_start[sol_nodes])]
        path_to_sol = [closest_sol_node]
        while self.parent_node[closest_sol_node] != -1:
            path_to_sol.append(self.parent_node[closest_sol_node])
            closest_sol_node = self.parent_node[closest_sol_node]
        path = path_to_sol[::-1]
        self.opt_path = np.array(list(zip(path[:-1], path[1:])), dtype=int)
        self.min_dist = self.distance_from_start[closest_sol_node].copy()
        return True

    def near(self, new_state: np.ndarray) -> np.ndarray:
        r = self.sigma * (np.log(self.n_nodes) / self.n_nodes) ** (1 / self.env.dim) if self.n_nodes > 1 else np.inf
        # r = np.inf
        nodes_near = self.find_near(new_state, r)
        return nodes_near

    def choose_parent(self, neighbor_nodes: np.ndarray, new_state: np.ndarray) -> tuple[int, float]:
        min_cost = np.inf
        closest_parent = -1
        for node in neighbor_nodes:
            dist = self.env.distance(new_state, self.tree_nodes[node])
            if not self.check_collision_free(self.tree_nodes[node], new_state):
                continue
            cost = self.distance_from_start[node] + dist
            if cost < min_cost:
                min_cost = cost
                closest_parent = node
        return closest_parent, min_cost

    def rewire(self, new_node: np.ndarray, neighbor_nodes: np.ndarray, new_node_parent: int):
        new_state = self.tree_nodes[new_node]
        for node in neighbor_nodes:
            if node == new_node_parent or not self.check_collision_free(self.tree_nodes[node], new_state):
                continue
            cost = self.distance_from_start[new_node] + self.env.distance(self.tree_nodes[node], new_state)
            if cost < self.distance_from_start[node]:
                self.distance_from_start[node] = cost
                self.add_edges([(node, new_node)])
                self.remove_edges([(self.parent_node[node], node)])
                self.parent_node[node] = new_node

    def check_collision_free(self, s1: np.ndarray, s2: np.ndarray) -> bool:
        path = self.extend(s1, s2)
        if len(path) == 0 or (path[-1] - s2 > 1e-3).any():
            return False
        return True

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
        sigma: float = 0.1,
    ):
        super().__init__(start, goal, env, sol_threshold, extend_step=extend_step)
        self.sigma = sigma

    def step(self) -> bool:
        super().step()

        new_state = self.sample_state()
        near_states = self.near(new_state)
        x_min = self.choose_parent(new_state, near_states)
        if x_min == -1:
            return False

        new_node = self.add_nodes(new_state.reshape(1, -1))
        self.add_edges(np.array([(x_min, new_node[0])]))

        self.rewire(new_state, near_states)
        return self.check_solution()

    def near(self, new_state: np.ndarray) -> np.ndarray:
        radius = self.sigma * (np.log(self.n_nodes) / self.n_nodes) ** (1 / self.env.dim)
        return self.find_near(new_state, radius)

    def choose_parent(self, new_state: np.ndarray, neighbors: np.ndarray) -> int:
        """
        Choose the parent node for the new state based on the minimum cost.
        """

        min_cost: float = np.inf
        best_neighbor: int = -1
        for neighbor in neighbors:
            if self.env.check_collision(new_state, self.tree_nodes[neighbor]):
                continue
            new_cost = self.distance_from_start[neighbor] + self.env.distance(self.tree_nodes[neighbor], new_state)
            if new_cost < min_cost:
                if not self.check_collision_free(self.tree_nodes[neighbor], new_state):
                    continue
                min_cost = new_cost
                best_neighbor = neighbor
        return best_neighbor

    def rewire(self, new_state: np.ndarray, neighbors: np.ndarray):
        for neighbor in neighbors:
            if not self.check_collision_free(self.tree_nodes[neighbor], new_state):
                continue
            d = self.env.distance(self.tree_nodes[neighbor], new_state)
            if self.distance_from_start[neighbor] > self.distance_from_start[new_state] + d:
                parent = self.parent_node[neighbor]
                self.remove_edges(np.array([(parent, neighbor)]))
                self.add_edges(np.array([(new_state, neighbor)]))

    def check_collision_free(self, s1: np.ndarray, s2: np.ndarray) -> bool:
        """
        Check if the path between s1 and s2 is collision-free.
        """
        rollout = self.extend(self.tree_nodes[s1], s2)
        is_rollout_valid = True
        for rlt in rollout:
            if not self.env.is_state_valid(rlt):
                is_rollout_valid = False
                break

        return is_rollout_valid
